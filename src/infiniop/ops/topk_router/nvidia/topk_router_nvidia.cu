#include "topk_router_nvidia.cuh"
#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../../utils/check.h"
#include "../../../../utils/result.hpp"
#include "infinicore.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>
#include <cmath>
#include <algorithm>

namespace op::topk_router::nvidia {

struct Descriptor::Opaque {
    cublasHandle_t cublas_handle;
    
    Opaque() {
        cublasCreate(&cublas_handle);
    }
    
    ~Opaque() {
        cublasDestroy(cublas_handle);
    }
};

Descriptor::~Descriptor() {
    delete _opaque;
}

// CUDA kernel for top-k selection with softmax
template<typename T>
__global__ void topk_softmax_kernel(
    const T* gate_scores,
    int* topk_indices,
    T* topk_weights,
    size_t batch_size,
    size_t seq_len,
    size_t num_experts,
    int k) {
    
    int batch_idx = blockIdx.x;
    int seq_idx = blockIdx.y;
    
    if (batch_idx >= batch_size || seq_idx >= seq_len) return;
    
    extern __shared__ char shared_mem[];
    T* shared_scores = (T*)shared_mem;
    int* shared_indices = (int*)(shared_scores + num_experts);
    
    size_t gate_offset = batch_idx * seq_len * num_experts + seq_idx * num_experts;
    
    // 复制到共享内存并初始化索引
    for (int i = threadIdx.x; i < num_experts; i += blockDim.x) {
        shared_scores[i] = gate_scores[gate_offset + i];
        shared_indices[i] = i;
    }
    __syncthreads();
    
    // 简单的选择排序来找到top-k（对于小的num_experts）
    for (int i = 0; i < k && threadIdx.x == 0; ++i) {
        int max_idx = i;
        for (int j = i + 1; j < num_experts; ++j) {
            if (shared_scores[j] > shared_scores[max_idx]) {
                max_idx = j;
            }
        }
        if (max_idx != i) {
            T temp_score = shared_scores[i];
            int temp_idx = shared_indices[i];
            shared_scores[i] = shared_scores[max_idx];
            shared_indices[i] = shared_indices[max_idx];
            shared_scores[max_idx] = temp_score;
            shared_indices[max_idx] = temp_idx;
        }
    }
    __syncthreads();
    
    // 计算softmax
    if (threadIdx.x == 0) {
        T max_score = shared_scores[0];
        T sum_exp = 0;
        
        for (int i = 0; i < k; ++i) {
            T exp_val;
            if constexpr (std::is_same_v<T, float>) {
                exp_val = expf(shared_scores[i] - max_score);
            } else if constexpr (std::is_same_v<T, __half>) {
                exp_val = hexp(shared_scores[i] - max_score);
            } else {
                exp_val = exp(shared_scores[i] - max_score);
            }
            shared_scores[i] = exp_val;
            sum_exp += exp_val;
        }
        
        size_t output_offset = batch_idx * seq_len * k + seq_idx * k;
        for (int i = 0; i < k; ++i) {
            topk_indices[output_offset + i] = shared_indices[i];
            topk_weights[output_offset + i] = shared_scores[i] / sum_exp;
        }
    }
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t topk_indices_desc,
    infiniopTensorDescriptor_t topk_weights_desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t w_gate_desc,
    int top_k,
    int n_group,
    int topk_group,
    float routed_scaling_factor,
    bool norm_topk_prob) {
    
    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);
    
    // 验证数据类型一致性
    auto dtype = input_desc->dtype();
    if (w_gate_desc->dtype() != dtype ||
        topk_weights_desc->dtype() != dtype) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);
    
    // 创建TopkRouterInfo
    auto info_result = TopkRouterInfo::create(
        topk_indices_desc, topk_weights_desc, input_desc, w_gate_desc,
        top_k, n_group, topk_group, routed_scaling_factor, norm_topk_prob);
    
    CHECK_RESULT(info_result);
    
    auto info = info_result.take();
    
    // 计算工作空间大小（用于存储中间的router_logits）
    size_t workspace_size = info.batch_size * info.seq_len * info.n_routed_experts * 
                           infiniSizeOf(dtype);
    
    auto opaque = new Opaque{};
    
    *desc_ptr = new Descriptor(
        dtype,
        info,
        workspace_size,
        opaque,
        handle->device,
        handle->device_id
    );
    
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size,
    void *topk_indices,
    void *topk_weights,
    const void *input,
    const void *w_gate,
    void *stream) const {
    
    // 验证工作空间大小
    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }
    
    cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    cublasSetStream(_opaque->cublas_handle, cuda_stream);
    
    // 根据数据类型分发
    switch (_dtype) {
        case INFINI_DTYPE_F32: {
            auto gate_scores = static_cast<float*>(workspace);
            
            // 步骤1: 使用cuBLAS进行矩阵乘法
            // gate_scores = input @ w_gate
            // input: [batch_size * seq_len, hidden_size]
            // w_gate: [n_routed_experts, hidden_size]
            // gate_scores: [batch_size * seq_len, n_routed_experts]
            
            const float alpha = 1.0f, beta = 0.0f;
            auto status = cublasSgemm(
                _opaque->cublas_handle,
                CUBLAS_OP_T, CUBLAS_OP_N,  // 转置w_gate来匹配DeepSeek v3布局
                _info.n_routed_experts, _info.batch_size * _info.seq_len, _info.hidden_size,
                &alpha,
                static_cast<const float*>(w_gate), _info.hidden_size,
                static_cast<const float*>(input), _info.hidden_size,
                &beta,
                gate_scores, _info.n_routed_experts
            );
            
            if (status != CUBLAS_STATUS_SUCCESS) {
                return INFINI_STATUS_INTERNAL_ERROR;
            }
            
            // 步骤2: 启动top-k kernel
            dim3 grid(_info.batch_size, _info.seq_len);
            dim3 block(min(256, (int)_info.n_routed_experts));
            
            size_t shared_mem_size = (sizeof(float) + sizeof(int)) * _info.n_routed_experts;
            
            // Check for potential overflow
            if (shared_mem_size > 48 * 1024) {
                return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
            }
            
            topk_softmax_kernel<<<grid, block, shared_mem_size, cuda_stream>>>(
                gate_scores,
                static_cast<int*>(topk_indices),
                static_cast<float*>(topk_weights),
                _info.batch_size,
                _info.seq_len,
                _info.n_routed_experts,
                _info.top_k
            );
            
            cudaError_t cuda_error = cudaGetLastError();
            if (cuda_error != cudaSuccess) {
                return INFINI_STATUS_INTERNAL_ERROR;
            }
            
            break;
        }
        case INFINI_DTYPE_F16: {
            auto gate_scores = static_cast<__half*>(workspace);
            
            const __half alpha = __float2half(1.0f), beta = __float2half(0.0f);
            auto status = cublasHgemm(
                _opaque->cublas_handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                _info.n_routed_experts, _info.batch_size * _info.seq_len, _info.hidden_size,
                &alpha,
                static_cast<const __half*>(w_gate), _info.hidden_size,
                static_cast<const __half*>(input), _info.hidden_size,
                &beta,
                gate_scores, _info.n_routed_experts
            );
            
            if (status != CUBLAS_STATUS_SUCCESS) {
                return INFINI_STATUS_INTERNAL_ERROR;
            }
            
            dim3 grid(_info.batch_size, _info.seq_len);
            dim3 block(min(256, (int)_info.n_routed_experts));
            
            size_t shared_mem_size = (sizeof(__half) + sizeof(int)) * _info.n_routed_experts;
            
            if (shared_mem_size > 48 * 1024) {
                return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
            }
            
            topk_softmax_kernel<<<grid, block, shared_mem_size, cuda_stream>>>(
                gate_scores,
                static_cast<int*>(topk_indices),
                static_cast<__half*>(topk_weights),
                _info.batch_size,
                _info.seq_len,
                _info.n_routed_experts,
                _info.top_k
            );
            
            cudaError_t cuda_error = cudaGetLastError();
            if (cuda_error != cudaSuccess) {
                return INFINI_STATUS_INTERNAL_ERROR;
            }
            
            break;
        }
        case INFINI_DTYPE_BF16:
            // BF16 support requires specific CUDA version and hardware
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        default:
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::topk_router::nvidia