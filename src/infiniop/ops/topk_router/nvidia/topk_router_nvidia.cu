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

// Sigmoid activation applied elementwise on router logits
template<typename T>
__device__ inline T sigmoid_device(T x) {
    if constexpr (std::is_same_v<T, float>) {
        return 1.0f / (1.0f + expf(-x));
    } else if constexpr (std::is_same_v<T, __half>) {
        float xf = __half2float(x);
        float yf = 1.0f / (1.0f + expf(-xf));
        return __float2half(yf);
    } else {
        // fallback
        float xf = static_cast<float>(x);
        float yf = 1.0f / (1.0f + expf(-xf));
        return static_cast<T>(yf);
    }
}

template<typename T>
__global__ void topk_router_apply_sigmoid_kernel(T* data, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = sigmoid_device<T>(data[idx]);
    }
}

// Group-masked TopK selection aligned with DeepSeek V3
// Steps per token:
// 1) compute per-group score as sum of top2 experts within group
// 2) select topk_group groups by group score (desc)
// 3) among experts of selected groups, select global top_k by score
// 4) write indices and weights (weights are already sigmoid probs); optionally normalize and scale
template<typename T>
__global__ void topk_router_group_masked_topk_kernel(
    const T* __restrict__ probs,  // already sigmoid'ed
    int* __restrict__ topk_indices,
    T* __restrict__ topk_weights,
    int batch_size,
    int seq_len,
    int n_routed_experts,
    int n_group,
    int topk_group,
    int top_k,
    float routed_scaling_factor,
    bool norm_topk_prob) {
    int b = blockIdx.x;
    int s = blockIdx.y;
    if (b >= batch_size || s >= seq_len) return;

    int token_offset = (b * seq_len + s) * n_routed_experts;
    const T* token_probs = probs + token_offset;
    int experts_per_group = n_routed_experts / n_group;

    extern __shared__ unsigned char smem[];
    T* group_scores = reinterpret_cast<T*>(smem);            // size n_group
    int* group_indices = reinterpret_cast<int*>(group_scores + n_group); // size n_group
    T* sel_vals = reinterpret_cast<T*>(group_indices + n_group);         // size top_k
    int* sel_idx = reinterpret_cast<int*>(sel_vals + top_k);             // size top_k

    // Each thread up to n_group computes its group's top2 sum
    if (threadIdx.x < n_group) {
        int g = threadIdx.x;
        T best1 = static_cast<T>(-1e30f);
        T best2 = static_cast<T>(-1e30f);
        int base = g * experts_per_group;
        for (int e = 0; e < experts_per_group; ++e) {
            T v = token_probs[base + e];
            if (v > best1) {
                best2 = best1;
                best1 = v;
            } else if (v > best2) {
                best2 = v;
            }
        }
        group_scores[g] = best1 + best2;
        group_indices[g] = g;
    }
    __syncthreads();
    
    // thread 0: select topk_group groups by simple partial selection (n_group is small typically)
    if (threadIdx.x == 0) {
        // partial sort by descending score
        for (int i = 0; i < topk_group; ++i) {
            int max_pos = i;
            for (int j = i + 1; j < n_group; ++j) {
                if (group_scores[j] > group_scores[max_pos]) max_pos = j;
            }
            if (max_pos != i) {
                T ts = group_scores[i]; group_scores[i] = group_scores[max_pos]; group_scores[max_pos] = ts;
                int ti = group_indices[i]; group_indices[i] = group_indices[max_pos]; group_indices[max_pos] = ti;
            }
        }

        // global top_k selection among experts within selected groups
        int used_k = top_k;
        for (int i = 0; i < used_k; ++i) { sel_idx[i] = -1; sel_vals[i] = static_cast<T>(-1e30f); }

        for (int gi = 0; gi < topk_group; ++gi) {
            int g = group_indices[gi];
            int base = g * experts_per_group;
            for (int e = 0; e < experts_per_group; ++e) {
                int expert_idx = base + e;
                T v = token_probs[expert_idx];
                // insert into top-k buffers
                int pos = used_k - 1;
                if (v > sel_vals[pos]) {
                    // find position
                    while (pos > 0 && v > sel_vals[pos - 1]) {
                        sel_vals[pos] = sel_vals[pos - 1];
                        sel_idx[pos] = sel_idx[pos - 1];
                        --pos;
                    }
                    sel_vals[pos] = v;
                    sel_idx[pos] = expert_idx;
                }
            }
        }

        // write outputs
        int out_offset = (b * seq_len + s) * top_k;
        T denom = static_cast<T>(0);
        for (int i = 0; i < used_k; ++i) {
            reinterpret_cast<int*>(topk_indices)[out_offset + i] = sel_idx[i];
            reinterpret_cast<T*>(topk_weights)[out_offset + i] = sel_vals[i];
            denom += sel_vals[i];
        }
        if (norm_topk_prob && denom > static_cast<T>(1e-20f)) {
            for (int i = 0; i < used_k; ++i) {
                reinterpret_cast<T*>(topk_weights)[out_offset + i] = reinterpret_cast<T*>(topk_weights)[out_offset + i] / denom;
            }
        }
        if (routed_scaling_factor != 1.0f) {
            for (int i = 0; i < used_k; ++i) {
                reinterpret_cast<T*>(topk_weights)[out_offset + i] = reinterpret_cast<T*>(topk_weights)[out_offset + i] * static_cast<T>(routed_scaling_factor);
            }
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
            auto logits = static_cast<float*>(workspace);
            // 1) GEMM: logits = input @ w_gate^T -> [B*S, E]
            const float alpha = 1.0f, beta = 0.0f;
            auto status = cublasSgemm(
                _opaque->cublas_handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                _info.n_routed_experts, _info.batch_size * _info.seq_len, _info.hidden_size,
                &alpha,
                static_cast<const float*>(w_gate), _info.hidden_size,
                static_cast<const float*>(input), _info.hidden_size,
                &beta,
                logits, _info.n_routed_experts
            );
            if (status != CUBLAS_STATUS_SUCCESS) return INFINI_STATUS_INTERNAL_ERROR;

            // 2) Sigmoid
            size_t total = _info.batch_size * _info.seq_len * _info.n_routed_experts;
            int threads = 256;
            int blocks = (total + threads - 1) / threads;
            topk_router_apply_sigmoid_kernel<float><<<blocks, threads, 0, cuda_stream>>>(logits, total);
            if (cudaGetLastError() != cudaSuccess) return INFINI_STATUS_INTERNAL_ERROR;

            // 3) Group-masked TopK per token
            dim3 grid(_info.batch_size, _info.seq_len);
            dim3 block(std::max(32, (int)_info.n_group));
            size_t smem = sizeof(float) * _info.n_group + sizeof(int) * _info.n_group + sizeof(float) * _info.top_k + sizeof(int) * _info.top_k;
            topk_router_group_masked_topk_kernel<float><<<grid, block, smem, cuda_stream>>>(
                logits,
                static_cast<int*>(topk_indices),
                static_cast<float*>(topk_weights),
                (int)_info.batch_size,
                (int)_info.seq_len,
                (int)_info.n_routed_experts,
                (int)_info.n_group,
                (int)_info.topk_group,
                (int)_info.top_k,
                _info.routed_scaling_factor,
                _info.norm_topk_prob
            );
            if (cudaGetLastError() != cudaSuccess) return INFINI_STATUS_INTERNAL_ERROR;
            break;
        }
        case INFINI_DTYPE_F16: {
            auto logits = static_cast<__half*>(workspace);
            const __half alpha = __float2half(1.0f), beta = __float2half(0.0f);
            auto status = cublasHgemm(
                _opaque->cublas_handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                _info.n_routed_experts, _info.batch_size * _info.seq_len, _info.hidden_size,
                &alpha,
                static_cast<const __half*>(w_gate), _info.hidden_size,
                static_cast<const __half*>(input), _info.hidden_size,
                &beta,
                logits, _info.n_routed_experts
            );
            if (status != CUBLAS_STATUS_SUCCESS) return INFINI_STATUS_INTERNAL_ERROR;

            size_t total = _info.batch_size * _info.seq_len * _info.n_routed_experts;
            int threads = 256;
            int blocks = (total + threads - 1) / threads;
            topk_router_apply_sigmoid_kernel<__half><<<blocks, threads, 0, cuda_stream>>>(logits, total);
            if (cudaGetLastError() != cudaSuccess) return INFINI_STATUS_INTERNAL_ERROR;
            
            dim3 grid(_info.batch_size, _info.seq_len);
            dim3 block(std::max(32, (int)_info.n_group));
            size_t smem = sizeof(__half) * _info.n_group + sizeof(int) * _info.n_group + sizeof(__half) * _info.top_k + sizeof(int) * _info.top_k;
            topk_router_group_masked_topk_kernel<__half><<<grid, block, smem, cuda_stream>>>(
                logits,
                static_cast<int*>(topk_indices),
                static_cast<__half*>(topk_weights),
                (int)_info.batch_size,
                (int)_info.seq_len,
                (int)_info.n_routed_experts,
                (int)_info.n_group,
                (int)_info.topk_group,
                (int)_info.top_k,
                _info.routed_scaling_factor,
                _info.norm_topk_prob
            );
            if (cudaGetLastError() != cudaSuccess) return INFINI_STATUS_INTERNAL_ERROR;
            break;
        }
        case INFINI_DTYPE_BF16:
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        default:
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::topk_router::nvidia