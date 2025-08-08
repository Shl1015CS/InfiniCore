#include "topk_router_metax.h"
#include "../../../devices/metax/metax_common.h"
#include "../../../devices/metax/metax_handle.h"
#include "../../../../utils/check.h"
#include "../../../../utils/result.hpp"
#include "../../../../utils/custom_types.h"
#include <vector>
#include <cmath>
#include <algorithm>

namespace op::topk_router::metax {

struct Descriptor::Opaque {
    std::shared_ptr<device::metax::Handle::Internal> internal;
    // Metax特定的kernel配置
    hcdnnTensorDescriptor_t input_desc;
    hcdnnTensorDescriptor_t w_gate_desc;
    hcdnnTensorDescriptor_t logits_desc;
    hcdnnActivationDescriptor_t softmax_desc;
};

Descriptor::~Descriptor() {
    if (_opaque) {
        // 清理Metax描述符
        if (_opaque->input_desc) hcdnnDestroyTensorDescriptor(_opaque->input_desc);
        if (_opaque->w_gate_desc) hcdnnDestroyTensorDescriptor(_opaque->w_gate_desc);
        if (_opaque->logits_desc) hcdnnDestroyTensorDescriptor(_opaque->logits_desc);
        if (_opaque->softmax_desc) hcdnnDestroyActivationDescriptor(_opaque->softmax_desc);
        delete _opaque;
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
    
    auto handle = reinterpret_cast<device::metax::Handle *>(handle_);
    
    // 验证数据类型一致性
    auto dtype = input_desc->dtype();
    if (w_gate_desc->dtype() != dtype ||
        topk_weights_desc->dtype() != dtype) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    
    // Metax支持的数据类型
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);
    
    // 创建TopkRouterInfo
    auto info_result = TopkRouterInfo::create(
        topk_indices_desc, topk_weights_desc, input_desc, w_gate_desc, 
        top_k, n_group, topk_group, routed_scaling_factor, norm_topk_prob);
    
    CHECK_RESULT(info_result);
    
    auto info = info_result.take();
    
    // 计算工作空间大小
    size_t logits_size = info.batch_size * info.seq_len * info.n_routed_experts * 
                        infiniSizeOf(dtype);
    size_t sort_workspace = info.batch_size * info.seq_len * 
                           (info.n_routed_experts * sizeof(int) + // 索引
                            info.n_routed_experts * infiniSizeOf(dtype)); // 值
    
    size_t workspace_size = logits_size + sort_workspace;
    
    // 创建Metax描述符
    auto opaque = new Opaque{handle->internal()};
    
    hcdnnDataType_t hcdnn_dtype = device::metax::getHcdnnDtype(dtype);
    
    // 创建输入张量描述符
    CHECK_MCDNN(hcdnnCreateTensorDescriptor(&opaque->input_desc));
    CHECK_MCDNN(hcdnnSetTensorNdDescriptor(
        opaque->input_desc, hcdnn_dtype, 3,
        (int[]){(int)info.batch_size, (int)info.seq_len, (int)info.hidden_size},
        (int[]){(int)(info.seq_len * info.hidden_size), (int)info.hidden_size, 1}));
    
    // 创建权重张量描述符
    CHECK_MCDNN(hcdnnCreateTensorDescriptor(&opaque->w_gate_desc));
    CHECK_MCDNN(hcdnnSetTensorNdDescriptor(
        opaque->w_gate_desc, hcdnn_dtype, 2,
        (int[]){(int)info.n_routed_experts, (int)info.hidden_size},
        (int[]){(int)info.hidden_size, 1}));
    
    // 创建logits张量描述符
    CHECK_MCDNN(hcdnnCreateTensorDescriptor(&opaque->logits_desc));
    CHECK_MCDNN(hcdnnSetTensorNdDescriptor(
        opaque->logits_desc, hcdnn_dtype, 3,
        (int[]){(int)info.batch_size, (int)info.seq_len, (int)info.n_routed_experts},
        (int[]){(int)(info.seq_len * info.n_routed_experts), (int)info.n_routed_experts, 1}));
    
    // 创建softmax描述符
    CHECK_MCDNN(hcdnnCreateActivationDescriptor(&opaque->softmax_desc));
    CHECK_MCDNN(hcdnnSetActivationDescriptor(
        opaque->softmax_desc, HCDNN_SOFTMAX, HCDNN_NOT_PROPAGATE_NAN, 1.0));
    
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

// Sigmoid 激活
template<typename T>
__global__ void topk_router_metax_apply_sigmoid_kernel(T* data, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = static_cast<float>(data[idx]);
        float y = 1.0f / (1.0f + expf(-x));
        data[idx] = static_cast<T>(y);
    }
}

// DeepSeek v3 分组掩码 TopK：先选 group，再在被选 group 中全局选 top_k；按和归一并缩放
template<typename T>
__global__ void topk_router_metax_group_masked_topk_kernel(
    const T* __restrict__ probs,  // 已经 sigmoid 后的概率
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

    __shared__ T group_scores[64];
    __shared__ int group_indices[64];

    if (threadIdx.x < n_group) {
        int g = threadIdx.x;
        T best1 = (T)-1e30f, best2 = (T)-1e30f;
        int base = g * experts_per_group;
        for (int e = 0; e < experts_per_group; ++e) {
            T v = token_probs[base + e];
            if (v > best1) { best2 = best1; best1 = v; }
            else if (v > best2) { best2 = v; }
        }
        group_scores[g] = best1 + best2;
        group_indices[g] = g;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        // 选 topk_group 个 group
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

        const int MAX_TOPK = 64;
        int idx_buf[MAX_TOPK];
        T val_buf[MAX_TOPK];
        int used_k = top_k < MAX_TOPK ? top_k : MAX_TOPK;
        for (int i = 0; i < used_k; ++i) { idx_buf[i] = -1; val_buf[i] = (T)-1e30f; }

        // 在选中 group 内进行全局 top-k
        for (int gi = 0; gi < topk_group; ++gi) {
            int g = group_indices[gi];
            int base = g * experts_per_group;
            for (int e = 0; e < experts_per_group; ++e) {
                int expert_idx = base + e;
                T v = token_probs[expert_idx];
                int pos = used_k - 1;
                if (v > val_buf[pos]) {
                    while (pos > 0 && v > val_buf[pos - 1]) {
                        val_buf[pos] = val_buf[pos - 1];
                        idx_buf[pos] = idx_buf[pos - 1];
                        --pos;
                    }
                    val_buf[pos] = v;
                    idx_buf[pos] = expert_idx;
                }
            }
        }

        int out_offset = (b * seq_len + s) * top_k;
        T denom = (T)0;
        for (int i = 0; i < used_k; ++i) {
            topk_indices[out_offset + i] = idx_buf[i];
            topk_weights[out_offset + i] = val_buf[i];
            denom += val_buf[i];
        }
        if (norm_topk_prob && denom > (T)1e-20f) {
            for (int i = 0; i < used_k; ++i) {
                topk_weights[out_offset + i] = topk_weights[out_offset + i] / denom;
            }
        }
        if (routed_scaling_factor != 1.0f) {
            for (int i = 0; i < used_k; ++i) {
                topk_weights[out_offset + i] = topk_weights[out_offset + i] * (T)routed_scaling_factor;
            }
        }
    }
}

// Metax TopK Router主要实现函数
template<typename T>
static infiniStatus_t metax_topk_router_impl(
    void* workspace,
    size_t workspace_size,
    void* topk_indices_ptr,
    void* topk_weights_ptr,
    const void* input_ptr,
    const void* w_gate_ptr,
    const TopkRouterInfo& info,
    const Descriptor::Opaque* opaque,
    hcStream_t stream) {
    
    const T* input = static_cast<const T*>(input_ptr);
    const T* w_gate = static_cast<const T*>(w_gate_ptr);
    int* topk_indices = static_cast<int*>(topk_indices_ptr);
    T* topk_weights = static_cast<T*>(topk_weights_ptr);
    
    size_t logits_size = info.batch_size * info.seq_len * info.n_routed_experts * sizeof(T);
    T* router_logits = static_cast<T*>(workspace);
    
    // Step 1: 使用Metax BLAS进行矩阵乘法: input @ w_gate -> router_logits
    auto status = opaque->internal->useMcblas(stream, [&](hcblasHandle_t handle) -> infiniStatus_t {
        const T alpha = 1.0f;
        const T beta = 0.0f;
        
        hcblasStatus_t blas_status;
        if constexpr (std::is_same_v<T, float>) {
            blas_status = hcblasSgemm(
                handle, HCBLAS_OP_N, HCBLAS_OP_T,
                info.n_routed_experts, info.batch_size * info.seq_len, info.hidden_size,
                &alpha,
                w_gate, info.n_routed_experts,
                input, info.hidden_size,
                &beta,
                router_logits, info.n_routed_experts);
        } else if constexpr (std::is_same_v<T, __fp16>) {
            blas_status = hcblasHgemm(
                handle, HCBLAS_OP_N, HCBLAS_OP_T,
                info.n_routed_experts, info.batch_size * info.seq_len, info.hidden_size,
                reinterpret_cast<const __fp16*>(&alpha),
                reinterpret_cast<const __fp16*>(w_gate), info.n_routed_experts,
                reinterpret_cast<const __fp16*>(input), info.hidden_size,
                reinterpret_cast<const __fp16*>(&beta),
                reinterpret_cast<__fp16*>(router_logits), info.n_routed_experts);
        }
        
        CHECK_MCBLAS(blas_status);
        return INFINI_STATUS_SUCCESS;
    });
    
    if (status != INFINI_STATUS_SUCCESS) return status;
    
    // Step 2: 对router_logits做sigmoid
    size_t total = info.batch_size * info.seq_len * info.n_routed_experts;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    topk_router_metax_apply_sigmoid_kernel<T><<<blocks, threads, 0, stream>>>(router_logits, total);

    // Step 3: 启动分组掩码TopK kernel
    dim3 grid(info.batch_size, info.seq_len);
    dim3 block(std::max(32, std::min(info.n_group, 256))); // 至少32线程以便有足够并行
    topk_router_metax_group_masked_topk_kernel<T><<<grid, block, 0, stream>>>(
        router_logits,
        topk_indices,
        topk_weights,
        (int)info.batch_size,
        (int)info.seq_len,
        (int)info.n_routed_experts,
        (int)info.n_group,
        (int)info.topk_group,
        (int)info.top_k,
        info.routed_scaling_factor,
        info.norm_topk_prob
    );
    
    // 检查kernel执行错误
    hcError_t kernel_error = hcGetLastError();
    if (kernel_error != hcSuccess) {
        return INFINI_STATUS_EXECUTION_FAILED;
    }
    
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::launch(
    void* workspace,
    size_t workspace_size,
    void* topk_indices_ptr,
    void* topk_weights_ptr,
    const void* input_ptr,
    const void* w_gate_ptr,
    void* stream_) const {
    
    CHECK_WORKSPACE_SIZE(workspace_size, _workspace_size);
    
    auto stream = reinterpret_cast<hcStream_t>(stream_);
    
    switch (_dtype) {
        case INFINI_DTYPE_F32:
            return metax_topk_router_impl<float>(
                workspace, workspace_size,
                topk_indices_ptr, topk_weights_ptr,
                input_ptr, w_gate_ptr,
                _info, _opaque, stream);
        
        case INFINI_DTYPE_F16:
            return metax_topk_router_impl<__fp16>(
                workspace, workspace_size,
                topk_indices_ptr, topk_weights_ptr,
                input_ptr, w_gate_ptr,
                _info, _opaque, stream);
                
        case INFINI_DTYPE_BF16:
            return metax_topk_router_impl<bfloat16>(
                workspace, workspace_size,
                topk_indices_ptr, topk_weights_ptr,
                input_ptr, w_gate_ptr,
                _info, _opaque, stream);
        
        default:
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::topk_router::metax
