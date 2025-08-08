// TopK Router implementation for Iluvatar (天数) GPU
// 天数GPU与CUDA兼容，复用NVIDIA的CUDA kernel但使用独立的描述符管理

#include "topk_router_iluvatar.h"
#include "../../../../utils/check.h"
#include "../../../../utils/result.hpp"
#include "../../../../utils/custom_types.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <algorithm>

#ifdef INFINI_USE_ILUVATAR

namespace op::topk_router::iluvatar {

struct Descriptor::Opaque {
    cublasHandle_t cublas_handle;
    Opaque() { cublasCreate(&cublas_handle); }
    ~Opaque() { cublasDestroy(cublas_handle); }
};

// 内联内核：Sigmoid 与 分组掩码 TopK（与 NVIDIA 实现一致）
template<typename T>
__device__ inline T ilu_sigmoid_dev(T x) {
    if constexpr (std::is_same_v<T, float>) return 1.0f / (1.0f + expf(-x));
    else { float xf = static_cast<float>(x); return static_cast<T>(1.0f / (1.0f + expf(-xf))); }
}

template<typename T>
__global__ void ilu_apply_sigmoid(T* data, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x; if (idx < n) data[idx] = ilu_sigmoid_dev<T>(data[idx]);
}

template<typename T>
__global__ void ilu_group_masked_topk(
    const T* __restrict__ probs,
    int* __restrict__ topk_indices,
    T* __restrict__ topk_weights,
    int B,int S,int E,int G,int KG,int K,float scaling,bool norm_prob) {
    int b = blockIdx.x, s = blockIdx.y; if (b>=B||s>=S) return; int exp_per_g = E / G;
    extern __shared__ unsigned char smem[];
    T* g_scores = reinterpret_cast<T*>(smem); int* g_idx = reinterpret_cast<int*>(g_scores + G);
    T* sel_vals = reinterpret_cast<T*>(g_idx + G); int* sel_idx = reinterpret_cast<int*>(sel_vals + K);
    const T* token = probs + (b*S + s) * E;
    if (threadIdx.x < G) {
        int g = threadIdx.x; T a = (T)-1e30f, b2 = (T)-1e30f; int base = g*exp_per_g;
        for (int e=0;e<exp_per_g;++e){T v=token[base+e]; if(v>a){b2=a;a=v;} else if(v>b2){b2=v;}}
        g_scores[g] = a + b2; g_idx[g]=g;
    }
    __syncthreads();
    if (threadIdx.x==0){
        for(int i=0;i<KG;++i){int m=i;for(int j=i+1;j<G;++j) if(g_scores[j]>g_scores[m]) m=j; if(m!=i){T ts=g_scores[i];g_scores[i]=g_scores[m];g_scores[m]=ts; int ti=g_idx[i];g_idx[i]=g_idx[m];g_idx[m]=ti;}}
        for(int i=0;i<K;++i){sel_idx[i]=-1; sel_vals[i]=(T)-1e30f;}
        for(int gi=0;gi<KG;++gi){int g=g_idx[gi]; int base=g*exp_per_g; for(int e=0;e<exp_per_g;++e){int ex=base+e; T v=token[ex]; int pos=K-1; if(v>sel_vals[pos]){while(pos>0 && v>sel_vals[pos-1]){sel_vals[pos]=sel_vals[pos-1]; sel_idx[pos]=sel_idx[pos-1]; --pos;} sel_vals[pos]=v; sel_idx[pos]=ex;}}}
        int out = (b*S + s)*K; T denom = (T)0; for(int i=0;i<K;++i){ topk_indices[out+i]=sel_idx[i]; topk_weights[out+i]=sel_vals[i]; denom+=sel_vals[i]; }
        if (norm_prob && denom>(T)1e-20f) for(int i=0;i<K;++i) topk_weights[out+i]=topk_weights[out+i]/denom;
        if (scaling!=1.0f) for(int i=0;i<K;++i) topk_weights[out+i]=topk_weights[out+i]*(T)scaling;
    }
}

Descriptor::~Descriptor() { delete _opaque; }

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
    
    // 天数GPU支持的数据类型（与CUDA兼容）
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);
    
    // 创建TopkRouterInfo
    auto info_result = TopkRouterInfo::create(
        topk_indices_desc, topk_weights_desc, input_desc, w_gate_desc, 
        top_k, n_group, topk_group, routed_scaling_factor, norm_topk_prob);
    
    CHECK_RESULT(info_result);
    
    auto info = info_result.take();
    
    // 计算工作空间大小（router logits）
    size_t logits_size = info.batch_size * info.seq_len * info.n_routed_experts * 
                        infiniSizeOf(dtype);
    size_t workspace_size = logits_size;
    
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

infiniStatus_t Descriptor::launch(
    void* workspace,
    size_t workspace_size,
    void* topk_indices_ptr,
    void* topk_weights_ptr,
    const void* input_ptr,
    const void* w_gate_ptr,
    void* stream_) const {
    
    CHECK_WORKSPACE_SIZE(workspace_size, _workspace_size);
    
    auto stream = reinterpret_cast<cudaStream_t>(stream_);
    cublasSetStream(_opaque->cublas_handle, stream);

    switch (_dtype) {
        case INFINI_DTYPE_F32: {
            auto logits = static_cast<float*>(workspace);
            const float alpha = 1.0f, beta = 0.0f;
            auto st = cublasSgemm(
                _opaque->cublas_handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                _info.n_routed_experts, _info.batch_size * _info.seq_len, _info.hidden_size,
                &alpha,
                static_cast<const float*>(w_gate_ptr), _info.hidden_size,
                static_cast<const float*>(input_ptr), _info.hidden_size,
                &beta,
                logits, _info.n_routed_experts);
            if (st != CUBLAS_STATUS_SUCCESS) return INFINI_STATUS_INTERNAL_ERROR;
            size_t total = _info.batch_size * _info.seq_len * _info.n_routed_experts;
            int threads = 256; int blocks = (total + threads - 1) / threads;
            ilu_apply_sigmoid<float><<<blocks, threads, 0, stream>>>(logits, total);
            if (cudaGetLastError() != cudaSuccess) return INFINI_STATUS_INTERNAL_ERROR;
            dim3 grid(_info.batch_size, _info.seq_len); dim3 block(std::max(32, (int)_info.n_group));
            size_t smem = sizeof(float)*_info.n_group + sizeof(int)*_info.n_group + sizeof(float)*_info.top_k + sizeof(int)*_info.top_k;
            ilu_group_masked_topk<float><<<grid, block, smem, stream>>>(
                logits,
                static_cast<int*>(topk_indices_ptr),
                static_cast<float*>(topk_weights_ptr),
                (int)_info.batch_size,(int)_info.seq_len,(int)_info.n_routed_experts,
                (int)_info.n_group,(int)_info.topk_group,(int)_info.top_k,
                _info.routed_scaling_factor,_info.norm_topk_prob);
            if (cudaGetLastError() != cudaSuccess) return INFINI_STATUS_INTERNAL_ERROR;
            return INFINI_STATUS_SUCCESS;
        }
        case INFINI_DTYPE_F16: {
            auto logits = static_cast<__half*>(workspace);
            const __half alpha = __float2half(1.0f), beta = __float2half(0.0f);
            auto st = cublasHgemm(
                _opaque->cublas_handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                _info.n_routed_experts, _info.batch_size * _info.seq_len, _info.hidden_size,
                &alpha,
                static_cast<const __half*>(w_gate_ptr), _info.hidden_size,
                static_cast<const __half*>(input_ptr), _info.hidden_size,
                &beta,
                logits, _info.n_routed_experts);
            if (st != CUBLAS_STATUS_SUCCESS) return INFINI_STATUS_INTERNAL_ERROR;
            size_t total = _info.batch_size * _info.seq_len * _info.n_routed_experts;
            int threads = 256; int blocks = (total + threads - 1) / threads;
            ilu_apply_sigmoid<__half><<<blocks, threads, 0, stream>>>(logits, total);
            if (cudaGetLastError() != cudaSuccess) return INFINI_STATUS_INTERNAL_ERROR;
            dim3 grid(_info.batch_size, _info.seq_len); dim3 block(std::max(32, (int)_info.n_group));
            size_t smem = sizeof(__half)*_info.n_group + sizeof(int)*_info.n_group + sizeof(__half)*_info.top_k + sizeof(int)*_info.top_k;
            ilu_group_masked_topk<__half><<<grid, block, smem, stream>>>(
                logits,
                static_cast<int*>(topk_indices_ptr),
                static_cast<__half*>(topk_weights_ptr),
                (int)_info.batch_size,(int)_info.seq_len,(int)_info.n_routed_experts,
                (int)_info.n_group,(int)_info.topk_group,(int)_info.top_k,
                _info.routed_scaling_factor,_info.norm_topk_prob);
            if (cudaGetLastError() != cudaSuccess) return INFINI_STATUS_INTERNAL_ERROR;
            return INFINI_STATUS_SUCCESS;
        }
        case INFINI_DTYPE_BF16:
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        default:
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::topk_router::iluvatar

#endif // INFINI_USE_ILUVATAR
