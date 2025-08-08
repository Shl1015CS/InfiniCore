#include "topk_router_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../../../utils/check.h"
#include "../../../../utils/result.hpp"
#include "../../../../utils/custom_types.h"
#include <algorithm>
#include <vector>
#include <cmath>
#include <queue>

namespace op::topk_router::cpu {

struct Descriptor::Opaque {
    // CPU实现不需要额外的不透明数据
};

Descriptor::~Descriptor() {
    delete _opaque;
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
    
    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    
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

// DeepSeek v3 style TopkRouter implementation
template<typename T>
static void topk_router_group_masked_topk_indices(
    const T* scores,                  // after sigmoid
    const T* logits_raw_or_null,      // pre-sigmoid logits for tie-break (can be nullptr)
    int* topk_indices,
    size_t n_routed_experts,
    int n_group,
    int topk_group,
    int top_k) {
    const size_t experts_per_group = n_routed_experts / n_group;

    // 1) 每组计算前2名之和
    std::vector<T> group_scores(n_group);
    for (int g = 0; g < n_group; ++g) {
        T best1 = static_cast<T>(-1e30);
        T best2 = static_cast<T>(-1e30);
        const size_t base = static_cast<size_t>(g) * experts_per_group;
        for (size_t e = 0; e < experts_per_group; ++e) {
            T v = scores[base + e];
            if (v > best1) { best2 = best1; best1 = v; }
            else if (v > best2) { best2 = v; }
        }
        group_scores[g] = best1 + best2;
    }
    
    // 2) 选择 topk_group 个组（部分选择 + 可选排序）
    std::vector<int> group_indices(n_group);
    for (int i = 0; i < n_group; ++i) group_indices[i] = i;
    // 常规排序逻辑（与 torch 行为更接近：同分时组 id 较大者优先）
    auto cmp_group = [&](int a, int b) {
        if (group_scores[a] == group_scores[b]) return a > b; // tie: higher group id first
        return group_scores[a] > group_scores[b];              // higher score first
    };
    if (topk_group < n_group) {
        std::partial_sort(group_indices.begin(), group_indices.begin() + topk_group, group_indices.end(), cmp_group);
        group_indices.resize(topk_group);
    } else {
        std::sort(group_indices.begin(), group_indices.end(), cmp_group);
    }

    // 3) 全局 top_k on masked scores（严格模拟 PyTorch：对未选中组的分数置零/极小值，再在全体 expert 上选 top_k）
    using Pair = std::pair<T, int>;
    std::vector<bool> group_selected(n_group, false);
    for (int i = 0; i < (int)group_indices.size(); ++i) group_selected[group_indices[i]] = true;

    std::vector<Pair> candidates;
    candidates.reserve(n_routed_experts);
    for (size_t e = 0; e < n_routed_experts; ++e) {
        int g = (int)(e / experts_per_group);
        T v = group_selected[g] ? scores[e] : (T)0; // 与参考保持一致：未选中组置0
        candidates.emplace_back(v, (int)e);
    }
    
    // 全局 top-k：同分时用 pre-sigmoid logits 打破平局，再以索引升序作为最终兜底
    auto cmp_desc_score_tie_logits_then_idx = [&](const Pair& a, const Pair& b) {
        if (a.first == b.first) {
            if (logits_raw_or_null != nullptr) {
                // 仅比较被选中组内的 raw logits；未选中组为 0，不影响
                T la = logits_raw_or_null[a.second];
                T lb = logits_raw_or_null[b.second];
                if (la != lb) return la > lb; // higher raw logit first
            }
            return a.second < b.second; // finally: lower index first
        }
        return a.first > b.first; // higher score first
    };
    const int used_k = std::min(top_k, (int)candidates.size());
    std::partial_sort(candidates.begin(), candidates.begin() + used_k, candidates.end(), cmp_desc_score_tie_logits_then_idx);
    for (int i = 0; i < used_k; ++i) topk_indices[i] = candidates[i].second;
}

template<typename T>
static void topk_router_sigmoid(const T* input, T* output, size_t size) {
    #pragma omp parallel for if (size > 4096)
    for (size_t i = 0; i < size; ++i) {
        double x = static_cast<double>(input[i]);
        output[i] = static_cast<T>(1.0 / (1.0 + std::exp(-x)));
    }
}

template<typename T>
static void topk_router_matmul_expert_logits(
    const T* input,    // [batch_size * seq_len, hidden_size]
    const T* weight,   // [n_routed_experts, hidden_size] (DeepSeek v3 layout)
    T* output,         // [batch_size * seq_len, n_routed_experts]
    size_t batch_seq,
    size_t hidden_size,
    size_t n_routed_experts) {
    #pragma omp parallel for collapse(2) if ((long)(batch_seq * n_routed_experts) > 512)
    for (size_t bs = 0; bs < batch_seq; ++bs) {
        for (size_t e = 0; e < n_routed_experts; ++e) {
            T sum = 0;
            const size_t input_base = bs * hidden_size;
            const size_t weight_base = e * hidden_size;
            #pragma omp simd reduction(+:sum)
            for (size_t h = 0; h < hidden_size; ++h) {
                sum += input[input_base + h] * weight[weight_base + h];
            }
            output[bs * n_routed_experts + e] = sum;
        }
    }
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
    
    // 根据数据类型分发
    switch (_dtype) {
        case INFINI_DTYPE_F32: {
            auto router_logits = static_cast<float*>(workspace);
            
            // Step 1: 计算router logits (矩阵乘法)
            // input: [batch_size, seq_len, hidden_size] -> reshape to [batch_size*seq_len, hidden_size]
            // weight: [n_routed_experts, hidden_size]
            // router_logits: [batch_size*seq_len, n_routed_experts]
            
            size_t batch_seq = _info.batch_size * _info.seq_len;
            
            topk_router_matmul_expert_logits(
                static_cast<const float*>(input),
                static_cast<const float*>(w_gate),
                router_logits,
                batch_seq,
                _info.hidden_size,
                _info.n_routed_experts
            );
            // Step 2: 对每个token进行分组掩码 top-k（逐行计算 sigmoid，保留 raw logits 作为并列打分依据）
            auto indices_ptr = static_cast<int*>(topk_indices);
            auto weights_ptr = static_cast<float*>(topk_weights);
            
            #pragma omp parallel for if ((long)batch_seq > 4)
            for (long bs = 0; bs < (long)batch_seq; ++bs) {
                size_t logits_offset = bs * _info.n_routed_experts;
                size_t output_offset = bs * _info.top_k;
                
                // 为该行计算 sigmoid 分数，同时保留原始 logits 作为并列打分依据
                std::vector<float> scores_row(_info.n_routed_experts);
                for (size_t e = 0; e < _info.n_routed_experts; ++e) {
                    float x = router_logits[logits_offset + e];
                    scores_row[e] = 1.0f / (1.0f + std::exp(-x));
                }

                // 获取top-k indices（并列时优先 raw logits 更大者，其次索引小者）
                topk_router_group_masked_topk_indices(
                    scores_row.data(),                 // scores after sigmoid
                    &router_logits[logits_offset],     // raw logits for tie-break
                    &indices_ptr[output_offset],
                    _info.n_routed_experts,
                    _info.n_group,
                    _info.topk_group,
                    _info.top_k);
                
                // 收集对应的权重
                float weight_sum = 0.0f;
                for (int k = 0; k < _info.top_k; ++k) {
                    int expert_idx = indices_ptr[output_offset + k];
                    weights_ptr[output_offset + k] = scores_row[(size_t)expert_idx];
                    weight_sum += weights_ptr[output_offset + k];
                }
                
                // 归一化权重（如果启用）
                if (_info.norm_topk_prob && weight_sum > 1e-20f) {
                    for (int k = 0; k < _info.top_k; ++k) {
                        weights_ptr[output_offset + k] /= weight_sum;
                    }
                }
                
                // 应用scaling factor
                for (int k = 0; k < _info.top_k; ++k) {
                    weights_ptr[output_offset + k] *= _info.routed_scaling_factor;
                }
            }
            break;
        }
        case INFINI_DTYPE_F16: {
            size_t batch_seq = _info.batch_size * _info.seq_len;
            
            // 使用float进行计算，然后转换回fp16
            std::vector<float> temp_logits(batch_seq * _info.n_routed_experts);
            
            // Step 1: 矩阵乘法 (转换为float计算，OMP)
            #pragma omp parallel for collapse(2) if ((long)(batch_seq * _info.n_routed_experts) > 512)
            for (long bs = 0; bs < (long)batch_seq; ++bs) {
                for (long e = 0; e < (long)_info.n_routed_experts; ++e) {
                    float sum = 0.0f;
                    const size_t input_base = (size_t)bs * _info.hidden_size;
                    const size_t weight_base = (size_t)e * _info.hidden_size;
                    #pragma omp simd reduction(+:sum)
                    for (size_t h = 0; h < _info.hidden_size; ++h) {
                        float input_val = utils::cast<float>(static_cast<const fp16_t*>(input)[input_base + h]);
                        float weight_val = utils::cast<float>(static_cast<const fp16_t*>(w_gate)[weight_base + h]);
                        sum += input_val * weight_val;
                    }
                    temp_logits[(size_t)bs * _info.n_routed_experts + (size_t)e] = sum;
                }
            }
            
            // Step 2: Top-k选择和权重计算（逐行计算 sigmoid，保留 raw logits 作为并列打分依据）
            auto indices_ptr = static_cast<int*>(topk_indices);
            auto weights_ptr = static_cast<fp16_t*>(topk_weights);
            
            #pragma omp parallel for if ((long)batch_seq > 4)
            for (long bs = 0; bs < (long)batch_seq; ++bs) {
                size_t logits_offset = (size_t)bs * _info.n_routed_experts;
                size_t output_offset = (size_t)bs * _info.top_k;
                
                std::vector<float> scores_row(_info.n_routed_experts);
                for (size_t e = 0; e < _info.n_routed_experts; ++e) {
                    float x = temp_logits[logits_offset + e];
                    scores_row[e] = 1.0f / (1.0f + std::exp(-x));
                }

                topk_router_group_masked_topk_indices(
                    scores_row.data(),              // scores after sigmoid
                    &temp_logits[logits_offset],    // raw logits for tie-break
                    &indices_ptr[output_offset],
                    _info.n_routed_experts,
                    _info.n_group,
                    _info.topk_group,
                    _info.top_k
                );
                
                // 收集对应的权重并转换回fp16
                float weight_sum = 0.0f;
                std::vector<float> temp_weights(_info.top_k);
                for (int k = 0; k < _info.top_k; ++k) {
                    int expert_idx = indices_ptr[output_offset + k];
                    temp_weights[k] = scores_row[(size_t)expert_idx];
                    weight_sum += temp_weights[k];
                }
                
                if (_info.norm_topk_prob && weight_sum > 1e-20f) {
                    for (int k = 0; k < _info.top_k; ++k) temp_weights[k] /= weight_sum;
                }
                for (int k = 0; k < _info.top_k; ++k) {
                    temp_weights[k] *= _info.routed_scaling_factor;
                    weights_ptr[output_offset + k] = utils::cast<fp16_t>(temp_weights[k]);
                }
            }
            break;
        }
        case INFINI_DTYPE_BF16: {
            size_t batch_seq = _info.batch_size * _info.seq_len;
            
            // 使用float进行计算，然后转换回bf16
            std::vector<float> temp_logits(batch_seq * _info.n_routed_experts);
            
            // Step 1: 矩阵乘法 (转换为float计算，OMP)
            #pragma omp parallel for collapse(2) if ((long)(batch_seq * _info.n_routed_experts) > 512)
            for (long bs = 0; bs < (long)batch_seq; ++bs) {
                for (long e = 0; e < (long)_info.n_routed_experts; ++e) {
                    float sum = 0.0f;
                    const size_t input_base = (size_t)bs * _info.hidden_size;
                    const size_t weight_base = (size_t)e * _info.hidden_size;
                    #pragma omp simd reduction(+:sum)
                    for (size_t h = 0; h < _info.hidden_size; ++h) {
                        float input_val = utils::cast<float>(static_cast<const bf16_t*>(input)[input_base + h]);
                        float weight_val = utils::cast<float>(static_cast<const bf16_t*>(w_gate)[weight_base + h]);
                        sum += input_val * weight_val;
                    }
                    temp_logits[(size_t)bs * _info.n_routed_experts + (size_t)e] = sum;
                }
            }
            
            // Step 2: Top-k选择和权重计算（逐行计算 sigmoid，保留 raw logits 作为并列打分依据）
            auto indices_ptr = static_cast<int*>(topk_indices);
            auto weights_ptr = static_cast<bf16_t*>(topk_weights);
            
            #pragma omp parallel for if ((long)batch_seq > 4)
            for (long bs = 0; bs < (long)batch_seq; ++bs) {
                size_t logits_offset = (size_t)bs * _info.n_routed_experts;
                size_t output_offset = (size_t)bs * _info.top_k;
                
                std::vector<float> scores_row(_info.n_routed_experts);
                for (size_t e = 0; e < _info.n_routed_experts; ++e) {
                    float x = temp_logits[logits_offset + e];
                    scores_row[e] = 1.0f / (1.0f + std::exp(-x));
                }

                topk_router_group_masked_topk_indices(
                    scores_row.data(),              // scores after sigmoid
                    &temp_logits[logits_offset],    // raw logits for tie-break
                    &indices_ptr[output_offset],
                    _info.n_routed_experts,
                    _info.n_group,
                    _info.topk_group,
                    _info.top_k
                );
                
                // 收集对应的权重并转换回bf16
                float weight_sum = 0.0f;
                std::vector<float> temp_weights(_info.top_k);
                for (int k = 0; k < _info.top_k; ++k) {
                    int expert_idx = indices_ptr[output_offset + k];
                    temp_weights[k] = scores_row[(size_t)expert_idx];
                    weight_sum += temp_weights[k];
                }
                
                if (_info.norm_topk_prob && weight_sum > 1e-20f) {
                    for (int k = 0; k < _info.top_k; ++k) temp_weights[k] /= weight_sum;
                }
                for (int k = 0; k < _info.top_k; ++k) {
                    temp_weights[k] *= _info.routed_scaling_factor;
                    weights_ptr[output_offset + k] = utils::cast<bf16_t>(temp_weights[k]);
                }
            }
            break;
        }
        default:
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::topk_router::cpu