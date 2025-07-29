#include "topk_router_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../../../utils/check.h"
#include "../../../../utils/result.hpp"
#include "../../../../utils/custom_types.h"
#include <algorithm>
#include <vector>
#include <cmath>

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
static void deepseek_v3_get_topk_indices(
    const T* scores,
    int* topk_indices,
    size_t n_routed_experts,
    int n_group,
    int topk_group,
    int top_k) {
    
    size_t experts_per_group = n_routed_experts / n_group;
    
    // Step 1: 计算每个group的得分 (取每组前2名的和)
    std::vector<T> group_scores(n_group, 0.0);
    for (int g = 0; g < n_group; ++g) {
        std::vector<std::pair<T, int>> group_experts;
        for (size_t e = 0; e < experts_per_group; ++e) {
            size_t expert_idx = g * experts_per_group + e;
            group_experts.emplace_back(scores[expert_idx], expert_idx);
        }
        
        // 取前2名
        std::partial_sort(group_experts.begin(), group_experts.begin() + 2, group_experts.end(),
                         [](const auto& a, const auto& b) { return a.first > b.first; });
        
        group_scores[g] = group_experts[0].first + group_experts[1].first;
    }
    
    // Step 2: 选择top_k个group
    std::vector<std::pair<T, int>> group_pairs;
    for (int g = 0; g < n_group; ++g) {
        group_pairs.emplace_back(group_scores[g], g);
    }
    std::partial_sort(group_pairs.begin(), group_pairs.begin() + topk_group, group_pairs.end(),
                     [](const auto& a, const auto& b) { return a.first > b.first; });
    
    // Step 3: 创建group mask
    std::vector<bool> group_mask(n_group, false);
    for (int i = 0; i < topk_group; ++i) {
        group_mask[group_pairs[i].second] = true;
    }
    
    // Step 4: 基于group mask筛选experts并进行最终的top-k选择
    std::vector<std::pair<T, int>> valid_experts;
    for (size_t e = 0; e < n_routed_experts; ++e) {
        int group_id = e / experts_per_group;
        if (group_mask[group_id]) {
            valid_experts.emplace_back(scores[e], e);
        }
    }
    
    // 从有效的experts中选择top_k
    std::partial_sort(valid_experts.begin(), 
                     valid_experts.begin() + std::min(top_k, (int)valid_experts.size()),
                     valid_experts.end(),
                     [](const auto& a, const auto& b) { return a.first > b.first; });
    
    // 输出结果
    for (int i = 0; i < top_k && i < static_cast<int>(valid_experts.size()); ++i) {
        topk_indices[i] = valid_experts[i].second;
    }
}

template<typename T>
static void sigmoid_activation(const T* input, T* output, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        output[i] = static_cast<T>(1.0 / (1.0 + std::exp(-static_cast<double>(input[i]))));
    }
}

template<typename T>
static void matrix_multiply_deepseek_v3(
    const T* input,    // [batch_size * seq_len, hidden_size]
    const T* weight,   // [n_routed_experts, hidden_size] (DeepSeek v3 layout)
    T* output,         // [batch_size * seq_len, n_routed_experts]
    size_t batch_seq,
    size_t hidden_size,
    size_t n_routed_experts) {
    
    for (size_t bs = 0; bs < batch_seq; ++bs) {
        for (size_t e = 0; e < n_routed_experts; ++e) {
            T sum = 0;
            for (size_t h = 0; h < hidden_size; ++h) {
                size_t input_idx = bs * hidden_size + h;
                size_t weight_idx = e * hidden_size + h;
                sum += input[input_idx] * weight[weight_idx];
            }
            size_t output_idx = bs * n_routed_experts + e;
            output[output_idx] = sum;
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
            
            matrix_multiply_deepseek_v3(
                static_cast<const float*>(input),
                static_cast<const float*>(w_gate),
                router_logits,
                batch_seq,
                _info.hidden_size,
                _info.n_routed_experts
            );
            
            // Step 2: 应用sigmoid激活函数
            sigmoid_activation(router_logits, router_logits, batch_seq * _info.n_routed_experts);
            
            // Step 3: 对每个token进行DeepSeek v3 style top-k选择
            auto indices_ptr = static_cast<int*>(topk_indices);
            auto weights_ptr = static_cast<float*>(topk_weights);
            
            for (size_t bs = 0; bs < batch_seq; ++bs) {
                size_t logits_offset = bs * _info.n_routed_experts;
                size_t output_offset = bs * _info.top_k;
                
                // 获取top-k indices
                deepseek_v3_get_topk_indices(
                    &router_logits[logits_offset],
                    &indices_ptr[output_offset],
                    _info.n_routed_experts,
                    _info.n_group,
                    _info.topk_group,
                    _info.top_k
                );
                
                // 收集对应的权重
                float weight_sum = 0.0f;
                for (int k = 0; k < _info.top_k; ++k) {
                    int expert_idx = indices_ptr[output_offset + k];
                    weights_ptr[output_offset + k] = router_logits[logits_offset + expert_idx];
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
            
            // Step 1: 矩阵乘法 (转换为float计算)
            for (size_t bs = 0; bs < batch_seq; ++bs) {
                for (size_t e = 0; e < _info.n_routed_experts; ++e) {
                    float sum = 0;
                    for (size_t h = 0; h < _info.hidden_size; ++h) {
                        size_t input_idx = bs * _info.hidden_size + h;
                        size_t weight_idx = e * _info.hidden_size + h;
                        float input_val = utils::cast<float>(static_cast<const fp16_t*>(input)[input_idx]);
                        float weight_val = utils::cast<float>(static_cast<const fp16_t*>(w_gate)[weight_idx]);
                        sum += input_val * weight_val;
                    }
                    temp_logits[bs * _info.n_routed_experts + e] = sum;
                }
            }
            
            // Step 2: 应用sigmoid激活函数
            sigmoid_activation(temp_logits.data(), temp_logits.data(), batch_seq * _info.n_routed_experts);
            
            // Step 3: Top-k选择和权重计算
            auto indices_ptr = static_cast<int*>(topk_indices);
            auto weights_ptr = static_cast<fp16_t*>(topk_weights);
            
            for (size_t bs = 0; bs < batch_seq; ++bs) {
                size_t logits_offset = bs * _info.n_routed_experts;
                size_t output_offset = bs * _info.top_k;
                
                deepseek_v3_get_topk_indices(
                    &temp_logits[logits_offset],
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
                    temp_weights[k] = temp_logits[logits_offset + expert_idx];
                    weight_sum += temp_weights[k];
                }
                
                if (_info.norm_topk_prob && weight_sum > 1e-20f) {
                    for (int k = 0; k < _info.top_k; ++k) {
                        temp_weights[k] /= weight_sum;
                    }
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
            
            // Step 1: 矩阵乘法 (转换为float计算)
            for (size_t bs = 0; bs < batch_seq; ++bs) {
                for (size_t e = 0; e < _info.n_routed_experts; ++e) {
                    float sum = 0;
                    for (size_t h = 0; h < _info.hidden_size; ++h) {
                        size_t input_idx = bs * _info.hidden_size + h;
                        size_t weight_idx = e * _info.hidden_size + h;
                        float input_val = utils::cast<float>(static_cast<const bf16_t*>(input)[input_idx]);
                        float weight_val = utils::cast<float>(static_cast<const bf16_t*>(w_gate)[weight_idx]);
                        sum += input_val * weight_val;
                    }
                    temp_logits[bs * _info.n_routed_experts + e] = sum;
                }
            }
            
            // Step 2: 应用sigmoid激活函数
            sigmoid_activation(temp_logits.data(), temp_logits.data(), batch_seq * _info.n_routed_experts);
            
            // Step 3: Top-k选择和权重计算
            auto indices_ptr = static_cast<int*>(topk_indices);
            auto weights_ptr = static_cast<bf16_t*>(topk_weights);
            
            for (size_t bs = 0; bs < batch_seq; ++bs) {
                size_t logits_offset = bs * _info.n_routed_experts;
                size_t output_offset = bs * _info.top_k;
                
                deepseek_v3_get_topk_indices(
                    &temp_logits[logits_offset],
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
                    temp_weights[k] = temp_logits[logits_offset + expert_idx];
                    weight_sum += temp_weights[k];
                }
                
                if (_info.norm_topk_prob && weight_sum > 1e-20f) {
                    for (int k = 0; k < _info.top_k; ++k) {
                        temp_weights[k] /= weight_sum;
                    }
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