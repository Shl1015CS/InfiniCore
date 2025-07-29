#ifndef __TOPK_ROUTER_INFO_H__
#define __TOPK_ROUTER_INFO_H__

#include "../../../utils.h"
#include "../../operator.h"
#include "../../tensor.h"

namespace op::topk_router {

class TopkRouterInfo {
    TopkRouterInfo() = default;

public:
    size_t batch_size;
    size_t seq_len;
    size_t hidden_size;
    size_t n_routed_experts;
    int top_k;
    int n_group;
    int topk_group;
    float routed_scaling_factor;
    bool norm_topk_prob;
    
    static utils::Result<TopkRouterInfo> create(
        infiniopTensorDescriptor_t topk_indices_desc,
        infiniopTensorDescriptor_t topk_weights_desc,
        infiniopTensorDescriptor_t input_desc,
        infiniopTensorDescriptor_t w_gate_desc,
        int top_k_value,
        int n_group_value = 1,
        int topk_group_value = 1,
        float routed_scaling_factor_value = 1.0f,
        bool norm_topk_prob_value = true) {

        // Validate input tensor shape: [batch_size, seq_len, hidden_size]
        if (input_desc->ndim() != 3) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        
        // Validate w_gate tensor shape: [n_routed_experts, hidden_size] (注意：DeepSeek v3中是转置的)
        if (w_gate_desc->ndim() != 2) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        
        // Validate topk_indices tensor shape: [batch_size * seq_len, top_k]
        if (topk_indices_desc->ndim() != 2) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        
        // Validate topk_weights tensor shape: [batch_size * seq_len, top_k]
        if (topk_weights_desc->ndim() != 2) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        auto batch_size = input_desc->dim(0);
        auto seq_len = input_desc->dim(1);
        auto hidden_size = input_desc->dim(2);
        auto n_routed_experts = w_gate_desc->dim(0);

        // Validate dimensions consistency
        if (w_gate_desc->dim(1) != hidden_size) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        
        if (topk_indices_desc->dim(0) != batch_size * seq_len ||
            topk_indices_desc->dim(1) != static_cast<size_t>(top_k_value)) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        
        if (topk_weights_desc->dim(0) != batch_size * seq_len ||
            topk_weights_desc->dim(1) != static_cast<size_t>(top_k_value)) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        // Validate group parameters
        if (n_routed_experts % n_group_value != 0) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        return utils::Result<TopkRouterInfo>(TopkRouterInfo{
            batch_size,
            seq_len,
            hidden_size,
            n_routed_experts,
            top_k_value,
            n_group_value,
            topk_group_value,
            routed_scaling_factor_value,
            norm_topk_prob_value
        });
    }
};

} // namespace op::topk_router

#endif // __TOPK_ROUTER_INFO_H__