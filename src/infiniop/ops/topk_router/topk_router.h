#ifndef __TOPK_ROUTER_H__
#define __TOPK_ROUTER_H__

#include "../../operator.h"
#include "info.h"

/**
 * Top-K Router算子描述符宏定义
 * 
 * 该宏定义了一个 `Descriptor` 类，用于Top-K Router算子的描述和操作
 * Top-K Router是Deepseek v3模型MoE层的核心计算组件
 * 
 * 核心功能：
 * 1. gate_scores = input @ w_gate  # 矩阵乘法计算expert分数
 * 2. topk_indices, topk_weights = topk(gate_scores, k=topk)  # Top-k选择
 */

#define TOPK_ROUTER_DESCRIPTOR(NAMESPACE)                        \
                                                                 \
    namespace op::topk_router::NAMESPACE {                       \
    class Descriptor final : public InfiniopDescriptor {         \
        struct Opaque;                                           \
        Opaque *_opaque;                                         \
        infiniDtype_t _dtype;                                    \
        TopkRouterInfo _info;                                    \
        size_t _workspace_size;                                  \
                                                                 \
        Descriptor(                                              \
            infiniDtype_t dtype,                                 \
            TopkRouterInfo info,                                 \
            size_t workspace_size_,                              \
            Opaque *opaque,                                      \
            infiniDevice_t device_type,                          \
            int device_id)                                       \
            : InfiniopDescriptor{device_type, device_id},        \
              _opaque(opaque),                                   \
              _dtype(dtype),                                     \
              _info(info),                                       \
              _workspace_size(workspace_size_) {}                \
                                                                 \
    public:                                                      \
        ~Descriptor();                                           \
                                                                 \
        size_t workspaceSize() const { return _workspace_size; } \
                                                                 \
        static infiniStatus_t create(                            \
            infiniopHandle_t handle,                             \
            Descriptor **desc_ptr,                               \
            infiniopTensorDescriptor_t topk_indices_desc,        \
            infiniopTensorDescriptor_t topk_weights_desc,        \
            infiniopTensorDescriptor_t input_desc,               \
            infiniopTensorDescriptor_t w_gate_desc,              \
            int top_k,                                           \
            int n_group,                                         \
            int topk_group,                                      \
            float routed_scaling_factor,                         \
            bool norm_topk_prob);                                \
                                                                 \
        infiniStatus_t calculate(                                \
            void *workspace, size_t workspace_size,              \
            void *topk_indices,                                  \
            void *topk_weights,                                  \
            const void *input,                                   \
            const void *w_gate,                                  \
            void *stream) const;                                 \
    };                                                           \
    }

#endif // __TOPK_ROUTER_H__