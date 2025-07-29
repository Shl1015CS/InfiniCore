#ifndef __INFINIOP_TOPK_ROUTER_API_H__
#define __INFINIOP_TOPK_ROUTER_API_H__

#include "../operator_descriptor.h"

typedef struct InfiniopDescriptor *infiniopTopkRouterDescriptor_t;

__C __export infiniStatus_t infiniopCreateTopkRouterDescriptor(infiniopHandle_t handle,
                                                               infiniopTopkRouterDescriptor_t *desc_ptr,
                                                               infiniopTensorDescriptor_t topk_indices_desc,
                                                               infiniopTensorDescriptor_t topk_weights_desc,
                                                               infiniopTensorDescriptor_t input_desc,
                                                               infiniopTensorDescriptor_t w_gate_desc,
                                                               int top_k,
                                                               int n_group,
                                                               int topk_group,
                                                               float routed_scaling_factor,
                                                               bool norm_topk_prob);

__C __export infiniStatus_t infiniopGetTopkRouterWorkspaceSize(infiniopTopkRouterDescriptor_t desc, size_t *size);

__C __export infiniStatus_t infiniopTopkRouter(infiniopTopkRouterDescriptor_t desc,
                                               void *workspace,
                                               size_t workspace_size,
                                               void *topk_indices,
                                               void *topk_weights,
                                               const void *input,
                                               const void *w_gate,
                                               void *stream);

__C __export infiniStatus_t infiniopDestroyTopkRouterDescriptor(infiniopTopkRouterDescriptor_t desc);

#endif