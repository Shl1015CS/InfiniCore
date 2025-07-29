#include "../../operator.h"
#include "../../handle.h"
#include "infiniop/ops/topk_router.h"

#ifdef ENABLE_CPU_API
#include "cpu/topk_router_cpu.h"
#endif
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_ILUVATAR_API)
#include "nvidia/topk_router_nvidia.cuh"
#endif
#ifdef ENABLE_CAMBRICON_API
#include "bang/topk_router_bang.h"
#endif
#ifdef ENABLE_ASCEND_API
#include "ascend/topk_router_ascend.h"
#endif
#ifdef ENABLE_METAX_API
#include "metax/topk_router_metax.h"
#endif
#ifdef ENABLE_MOORE_API
#include "musa/topk_router_musa.h"
#endif
#ifdef ENABLE_KUNLUN_API
#include "kunlun/topk_router_kunlun.h"
#endif

__C infiniStatus_t infiniopCreateTopkRouterDescriptor(
    infiniopHandle_t handle,
    infiniopTopkRouterDescriptor_t *desc_ptr,
    infiniopTensorDescriptor_t topk_indices_desc,
    infiniopTensorDescriptor_t topk_weights_desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t w_gate_desc,
    int top_k,
    int n_group,
    int topk_group,
    float routed_scaling_factor,
    bool norm_topk_prob) {

#define CREATE(CASE, NAMESPACE)                                                    \
    case CASE:                                                                     \
        return op::topk_router::NAMESPACE::Descriptor::create(                     \
            handle,                                                                \
            reinterpret_cast<op::topk_router::NAMESPACE::Descriptor **>(desc_ptr), \
            topk_indices_desc,                                                     \
            topk_weights_desc,                                                     \
            input_desc,                                                            \
            w_gate_desc,                                                           \
            top_k,                                                                 \
            n_group,                                                               \
            topk_group,                                                            \
            routed_scaling_factor,                                                 \
            norm_topk_prob)

    switch (handle->device) {

#ifdef ENABLE_CPU_API
        CREATE(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_NVIDIA_API
        CREATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        CREATE(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_CAMBRICON_API
        CREATE(INFINI_DEVICE_CAMBRICON, bang);
#endif
#ifdef ENABLE_ASCEND_API
        CREATE(INFINI_DEVICE_ASCEND, ascend);
#endif
#ifdef ENABLE_METAX_API
        CREATE(INFINI_DEVICE_METAX, metax);
#endif
#ifdef ENABLE_MOORE_API
        CREATE(INFINI_DEVICE_MOORE, musa);
#endif
#ifdef ENABLE_KUNLUN_API
        CREATE(INFINI_DEVICE_KUNLUN, kunlun);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CREATE
}

__C infiniStatus_t
infiniopGetTopkRouterWorkspaceSize(
    infiniopTopkRouterDescriptor_t desc,
    size_t *size) {

#define GET(CASE, NAMESPACE)                                                                             \
    case CASE:                                                                                           \
        *size = reinterpret_cast<const op::topk_router::NAMESPACE::Descriptor *>(desc)->workspaceSize(); \
        return INFINI_STATUS_SUCCESS

    switch (desc->device_type) {

#ifdef ENABLE_CPU_API
        GET(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_NVIDIA_API
        GET(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        GET(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_CAMBRICON_API
        GET(INFINI_DEVICE_CAMBRICON, bang);
#endif
#ifdef ENABLE_ASCEND_API
        GET(INFINI_DEVICE_ASCEND, ascend);
#endif
#ifdef ENABLE_METAX_API
        GET(INFINI_DEVICE_METAX, metax);
#endif
#ifdef ENABLE_MOORE_API
        GET(INFINI_DEVICE_MOORE, musa);
#endif
#ifdef ENABLE_KUNLUN_API
        GET(INFINI_DEVICE_KUNLUN, kunlun);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef GET
}

__C infiniStatus_t infiniopTopkRouter(
    infiniopTopkRouterDescriptor_t desc,
    void *workspace, size_t workspace_size,
    void *topk_indices,
    void *topk_weights,
    const void *input,
    const void *w_gate,
    void *stream) {

#define CALCULATE(CASE, NAMESPACE)                                                    \
    case CASE:                                                                        \
        return reinterpret_cast<const op::topk_router::NAMESPACE::Descriptor *>(desc) \
            ->calculate(workspace, workspace_size,                                    \
                        topk_indices, topk_weights,                                   \
                        input, w_gate,                                                \
                        stream)

    switch (desc->device_type) {

#ifdef ENABLE_CPU_API
        CALCULATE(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_NVIDIA_API
        CALCULATE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        CALCULATE(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_CAMBRICON_API
        CALCULATE(INFINI_DEVICE_CAMBRICON, bang);
#endif
#ifdef ENABLE_ASCEND_API
        CALCULATE(INFINI_DEVICE_ASCEND, ascend);
#endif
#ifdef ENABLE_METAX_API
        CALCULATE(INFINI_DEVICE_METAX, metax);
#endif
#ifdef ENABLE_MOORE_API
        CALCULATE(INFINI_DEVICE_MOORE, musa);
#endif
#ifdef ENABLE_KUNLUN_API
        CALCULATE(INFINI_DEVICE_KUNLUN, kunlun);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef CALCULATE
}

__C infiniStatus_t
infiniopDestroyTopkRouterDescriptor(infiniopTopkRouterDescriptor_t desc) {

#define DELETE(CASE, NAMESPACE)                                                        \
    case CASE:                                                                         \
        delete reinterpret_cast<const op::topk_router::NAMESPACE::Descriptor *>(desc); \
        return INFINI_STATUS_SUCCESS;

    switch (desc->device_type) {

#ifdef ENABLE_CPU_API
        DELETE(INFINI_DEVICE_CPU, cpu);
#endif
#ifdef ENABLE_NVIDIA_API
        DELETE(INFINI_DEVICE_NVIDIA, nvidia);
#endif
#ifdef ENABLE_ILUVATAR_API
        DELETE(INFINI_DEVICE_ILUVATAR, nvidia);
#endif
#ifdef ENABLE_CAMBRICON_API
        DELETE(INFINI_DEVICE_CAMBRICON, bang);
#endif
#ifdef ENABLE_ASCEND_API
        DELETE(INFINI_DEVICE_ASCEND, ascend);
#endif
#ifdef ENABLE_METAX_API
        DELETE(INFINI_DEVICE_METAX, metax);
#endif
#ifdef ENABLE_MOORE_API
        DELETE(INFINI_DEVICE_MOORE, musa);
#endif
#ifdef ENABLE_KUNLUN_API
        DELETE(INFINI_DEVICE_KUNLUN, kunlun);
#endif

    default:
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

#undef DELETE
}