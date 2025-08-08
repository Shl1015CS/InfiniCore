#ifndef __TOPK_ROUTER_ILUVATAR_H__
#define __TOPK_ROUTER_ILUVATAR_H__

#include "../topk_router.h"

// 天数(Iluvatar)GPU使用兼容CUDA的API
// 可以复用NVIDIA的kernel实现，但需要独立的描述符管理
TOPK_ROUTER_DESCRIPTOR(iluvatar)

#endif // __TOPK_ROUTER_ILUVATAR_H__
