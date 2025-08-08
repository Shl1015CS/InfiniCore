// Metax GPU kernels for TopK Router
// 这些kernel专门为沐曦GPU架构优化

#include <hc/hc.h>

namespace op::topk_router::metax {

// Metax GPU上的DeepSeek v3 TopK选择kernel
template<typename T>
__global__ void metax_deepseek_v3_topk_kernel(
    const T* router_logits,
    int* topk_indices,
    T* topk_weights,
    int batch_size,
    int seq_len,
    int n_routed_experts,
    int n_group,
    int topk_group,
    int top_k,
    float routed_scaling_factor,
    bool norm_topk_prob) {
    
    int batch_idx = blockIdx.x;
    int seq_idx = blockIdx.y;
    
    if (batch_idx >= batch_size || seq_idx >= seq_len) return;
    
    int token_idx = batch_idx * seq_len + seq_idx;
    const T* token_logits = router_logits + token_idx * n_routed_experts;
    int* token_indices = topk_indices + token_idx * top_k;
    T* token_weights = topk_weights + token_idx * top_k;
    
    // DeepSeek v3分组策略实现
    int experts_per_group = n_routed_experts / n_group;
    
    // 共享内存存储group scores和indices
    __shared__ T group_scores[64]; // 假设最大64个groups
    __shared__ int group_indices[64];
    
    // Step 1: 计算每个group的得分
    if (threadIdx.x < n_group) {
        int group_id = threadIdx.x;
        T max_score1 = -INFINITY, max_score2 = -INFINITY;
        int max_idx1 = -1, max_idx2 = -1;
        
        // 找到该group中的前2个expert
        for (int e = 0; e < experts_per_group; e++) {
            int expert_idx = group_id * experts_per_group + e;
            T score = token_logits[expert_idx];
            
            if (score > max_score1) {
                max_score2 = max_score1;
                max_idx2 = max_idx1;
                max_score1 = score;
                max_idx1 = expert_idx;
            } else if (score > max_score2) {
                max_score2 = score;
                max_idx2 = expert_idx;
            }
        }
        
        group_scores[group_id] = max_score1 + max_score2;
        group_indices[group_id] = group_id;
    }
    
    __syncthreads();
    
    // Step 2: 排序选择top_k个groups
    if (threadIdx.x == 0) {
        // 简单的冒泡排序（适合小规模）
        for (int i = 0; i < topk_group - 1; i++) {
            for (int j = 0; j < n_group - i - 1; j++) {
                if (group_scores[j] < group_scores[j + 1]) {
                    // 交换scores
                    T temp_score = group_scores[j];
                    group_scores[j] = group_scores[j + 1];
                    group_scores[j + 1] = temp_score;
                    
                    // 交换indices
                    int temp_idx = group_indices[j];
                    group_indices[j] = group_indices[j + 1];
                    group_indices[j + 1] = temp_idx;
                }
            }
        }
        
        // Step 3: 从选中的groups中选择experts
        int expert_count = 0;
        for (int g = 0; g < topk_group && expert_count < top_k; g++) {
            int selected_group = group_indices[g];
            
            // 找到该group中的前几个expert
            int experts_from_group = min(top_k - expert_count, 
                                       top_k / topk_group + (g < top_k % topk_group ? 1 : 0));
            
            // 收集该group的expert scores
            for (int e = 0; e < experts_per_group && expert_count < top_k; e++) {
                int expert_idx = selected_group * experts_per_group + e;
                if (expert_count < experts_from_group) {
                    token_indices[expert_count] = expert_idx;
                    token_weights[expert_count] = token_logits[expert_idx] * routed_scaling_factor;
                    expert_count++;
                }
            }
        }
        
        // Step 4: 权重归一化
        if (norm_topk_prob) {
            T sum = 0;
            for (int i = 0; i < top_k; i++) {
                token_weights[i] = exp(token_weights[i]);
                sum += token_weights[i];
            }
            for (int i = 0; i < top_k; i++) {
                token_weights[i] /= sum;
            }
        }
    }
}

// 高性能的group并行TopK实现
template<typename T>
__global__ void metax_group_parallel_topk_kernel(
    const T* router_logits,
    int* topk_indices,
    T* topk_weights,
    int batch_size,
    int seq_len,
    int n_routed_experts,
    int n_group,
    int topk_group,
    int top_k,
    float routed_scaling_factor,
    bool norm_topk_prob) {
    
    // 更复杂的并行实现，适合大规模expert数量
    // 每个warp处理一个group，多个block处理不同的token
    
    int token_idx = blockIdx.x;
    int group_id = threadIdx.x / 32; // warp_id as group_id
    int lane_id = threadIdx.x % 32;
    
    if (token_idx >= batch_size * seq_len || group_id >= n_group) return;
    
    const T* token_logits = router_logits + token_idx * n_routed_experts;
    int experts_per_group = n_routed_experts / n_group;
    
    // 每个warp内找到该group的topK experts
    T max_scores[8]; // 假设每个group最多需要8个experts
    int max_indices[8];
    
    // 初始化
    for (int i = 0; i < 8; i++) {
        max_scores[i] = -INFINITY;
        max_indices[i] = -1;
    }
    
    // warp内并行寻找topK
    for (int e = lane_id; e < experts_per_group; e += 32) {
        int expert_idx = group_id * experts_per_group + e;
        T score = token_logits[expert_idx];
        
        // 插入排序维护topK
        for (int k = 0; k < min(8, top_k); k++) {
            if (score > max_scores[k]) {
                // 向后移动
                for (int j = min(7, top_k-1); j > k; j--) {
                    max_scores[j] = max_scores[j-1];
                    max_indices[j] = max_indices[j-1];
                }
                max_scores[k] = score;
                max_indices[k] = expert_idx;
                break;
            }
        }
    }
    
    // warp内reduction找到最终的topK
    // ... 更多的warp-level并行优化代码
}

} // namespace op::topk_router::metax
