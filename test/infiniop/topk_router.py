import torch
import ctypes
from ctypes import c_uint64
from libinfiniop import (
    LIBINFINIOP,
    TestTensor,
    get_test_devices,
    check_error,
    test_operator,
    get_args,
    debug,
    get_tolerance,
    profile_operation,
    TestWorkspace,
    InfiniDtype,
    InfiniDtypeNames,
    InfiniDeviceNames,
    infiniopOperatorDescriptor_t,
)

# ==============================================================================
#  Configuration (Internal Use Only)
# ==============================================================================
# These are not meant to be imported from other modules
_TEST_CASES = [
    # batch_size, seq_len, hidden_size, n_routed_experts, top_k, n_group, topk_group, routed_scaling_factor, norm_topk_prob
    (2, 4, 512, 64, 6, 8, 2, 1.0, True),      # 典型DeepSeek v3配置
    (1, 8, 1024, 128, 8, 16, 4, 1.0, True),   # 更大的expert数量
    (4, 16, 2048, 256, 16, 32, 8, 2.0, False), # 不同的scaling factor
]

# Data types used for testing  
_TENSOR_DTYPES = [InfiniDtype.F32, InfiniDtype.F16, InfiniDtype.BF16]

# Tolerance map for different data types
_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-2},
    InfiniDtype.F32: {"atol": 1e-5, "rtol": 1e-4},
    InfiniDtype.BF16: {"atol": 1e-2, "rtol": 5e-2},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 100


# PyTorch reference implementation matching DeepSeek v3
def deepseek_v3_topk_router_reference(input_tensor, weight, top_k, n_group, topk_group, routed_scaling_factor, norm_topk_prob):
    """
    DeepSeek v3 TopkRouter reference implementation using PyTorch
    
    Args:
        input_tensor: [batch_size, seq_len, hidden_size]
        weight: [n_routed_experts, hidden_size] (DeepSeek v3 layout)
        top_k: int
        n_group: int 
        topk_group: int
        routed_scaling_factor: float
        norm_topk_prob: bool
        
    Returns:
        topk_indices: [batch_size * seq_len, top_k] (int32)
        topk_weights: [batch_size * seq_len, top_k] (float)
    """
    batch_size, seq_len, hidden_size = input_tensor.shape
    n_routed_experts, _ = weight.shape
    
    # Reshape input to [batch_size * seq_len, hidden_size]
    hidden_states = input_tensor.view(-1, hidden_size)
    
    # 计算router logits: [batch_size * seq_len, n_routed_experts]
    router_logits = torch.nn.functional.linear(
        hidden_states.type(torch.float32), 
        weight.type(torch.float32)
    )
    
    # 应用sigmoid激活
    scores = router_logits.sigmoid()
    
    # DeepSeek v3 style top-k selection with grouping
    def get_topk_indices_deepseek_v3(scores):
        # scores: [batch_size * seq_len, n_routed_experts]
        batch_seq_len = scores.shape[0]
        experts_per_group = n_routed_experts // n_group
        
        # Step 1: 计算group scores (每组前2名的和)
        scores_reshaped = scores.view(batch_seq_len, n_group, experts_per_group)
        group_scores = scores_reshaped.topk(2, dim=-1)[0].sum(dim=-1)  # [batch_seq_len, n_group]
        
        # Step 2: 选择top groups
        group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]
        
        # Step 3: 创建group mask
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        
        # Step 4: 应用mask并选择最终的top-k experts
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, n_group, experts_per_group)
            .reshape(batch_seq_len, n_routed_experts)
        )
        
        scores_masked = scores.masked_fill(~score_mask.bool(), 0.0)
        topk_indices = torch.topk(scores_masked, k=top_k, dim=-1, sorted=False)[1]
        
        return topk_indices
    
    # 获取top-k indices
    topk_indices = get_topk_indices_deepseek_v3(scores)
    
    # 收集对应的权重
    topk_weights = scores.gather(1, topk_indices)
    
    # 归一化权重（如果启用）
    if norm_topk_prob:
        denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
        topk_weights = topk_weights / denominator
    
    # 应用scaling factor
    topk_weights = topk_weights * routed_scaling_factor
    
    return topk_indices.to(torch.int32), topk_weights


def test(
    handle,
    device,
    batch_size,
    seq_len,
    hidden_size,
    n_routed_experts,
    top_k,
    n_group,
    topk_group,
    routed_scaling_factor,
    norm_topk_prob,
    dtype=InfiniDtype.F32,
    sync=None,
):
    print(
        f"Testing DeepSeekV3TopkRouter on {InfiniDeviceNames[device]} with "
        f"batch_size:{batch_size}, seq_len:{seq_len}, hidden_size:{hidden_size}, "
        f"n_routed_experts:{n_routed_experts}, top_k:{top_k}, n_group:{n_group}, "
        f"topk_group:{topk_group}, routed_scaling_factor:{routed_scaling_factor}, "
        f"norm_topk_prob:{norm_topk_prob}, dtype:{InfiniDtypeNames[dtype]}"
    )

    # Initialize input tensors
    input_shape = (batch_size, seq_len, hidden_size)
    weight_shape = (n_routed_experts, hidden_size)  # DeepSeek v3 layout
    
    input_tensor = TestTensor(input_shape, None, dtype, device, mode="random")
    weight_tensor = TestTensor(weight_shape, None, dtype, device, mode="random")
    
    # Initialize output tensors - DeepSeek v3 uses flattened output
    topk_indices_shape = (batch_size * seq_len, top_k)
    topk_weights_shape = (batch_size * seq_len, top_k)
    
    topk_indices = TestTensor(topk_indices_shape, None, InfiniDtype.I32, device, mode="zeros")
    topk_weights = TestTensor(topk_weights_shape, None, dtype, device, mode="zeros")

    # Compute the PyTorch reference result
    def torch_topk_router():
        ref_indices, ref_weights = deepseek_v3_topk_router_reference(
            input_tensor.torch_tensor(),
            weight_tensor.torch_tensor(),
            top_k,
            n_group,
            topk_group,
            routed_scaling_factor,
            norm_topk_prob
        )
        return ref_indices, ref_weights

    ref_indices, ref_weights = torch_topk_router()

    if sync is not None:
        sync()

    # Create operator descriptor
    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateTopkRouterDescriptor(
            handle,
            ctypes.byref(descriptor),
            topk_indices.descriptor,
            topk_weights.descriptor,
            input_tensor.descriptor,
            weight_tensor.descriptor,
            top_k,
            n_group,
            topk_group,
            ctypes.c_float(routed_scaling_factor),
            ctypes.c_bool(norm_topk_prob),
        )
    )

    # Invalidate the shape and strides in the descriptor
    for tensor in [input_tensor, weight_tensor, topk_indices, topk_weights]:
        tensor.destroy_desc()

    # Get workspace size and create workspace
    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetTopkRouterWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, device)

    # Execute infiniop topk_router operator
    def lib_topk_router():
        check_error(
            LIBINFINIOP.infiniopTopkRouter(
                descriptor,
                workspace.data(),
                workspace_size.value,
                topk_indices.data(),
                topk_weights.data(),
                input_tensor.data(),
                weight_tensor.data(),
                None,
            )
        )

    lib_topk_router()

    # Validate results
    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)

    if DEBUG:
        print("Reference indices:", ref_indices)
        print("Library indices:", topk_indices.actual_tensor())
        print("Reference weights:", ref_weights) 
        print("Library weights:", topk_weights.actual_tensor())

    # For indices, we check if they are valid expert indices
    assert torch.all(topk_indices.actual_tensor() >= 0)
    assert torch.all(topk_indices.actual_tensor() < n_routed_experts)
    
    # For weights, we check basic properties
    lib_weights = topk_weights.actual_tensor()
    assert torch.all(lib_weights >= 0)  # Weights should be non-negative
    
    # Check if normalization was applied correctly (when enabled)
    if norm_topk_prob:
        weights_sum = lib_weights.sum(dim=-1)
        expected_sum = torch.ones_like(weights_sum) * routed_scaling_factor
        assert torch.allclose(weights_sum, expected_sum, atol=atol*10, rtol=rtol*10)

    print("✓ DeepSeekV3TopkRouter test passed: indices are valid and weights are properly normalized")

    # Profiling workflow
    if PROFILE:
        profile_operation("PyTorch", lambda: torch_topk_router(), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_topk_router(), device, NUM_PRERUN, NUM_ITERATIONS)

    check_error(LIBINFINIOP.infiniopDestroyTopkRouterDescriptor(descriptor))


# ==============================================================================
#  Main Execution
# ==============================================================================
if __name__ == "__main__":
    args = get_args()

    # Configure testing options
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    # Execute tests
    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mDeepSeekV3TopkRouter test passed!\033[0m")