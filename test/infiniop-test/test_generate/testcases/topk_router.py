#!/usr/bin/env python3

import numpy as np
import torch

def generate_topk_router_test_cases():
    """
    Generate test cases for TopkRouter operator
    """
    test_cases = [
        {
            "name": "small_case",
            "batch_size": 2,
            "seq_len": 4, 
            "hidden_size": 256,
            "num_experts": 8,
            "topk": 2
        },
        {
            "name": "medium_case",
            "batch_size": 1,
            "seq_len": 8,
            "hidden_size": 1024, 
            "num_experts": 16,
            "topk": 4
        },
        {
            "name": "large_case",
            "batch_size": 4,
            "seq_len": 32,
            "hidden_size": 4096,
            "num_experts": 64,
            "topk": 8
        }
    ]
    
    print("=== TopkRouter Test Case Generation ===")
    
    for case in test_cases:
        print(f"\nGenerating {case['name']}:")
        print(f"  batch_size: {case['batch_size']}")
        print(f"  seq_len: {case['seq_len']}")
        print(f"  hidden_size: {case['hidden_size']}")
        print(f"  num_experts: {case['num_experts']}")
        print(f"  topk: {case['topk']}")
        
        # Generate input data
        input_shape = (case['batch_size'], case['seq_len'], case['hidden_size'])
        w_gate_shape = (case['hidden_size'], case['num_experts'])
        
        # Create random input
        np.random.seed(42)  # For reproducible results
        input_data = np.random.randn(*input_shape).astype(np.float32)
        w_gate_data = np.random.randn(*w_gate_shape).astype(np.float32)
        
        # Convert to PyTorch tensors
        input_tensor = torch.from_numpy(input_data)
        w_gate_tensor = torch.from_numpy(w_gate_data)
        
        # Compute reference results
        gate_scores = torch.matmul(input_tensor, w_gate_tensor)
        gate_probs = torch.softmax(gate_scores, dim=-1)
        topk_weights, topk_indices = torch.topk(gate_probs, k=case['topk'], dim=-1)
        
        # Renormalize
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        
        print(f"  Input range: [{input_data.min():.3f}, {input_data.max():.3f}]")
        print(f"  W_gate range: [{w_gate_data.min():.3f}, {w_gate_data.max():.3f}]")
        print(f"  Output indices range: [{topk_indices.min()}, {topk_indices.max()}]")
        print(f"  Output weights range: [{topk_weights.min():.6f}, {topk_weights.max():.6f}]")
        print(f"  Weights sum check: {topk_weights.sum(dim=-1).mean():.6f} (should be ~1.0)")
        
        # Save test case (optional)
        case_data = {
            'input': input_data,
            'w_gate': w_gate_data,
            'expected_indices': topk_indices.numpy().astype(np.int32),
            'expected_weights': topk_weights.numpy()
        }
        
        # You can save to file if needed:
        # np.savez(f"topk_router_{case['name']}.npz", **case_data)

def verify_topk_router_properties():
    """
    Verify mathematical properties of TopkRouter
    """
    print("\n=== TopkRouter Properties Verification ===")
    
    # Simple test case
    batch_size, seq_len, hidden_size, num_experts, topk = 1, 1, 4, 8, 3
    
    # Create simple test data
    input_data = torch.randn(batch_size, seq_len, hidden_size)
    w_gate_data = torch.randn(hidden_size, num_experts)
    
    # Compute
    gate_scores = torch.matmul(input_data, w_gate_data)
    gate_probs = torch.softmax(gate_scores, dim=-1)
    topk_weights, topk_indices = torch.topk(gate_probs, k=topk, dim=-1)
    
    # Renormalize
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    
    print(f"Gate scores: {gate_scores.squeeze()}")
    print(f"Gate probs: {gate_probs.squeeze()}")
    print(f"Topk indices: {topk_indices.squeeze()}")
    print(f"Topk weights: {topk_weights.squeeze()}")
    print(f"Weights sum: {topk_weights.sum():.6f}")
    
    # Verify properties
    assert torch.all(topk_indices >= 0) and torch.all(topk_indices < num_experts)
    assert torch.all(topk_weights >= 0)
    assert torch.allclose(topk_weights.sum(dim=-1), torch.ones(batch_size, seq_len))
    
    print("✓ All properties verified!")

if __name__ == "__main__":
    generate_topk_router_test_cases()
    verify_topk_router_properties()
    print("\n✓ Test case generation completed!")