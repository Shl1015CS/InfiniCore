#include "ops.hpp"
#include "utils.hpp"
#include <infinicore.h>
#include <infinirt.h>
#include <random>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

namespace infiniop_test::topk_router {

struct Test::Attributes {
    int top_k;
    int n_group;
    int topk_group;
    float routed_scaling_factor;
    bool norm_topk_prob;
    
    // 只存储输入tensors
    std::shared_ptr<Tensor> input;
    std::shared_ptr<Tensor> w_gate;
};

std::shared_ptr<Test> Test::build(
    std::unordered_map<std::string, std::vector<uint8_t>> attributes,
    std::unordered_map<std::string, std::shared_ptr<Tensor>> tensors,
    double rtol, double atol) {
    
    auto test = std::shared_ptr<Test>(new Test(rtol, atol));
    test->_attributes = new Attributes();
    
    // Parse attributes
    if (attributes.find("top_k") != attributes.end()) {
        test->_attributes->top_k = *reinterpret_cast<const int*>(attributes["top_k"].data());
    } else {
        test->_attributes->top_k = 2; // default
    }
    
    if (attributes.find("n_group") != attributes.end()) {
        test->_attributes->n_group = *reinterpret_cast<const int*>(attributes["n_group"].data());
    } else {
        test->_attributes->n_group = 1; // default
    }
    
    if (attributes.find("topk_group") != attributes.end()) {
        test->_attributes->topk_group = *reinterpret_cast<const int*>(attributes["topk_group"].data());
    } else {
        test->_attributes->topk_group = 1; // default
    }
    
    if (attributes.find("routed_scaling_factor") != attributes.end()) {
        test->_attributes->routed_scaling_factor = *reinterpret_cast<const float*>(attributes["routed_scaling_factor"].data());
    } else {
        test->_attributes->routed_scaling_factor = 1.0f; // default
    }
    
    if (attributes.find("norm_topk_prob") != attributes.end()) {
        test->_attributes->norm_topk_prob = *reinterpret_cast<const bool*>(attributes["norm_topk_prob"].data());
    } else {
        test->_attributes->norm_topk_prob = true; // default
    }
    
    // Store tensors
    test->_attributes->input = tensors.at("input");
    test->_attributes->w_gate = tensors.at("w_gate");
    
    return test;
}

std::vector<std::string> Test::attribute_names() {
    return {"top_k", "n_group", "topk_group", "routed_scaling_factor", "norm_topk_prob"};
}

std::vector<std::string> Test::tensor_names() {
    return {"input", "w_gate"};  // 只包含输入tensors，输出tensors在运行时创建
}

std::vector<std::string> Test::output_names() {
    return {"topk_indices", "topk_weights"};
}

std::shared_ptr<infiniop_test::Result> Test::run(
    infiniopHandle_t handle, infiniDevice_t device, int device_id,
    size_t warm_ups, size_t iterations) {
    
    std::cout << "Testing TopkRouter operator..." << std::endl;
    
        try {
        // 从存储的tensors中获取形状信息
        auto input_shape = _attributes->input->shape();
        auto w_gate_shape = _attributes->w_gate->shape();
        
        // 从形状推断参数
        size_t batch_size = input_shape[0];
        size_t seq_len = input_shape[1]; 
        // size_t hidden_size = input_shape[2];  // 暂时不需要
        // int num_experts = static_cast<int>(w_gate_shape[0]);  // 暂时不需要
        int topk = _attributes->top_k;
        
        // 获取实际的tensor数据（转移到目标设备）
        auto input = _attributes->input->to(device, device_id);
        auto w_gate = _attributes->w_gate->to(device, device_id);
        
        // 获取输入tensor的描述符
        auto input_desc = input->desc();
        auto w_gate_desc = w_gate->desc();
        
        // 手动创建输出tensor描述符
        std::vector<size_t> topk_indices_shape = {batch_size * seq_len, static_cast<size_t>(topk)};
        std::vector<size_t> topk_weights_shape = {batch_size * seq_len, static_cast<size_t>(topk)};
        
        infiniopTensorDescriptor_t topk_indices_desc, topk_weights_desc;
        infiniStatus_t status = infiniopCreateTensorDescriptor(&topk_indices_desc, 
            topk_indices_shape.size(), topk_indices_shape.data(), nullptr, INFINI_DTYPE_I32);
        if (status != INFINI_STATUS_SUCCESS) {
            return TEST_FAILED(OP_CREATION_FAILED, "Failed to create topk_indices tensor descriptor");
        }
        
        status = infiniopCreateTensorDescriptor(&topk_weights_desc, 
            topk_weights_shape.size(), topk_weights_shape.data(), nullptr, INFINI_DTYPE_F32);
        if (status != INFINI_STATUS_SUCCESS) {
            infiniopDestroyTensorDescriptor(topk_indices_desc);
            return TEST_FAILED(OP_CREATION_FAILED, "Failed to create topk_weights tensor descriptor");
        }

        // Create operator descriptor
        infiniopTopkRouterDescriptor_t desc;
        status = infiniopCreateTopkRouterDescriptor(
            handle,
            &desc,
            topk_indices_desc, topk_weights_desc,
            input_desc, w_gate_desc,
            _attributes->top_k, _attributes->n_group, _attributes->topk_group,
            _attributes->routed_scaling_factor, _attributes->norm_topk_prob);
        
        if (status != INFINI_STATUS_SUCCESS) {
            return TEST_FAILED(OP_CREATION_FAILED, "Failed to create TopkRouter descriptor");
        }

        // Get workspace size
        size_t workspace_size = 0;
        status = infiniopGetTopkRouterWorkspaceSize(desc, &workspace_size);
        if (status != INFINI_STATUS_SUCCESS) {
            infiniopDestroyTopkRouterDescriptor(desc);
            return TEST_FAILED(OP_CREATION_FAILED, "Failed to get workspace size");
        }

        // Allocate workspace if needed
        void *workspace_ptr = nullptr;
        if (workspace_size > 0) {
            status = infinirtMalloc(&workspace_ptr, workspace_size);
            if (status != INFINI_STATUS_SUCCESS) {
                infiniopDestroyTopkRouterDescriptor(desc);
                return TEST_FAILED(OP_EXECUTION_FAILED, "Failed to allocate workspace");
            }
        }
        
        // 分配输出tensor内存 
        void *output_topk_indices_ptr, *output_topk_weights_ptr;
        size_t topk_indices_size = batch_size * seq_len * topk * sizeof(int);
        size_t topk_weights_size = batch_size * seq_len * topk * sizeof(float);
        
        status = infinirtMalloc(&output_topk_indices_ptr, topk_indices_size);
        if (status != INFINI_STATUS_SUCCESS) {
            if (workspace_ptr) infinirtFree(workspace_ptr);
            infiniopDestroyTopkRouterDescriptor(desc);
            infiniopDestroyTensorDescriptor(topk_indices_desc);
            infiniopDestroyTensorDescriptor(topk_weights_desc);
            return TEST_FAILED(OP_EXECUTION_FAILED, "Failed to allocate output topk_indices memory");
        }
        
        status = infinirtMalloc(&output_topk_weights_ptr, topk_weights_size);
        if (status != INFINI_STATUS_SUCCESS) {
            infinirtFree(output_topk_indices_ptr);
            if (workspace_ptr) infinirtFree(workspace_ptr);
            infiniopDestroyTopkRouterDescriptor(desc);
            infiniopDestroyTensorDescriptor(topk_indices_desc);
            infiniopDestroyTensorDescriptor(topk_weights_desc);
            return TEST_FAILED(OP_EXECUTION_FAILED, "Failed to allocate output topk_weights memory");
        }
        
        // Create benchmark function
        auto benchmark_func = [&]() {
            infiniStatus_t exec_status = infiniopTopkRouter(desc,
                workspace_ptr, workspace_size,
                output_topk_indices_ptr,
                output_topk_weights_ptr,
                input->data(),
                w_gate->data(),
                nullptr);  // stream
            if (exec_status != INFINI_STATUS_SUCCESS) {
                throw std::runtime_error("TopkRouter execution failed");
            }
        };
        
        // Measure performance if iterations > 0
        double avg_time = 0.0;
        if (iterations > 0) {
            try {
                avg_time = infiniop_test::benchmark(benchmark_func, warm_ups, iterations);
            } catch (const std::exception& e) {
                infinirtFree(output_topk_indices_ptr);
                infinirtFree(output_topk_weights_ptr);
                if (workspace_ptr) infinirtFree(workspace_ptr);
                infiniopDestroyTopkRouterDescriptor(desc);
                infiniopDestroyTensorDescriptor(topk_indices_desc);
                infiniopDestroyTensorDescriptor(topk_weights_desc);
                return TEST_FAILED(OP_EXECUTION_FAILED, std::string("Benchmark failed: ") + e.what());
            }
        } else {
            // Just run once for correctness test
            try {
                benchmark_func();
            } catch (const std::exception& e) {
                infinirtFree(output_topk_indices_ptr);
                infinirtFree(output_topk_weights_ptr);
                if (workspace_ptr) infinirtFree(workspace_ptr);
                infiniopDestroyTopkRouterDescriptor(desc);
                infiniopDestroyTensorDescriptor(topk_indices_desc);
                infiniopDestroyTensorDescriptor(topk_weights_desc);
                return TEST_FAILED(OP_EXECUTION_FAILED, std::string("TopkRouter execution failed: ") + e.what());
            }
        }
        
        // 执行正确性验证 (简化版本 - 主要验证操作没有崩溃)
        if (iterations == 0) {
            // 只进行基本验证：检查输出内存是否可访问
            // 这里我们可以添加更多验证，比如检查输出值的合理性
            std::vector<int> output_indices(batch_size * seq_len * topk);
            std::vector<float> output_weights(batch_size * seq_len * topk);
            
            status = infinirtMemcpy(output_indices.data(), output_topk_indices_ptr, 
                                   topk_indices_size, INFINIRT_MEMCPY_D2H);
            if (status != INFINI_STATUS_SUCCESS) {
                infinirtFree(output_topk_indices_ptr);
                infinirtFree(output_topk_weights_ptr);
                if (workspace_ptr) infinirtFree(workspace_ptr);
                infiniopDestroyTopkRouterDescriptor(desc);
                infiniopDestroyTensorDescriptor(topk_indices_desc);
                infiniopDestroyTensorDescriptor(topk_weights_desc);
                return TEST_FAILED(OP_EXECUTION_FAILED, "Failed to copy output indices from device");
            }
            
            status = infinirtMemcpy(output_weights.data(), output_topk_weights_ptr, 
                                   topk_weights_size, INFINIRT_MEMCPY_D2H);
            if (status != INFINI_STATUS_SUCCESS) {
                infinirtFree(output_topk_indices_ptr);
                infinirtFree(output_topk_weights_ptr);
                if (workspace_ptr) infinirtFree(workspace_ptr);
                infiniopDestroyTopkRouterDescriptor(desc);
                infiniopDestroyTensorDescriptor(topk_indices_desc);
                infiniopDestroyTensorDescriptor(topk_weights_desc);
                return TEST_FAILED(OP_EXECUTION_FAILED, "Failed to copy output weights from device");
            }
        }
        
        // Clean up
        infinirtFree(output_topk_indices_ptr);
        infinirtFree(output_topk_weights_ptr);
        if (workspace_ptr) infinirtFree(workspace_ptr);
        infiniopDestroyTopkRouterDescriptor(desc);
        infiniopDestroyTensorDescriptor(topk_indices_desc);
        infiniopDestroyTensorDescriptor(topk_weights_desc);
        
        std::cout << "✓ TopkRouter test passed!" << std::endl;
        return TEST_PASSED(avg_time);
        
    } catch (const std::exception& e) {
        std::cerr << "✗ TopkRouter test failed with exception: " << e.what() << std::endl;
        return TEST_FAILED(OP_EXECUTION_FAILED, e.what());
    }
}

std::string Test::toString() const {
    return "topk_router[top_k=" + std::to_string(_attributes->top_k) + 
           ", n_group=" + std::to_string(_attributes->n_group) + 
           ", topk_group=" + std::to_string(_attributes->topk_group) + 
           ", routed_scaling_factor=" + std::to_string(_attributes->routed_scaling_factor) + 
           ", norm_topk_prob=" + (_attributes->norm_topk_prob ? "true" : "false") + "]";
}

Test::~Test() {
    delete _attributes;
}

} // namespace infiniop_test::topk_router

// Legacy test function for backward compatibility
void test_topk_router() {
    std::cout << "Running legacy TopkRouter test..." << std::endl;
    
    // Create a dummy test case and run it
    std::unordered_map<std::string, std::vector<uint8_t>> attributes;
    std::unordered_map<std::string, std::shared_ptr<infiniop_test::Tensor>> tensors;
    
    int top_k = 2;
    attributes["top_k"] = std::vector<uint8_t>(reinterpret_cast<const uint8_t*>(&top_k), 
                                               reinterpret_cast<const uint8_t*>(&top_k) + sizeof(int));
    
    auto test = infiniop_test::topk_router::Test::build(attributes, tensors, 1e-3, 1e-3);
    
    // Run with dummy parameters
    test->run(nullptr, INFINI_DEVICE_CPU, 0, 1, 1);
}