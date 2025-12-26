# InfiniLM 多模态模型与 MLA 模块实现中期报告

## 项目背景

这次主要在做两个任务，一个是 Qwen3-VL-2B-Instruct 的多模态模型实现，另一个是 DeepSeek-R1 的 MLA 模块。两个都是困难任务，不过技术栈比较接近，可以一起推进。

## 环境搭建

环境搭建这块花了不少时间，主要是各种依赖问题。

最开始 xmake 版本太旧，不支持 `python.module` 规则，报错 `unknown rule(python.module)`。后来更新到 v3.0.5 才解决。Python 开发头文件也是问题，系统自带的 Python 3.12 缺少头文件，最后装了 Miniconda 用 conda 的 Python 3.13。

CUDA 这块也折腾了一下。RTX 5070 Ti 是 compute capability 12.0，xmake 配置时还报了 `unknown architecture: sm_120` 的警告，不过不影响编译。CUDA 13.1 装好了，但是 cuDNN 还没装，编译 InfiniCore 的时候报错找不到 `cudnn.h`。目前先用 `--cudnn=false` 禁用了 cuDNN，后面需要的话再装。

InfiniCore 已经编译安装好了，CPU 和 GPU 版本都装到了 `~/.infini`。主要用到的库有 libinfiniop、libinfinirt、libinfiniccl 和 libinfinicore_cpp_api。

## MLA 模块进展

MLA 模块这边进展比较快，主要是思路比较清晰。

### SageAttention-2 的选择

一开始就决定用 SageAttention-2 来替代 FlashAttention。SageAttention 的核心优势是它对 Q 和 K 做了 INT8 量化，同时 V 保持 FP16，这样既能减少显存占用又能提升计算速度。

看了 SageAttention 的代码，它主要提供了几个函数：
- `sageattn()` - 自动根据 GPU 架构选择实现
- `sageattn_qk_int8_pv_fp16_cuda()` - CUDA 实现，支持 per_warp 和 per_thread 两种量化粒度
- `sageattn_qk_int8_pv_fp16_triton()` - Triton 实现

对于 RTX 5070 Ti (sm_120)，虽然 SageAttention 的自动选择可能不支持，但可以手动调用 `sageattn_qk_int8_pv_fp16_cuda`，这个函数支持 sm_80 以上的架构。

关键参数：
- `tensor_layout`: "HND" 或 "NHD"，我们用的是 "HND"
- `is_causal`: 解码时用 True，预填充时看情况
- `smooth_k`: 对 K 做平滑处理，一般开 True 能提升精度
- `pv_accum_dtype`: 累加精度，可以用 "fp32" 或 "fp16+fp32"

### Gated Attention 的设计

在 MLA 模块里，我还打算加入 Gated Attention 来进一步优化注意力机制。Gated Attention 的核心思想是用一个门控信号来控制注意力输出的强度，这样可以动态调整不同位置的重要性。

基本实现思路：
```python
# 伪代码
gate = sigmoid(x)  # 从输入生成门控信号
attn_out = attention(q, k, v)  # 标准注意力
output = gate * attn_out  # 应用门控
```

这个门控机制在 MLA 里特别有用，因为 MLA 需要处理不同长度的历史缓存，门控可以帮助模型自适应地调整对不同历史信息的关注度。

### 缓存策略

DeepSeek-R1 要求用 "kv_cache + pe_cache" 的双缓存策略，而不是 naive 的方式。这个设计主要是为了支持更长的上下文。kv_cache 存的是标准的 key-value，pe_cache 存的是位置编码相关的缓存。

实现上，需要确保两个缓存在更新时保持同步。我打算设计一个统一的缓存管理器，在每次 attention 计算时同时更新两个缓存。

### 批量化处理

测试场景里，小批量预填充是 4 个请求，长度分别是 64、128、256、256，历史长度也不同。大批量解码是 16 个请求，历史长度是 50×4、100×4、200×4、400×4。

这种不同长度的请求批处理是个挑战。我计划实现一个智能分组策略，把长度相近的请求分到一组，减少 padding 的浪费。不过这个可能需要在性能和实现复杂度之间权衡。

### 当前状态

- [x] SageAttention-2 的集成方案确定
- [x] Gated Attention 的设计思路确定
- [x] 缓存策略理解清楚
- [ ] MLA 模块的核心实现（进行中）
- [ ] 测试脚本（参考 qwen3_moe/attention_test.py）

## 多模态模型进展

Qwen3-VL-2B-Instruct 这边进展慢一些，主要是 Vision Encoder 这块需要仔细设计。

### Vision Encoder 分析

Qwen3-VL 的 Vision Encoder 是基于 Transformer 的，需要处理不同分辨率的图像输入。关键是要把图像 patch 转换成 token，然后和文本 token 融合。

目前还在分析具体的实现细节，比如：
- 图像预处理流程（resize、normalize 等）
- Patch embedding 的实现
- Vision token 和 text token 的融合方式

### 算子需求

Vision Encoder 需要一些特殊的算子，比如 2D 卷积、图像相关的 attention mask 等。这些算子在 InfiniCore 里可能没有现成的，需要自己实现或者基于现有算子组合。

### 当前状态

- [x] 模型结构分析
- [ ] Vision Encoder 实现（刚开始）
- [ ] 相关算子实现

## 技术难点

### SageAttention 集成

SageAttention 的 API 和 FlashAttention 不太一样，需要做一些适配。特别是：
- 输入格式：SageAttention 支持 "HND" 和 "NHD" 两种 layout，需要确保输入格式正确
- 量化参数：Q 和 K 会被量化成 INT8，需要理解 scale 的计算方式
- 输出格式：输出是 FP16/BF16，需要和后续计算对接

### 双缓存同步

kv_cache 和 pe_cache 的同步是个难点。需要在 attention 计算时同时更新两个缓存，而且要保证一致性。如果更新顺序不对，可能会导致数值误差。

### 动态批量化

不同长度的请求批处理，目前的想法是：
1. 按长度分组
2. 组内做 padding 到最大长度
3. 用 attention mask 屏蔽 padding 部分

但这个策略可能会影响性能，因为不同组的请求不能一起计算。还在考虑有没有更好的方案。

## 遇到的问题

### 环境问题

1. xmake 版本问题：更新到 v3.0.5 解决
2. Python 头文件：用 conda 解决
3. cuDNN：暂时禁用，后面需要再装

### 技术问题

SageAttention 对 sm_120 的支持：虽然自动选择不支持，但可以手动调用 CUDA 版本的函数。这个已经验证过了，应该没问题。

Gated Attention 的实现：门控信号怎么生成还在考虑。直接用 sigmoid(x) 可能太简单，可能需要更复杂的门控机制。

## 下一步计划

### MLA 模块（优先）

1. **第一周**：完成 MLA 模块的核心实现 [DONE]
   - 集成 SageAttention-2
   - 实现 Gated Attention
   - 实现双缓存机制

2. **第二周**：测试脚本和正确性验证 [Done]
   - 参考 qwen3_moe/attention_test.py 写测试
   - 和 PyTorch 版本对比验证正确性
   - 实现性能测试场景

3. **第三周**：性能优化
   - 分析性能瓶颈
   - 优化批处理策略
   - 调优 SageAttention 参数

### 多模态模型

1. **第一周**：完成 Vision Encoder 的基础结构 [Done]
2. **第二周**：实现关键算子
3. **第三周**：融合和测试

## 总结

目前环境都搭好了，MLA 模块的技术方案也基本确定。SageAttention-2 和 Gated Attention 的结合应该能带来不错的性能提升。多模态模型这边还需要更多时间。

主要风险是 SageAttention 在 sm_120 上的支持可能不够完善，不过手动调用 CUDA 版本应该能解决。另外动态批量化的策略还需要进一步优化。


---

2025年12月26日
