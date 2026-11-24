# Qwen3-1.7B 的 2–4 bit 权重量化与混合位宽评测

项目回答两个问题。把同一个模型量化到 2–4 bit 档位，能节省多少空间，困惑度和推理吞吐如何变化？在超低位宽基线上增加同样多的字节，提升敏感张量的位宽能否比随机分配恢复更多质量？

[查看实测报告](results/report.md) · [结果图](results/size_vs_perplexity.png) · [完整指标](results/metrics.csv)

## 运行

采用 Windows x64、Python 3.11、Git、NVIDIA 驱动和 `llama.cpp b12100` 的 CUDA 13.4 构建。实测环境为 RTX 5070 Ti Laptop GPU，约 12 GB 显存。首次运行会下载约 3.44 GB 的原始权重、CUDA 运行包和评测数据，建议预留至少 20 GiB 可用磁盘空间。PyTorch 参考实现使用 CPU 版，GGUF 评测使用 NVIDIA CUDA。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File .\run.ps1 -Stage all
```

也可单独运行 `test`、`int4`、`prepare`、`baselines`、`scan`、`evaluate`、`report`。GGUF、成功执行的原始日志和已锁定的分配会供同一实验复用。指标或模型变更时，在新目录重新运行，以保留原实验及其证据。

## 实验设计

1. **手写分组 INT4。** 每行按输入方向分组，以 `max(abs(w)) / 7` 计算 FP32 scale。量化整数用每字节两个值存储。依次用 group size 32、64、128，测试 Qwen3 第 0 层的 `q_proj` 和 `down_proj`。固定种子，使用 128 条高斯输入，计算权重与线性层输出的相对 L2 误差。存储值含打包权重、FP32 scale 和可能的组尾填充。
2. **从同一个 F16 建立基线。** 用固定 revision 的 `Qwen/Qwen3-1.7B-Base` 生成 F16 GGUF。由这份文件分别生成 `Q4_K_M`、`Q4_K_M + imatrix`、`Q3_K_M + imatrix` 和 `Q2_K + imatrix`。Q4 的有无校准对照共用同一个预设。GGUF 预设内部可能包含多种张量类型，压缩用实测文件字节数表示。
3. **按预算筛选张量。** 对 `Q2_K + imatrix`，从同一形状和量化类型的 FFN down 张量中等距抽取 8 个。每次只提升一个张量为 `Q4_K`，按验证集平均负对数似然改善排序。固定提升 3 个张量，比较排名前三的分配和 3 组固定种子的随机分配。逐文件核对目标张量类型、未覆盖张量的类型和形状，以及文件字节数。
4. **锁定后测独立测试集。** WikiText-2 训练集的前 32 个块用于 `imatrix`，验证集的前 32 个块用于筛选，完整测试文件用于最终评测，工具只使用其中完整的 512-token 块。运行前核对各分片的 SHA-256。困惑度测量统一上下文 512、batch 和 microbatch 512、F16 KV cache，日志检查显存中存在模型权重且推理层全部 offload。`llama-bench` 独立运行 `pp512` 和 `tg128`，各预热并重复 5 次。显存是 `nvidia-smi` 约每 100 ms 采样得到的设备峰值，另外记录运行前空闲值和两者之差。

当前 Windows 构建的 `llama-imatrix` 在分词整份训练文件时崩溃。脚本提供保留原始换行的训练集 UTF-8 前缀，并检查工具日志中实际处理了 32 个块。验证和测试均读取原始分片。

项目只用 WikiText-2 的下一个 token 预测来衡量质量。测试结果适用于这个 checkpoint、语料、推理引擎和硬件。三个同预算随机分配能提供对照范围，样本规模有限。

## 代码和实验结果

| 入口 | 内容 |
|---|---|
| `int4.py` / `tests/` | 参考量化器与单元测试 |
| `measure_int4.py` | 对真实模型矩阵测相对误差和存储开销 |
| `experiment.py` / `gguf_tensors.py` | 创建基线、扫描、锁定分配、评测和 GGUF 张量审计 |
| `setup.ps1` / `run.ps1` | 固定版本安装和实验运行 |
| `results/int4_layer_errors.csv` | 真实权重的 6 条 INT4 结果 |
| `results/validation_baselines.csv` / `results/sensitivity.csv` | 验证集基线和逐张量敏感度 |
| `results/protocol.json` / `results/allocations.json` / `results/allocation_audit.csv` | 数据散列、参数、选层及字节预算 |
| `results/metrics.csv` / `results/comparison.json` | CUDA 实测结果及同预算比较 |
| `results/size_vs_perplexity.png` / `results/report.md` | 结果图和中文实验报告 |
| `results/logs/` | 仓库附带最终核验摘要，量化、推理、困惑度、吞吐和显存的原始日志在本地复跑时生成 |

参考 [Qwen 中文量化指南](https://qwen.readthedocs.io/zh-cn/stable/quantization/llama.cpp.html) · [llama.cpp 量化工具](https://github.com/ggml-org/llama.cpp/blob/b12100/tools/quantize/README.md) · [llama.cpp 性能工具](https://github.com/ggml-org/llama.cpp/blob/b12100/tools/llama-bench/README.md)。
