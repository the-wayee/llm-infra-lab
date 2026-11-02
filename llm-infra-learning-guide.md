# LLM 推理框架 / AI Infra 转型学习指南

> 面向 Java 后端工程师 · 36 周 · 每周 10 小时 · 约 360 小时  
> 更新日期：2026-10-28。学习路线保存在本文件，进度与反馈见 [TODO.md](TODO.md)。  
> 主线：Python / 张量 → Transformer 推理 → KV Cache → 推理服务与调度 → GPU / 算子 → 框架贡献与作品集。

## 0. 阅读说明与文档分工

你的起点：有 Java 后端开发经验；Python 目前只掌握基础语法；数学需要补基础；能在 Linux 上部署服务；每周可投入 10 小时；希望用约 9 个月转向 LLM 推理框架 / AI Infra。

**关于历史课程的边界**：本次读取了“解释 Git 命令”对话。可读取的历史回答只有内容引用占位符，无法恢复原文；能确认的提问涉及自动求导、forward/backward、benchmark 预热，以及 `03_cosine_similarity`。因此，第 1–4 课以下采用“连续性重建版”：保留这些已知知识点，重新组织成可执行课程；课名、文件目录和另外两个第 4 课实验是本指南补建的安排，不声称逐字还原此前课程。

**文档分工**：本文件记录路线、知识依赖和验收标准；[TODO.md](TODO.md) 记录当前任务、完成证据、问题与反馈；`labs/` 保存可运行实验；`reports/` 保存详细实验报告。代码存在或脚本通过，都不自动代表已经掌握。

**使用方式**：先从 TODO 确认当前任务，再按本指南查阅概念并运行对应实验。已掌握的内容可通过验收后跳过。课次是教学单元，周次是投入预算，两者不要求一一对应；第 1–5 课覆盖路线前 4 周。

**范围**：本文件包含 36 周路线和后期项目规格；第 1–5 课的完整复盘实现已拆到 `labs/`，不在文档中重复维护。现有实验文件保留；新增的 `review` 文件用于补充复盘与断言，不覆盖原练习。

### 目录

1. [目标、时间与硬件](#1-目标时间与硬件)
2. [知识依赖与课程大纲](#2-知识依赖与课程大纲)
3. [项目结构与工作方式](#3-项目结构与工作方式)
4. [36 周逐周安排](#4-36-周逐周安排)
5. [阶段验收](#5-阶段验收)
6. [源码阅读顺序](#6-源码阅读顺序)
7. [第 1–4 课复盘与实验入口](#7-第-14-课复盘与实验入口)
8. [第 5 课计划与实验入口](#8-第-5-课计划与实验入口)
9. [后期项目规格与性能测量](#9-后期项目规格与性能测量)
10. [数学、排障、资源与执行清单](#10-数学排障资源与执行清单)

## 1. 目标、时间与硬件

### 1.1 36 周结束时应具备的能力

1. 读懂 Python 推理项目中的核心调用关系，能追踪张量形状、请求状态和内存归属。
2. 用 PyTorch 实现小型 decoder-only 模型，解释 causal attention、prefill、decode、KV Cache，并验证缓存与非缓存结果一致。
3. 运行一个真实推理引擎，记录版本、模型修订号、输入分布、延迟、吞吐和显存；能解释一次性能变化。
4. 理解 continuous batching、分页 KV 管理、前缀缓存的目标与代价，能做简化模拟。
5. 编写并分析一个简单 Triton 或 CUDA 算子，能用正确性基线和性能数据评估优化。
6. 在一个主框架完成范围明确、带复现和验证的改动，整理可用于面试的实验报告。

360 小时是形成入门工程能力与作品集的预算，并不保证达到资深 kernel 专家水平或获得岗位。优先瞄准推理服务工程、推理引擎工程中的调度 / 缓存方向；算子工程作为加强项。Java 的并发、排队、资源池、超时、监控和服务治理经验可以直接迁移。

### 1.2 每周固定 10 小时

| 时间块 | 小时 | 做什么 | 结束标志 |
|---|---:|---|---|
| 概念与必要数学 | 2 | 只补本周实验需要的概念，手算一个例子 | 半页自己的解释 |
| 编码与调试 | 4 | 实现本周最小实验，先 CPU 正确，再考虑 GPU | 一条运行命令、断言通过 |
| 源码 / 官方材料 | 2 | 带一个问题追踪 1–3 个函数 | 调用链与 shape 表 |
| 测量与复盘 | 2 | 保存环境、结果、失败原因并提交 Git | `reports/weekNN.md` |

推荐安排：两个工作日晚间各 1 小时，周末两段各 4 小时。每次结束写下“下次第一步”，减少重新进入状态的成本。

**落后时的裁剪顺序**：先删第二个框架、额外论文、复杂量化实现，再删多卡实测，最后缩小模型尺寸；保留缓存正确性、可信 benchmark、源码调用链、最终项目。某周验收未通过，下周先修复，不靠继续看视频掩盖问题。连续两周都不足 6 小时，延长总周期。

### 1.3 硬件与环境

已知设备：Windows + RTX 4060 8GB，以及 Mac M1 Pro（内存容量待补充）。**默认使用现有设备完成主线，云 GPU 预算先按 0 安排。**

- Mac：日常编码、CPU 基础实验、源码阅读与报告。前期代码默认 CPU；MPS 可作补充，但不可与 CUDA 混用性能基线。
- Windows：通过 WSL2 + Ubuntu 使用 RTX 4060，承担 CUDA、真实模型推理、推理服务和 Triton / CUDA 算子实验。具体引擎与工具版本须核对平台、驱动和设备支持；WSL 环境目前尚未确认配置完成。
- 第 1–8 周：CPU 足够；逐步实现小型 decoder，无需下载大模型。
- 第 9–18 周：先用约 0.5B–1.5B 的兼容模型，限制上下文和并发；显存允许时再尝试更大模型。是否能运行取决于权重、KV Cache 和运行时的总占用。
- 第 19–28 周：优先在 4060 上完成 profiling 和小算子实验，报告注明消费级 GPU 与 WSL 环境的限制。
- 第 31–32 周：多 GPU 性能实测为选修；没有多卡时完成两进程 CPU 通信正确性实验和容量分析，注明没有 GPU 多卡测量。
- 只有遇到明确的显存、设备特性或多卡需求时才评估租卡；先本地准备脚本，再集中运行、保存结果并释放实例。租卡不是完成主线的前置条件。

模型内存粗估：参数量 × 参数字节数只是权重本体，还要加 KV Cache、临时激活、运行时、图捕获等空间。例如 1B 参数 FP16 权重约 2 GB（十进制），不能据此推断 2 GB 显存就能运行。

### 1.4 创建学习环境

前期可选 Python 3.11 或 3.12；这只是教学环境建议。真实推理引擎另建环境，采用该版本支持的 Python / PyTorch / CUDA 组合。不要强行共用一个环境。

```bash
# 在你希望保存项目的父目录执行；已存在仓库则直接进入它
mkdir -p llm-infra-lab
cd llm-infra-lab
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# 初期只做 CPU 实验。GPU 安装命令以 PyTorch 官方安装选择器为准。
python -m pip install torch
mkdir -p labs reports results configs tests projects scripts reading external
python -m pip freeze > requirements-lock.txt
python -c "import platform,torch; print(platform.platform()); print(torch.__version__); print(torch.cuda.is_available())"
```

`venv` 相当于项目专属依赖目录；`source` 只影响当前 shell；`python -m pip` 确保包安装到当前 Python 对应环境。锁定文件仅代表当前平台环境，不保证能在另一操作系统直接复用。

`.gitignore` 至少包含：

```gitignore
.venv/
__pycache__/
*.pyc
external/
checkpoints/
.env
results/*.pt
results/*.bin
results/*.safetensors
```

实验报告和少量 JSON / CSV 结果应提交；模型权重、密钥与虚拟环境不提交。

## 2. 知识依赖与课程大纲

### 2.1 先后关系

```text
Python 函数 / 模块 / 容器 / 类 / 异常
  └─ Tensor(shape, dtype, device, stride)
      ├─ 向量 / 点积 / 矩阵乘法 / 广播
      │   └─ Linear / Embedding / softmax
      │       └─ 单头 attention → 多头 / causal mask → decoder
      │           └─ 自回归生成 → prefill / decode → KV Cache
      │               └─ 请求调度 / 分页内存 / 前缀缓存 → 推理引擎
      ├─ 最少求导 / autograd → 理解训练与推理差别
      └─ 计时 / 内存 / profiler → GPU 层次 → Triton / CUDA → 算子优化
Linux / 并发 / 排队 / 服务治理 ────────────────────────┘
```

不要等“数学全补完”才开始编程。推理路线先掌握 shape、点积、矩阵乘法、softmax 和缓存，求导理解到能够读懂训练示例即可。

### 2.2 六阶段课程

| 阶段 | 周数 | 核心问题 | 主交付 |
|---|---|---|---|
| A 基础 | 1–6 | 张量怎么算，程序怎么组织，结果怎么测 | 基础实验集 + 简单模型 |
| B 模型推理 | 7–12 | 一个 token 如何算出，哪些结果可以缓存 | `tiny_decoder` + KV 对照报告 |
| C 推理服务 | 13–18 | 请求如何排队、拼批、分配 KV | `mini_serving` + 调度模拟 |
| D GPU 与算子 | 19–24 | 时间花在哪，如何减少数据搬运 | profiler 报告 + 1 个算子 |
| E 框架进阶 | 25–30 | 真实框架怎样衔接调度与执行 | 主框架阅读档案 + 小改动 |
| F 工程与求职 | 31–36 | 怎样形成可复现、可解释的工程成果 | 完整项目、报告、演示与简历 |

## 3. 项目结构与工作方式

```text
llm-infra-lab/
├── README.md                       # 项目导航
├── llm-infra-learning-guide.md      # 路线、知识与验收
├── TODO.md                         # 进度、反馈与下次第一步
├── labs/                           # 所有学习实验；保留已有文件
│   ├── 01-tensor/                  # 已有张量练习
│   ├── 02-memory/                  # 已有内存练习
│   ├── 03-pytorch/                 # Tensor 复盘、autograd
│   ├── 04-vector/                  # 向量、矩阵与余弦实验
│   ├── 05-benchmark/               # 性能测量入门
│   └── 06-attention/               # softmax、causal attention
├── projects/                       # 后期项目，按需创建
│   ├── tiny_decoder/               # W7–12
│   ├── mini_serving/               # W13–18
│   ├── kernels/                    # W19–24
│   └── capstone/                   # 主作品
├── tests/                          # 项目正确性与回归检查
├── reports/                        # 周报与详细实验记录
├── results/                        # 小型 JSON / CSV / 图表
├── configs/                        # 模型与负载配置
├── scripts/                        # 工程辅助脚本
├── reading/                        # 固定 commit 的源码笔记
└── external/                       # 上游源码，不纳入本仓库
```

第 4 课的点积、矩阵乘法和余弦复盘三个文件属于同一组：点积是矩阵乘法与相似度的基础，矩阵乘法连接到模型中的 Linear，余弦相似度连接到向量检索。不能只学习 `03_cosine_similarity.py` 就算完成第 4 课。

每个实验至少包含：输入定义、shape 注释、参考结果、自动断言、运行命令、结论。路径名用两位数保持排序；从项目根目录运行对应的 `python labs/.../文件名.py`；具体路径见各课入口。

周报模板（每周复制一次）：

```markdown
# Week NN：主题
投入：__ / 10 h
前置知识：
本周问题：
环境：OS / CPU / GPU / 驱动 / Python / 包版本 / 上游 commit
复现命令：
输入：shape / dtype / batch / prompt 与输出长度 / 随机种子
正确性：基线、误差容限、通过与失败案例
测量：计时范围、预热、重复次数、原始数据位置
结论：一条数据能支持的结论
局限：没测什么、不能推广到什么
下次第一步：
```

## 4. 36 周逐周安排

各周均沿用“概念 2h + 编码 4h + 阅读 2h + 测量复盘 2h”。下面的验收用于决定能否前进，不以“看完课程”代替。

### A 阶段：W1–6，基础与可测量性

| 周 | 前置 | 学习目标 / 重点 | 编码与阅读任务 | 当周验收 / 交付 |
|---|---|---|---|---|
| 1 | Python 基础语法 | 补函数、模块、异常、列表推导、`with`；掌握 shape / dtype / device | 第 1 课；把一个脚本改成函数 + `main`；读 Python 模块章节 | 解释 `[2,3]`；运行 Tensor 实验；保存环境与 week01 |
| 2 | W1 | 分清 forward、backward、梯度累加与推理；理解预热 / 同步 | 第 2、3 课全部运行；手算 `x²+3x` 在 x=2 的导数 | 三种梯度值一致；benchmark 输出多组耗时而非单次结论 |
| 3 | W1–2 | 点积、矩阵乘法、归一化、余弦相似度 | 第 4 课三个实验；每个各改一组输入 | 手算结果与 PyTorch 对齐；说清 `*` 与 `@`；3/3 通过 |
| 4 | W3 | softmax、概率、causal attention；认识 Q/K/V | 第 5 课；画 `[B,H,T,Dh]` shape 流；读 PyTorch softmax / SDPA 接口 | future 权重为 0、每行和为 1、输出 shape 正确 |
| 5 | W4 | `nn.Module`、参数、Linear、Embedding；最少训练循环 | 做 `Embedding → Linear` 的玩具 token 分类器；使用固定种子 | loss 有下降；能列参数数目；解释 `eval()` 不等于关闭梯度 |
| 6 | W1–5 | dtype、精度误差、stride / view / reshape 与 profiler 入门 | 比较 transpose 后 view/reshape；构建 5 个边界断言 | 阶段 A 演示：10 分钟解释并运行 1–5 课；提交阶段报告 |

### B 阶段：W7–12，从 attention 到缓存推理

| 周 | 前置 | 学习目标 / 重点 | 编码与阅读任务 | 当周验收 / 交付 |
|---|---|---|---|---|
| 7 | W4–6 | 单头与多头 attention、mask、残差 | `tiny_decoder/attention.py`；读选定 HF 模型的 attention forward | 输出 `[B,T,D]`；用 future 扰动证明过去输出不变 |
| 8 | W7 | Embedding、位置编码、norm、MLP、LM head | 完成一个小 decoder block，再堆 2 层；先用简单位置编码 | `[B,T] → [B,T,V]`；参数 / shape 表齐全，无需训练出好语言能力 |
| 9 | W8 | tokenizer、logits、greedy、temperature、top-k | 编写 `generate.py`；先 greedy，再采样；加载一个适合机器的小模型做对照 | 区分 logits 与 token id；固定 seed、固定输入可复现；记录模型 revision |
| 10 | W9 | prefill / decode，KV 缓存的时间与空间代价 | 在 tiny decoder 实现每层 K/V 缓存；画一次 prefill + 两次 decode | 缓存与无缓存同前缀 logits `assert_close`；先用 FP32 CPU |
| 11 | W10 | 缓存位置、长度、mask 偏移与重置 | 增加 T=1、多 token chunk、不同 prompt 长度、两轮请求测试 | 排除错位、串请求、遗漏新 token；记录峰值内存与 KV 估算 |
| 12 | W7–11 | 可复现推理基准 | prompt 长度 × output 长度 × batch 做小矩阵；区分计算与 tokenization | 阶段 B：正确性全部通过；提交缓存开关对比与无加速场景解释 |

### C 阶段：W13–18，推理服务与资源调度

| 周 | 前置 | 学习目标 / 重点 | 编码与阅读任务 | 当周验收 / 交付 |
|---|---|---|---|---|
| 13 | W12、Linux | 真实引擎安装、模型加载、一次在线 / 离线请求 | 主选 vLLM；按对应版文档安装；读入口到请求入队路径 | 环境与命令完整；一条请求可复现；测量不把模型加载混入稳态 |
| 14 | W13 | TTFT、ITL、吞吐、排队时间，流式边界 | 做负载生成器；保存单请求时间戳、输出 token 数、错误 | batch/并发 1、2、4 各一轮；报告 p50/p95 与样本数 |
| 15 | W10、W14 | 静态拼批与 continuous batching | 写离散时间调度模拟器，混合长短请求；读 scheduler | 请求可完成 / 取消；完成后插入新请求；对比等待与完成时间 |
| 16 | W15 | KV block、逻辑 / 物理索引、内存碎片 | 实现 block pool、allocate/free、block table 模拟 | 容量守恒；回收后可复用；无重复所有权与 double-free |
| 17 | W16 | 前缀缓存、引用计数、命中与隔离边界 | 在模拟器加前缀块复用；对照真实引擎功能 | 相同前缀命中；不同后缀正确；请求退出后引用计数正确 |
| 18 | W13–17 | 背压、取消、超时、显存不足与状态清理 | 最小服务接入 toy engine；加入长度上限和有界队列 | 阶段 C：成功、拒绝、取消、失败均能释放资源；输出调度报告 |

### D 阶段：W19–24，GPU 与算子

| 周 | 前置 | 学习目标 / 重点 | 编码与阅读任务 | 当周验收 / 交付 |
|---|---|---|---|---|
| 19 | W6、W12 | GPU 执行、warp/block、寄存器/shared/global memory | 跑 profiler；区分 CPU dispatch、拷贝与 kernel；读 GPU 编程入门 | 能给真实 trace 中三段耗时归因；注明计时范围 |
| 20 | W19 | 带宽、计算量、arithmetic intensity、roofline 基本判断 | 算 GEMM FLOPs 和 softmax 数据读写；测不同 shape | 给出受带宽 / 计算约束的假设，并用数据检验 |
| 21 | W19–20 | Triton program id、arange、mask、load/store | 从官方 vector add 开始；实现非整块长度 | 与 PyTorch 对照，覆盖尾块；编译与稳态耗时分开 |
| 22 | W21、softmax | reduction、数值稳定、算子融合 | 实现行 softmax；固定支持范围；比较非融合 reference | 极端输入 / 非方阵通过；至少 5 个 shape；报告限制 |
| 23 | W21–22 | 分块 GEMM、访存重用、配置选择 | 阅读官方 matmul 教程；改一个 tile 参数做受控实验 | 解释 tile 变化为何影响性能；不强求超过库实现 |
| 24 | W19–23 | 将优化与模型瓶颈联系 | 从模型 trace 选一个小算子；实现或改造并回归 | 阶段 D：正确性报告 + 微基准 + 模型侧影响；未加速也说明原因 |

### E 阶段：W25–30，框架源码与可审阅改动

| 周 | 前置 | 学习目标 / 重点 | 编码与阅读任务 | 当周验收 / 交付 |
|---|---|---|---|---|
| 25 | W13–24 | 固定主框架版本、建立完整调用链 | 锁定 vLLM tag/commit；离线入口 → engine → scheduler → worker | 每个节点写输入、输出、状态、所属进程；不超过一页主图 |
| 26 | W25 | 调度预算、token budget、抢占与公平性 | 跟踪 scheduler 一轮；构造长短请求负载与 block 紧张案例 | 能从日志还原“谁被调度、为什么”；解释队列与容量约束 |
| 27 | W26 | KV 生命周期、模型输入、attention 后端 | 跟一个请求进入 KV 分配到释放；定位实际启用的 backend | commit、入口、分配与释放路径齐全；区分设计图与本次运行分支 |
| 28 | W27、W24 | 图捕获 / 编译 / 动态 shape 的收益与边界 | 只选一个开关做 A/B；固定模型、输入与设备 | 首次成本与稳态收益分别记录；报告显存与 shape 约束 |
| 29 | W25–28 | 量化的格式、误差、加载兼容 | 做 toy int8 量化对照；有条件运行引擎已支持方案 | 对比绝对 / 相对误差、显存与速度；不声称所有 int8 都更快 |
| 30 | W25–29 | 将问题缩成最小复现 | 选测试、诊断、文档错误或小逻辑问题；写补丁和回归 | 阶段 E：本地可审阅 diff、复现命令、相关检查；是否外部提交由本人决定 |

### F 阶段：W31–36，整合与求职

| 周 | 前置 | 学习目标 / 重点 | 编码与阅读任务 | 当周验收 / 交付 |
|---|---|---|---|---|
| 31 | W20、W25 | TP/PP/DP、collective 与通信量 | 用矩阵分片手算 TP；用两进程 Gloo 验证 all-reduce；有多 GPU 再测 NCCL | 分清进程与设备；分片结果对齐；明确哪些只有模拟 |
| 32 | W31 | 分布式推理的瓶颈与故障 | 整理一次通信 / 网络 / OOM 排障；多卡条件不足时做容量模型 | 提供日志、假设、排除步骤与结论；不伪造硬件测试 |
| 33 | W12–32 | 确定主作品问题、基线和实验矩阵 | 锁定 capstone 范围；复用已有缓存、服务和基准；编写 README | 陌生人按文档可跑基线；只有一个主要优化变量 |
| 34 | W33 | 完成改进并验证负面场景 | 运行短长混合、低高并发、缓存命中 / 不命中等矩阵 | 改进与回归都有原始结果；取消 / 错误不泄漏 KV |
| 35 | W34 | 可复现、报告、简历与讲解 | 清理依赖；从新环境重跑；录 5 分钟演示；写两条项目经历 | 不依赖口头补充能复现；每条性能描述可追溯到数据 |
| 36 | W35 | 模拟面试、补齐短板、规划下一步 | 45 分钟项目问答；20 分钟源码追踪；回顾 36 份周报 | 阶段 F：作品集完成；列出可证明能力与尚未实测部分 |

## 5. 阶段验收

### A：基础可解释（W6）

- [ ] 不看答案写出向量点积、矩阵乘法维度规则、稳定 softmax。
- [ ] 说清 `shape`、`dtype`、`device`、广播、`dim=-1`。
- [ ] 第 1–5 课脚本运行完成；刻意制造一个 shape 错误并说明。
- [ ] 用自己的话解释 backward 和预热，不要求复杂微积分推导。

### B：缓存推理正确（W12）

- [ ] 小 decoder 具备生成与每层缓存；参考路径一直保留。
- [ ] FP32、`eval()` 下比较相同序列每一步 logits，先 `rtol=1e-4, atol=1e-5`，再按误差来源调整并记录理由。
- [ ] 覆盖 prompt 长度 1/4/16、多个生成步、分块输入和独立请求重置。
- [ ] 能估算 KV 内存，并展示缓存有额外开销的小输入场景。

### C：服务资源可控（W18）

- [ ] 明确 queued / running / finished / cancelled / failed 状态与转换。
- [ ] 资源池满足 free + owned = total，取消和失败后能够恢复容量。
- [ ] 报告包含负载、错误率、TTFT、ITL、吞吐、并发与采样数量。
- [ ] 能解释为什么提高并发可能增加吞吐同时恶化延迟。

### D：优化有证据（W24）

- [ ] 先证明算子数值正确，再做计时，包含极小、尾块、非方阵、极端值。
- [ ] 至少 5 个 shape，固定设备 / dtype / 线程；首次编译与稳态分开。
- [ ] 微基准收益与端到端收益分别给出，不把局部加速等同整体加速。

### E：源码改动可审阅（W30）

- [ ] 主框架锁到具体 commit，阅读笔记能在该版本定位。
- [ ] 改动有最小复现、预期行为、相关验证、局限。
- [ ] 可以完整追踪一次请求：入口 → 调度 → KV → 模型执行 → 输出 → 释放。

### F：成果可交付（W36）

- [ ] README 有环境、模型 revision、运行命令、输入、结果、成本与限制。
- [ ] 有一份 3–5 页工程报告、一份 5 分钟演示、两条有证据的项目经历。
- [ ] 能回答“为什么这么设计”“在哪些输入上变慢”“如果显存不足怎么办”。

## 6. 源码阅读顺序

### 6.1 版本与路径规则

下面是**候选入口和定位方法**。框架在重构，目录、类名和默认执行后端可能变化；固定源码版本后，以实际调用链为准。`main` 链接方便找到仓库，不是本实验的可复现版本。

```bash
# 在 llm-infra-lab 根目录，只下载源代码阅读；不要求立即编译
git clone https://github.com/huggingface/transformers.git external/transformers
git clone https://github.com/vllm-project/vllm.git external/vllm
git -C external/transformers rev-parse HEAD
git -C external/vllm rev-parse HEAD
```

第一次克隆后先记录实际 SHA。如果你选择某个已安装 release 对应的 tag，用 `git -C external/vllm tag --list` 查到真实 tag，再手动切换，避免把示例占位符误当命令。切换前确认阅读目录没有需要保留的本地修改。不要为了“最新”每周更新；阶段结束再决定升级。

源码笔记每条记录：`repo + commit + 文件 + 符号 + 输入/输出 shape + 状态变化 + 调用方 + 下一个问题`。只读与你的运行配置有关的分支。

### 6.2 第一层：先读自己的实现（W1–12）

顺序：点积 → GEMM → softmax → 单头 attention → 多头 → block → 生成循环 → KV 缓存。自己写过的 shape 流就是阅读上游源码的地图。

### 6.3 第二层：Hugging Face Transformers（W7–12）

官方仓库：[huggingface/transformers](https://github.com/huggingface/transformers)。阅读目标是理解一个模型从 token 到 logits / cache 的执行过程。

| 顺序 | 候选位置 / 符号 | 要回答的问题 |
|---|---|---|
| 1 | `src/transformers/models/<model>/configuration_*.py` | hidden size、head 数、KV head 数、层数来自哪里？ |
| 2 | 同目录 `modeling_*.py` 的 `*ForCausalLM.forward` | 输入 token id 如何到 logits？loss 分支是否执行？ |
| 3 | `*Model.forward` 与 `*DecoderLayer.forward` | 残差、norm、attention、MLP 顺序怎样？ |
| 4 | `*Attention.forward` 或委托的 attention 实现 | Q/K/V 的 shape 如何变化？mask 与位置编码怎样传入？ |
| 5 | `src/transformers/cache_utils.py` 中相关 cache 类 | 哪一层追加 K/V，谁记录当前长度？ |
| 6 | `src/transformers/generation/` 下生成相关实现 | 一次生成循环怎样准备输入、停止、更新缓存？ |

可选 Llama 风格实现作为阅读样本，不要求下载受访问限制的权重。若目标模型采用不同 attention 或混合架构，不能套用所有 Llama 假设。

```bash
rg --files external/transformers/src/transformers | rg 'modeling_llama|configuration_llama|cache_utils|generation'
rg -n 'class .*ForCausalLM|class .*Attention|def forward' external/transformers/src/transformers/models/llama
rg -n 'class .*Cache|def update' external/transformers/src/transformers/cache_utils.py
```

若文件不存在，先 `rg --files` 搜对应模型名，再找类定义与调用点；不要照着旧文章的行号硬找。

### 6.4 第三层：vLLM（W13–30 主线）

官方仓库：[vllm-project/vllm](https://github.com/vllm-project/vllm)；安装与平台支持见 [官方安装说明](https://docs.vllm.ai/en/latest/getting_started/installation/)。先用已支持的模型和后端跑通，再读当前引擎。旧文章中的 V0 目录与 V1 不能混读成一条调用链。

候选阅读顺序：

1. `examples/` 中最小离线推理示例；`vllm/entrypoints/llm.py` 或当前版本等价入口：用户输入怎样封装成请求。
2. 当前版本 engine 的 add/request/step 路径；V1 常从 `vllm/v1/engine/` 定位。先跟 Python 控制流，不钻进所有多进程分支。
3. V1 scheduler 候选 `vllm/v1/core/sched/scheduler.py`：每轮 token 预算、waiting/running、抢占与完成回收。
4. [`vllm/v1/core/kv_cache_manager.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_manager.py) 及其依赖：分配、查缓存、释放如何组合；细节可能委托给其他管理器。
5. worker / model runner，候选 `vllm/v1/worker/`：scheduler 输出怎样变成张量与执行输入。
6. 模型实现候选 `vllm/model_executor/models/`：已知的 Q/K/V、norm、MLP 在并行执行中如何组织。
7. attention 后端选择与 kernels：先确定实际选择了哪个后端，再阅读相关实现；不是所有 attention 都经过同一个 CUDA 文件。
8. 最后补在线服务入口、异步输出、流式、取消与故障清理。Java 服务经验在这里最有用。

```bash
rg --files external/vllm/vllm | rg 'scheduler|kv_cache|model_runner|engine|attention'
rg -n 'class Scheduler|def schedule|def update_from_output' external/vllm/vllm
rg -n 'class KVCacheManager|def allocate_slots|def free' external/vllm/vllm
rg -n 'def add_request|def step|def execute_model' external/vllm/vllm
```

每周只追一个问题。例如“请求被取消后，哪条路径释放物理 KV blocks？”比“今天阅读 vLLM 全部架构”更容易完成。

### 6.5 第四层：Triton / CUDA（W19–24）

按 [Triton 官方教程](https://triton-lang.org/main/getting-started/tutorials/) 的 vector add → fused softmax → matmul 顺序。每步先运行原例、再改输入、最后只改一个实现参数。教程代码与库版本要匹配。

C++ / CUDA 按需补：指针、引用、RAII、模板的基本识读、编译链接、kernel launch、线程索引、同步。先达到“能读懂一个算子”，不在 360 小时内试图学完完整 C++ 体系。

SGLang、TensorRT-LLM、llama.cpp 是后续可比较项目。前 30 周主选一个框架，避免每个都只会启动服务。

## 7. 第 1–4 课复盘与实验入口

以下复盘实验已保存在 `labs/`，仅依赖 Python 标准库和 PyTorch，不下载模型，不需要 API key。从项目根目录运行各节命令。

### 7.1 第 1 课：Tensor、shape、dtype、device、广播

#### 先理解什么

Tensor 是带形状、元素类型和设备信息的多维数组。Java 中的 `float[][]` 可以帮助理解二维形状，但 PyTorch 张量还有 stride、向量化运算、设备执行与自动求导能力。

- `[3]`：3 个元素的向量；`[2,3]`：2 行 3 列；`[B,T,D]`：batch、序列长度、隐藏维度。
- `shape` 是各轴长度；`numel()` 是元素总数；`element_size()` 是每元素字节数。
- `dtype` 决定存储与计算精度；token id 通常使用整数，权重和激活通常是浮点数。
- `device` 决定数据在哪儿；CPU 与 CUDA Tensor 不可随意直接相加。
- 广播从最右侧维度对齐，维度相等或其中一个为 1 时可扩展；不是任意两个 shape 都能算。
- `reshape` 改形状，必要时可能复制；`transpose` 交换维度，通常改变视图的 stride；`view` 对布局要求更严格。

#### 实验入口：`labs/03-pytorch/04_tensor_review.py`

实验实现见 [04_tensor_review.py](labs/03-pytorch/04_tensor_review.py)。本指南只保留概念、运行入口与验收要求。

```bash
python labs/03-pytorch/04_tensor_review.py
```

验收：不运行程序先预测 `x + bias`、`sum(dim=-1)` 和转置后的 shape。把 bias 改成 `[2,1]`，解释为什么变成每一行加一个数。

### 7.2 第 2 课：最少求导、forward 与 backward

#### 求导现在需要学到哪一步

需要理解“一个量变一点，结果变化多快”，能手算简单多项式和链式法则即可。现阶段不用系统重学高数。偏导是多输入函数中，先固定其他输入，只看一个输入改变时输出怎么变；梯度是各输入偏导组成的向量。

| 函数 | 导数 | 例子 |
|---|---|---|
| 常数 c | 0 | `d(5)/dx = 0` |
| x | 1 | x 增加多少，输出也增加多少 |
| xⁿ | n·xⁿ⁻¹ | `d(x²)/dx = 2x` |
| a·f(x)+b·g(x) | a·f′(x)+b·g′(x) | `d(x²+3x)/dx = 2x+3` |
| f(g(x)) | f′(g(x))·g′(x) | `d((3x+1)²)/dx = 2(3x+1)·3` |

**forward（前向计算）**：给输入和参数计算输出，比如 `y=x*x+3*x`。这是模型生成答案也要做的步骤。

**backward（反向传播）**：从一个输出沿计算图反向应用链式法则，算出输出对输入 / 参数的导数。`backward()` 不会自动更新参数。

**forward-mode 自动微分**是另一种求导算法，不能和“前向计算”混为一谈。`backward()` 对应常见的反向模式；本阶段暂不要求实现 forward-mode AD。

API 解释：

| API | 本课中的意义 | 常见误解 |
|---|---|---|
| `requires_grad=True` | 希望追踪相关计算，之后对该输入求导 | 不是立即开始训练 |
| `y.backward()` | 以标量 y 为起点求导，累加到叶子 Tensor 的 `.grad` | 不是“再算一次预测” |
| `x.grad` | 反向传播写入的梯度；开始通常是 `None` | 不是 x 自己的值 |
| `torch.autograd.grad(y,x)` | 返回指定梯度，通常不写入 `.grad` | 不等于优化器更新 |
| `x.grad = None` | 清理上次积累的梯度 | 不会改变 x 的数值 |
| `torch.no_grad()` | 在作用域内停用梯度记录 | 不会自动调用模型 `eval()` |
| `torch.inference_mode()` | 更适合纯推理的上下文，有额外限制 | 不适合无条件替换所有训练代码 |
| `model.eval()` | 切换 Dropout / BatchNorm 等模块行为 | 仍可能记录梯度 |
| `optimizer.step()` | 使用已有梯度更新参数 | 与 `backward()` 是不同步骤 |

官方说明：[自动微分入门](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)。非标量输出通常需要显式起始梯度或先归约成标量；不要对向量直接无参数 `backward()`。

#### 实验入口：`labs/03-pytorch/05_autograd.py`

实验实现见 [05_autograd.py](labs/03-pytorch/05_autograd.py)。本指南只保留概念、运行入口与验收要求。

```bash
python labs/03-pytorch/05_autograd.py
```

验收：解释为什么输出梯度依次是 7、14、7；知道“第二次对同一已释放的图 backward”与“重新 forward 后 backward”是两回事；知道推理优化重点通常不是反向传播。

### 7.3 第 3 课：benchmark、预热、同步与结果解释

**预热**：正式测量前先跑若干次相同操作，使线程池、缓存、运行库初始化、GPU 上下文或编译等一次性成本尽量不混进稳态测量。预热不保证每次耗时相等；冷启动本身也应该单独记录。

CPU 计时器看到的是主机时间。CUDA 通常异步执行，Python 调用返回时 GPU 可能还没算完，因此不能直接用一前一后的主机时间戳声称得到 GPU kernel 时间。下面用同步包围一批操作，测量的是**包含主机提交和同步边界影响的平均调用墙钟时间**，不是纯 kernel 时间。[PyTorch CUDA 语义](https://docs.pytorch.org/docs/main/notes/cuda.html)解释了异步执行与计时注意点。

`time.perf_counter()` 返回高精度单调时钟，差值单位为秒；乘 1000 得毫秒。中位数比单次结果更能抵抗偶发抖动，但仍要保留各组原始样本。

#### 实验入口：`labs/05-benchmark/01_benchmark.py`

实验实现见 [01_benchmark.py](labs/05-benchmark/01_benchmark.py)。本指南只保留概念、运行入口与验收要求。

```bash
python labs/05-benchmark/01_benchmark.py --device cpu --size 256
python labs/05-benchmark/01_benchmark.py --device cpu --size 512 --out results/lesson03_size512.json
# 有可用 CUDA 的机器再运行；不会自动使用 MPS
python labs/05-benchmark/01_benchmark.py --device cuda --out results/lesson03_cuda.json
```

验收：跑 3 次，记录波动；解释第一次为什么可能慢，也可能因噪声不慢。脚本输出的是“每组平均耗时的中位数”，不能当成单请求 p95。学习后续更可靠的计时工具时阅读 [PyTorch Benchmark 教程](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html)。

### 7.4 第 4 课：向量与矩阵——三个实验都要完成

本课的顺序是：**点积 → 矩阵乘法 → 余弦相似度**。前两项为后续 attention 铺路，第三项说明“向量长度”和“方向相似”的差别。它们既可以一次学完，也可以分三次各学约 1 小时。

#### 共同知识与 API

设 `a=[a1,a2,...,ad]`、`b=[b1,b2,...,bd]`：

- 点积 `a·b = Σ ai*bi`，结果是一个标量。
- 长度 `||a||₂ = sqrt(Σ ai²)`。
- 矩阵乘法 `A[m,k] @ B[k,n] = C[m,n]`，其中 `C[i,j]` 是 A 第 i 行与 B 第 j 列的点积。
- 余弦相似度 `cos(a,b)=(a·b)/(||a||₂ ||b||₂)`，非零向量下衡量方向的一致程度。

| API / 写法 | 用途 | 输入 / 输出与注意点 |
|---|---|---|
| `torch.dot(a,b)` | 两个一维向量点积 | `[D],[D] → []`；不用于二维矩阵 |
| `a * b` | 逐元素乘法 | 可广播；不等于矩阵乘法 |
| `(a*b).sum()` | 先逐元素乘再归约 | 一维时等于点积 |
| `A @ B` | 矩阵乘法，调用 matmul 语义 | 支持向量 / 矩阵 / 批量等情况 |
| `torch.mm(A,B)` | 仅二维矩阵相乘 | `[M,K],[K,N] → [M,N]` |
| `torch.bmm(A,B)` | 两个三维批量矩阵相乘 | `[B,M,K],[B,K,N] → [B,M,N]`，不自动广播 batch |
| `torch.matmul(A,B)` | 更一般的乘法 | 高维时最后两维做矩阵乘法，前面的 batch 维按广播规则处理 |
| `transpose(-2,-1)` | 交换最后两维 | attention 中常用来形成 `Kᵀ` |
| `torch.linalg.vector_norm(x,dim=-1)` | 沿特定维度求向量长度 | `[N,D] → [N]`；加 keepdim 则 `[N,1]` |
| `F.normalize(x,p=2,dim=-1,eps=...)` | 按行归一化 | 用 epsilon 下限保护分母；输入输出 shape 相同 |
| `F.cosine_similarity(a,b,dim=-1,eps=...)` | 对齐或广播后求余弦相似度 | 被归约的 dim 消失；不是自动生成全部两两组合 |
| `topk(k)` | 找最大的 k 个值与位置 | 返回 values 和 indices；是否稳定排序等细节看当前接口 |
| `.item()` | 单元素 Tensor 转 Python 数字 | CUDA 上可能触发同步，不放到计时热点里 |
| `torch.testing.assert_close` | 按容限验证数值 | 用于浮点数比对，比直接 `==` 更合适 |

零向量没有数学意义上的方向。API 为避免除零可返回有限值（本实验中为 0），这是一种数值约定，不能解释为“零向量与其他向量语义无关”。

#### 实验一：点积

实验路径：`labs/04-vector/04_dot_product.py`。

实验实现见 [04_dot_product.py](labs/04-vector/04_dot_product.py)。本指南只保留概念、运行入口与验收要求。

```bash
python labs/04-vector/04_dot_product.py
```

预期关键结果：`32.0`、正交点积 `0.0`、相反单位向量点积 `-1.0`。验收：手算 `[2,1]·[3,4]=10`，解释为什么点积不能直接当成纯方向相似度。

#### 实验二：矩阵乘法

实验路径：`labs/04-vector/05_matrix_multiplication.py`。

实验实现见 [05_matrix_multiplication.py](labs/04-vector/05_matrix_multiplication.py)。本指南只保留概念、运行入口与验收要求。

```bash
python labs/04-vector/05_matrix_multiplication.py
```

预期 `A@B=[[22,28],[49,64]]`，Linear 的两个输出行为 `[[-1.5,4.5],[-1.5,10.5]]`。验收：手算 `C[0,0]`；解释 `Linear(3,2)` 的权重为什么是 `[2,3]`；区分二维转置 `.T` 与高维最后两维转置。

#### 实验三：余弦相似度与最小向量检索

实验路径：`labs/04-vector/06_cosine_review.py`。

实验实现见 [06_cosine_review.py](labs/04-vector/06_cosine_review.py)。本指南只保留概念、运行入口与验收要求。

```bash
python labs/04-vector/06_cosine_review.py
```

预期：同方向余弦约为 1；相反方向约为 -1；检索前两名为 `direction_x` 和 `mostly_x`，后者约为 0.97014。验收：把 query 乘以正数，排序不变；把 query 取负，排序改变；解释 `dim=0` 为什么适用于单向量、`dim=-1` 为什么适用于一批行向量。

#### 第 4 课总验收

- [ ] 三个文件均运行通过，不能只完成第三个。
- [ ] 给定 `[B,T,D]` 与 `[D,V]`，写出 matmul 输出为 `[B,T,V]`。
- [ ] 知道点积会随长度变化，余弦对非零向量正数缩放不变。
- [ ] 知道标准 scaled dot-product attention 使用缩放点积，通常没有先把 Q/K 都单位化；不能把本课余弦相似度直接当成 attention。
- [ ] 记录一个真实出错案例（dim 错、shape 不匹配或零向量解释），写出修复原因。

## 8. 第 5 课计划与实验入口

### 8.1 下一课学习目标

把第 4 课的矩阵乘法变成一个最小 causal self-attention：

```text
Q[B,T,D] @ K.transpose(-2,-1)[B,D,T]
    -> scores[B,T,T]
    -> 除以 sqrt(D)
    -> 屏蔽未来位置
    -> softmax(dim=-1)
    -> weights[B,T,T] @ V[B,T,Dv]
    -> output[B,T,Dv]
```

每一行表示“当前 query 从哪些 key 位置取信息”。softmax 将分数变成非负且行和为 1 的权重，权重再用于对 V 加权求和。教学中先用单头和人工 Q/K/V；下一步再学习由同一个 X 通过三个 Linear 得到 Q/K/V。

### 8.2 时间预算与顺序（共 10 小时，可覆盖 W4）

| 时段 | 小时 | 任务 | 产物 |
|---|---:|---|---|
| 概念 | 2 | 手算 softmax；理解 `exp`、行归一化与减最大值 | 手写 `[0,0] → [0.5,0.5]` |
| 实验 A | 1.5 | 实现稳定 softmax，与 PyTorch 对照 | 极端值不会溢出 |
| 实验 B | 1.5 | 用 Q/K/V 算单头 attention，逐行标 shape | 每行权重和为 1 |
| 实验 C | 1 | 加 causal mask，扰动未来输入 | 过去输出保持不变 |
| 阅读 | 2 | 查 softmax、masked_fill、SDPA 参数；对照自己的实现 | 一张 shape 表与接口笔记 |
| 验收复盘 | 2 | 做下面练习，讲解 `sqrt(D)` 与 mask | week04 报告与提交 |

### 8.3 API 解释

- `torch.softmax(x, dim=-1)`：沿最后一维归一化；不是对整个 Tensor 所有元素一起归一化。
- `x.max(dim=-1, keepdim=True).values`：取每行最大值并保留长度为 1 的维度，以便广播减法。
- `masked_fill(mask, value)`：mask 为 True 的位置替换成 value；不同 API 的布尔 mask 约定可能相反，调用前必须看该 API 文档。
- `torch.triu(..., diagonal=1)`：保留严格上三角，用来标记未来位置。
- `torch.finfo(x.dtype).eps`：浮点类型的精度相关常量，不等同于所有归一化函数的 `eps` 参数。

### 8.4 实验入口：`labs/06-attention/01_softmax_attention.py`

实验实现见 [01_softmax_attention.py](labs/06-attention/01_softmax_attention.py)。本指南只保留概念、运行入口与验收要求。

```bash
python labs/06-attention/01_softmax_attention.py
```

### 8.5 进入下一单元前的练习

1. 把所有 logits 设成 0，预测 softmax 结果；在 causal 条件下，每一行可见位置数不同，权重也不同。
2. 关闭 `causal`，重新做 future 扰动；过去位置通常会受影响。保存这一反例。
3. 把 D 从 3 改成 8；检查分母和 shape，不改公式本身。
4. 在不看代码的情况下画出 `QKᵀ → softmax → V`。
5. 解释为何本课“等长上三角 mask”不能直接用于 `query_len=1, key_len=past_len+1` 的缓存 decode：此时 query 的绝对位置不是 0，必须按 past_len 偏移构造可见关系。

通过后再学习多头 attention、Q/K/V 投影和 decoder block。暂时不引入 FlashAttention 实现细节。

## 9. 后期项目规格与性能测量

### 9.1 项目一：`tiny_decoder`（W7–12）

建议接口：

```text
config.py          # vocab_size, hidden_size, num_heads, num_layers, max_seq_len
attention.py       # attention reference、mask、head reshape
model.py           # embedding、block、norm、MLP、lm_head
cache.py           # 每层缓存、长度、追加与 reset
generate.py        # greedy 为基线，采样为拓展
benchmark.py       # 缓存开关与长度 / batch 扫描
tests/             # causal、缓存等价、请求隔离、长度边界
```

接口设计建议：`forward(input_ids, cache=None, use_cache=False)` 返回 logits 和更新后的 cache；输入 `[B,T]`，logits `[B,T,V]`。先只支持同一个 batch 内等长 prompt，后续再处理 padding 和不同请求长度。不要一开始就同时实现复杂采样、量化和分页缓存。

缓存实现顺序：

1. 用 Python 列表按层保存 K/V，先 `cat` 追加，实现正确性。
2. 明确每层存储 `[B,Hkv,T,Dh]` 或其他布局，并在代码中统一。
3. 正确处理位置编码偏移；缓存多了 T 个 token，下次新 token 的位置从旧长度开始。
4. 验证无缓存完整序列与“prefill + 增量输入”的相同位置 logits。
5. 再改为预分配缓冲区与写入 offset，比较反复 `cat` 的分配成本。

**不要只比较生成文本**：两个错误实现也可能输出同一个 argmax。主校验应比较 logits 与中间 K/V 的 shape、长度、数值；greedy token 一致是补充校验。浮点精度降低后可能有小误差，接近并列最大值时 token 可能改变，需要报告而非直接放宽容限掩盖错误。

缓存内存对均匀全注意力层的估算：

```text
KV_bytes = 2 × layers × batch × cached_tokens × kv_heads × head_dim × bytes_per_element
            K/V
```

例如 L=12、B=1、T=1024、Hkv=8、Dh=64、FP16=2 bytes：总量为 25,165,824 bytes，约 24 MiB。GQA 的 `kv_heads` 不等于 query head 数。滑窗、混合层、量化 cache、分布式分片和对齐开销需要单独处理，不能直接套总公式。公式只估算 KV 本体。

### 9.2 项目二：`mini_serving`（W13–18）

先实现“离散 tick 模拟器”，后接真实模型，避免把网络问题与调度问题混在一起。

```text
request.py        # id、到达时间、输入/目标输出长度、状态、取消标记
scheduler.py      # waiting / running、token budget、pick_next
block_pool.py     # free list、block table、引用计数、容量约束
engine.py         # 单步 prefill/decode；初期可使用 toy engine
server.py         # 接口、stream、取消、长度校验、队列上限
loadgen.py        # 固定请求集、并发与到达过程
metrics.py        # 单请求原始时间戳、汇总
```

最低不变量：每个请求最多处于一个活动队列；每个已分配 block 有可追踪所有者；引用计数不为负；free 与 owned 不重叠；超时、取消、失败和成功都能走到最终状态；空闲容量最终恢复。测试调度逻辑时用确定性 toy engine，减少模型随机性干扰。

注意：一个 tick 的 toy 模拟不能当成真实 GPU 毫秒。模拟用于比较策略逻辑；真实吞吐与延迟必须在真实执行器上测量。

### 9.3 项目三：`kernels`（W19–24）

选 softmax 或 RMSNorm 作为主项目，matmul 以阅读与小改为主。接口规格先收窄：支持哪些 dtype、连续布局、轴长度、设备。超范围输入明确拒绝或回退参考实现。

```text
reference.py      # PyTorch 参考实现
kernel.py         # Triton / CUDA 实现
check.py          # shape、边界值、dtype、误差
bench.py          # 预热、同步、重复、结果 JSON
README.md         # 支持范围、命令、图表、未优化原因
```

正确性矩阵建议：行数 1/7/128；列数 1/31/128/513/2048；FP32 起步，FP16 / BF16 仅在设备支持时加入；测试大正值、大负值、全部相等值和尾块 mask。只接受显式支持的 stride，不能悄悄把非连续 Tensor 当连续内存读取。

性能结果至少包含输入读写规模、计时范围、参考实现和 kernel 参数。不要只选最有利的 shape，也不要把 PyTorch 首次编译与已预热自定义 kernel 比较。

### 9.4 最终作品：围绕一个问题做完整闭环（W25–36）

默认建议选 **“混合长短请求下的 KV 资源与调度分析”**，更能利用已有后端经验。另一个可选方向是“一个小算子从 trace 到端到端验证”，二者选一为主。

主作品应复用前期产物，控制新增范围：

- 基线：固定框架与模型，关闭待研究的可选功能，记录默认行为。
- 假设：例如“长 prompt 占用预算，短请求 TTFT 上升”；先从 trace / 指标找证据。
- 改进：只调整一个策略或一个明确开关；若实现自定义 scheduler，先在模拟器验证，不宣称已修改真实引擎。
- 负载：短 / 长 prompt，短 / 长输出，低 / 高并发，前缀重复 / 不重复。
- 对照：相同硬件、版本、精度、数据、到达过程；对每种配置多次运行。
- 结果：正确性、吞吐、TTFT / ITL、显存、错误率和回归情况。
- 解释：为何有效、何处变慢、是否达到设定目标、还有哪些未知。

最终 README 建议包含“问题 → 环境 → 快速运行 → 基线 → 改动 → 实验 → 局限”。简历只写可以复现的结果；没有加速也可展示定位瓶颈、验证假设与修复正确性的工程能力。

### 9.5 指标口径：先定义再测量

| 指标 | 本指南建议定义 | 必须额外记录 |
|---|---|---|
| E2E latency | 客户端发请求到完整结果接收 | 是否包括网络、排队、tokenization |
| TTFT | 发请求到收到首个可识别的生成 token | 不把角色/空文本等首个 SSE 事件当首 token |
| ITL | 相邻生成 token 的到达间隔 | 网络分块可能一块多 token，不能无条件视为真实 token 时间 |
| TPOT | `(末 token 时间-首 token 时间)/(N-1)`，N>1 | 这是平均后续 token 时间，N=1 时未定义 |
| 输出吞吐 | 统一观测窗口内完成输出的 token 总数 / 窗口秒数 | token 如何计数、窗口、是否排除失败请求 |
| 请求吞吐 | 窗口内完成请求数 / 秒 | 成功率、超时与拒绝数 |
| p50/p95/p99 | 对单请求或单 token 样本求分位数 | 样本单位、样本数、工具与分位数算法 |
| 峰值显存 | 按具体工具采集的最高值 | allocated / reserved / 设备进程显存口径 |

服务端 token 时间戳通常比客户端文本分块更适合研究引擎 decode；如果只能测 chunk，就报告“chunk 间隔”，不要伪装成精确 ITL。输出 token 数使用模型 tokenizer 或服务端 usage；字符数不等于 token 数。

压测方法：先 1 个请求确认正确，再运行固定并发 1/2/4/8 的小矩阵。标明是 closed-loop（完成后发下一个）还是 open-loop（按独立到达率发请求）。低样本试跑用于排障；正式报告争取每配置至少百级请求并重复多轮，根据成本与波动决定样本量，不能用几个请求推断可靠 p99。

OOM、超时和失败请求必须计入报告。只保留成功请求的延迟会带来幸存者偏差。报告单次最大值也不能替代分位数。

### 9.6 建议实验矩阵与结果文件

分三轮，而非一口气做所有组合：

| 轮次 | 固定项 | 变化项 | 目的 |
|---|---|---|---|
| 正确性 | FP32、CPU、小模型、固定 seed | T=1/4/16，缓存开关、chunk 长度 | 发现逻辑错误 |
| 单机性能 | 同一 GPU、dtype、模型 revision | batch=1/4，prompt=32/256，输出=16/64 | 判断缓存与长度关系 |
| 服务性能 | 同一输入集、服务配置 | 并发=1/2/4/8，前缀复用开关 | 观察排队、吞吐、显存与延迟 |

一个结果文件最少包括：日期、Git SHA、包版本、硬件、模型 revision、配置、随机种子、命令、warmup、测量次数、计时范围、正确性误差、原始样本路径。把不同机器结果放到不同文件，不直接相减当作同机优化收益。

## 10. 数学、排障、资源与执行清单

### 10.1 按需补数学：每次只解决当前问题

| 时机 | 必须掌握 | 可以暂缓 | 自检 |
|---|---|---|---|
| W1–3 | 向量、坐标、求和符号、长度、点积、矩阵维度 | 特征值分解、证明体系 | 手算第 4 课结果 |
| W2 | 导数直觉、多项式、链式法则、梯度 | 完整积分技巧、复杂反向传播推导 | 算出 7 与 42 |
| W4–8 | 指数、对数、概率归一化、加权平均、均值方差 | 测度论、复杂概率模型 | 解释 softmax 与 norm |
| W9–12 | 条件概率、负对数似然、复杂度估算 | 优化理论全体系 | 解释按前缀预测下一个 token |
| W19–24 | FLOPs、带宽、算术强度、数量级 | GPU 性能精确解析模型 | 算 GEMM 约 `2MNK` FLOPs |
| W29–31 | 量化误差、分片矩阵、通信量 | 从零推导高级量化论文 | 分片结果与未分片结果对齐 |

补数学方法：先用 2 维或 3 维数字手算，再运行代码，最后看公式。每次 20–30 分钟，避免连续几周只看高数。

### 10.2 常见问题与排障顺序

**“Python 能看懂，自己写不出来。”** 从第 4 课删掉输出和断言以外的主体，独立补写，再对比。日常优先补函数参数、可变对象、引用、类、迭代器、上下文管理器和模块导入；先不花大量时间钻 metaclass。

**“shape 总错。”** 每个关键变量旁写 `[B,H,T,Dh]`；逐步 `print(tuple(x.shape))`；先把 batch/head/sequence 全设成不同数字，避免错误的维度交换被相同长度掩盖。

**“不懂求导是不是学不了推理？”** 可以继续推理路线，但要理解参数、前向图与梯度的区别。第 2 课的范围足以支撑起步，复杂训练算法可后补。

**“跑通一个服务是不是已经会 AI Infra？”** 还需证明资源边界、缓存正确性、性能测量、源码追踪和故障清理。部署经验是起点，实验与解释是证据。

**“GPU 比 CPU 慢。”** 先看矩阵是否太小、是否包含数据搬运、是否频繁 `.item()` 同步、是否首次初始化、CPU 线程是否一致、GPU 是否共享。不能预设 GPU 在每个小输入上都更快。

**“改缓存后文本看起来没问题。”** 比较每步 logits；检查位置 offset、缓存长度、层号、请求隔离和 dtype。只看文本不能证明正确。

**“CUDA OOM。”** 先记录失败配置与显存口径；降低 batch、prompt、max output 或模型大小；估算权重与 KV；检查请求结束后的引用。`empty_cache()` 不能释放仍被活跃 Tensor 引用的内存，也不能替代资源生命周期管理。

**“源码目录找不到。”** 确认仓库与 commit，再用 `rg --files` 搜概念词，找定义与调用。记录“旧符号 → 新符号”映射，不复制旧路径假装已读。

**“一个 kernel 快了，服务却没变快。”** 先看它在总耗时中的比例。若只有 10% 时间可优化，即使这部分无限快，理论整体加速也只有约 `1/(1-0.1)=1.11x`，还未考虑新增开销。

**“Triton 跑不了。”** 检查当前版本平台与设备支持。CPU 阶段继续完成参考实现、形状和正确性；GPU 性能章节保留为待测，不能把 CPU 模拟当 GPU 验证。

### 10.3 精选参考资源与使用方式

以下以官方文档、原始论文和项目仓库为主。链接中的 `stable` / `latest` / `main` 会变化，复现实验时记录访问日期与实际版本；本文件不承诺所有链接对应同一个软件版本。前述源码路径是候选定位点，已给出搜索办法。

| 资源 | 用在何时 | 怎么读 |
|---|---|---|
| [Python 官方教程](https://docs.python.org/3/tutorial/) | W1 起 | 模块、类、异常、数据结构按需查；不从编程零基础重来 |
| [PyTorch 安装选择器](https://pytorch.org/get-started/locally/) | 环境准备 | 按 OS、包管理和设备选择命令；记录安装版本 |
| [PyTorch Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html) | W1–6 | Tensors、Autograd、Build Model、Optimization |
| [自动微分教程](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html) | 第 2 课 | 理解计算图和梯度，不扩展到复杂训练 |
| [Tensor.backward API](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.backward.html) | 第 2 课 | 看 gradient、累加与图释放行为 |
| [cosine_similarity API](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.cosine_similarity.html) | 第 4 课 | 对照 dim、广播和 eps；以本机安装版为准 |
| [PyTorch Benchmark](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html) | W2、W12、W24 | 学计时口径与受控比较 |
| [CUDA semantics](https://docs.pytorch.org/docs/main/notes/cuda.html) | W3、W19 | 异步执行、同步、流；版本参数要对照本机 |
| [Transformers 缓存说明](https://huggingface.co/docs/transformers/main/cache_explanation) | W10–11 | 结合自己缓存实现理解位置、长度和追加 |
| [Transformers Cache strategies](https://huggingface.co/docs/transformers/kv_cache) | W11、W28 | 用来查可选 cache 策略，具体支持依模型而异 |
| [Transformers 源码](https://github.com/huggingface/transformers) | W7–12 | 选一个模型，锁 commit，只追一条执行路径 |
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | W7–8 | 首轮看 attention 与结构图，暂不逐式复现训练 |
| [vLLM 文档](https://docs.vllm.ai/) / [安装](https://docs.vllm.ai/en/latest/getting_started/installation/) | W13 起 | 先匹配平台与版本，再查功能和配置 |
| [vLLM 源码](https://github.com/vllm-project/vllm) | W13–30 | 按第 6 节路线定位，安装版本与阅读版本一致 |
| [PagedAttention 原始论文](https://arxiv.org/abs/2309.06180) | W16–17 | 先理解 block table、共享与碎片，再联系代码 |
| [FlashAttention 原始论文](https://arxiv.org/abs/2205.14135) | W20–24 | 重点看 IO 约束与分块思路，暂不要求从零实现 |
| [Triton 官方教程](https://triton-lang.org/main/getting-started/tutorials/) | W21–23 | vector add → softmax → matmul |
| [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) | W19–24 | 查执行模型、内存与同步；按需要选读 |
| [NVIDIA GPU 计算能力](https://developer.nvidia.com/cuda/gpus) | 选择设备 / 排障 | 查硬件特征，再核对引擎要求 |
| [PyTorch Distributed](https://docs.pytorch.org/docs/stable/distributed.html) | W31 | 先看 process group 与 collective，再做分布式推理 |
| [MIT 线性代数公开课](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/) | 按需 | 向量、矩阵、线性变换相关章节，避免成为前置长课 |
| [Dive into Deep Learning](https://d2l.ai/) | W3–12 按需 | 作为数学和 attention 查阅资料，不要求整本完成 |

资源阅读纪律：每周官方材料主读不超过两项，论文只围绕一个实验问题。读完写三句话：解决了什么、还不懂什么、怎样用代码验证。

### 10.4 当前任务与反馈入口

具体执行清单统一维护在 [TODO.md](TODO.md)，不在本指南重复勾选。每次学习后更新任务状态、运行或解释证据、卡点和下次第一步；未完成的验收继续保留。

### 10.5 实验验证记录

7 个复盘脚本已从指南原样拆到 `labs/`。拆分后的运行结果记录在 [TODO.md](TODO.md) 的维护记录中；该结果只表示脚本检查，不代表学习者已完成对应课程。

修改实现后应重新运行受影响的脚本；CUDA、真实推理引擎、Triton 和多 GPU 的结果只能由相应环境实测证明，不能沿用 CPU 检查。
