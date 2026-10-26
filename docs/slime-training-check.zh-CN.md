# 在 Slime 训练入口中开启精度对照

开发版本：`precision-validator 0.3.0rc3`。这是 Slime 配套功能：用户沿用现有训练命令，通过参数指定验证步数和参考拓扑；无需为每个模型另写 `adapter.py`。

当前首版仅支持同步 `train.py`、文本 SFT、HF 初始权重、全新原生 distributed Adam、单个模型 chunk。发布候选版的验证状态以 validation.md 为准，不能沿用旧镜像的通过结论。

## 用户如何使用

维护者先安装开发版 wheel，并准备带接入点的 Slime 副本。之后，使用者在原训练命令后追加：

```bash
# TRAIN_ARGS 保持原有模型、数据、GBS、LR 和候选并行配置。
# 例如原命令的 TP=2；这里将 TP=1/ETP=1 作为参考。
python "$SLIME_ROOT/train.py" "${TRAIN_ARGS[@]}" \
  --precision-check-steps 3 \
  --precision-check-reference tp=1,etp=1 \
  --precision-check-dir /shared/checks/my-model-tp-3steps
```

- `--precision-check-steps 3`：每轮执行前 3 个真正的 optimizer step，随后退出。一个 step 可能包含多个 microbatch；它不是 3 个 rollout，也不是第 3 步才开始验证。
- `--precision-check-reference tp=1,etp=1`：只覆盖参考配置中的这些轴。其余 TP/PP/CP/EP/ETP 沿用训练配置；DP 根据同一分配卡数重新计算。原训练配置作为 candidate。
- `--precision-check-dir`：所有 rank 可访问的新目录。不能覆盖以前的验证结果。
- 只设置步数而不设置 reference 时，三轮拓扑相同，用于观察自身重复差异；这不构成 TP1/TP2 或 CP1/CP2 验收。

CP 对照可写 `--precision-check-reference cp=1`；同时检查 TP/CP 可写 `tp=1,etp=1,cp=1`。模型和 Slime 本身必须支持目标拓扑，配置参数不会替模型实现并行计算。TP/PP/CP 乘积需要整除分配卡数，expert 拓扑也须满足整除关系。参考拓扑需要能够装入显存。

## 并行策略示例：原训练参数与参考参数分别写什么

先看这一条规则：**原训练命令定义 candidate；`--precision-check-reference` 只覆盖参考轮中明确写出的轴。repeat 自动使用 reference 的配置。** 不写的轴沿用原训练配置，不会自动变成 1。

例如，分配 8 张训练卡，原训练配置是 TP1 / PP2 / CP4，额外写 `--precision-check-reference cp=1`，三轮实际运行如下：

```text
同一初始权重、同一组全局样本，每轮独立开始
  reference：TP1 / PP2 / CP1 / DP4，更新 1 → 2 → 3
  repeat：   TP1 / PP2 / CP1 / DP4，更新 1 → 2 → 3
  candidate：TP1 / PP2 / CP4 / DP1，更新 1 → 2 → 3

第 1 步：reference[1] 对 repeat[1]、candidate[1]
第 2 步：reference[2] 对 repeat[2]、candidate[2]
第 3 步：reference[3] 对 repeat[3]、candidate[3]
```

这是三个独立的三步训练，不是在同一个模型上连续更新九步。当前一键入口复用同一份资源，按 reference、repeat、candidate 的顺序启动；`--precision-check-reference` 也不会自动枚举所有并行组合。

### 参数名称对照

| 缩写 | Slime 原训练参数 | 在本例中的含义 |
| --- | --- | --- |
| TP | `--tensor-model-parallel-size` | 将层内张量计算分给多张卡 |
| PP | `--pipeline-model-parallel-size` | 将模型层分配到不同流水线阶段 |
| CP | `--context-parallel-size` | 将同一输入序列分给多张卡 |
| EP | `--expert-model-parallel-size` | 在 MoE 模型中分配不同 expert |
| ETP | `--expert-tensor-parallel-size` | 将单个 expert 的张量计算分给多张卡 |
| DP | 不在 reference 字符串中直接设置 | 用不同的数据副本参与同一个全局训练步骤，由卡数和 TP/PP/CP 推导 |

以下场景均分配 **1 台 × 8 张训练卡**。模型、数据、精度、loss、全局 batch 和 LR 保持原设置；为说明数据分配，假设示例的全局 batch 是 8。示例用来说明参数含义，不代表这些模型规模和拓扑均已通过 GPU 验收。

### 选择要比较的变化

表中箭头按「参考 → 原训练配置」排列。未显示的 EP、ETP 均为 1；TP 示例另行显示 ETP。

| 要比较的变化 | 原训练配置：candidate | 追加的 reference 值 | 自动得到的参考配置：reference / repeat |
| --- | --- | --- | --- |
| CP1 → CP2 | TP1 / PP2 / CP2 / DP2 | `cp=1` | TP1 / PP2 / CP1 / DP4 |
| CP1 → CP4 | TP1 / PP2 / CP4 / DP1 | `cp=1` | TP1 / PP2 / CP1 / DP4 |
| TP1 → TP2；ETP 随 TP 变化 | TP2 / PP2 / CP1 / ETP2 / DP2 | `tp=1,etp=1` | TP1 / PP2 / CP1 / ETP1 / DP4 |
| PP1 → PP2 | TP1 / PP2 / CP1 / DP4 | `pp=1` | TP1 / PP1 / CP1 / DP8 |
| MoE 的 EP1 → EP2 | TP1 / PP2 / CP1 / EP2 / DP4 | `ep=1` | TP1 / PP2 / CP1 / EP1 / DP4 |
| TP、CP 联合变化 | TP2 / PP2 / CP2 / ETP2 / DP1 | `tp=1,etp=1,cp=1` | TP1 / PP2 / CP1 / ETP1 / DP4 |
| 在 TP2 条件下只改变 CP | TP2 / PP2 / CP2 / ETP2 / DP1 | `cp=1` | TP2 / PP2 / CP1 / ETP2 / DP2 |
| 同一配置重复 | TP1 / PP2 / CP2 / DP2 | 不传该参数 | TP1 / PP2 / CP2 / DP2 |

MoE 模型中，`tp=1,etp=1` 同时改变普通层 TP 和 expert 内部 ETP。如果只想检查普通层 TP，应在原训练命令中先明确 ETP=1，参考也保留 ETP=1，再写 `--precision-check-reference tp=1`。只写 `tp=1` 不会重置已经确定的 ETP。

### 示例 1：已有 CP4 训练命令，只追加验证参数

假设 `TRAIN_ARGS` 已包含模型、数据、优化器，以及以下原训练配置：

```text
--actor-num-nodes 1
--actor-num-gpus-per-node 8
--tensor-model-parallel-size 1
--pipeline-model-parallel-size 2
--context-parallel-size 4
--expert-model-parallel-size 1
--expert-tensor-parallel-size 1
```

这组参数已经在 `TRAIN_ARGS` 中，无需再添加一次。只在原命令后追加：

```bash
python "$SLIME_ROOT/train.py" "${TRAIN_ARGS[@]}" \
  --precision-check-steps 3 \
  --precision-check-reference cp=1 \
  --precision-check-dir /shared/checks/cp1-vs-cp4-001
```

候选轮沿用 CP4；参考和重复轮改为 CP1，PP 仍为 2。若原训练配置是 CP2，同样写 `cp=1`，比较的就是 CP1 与 CP2，无需把候选 CP 值再写进验证参数。

### 示例 2：原配置 TP2 / PP2 / CP1，检查 TP1 与 TP2

假设原训练命令是 TP2 / PP2 / CP1 / EP1 / ETP2。追加：

```bash
python "$SLIME_ROOT/train.py" "${TRAIN_ARGS[@]}" \
  --precision-check-steps 3 \
  --precision-check-reference tp=1,etp=1 \
  --precision-check-dir /shared/checks/tp1-vs-tp2-001
```

两轮参考使用 TP1 / ETP1；候选使用原来的 TP2 / ETP2。PP 和 CP 不变。若原模型是 MoE，这个例子同时覆盖普通层 TP 和 expert 内部 ETP 的变化。

### 示例 3：原配置 PP2，检查流水线切分

假设原训练命令是 TP1 / PP2 / CP1 / EP1 / ETP1。追加：

```bash
python "$SLIME_ROOT/train.py" "${TRAIN_ARGS[@]}" \
  --precision-check-steps 3 \
  --precision-check-reference pp=1 \
  --precision-check-dir /shared/checks/pp1-vs-pp2-001
```

参考为 PP1 / DP8，候选为 PP2 / DP4。PP1 参考必须能够装入显存，PP2 的模型层数分配也必须有效。此示例未启用 virtual pipeline；当前入口仍不支持 virtual pipeline。

### 示例 4：MoE 原配置 EP2，检查 expert 分布

假设原训练命令是 TP1 / PP2 / CP1 / EP2 / ETP1，模型的 expert 数可按 EP2 分配。追加：

```bash
python "$SLIME_ROOT/train.py" "${TRAIN_ARGS[@]}" \
  --precision-check-steps 3 \
  --precision-check-reference ep=1 \
  --precision-check-dir /shared/checks/ep1-vs-ep2-001
```

参考将 expert 放在 EP1 配置下，候选使用 EP2。两边 dense DP 都为 4；expert 数据并行度 EDP 从参考的 4 变为候选的 2。模型 expert 总数、router 配置和训练数据不随 EP 改变。Dense 模型没有这项 expert 分布对照。

### 示例 5：原配置 TP2 / PP2 / CP2，联合验证与分别定位

如果关心这组生产配置整体是否对齐，追加：

```bash
python "$SLIME_ROOT/train.py" "${TRAIN_ARGS[@]}" \
  --precision-check-steps 3 \
  --precision-check-reference tp=1,etp=1,cp=1 \
  --precision-check-dir /shared/checks/tp2-cp2-joint-001
```

实际比较「TP1 / PP2 / CP1」与「TP2 / PP2 / CP2」。联合对照若超差，不能仅凭这份报告判断是 TP 还是 CP 引起。

若要保持 TP2，先定位 CP 的变化，则在另一次实验中只写 `cp=1`：

```bash
python "$SLIME_ROOT/train.py" "${TRAIN_ARGS[@]}" \
  --precision-check-steps 3 \
  --precision-check-reference cp=1 \
  --precision-check-dir /shared/checks/cp-under-tp2-001
```

此时实际比较「TP2 / PP2 / CP1」与「TP2 / PP2 / CP2」，ETP 保留原值。反过来，写 `tp=1,etp=1` 而不写 CP，会保留 CP2，检查 CP2 条件下的 TP/ETP 变化。它不会自动额外执行 TP1/CP1 的对照。

### 为什么 DP 跟着变，batch 是否也会变

这个入口在三轮中使用相同卡数。普通层拓扑满足：

```text
卡数 = TP × PP × CP × DP
```

8 卡、TP1、PP2 时，CP4 对应 DP1，CP1 对应 DP4。这是两套有效拓扑之间的对照，包含 DP 归约方式的变化，不能解释为「除 CP 外所有计算路径都完全不变」。

全局 batch 仍须保持一致。以全局 8 个样本为例：CP4/DP1 一侧的一个 DP 副本参与这 8 个样本；CP1/DP4 一侧在 4 个 DP 副本之间分配同一组 8 个样本。每卡 microbatch 数可以变化，但样本贡献、监督 mask 和 loss 归一化必须一致。库检查实际消费数据的一致性；它不会通过重复样本来自动修正不合适的 batch 配置。

EP 不再乘进上面的普通层卡数公式。本库对 expert 另行检查 `卡数 = PP × EP × ETP × EDP`。因此 EP1 → EP2 可以保持 dense DP 不变，同时改变 EDP。

当前一键入口不能为参考和候选分别申请不同卡数，也不能通过 `--precision-check-reference dp=...` 强行保持 DP。若要求 DP 固定，需要另一种资源编排方式。

### 从这些示例中选择

- 原配置用了 CP2 或 CP4，先看 CP 对齐：写 `cp=1`，保留必要的 PP、TP。
- 原配置用了 TP2，检查 TP/ETP 一起变化：写 `tp=1,etp=1`。
- 原配置用了 PP2，且 PP1 能装入显存：写 `pp=1`。
- MoE 原配置用了 EP2，且 EP1 能装入显存：写 `ep=1`。
- 只想先确认同配置的重复误差：不传 reference，三轮保持当前配置。

每个命令都是一项独立实验，需要新输出目录。示例未设置 `--precision-check-policy`，因此生成的是误差报告 `report_only`；需要正式 passed/failure 判定时，再追加已审查的门限文件。生产模型的完整原始数据可能很大，应按「磁盘和资源」中的计算方式设置磁盘预算。

## 开启参数后实际发生什么

启动入口自动执行以下过程：

1. 固定源码、验证库、参数选择和门限，生成 manifest。
2. 从相同 HF 初始权重独立启动 reference、repeat、candidate 三轮正常 Slime 训练。
3. 每个 GPU rank 在原有 `train_one_step` 外安装采集器。训练函数继续执行真实的前向、反向、归约、裁剪、Adam 更新和 scheduler 更新。
4. 每轮到达 N 个 optimizer step 后停止。即使当前 rollout 还计划了更多 step，也不会多执行一次更新。
5. 收齐全部 rank 和进程成功退出回执，在 CPU 上生成 `acceptance.json` 与 `report.md`。

用户只提交一次启动命令；当前驱动在同一资源分配上顺序重建三轮训练进程。训练计算量大致相当于三轮各 N 步，另加模型加载、数据校验、张量写入和比较开销。独立的 GPU 集成测试可将三轮分配到互不重叠的 GPU 并行运行。

当前模式在验证结束后退出，不会继续长训，也不会写入原训练 checkpoint 目录。模型、GBS、loss、LR 和 scheduler 计算不被验证器替换。正式训练仍使用原命令，并移除验证参数。

## 精度从哪里采集

Slime 的 `train_one_step` 已经调用了 optimizer.step，因此配套库只观察原调用，不额外执行第二次更新：

```text
原 train_one_step
├─ 原有 forward / backward / 梯度归约
├─ 原生 optimizer.prepare_grads
│  └─ 采集归约后、裁剪前的 FP32 梯度
├─ 原生 optimizer.step
│  └─ 采集实际 FP32 主参数的 after − before
└─ 原有 scheduler 更新与梯度清理
```

TP/PP/EP 参数映射来自 Megatron 原生分片元数据；DP 参数所有权来自原生 optimizer 分片范围。比较前恢复为同一参数的全局坐标，并检查覆盖、重复副本和实际通信组。未知分片结构会报错。

每一步还记录实际 DataIterator 返回的 token、mask 和其他已使用字段的哈希。在 CP 切片前采集，先检查同一 DP shard 上 TP/CP/PP 副本一致，再统计全局样本贡献。相同样本被使用两次时保留两次贡献，不做去重。三轮的输入、初始化、实际 optimizer 配置、学习率和 scheduler 进度不一致时，不能给出数值通过结论。

目前的前向观测量是 loss；没有自动采集完整 logits。Adam 状态检查 dtype、有限性和步数，但不做状态向量逐元素比较。报告会明确标出这些范围。

## 怎么判定通过

没有提供门限文件时，只输出 `report_only` 和误差，不自动宣布 passed。要进行验收，在运行前额外指定：

```bash
--precision-check-policy /shared/policies/my-model.json
```

这个 JSON 包含两个顶层字段：

- `policy`：完整梯度和更新的全局门限，包括自身重复噪声约束。
- `acceptance`：全部参数、分组及 loss 的绝对/相对门限，并包含接近零参考值的处理规则。

结构与已有验证库的 `spec.policy`、`spec.acceptance` 相同。可以从已经审查的模型策略导出；不能把小模型集成测试门限直接用于任意生产模型。策略在准备阶段冻结，不能看到误差之后再修改本轮门限。

逐步报告包含最大绝对误差、绝对 L2、RMSE、相对 L2、逐参数和分组误差，以及失败 gate 的原因。任意一个声明门限失败，都不会把整轮标记为 passed。

## 磁盘和资源

若可训练参数数量为 P，每轮每步保存梯度和更新两份 FP32 向量。三轮 N 步的原始数据约为 `24 × P × N` 字节，此外需要初始化校验和临时空间。27B 模型验证 1 步，仅这两类原始向量就约 648 GB。

默认磁盘预算为 1024 GiB，可用 `--precision-check-disk-gib` 调整；预算不足时明确报错。不要把“只跑一步”理解为只写少量日志。此首版保留原始证据，清理应在报告确认后另行执行。

## 维护者首次接入

安装 wheel 后执行一次：

```bash
python -m pip install /path/to/precision_validator-0.3.0rc3-py3-none-any.whl
python -m precision_validator.slime_check install-bridge \
  --source /path/to/existing-slime \
  --destination /path/to/new-isolated-slime
```

命令复制源码，再修改副本中的两个文件：

| 文件 | 接入内容 |
| --- | --- |
| `train.py` | 注册可选参数、组织三轮子进程、达到步数后结束驱动循环 |
| `slime/backends/megatron_utils/model.py` | 在实际训练循环选择观察器；达到步数后退出内部循环 |

不会原地修改 source，也不会替换 attention、GEMM、loss 或优化器算法。没有开启验证参数时，不导入配套库，仍走原训练调用。安装器只接受已识别的源代码边界；不能识别时拒绝修改。

需保证所有 Ray worker 都能使用安装后的库和同一份 Slime 副本。仅在 driver 环境安装而 worker 环境缺包，会导致启动失败。

## 当前限制与尚待验证项

- 首版不支持 `train_async.py`、RL 在线生成、多模态、virtual pipeline、多个模型 chunk、自定义 before-step hook 或 stateful model buffers。
- 从 checkpoint 恢复后的中途抽检，以及“验证通过后继续当前训练”尚未接入本入口。
- TP/PP/CP/EP 是否能够通过，需要在实际模型、卡型、输入和配置上运行；不能由接口支持这些轴推导所有组合已通过。
- 当前 GPU 集成用例是两层小模型、TP1/TP2、每轮 3 步，同时检查原生训练循环的步数上限。它不替代生产 Qwen 256K 或完整 Ray 驱动验收。

## 发布验证状态

当前是发布候选版，尚未完成最终验收。历史开发版结果不计入本版本通过数。
当前 B200、H800、A800 的小模型矩阵已完成：每种卡型 8 组通过、2 组 BF16 MoE 数值超差。完整 Ray 入口和真实 256K 对照仍在验证，不能把小模型结果扩大为这些范围的通过结论。各项结果和范围见 [validation.md](validation.md)。
