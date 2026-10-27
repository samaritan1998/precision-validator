# Precision Validator

Precision Validator 用于比较训练配置变化前后的完整梯度、实际优化器更新和前向观测量。可以在 Slime 训练命令中启用验证，也可以通过适配器接入其他训练代码。

默认 pip 安装只依赖 NumPy。运行模型时使用已有的 PyTorch、Slime 和 Megatron 环境。

当前版本是 `0.3.0rc3`。本轮验证已执行完：完整 CPU 训练环境下的 44 项仓库测试、140 项内置测试全部通过，零跳过。GPU 对照仍有模型数值超差，详情见[验证记录](docs/validation.md)。当前保留为候选版本，尚未发布到 GitHub 或 PyPI；已验证安装包的用法见[交付说明](docs/delivery.md)。

## 安装与入口

在本仓库目录安装：

```bash
python -m pip install .
precisionctl --version
precisionctl env
precisionctl selftest
```

也可以安装经过验证的 wheel。当前未声明已发布到 PyPI，请勿根据同名包推断来源。

| 使用场景 | 入口 |
|---|---|
| 对比现有 Slime SFT 训练配置 | [Slime 使用说明](docs/slime-training-check.zh-CN.md) |
| 接入自己的模型或训练循环 | [适配器与实现原理](docs/architecture.md) |
| 先运行不需要下载模型的例子 | [README 中的 CPU 示例](README.md#run-a-portable-cpu-example) |
| 确认哪些结果已经验证 | [验证记录](docs/validation.md) |

## 在 Slime 训练命令中启用

先创建独立的 Slime 源码副本。安装器只修改新副本：

```bash
python -m precision_validator.slime_check install-bridge \
  --source /path/to/slime \
  --destination /path/to/slime-with-precision-check
```

所有训练 rank 必须使用相同的 wheel 和新副本。保留原有训练参数，在命令末尾追加：

```bash
python /path/to/slime-with-precision-check/train.py "${TRAIN_ARGS[@]}" \
  --precision-check-steps 3 \
  --precision-check-reference tp=1,etp=1 \
  --precision-check-dir /shared/checks/tp-check-001
```

原训练配置是 `candidate`。程序从相同初始权重分别启动 `reference`、`repeat`、`candidate`，每轮执行 3 个真实 optimizer step，然后退出并生成报告。当前自动入口复用同一份资源，三轮依次运行。

`tp=1,etp=1` 只修改参考轮的 TP 和 ETP。未指定的 CP、PP、EP 沿用原配置。DP 根据总训练卡数计算。指定 3 步表示验证前 3 个更新步骤，不表示从训练第 3 步开始采样，也不表示验证结束后继续正式训练。

没有设置验收策略时，报告状态为 `report_only`。要获得 `passed` 或 `numerical_failure`，必须在运行前通过 `--precision-check-policy` 提供全局、局部参数和前向门限。

## 常见并行策略

下表是配置示例，不能代替对应模型的 GPU 验收。两种拓扑都需要能容纳模型和相同输入。

| 原配置中需要检查的变化 | 参考参数 | 保持的其他配置 |
|---|---|---|
| TP2 与 TP1 对比 | `tp=1,etp=1` | 原 PP、CP、EP |
| CP4 与 CP1 对比 | `cp=1` | 原 TP、PP、EP、ETP |
| PP2 与 PP1 对比 | `pp=1` | 原 TP、CP、EP、ETP |
| EP2 与 EP1 对比 | `ep=1` | 原 TP、PP、CP、ETP |
| TP2/CP2 与 TP1/CP1 对比 | `tp=1,etp=1,cp=1` | 原 PP、EP |

ETP 是 expert tensor parallelism。DP 是 dense data parallelism；expert 的数据并行分组还受 EP 和 ETP 约束，不能把所有并行度简单相乘来计算总卡数。

## 报告的含义

梯度是完成归约、反缩放后，裁剪前的训练梯度。更新是实际 `optimizer.step()` 前后优化器主参数的差值。使用 BF16 模型参数、FP32 主参数时，更新比较使用 FP32 主参数。

报告包括最大绝对误差、绝对 L2、RMSE、相对 L2，以及逐参数和参数组结果。参考值接近零时使用单独的绝对误差规则。局部参数失败或已声明的 loss 门限失败，会使严格验收失败。

`passed` 只表示本次输入、模型、配置和已启用门限通过。它不证明任意模型都能运行，也不证明两边没有共同的实现错误。模型在某个配置下产生 `numerical_failure`，可能是验证器正确发现了训练差异，需要结合原始张量和独立对照定位。

## 当前边界

Slime 自动入口支持同步 `train.py`、文本 SFT、全新优化器、HF 初始权重、原生 distributed Adam 和单个模型 chunk。尚不支持从任意训练中间步骤采样、检查后继续训练、RL、多模态或虚拟流水线。Slime 实际参数精度由其训练配置决定；库存 Slime 的默认训练精度不能仅由验证器改成 FP32。

多步、checkpoint 恢复、Adam 状态向量和 FP32/BF16 存储精度对比有单独的 trajectory 适配器。这些能力不能自动算作 Slime 一键入口的覆盖。

原始梯度和更新需要较大空间。若有 P 个训练参数、每轮 N 步，三轮 FP32 梯度和更新的基础存储量约为 `24 × P × N` 字节，另需临时空间。权重、数据集和原始训练张量不随开源仓库分发。
