# Name That LLM：Qwen3-8B 两阶段 LoRA

本仓库保存目前最佳方案的训练代码、实际提交推理代码和复现步骤，直接在 实验室服务器 上整理并上传。

| 指标 | 数值 |
|---|---:|
| 五折 OOF Balanced Accuracy | 0.556403 |
| 最终模型训练集 BA | 0.804362 |
| Kaggle Public Score | **0.524** |
| 用户记录的排名（2026-11-20） | **4096** |

线上成绩及排名由用户提供；排名所属榜单尚未独立核实。提交编号 58589746，Notebook 为 `yifeiorange/qwen8b-api-write-probe` 的 **version 8**。该名称沿用调试阶段，但本仓库保存的是正式提交的推理代码。

## 方法

基础模型为 Qwen/Qwen3-8B-Base，固定 revision：
`49e3418fbbbca6ecbdf9608b4d22e5a407081db4`。

保留前 24 层 decoder，最终 norm 改为 Identity；融合第 2、4、6、8、12、16、20、24 层。各层采用 masked mean pooling 和独立 FP32 LayerNorm，拼接为 32768 维表示，再分类为 18 类。

LoRA 覆盖 q/k/v/o/gate/up/down 投影，r=16、alpha=32、dropout=0.05。训练同时优化普通分类头和同提示三样本中心化辅助分类头，组合目标为两项交叉熵之和。提交时只使用普通头，不需要测试数据提供 prompt_id 或完整类别组。

第一阶段在 35,236 条清洗后的外部数据上训练 1 轮；第二阶段在官方数据上训练固定 2 轮。三样本组来自同一 prompt 的不同标签；每轮是按数据量确定的抽样预算，不保证每条样本恰好访问一次。

## 文件目录

| 文件 | 作用 |
|---|---|
| scripts/train.py | 可配置根目录的两阶段、五折及最终训练入口 |
| archive/train_original.py | 原实验完整训练脚本，保留原路径及旧实验校验 |
| scripts/download_model.py | 下载固定 revision 并保存准备状态 |
| scripts/prepare_pool.py | 清洗外部数据，不运行原脚本附带的 SVC 实验 |
| scripts/export_kaggle.py | 导出训练参数、tokenizer、配置及基础权重校验清单 |
| scripts/infer_kaggle.py | 实际提交的双 GPU FP32 推理代码 |
| scripts/inference.ipynb | 实际提交 Notebook |
| configs/protocol.json | 训练协议记录，不是训练入口自动读取的配置文件 |
| configs/kernel-metadata.json | 实际 Notebook 输入来源与离线设置 |
| results/training_history.csv | 每轮训练 BA、验证 BA、训练损失等原始记录 |
| results/ | 五折结果、分类报告、数据审计、显存及部署验证 |

## 训练环境

原训练使用单张 RTX 5090。创建全新环境，避免修改已有环境：

```bash
git clone https://github.com/YifiCheng/Name-That-LLM.git
cd Name-That-LLM
python3 -m venv .venv-train
source .venv-train/bin/activate
python -m pip install -r requirements-train.txt
export NTL_ROOT="$PWD"
python scripts/train.py --self-test
```

requirements 从实际 实验室服务器 环境读取，包含 PyTorch 2.10.0+cu128、Transformers 5.20.2、PEFT 0.21.2。还需要兼容的 NVIDIA 驱动。不同硬件、软件及算子可能带来数值差异，不保证逐位复现。

训练入口要求显卡空闲显存至少 27,000 MiB，连续三次空闲检查后才启动；进程显存上限比例为 0.8，并进行实际训练预检。原记录峰值 reserved 约 12.9 GiB，部署仍应保留加载及其他进程的显存余量。

## 数据准备和复现边界

首先接受[比赛规则并下载官方数据](https://www.kaggle.com/competitions/name-that-llm/data)。使用自己的 Kaggle 认证，不要向仓库提交 Token。

```text
$NTL_ROOT/
  data/official/train.csv
  data/external/two_stage_public/pool_v1.parquet
  data/external/extended_public/passages_clean.csv
```

官方 CSV 必须包含 id、text、model、prompt_id；pool 需要 text、model、prompt_key、source；扩展 CSV 需要 text、model、prompt_id、split。

策略与外部 pool 来源：
- [Two-stage LoRA on the public pool](https://www.kaggle.com/code/nishantdahal/two-stage-lora-on-the-public-pool)
- [nishantdahal/ntl-two-stage-lora](https://www.kaggle.com/datasets/nishantdahal/ntl-two-stage-lora)
- prvsiyan 的公开扩展数据在本实验保存为 passages_clean.csv。原始 Dataset ID、版本和快照尚未可靠固定，不能给出保证获取同一文件的下载命令。完整重训需要取得原始两份快照，或原实验的清洗 parquet。

```bash
python scripts/prepare_pool.py
python scripts/download_model.py
```

清洗产物位于 experiments/public_pool_strategy_pilot_20261113_v1/。删除精确重复、标签冲突、官方文本重合及已知重生成数据；按提示组 TF-IDF 相似度阈值 0.45 排除可能重叠组。预期剩余 35,236 条；对照 results/run_manifest.json 中的官方 CSV 和清洗 parquet SHA256。Parquet 序列化版本也可能影响文件哈希。

词面筛查无法保证语义提示独立，外部数据准备使用了官方训练数据的排版特征筛选。因此本地验证是探索性证据，不能宣称完全独立确认。数据不随本仓库重新分发，须遵守来源许可和比赛规则。

## 五折和最终训练

官方 2,699 条样本按 prompt_id 使用 GroupKFold(5)，同一个提示不跨训练集与验证集。前四折分别为 2,159/540 条，第五折为 2,160/539 条。

```bash
python -u scripts/train.py \
  --root "$NTL_ROOT" \
  --out "$NTL_ROOT/experiments/qwen3_8b_two_stage_20261113_v1"
```

使用新输出目录。第一折第二轮 BA 达到 0.46 后，自动完成其他四折；每折均恢复相同外部阶段权重，再训练两轮。OOF 达到 0.46 后，从外部阶段权重重新在全量官方数据上训练两轮。未通过门槛则不会自动生成最终模型。两轮是预先固定设置，不是事后证明的最佳轮数。

| Fold | 第二轮验证 BA |
|---|---:|
| 1 | 0.538889 |
| 2 | 0.568519 |
| 3 | 0.557407 |
| 4 | 0.557407 |
| 5 | 0.559770 |
| 汇总 OOF | 0.556403 |

最终提交权重为 full_fit/final_epoch2，没有直接挑选最高分折，也没有采用五折集成。训练集 BA 为 0.804362，不能作为泛化表现。逐轮数据见 results/training_history.csv：train_set_loss 为普通头评估交叉熵，optimization_loss 为实际组合训练目标，二者不可混用。

## Kaggle 推理复现

```bash
python scripts/export_kaggle.py
```

生成 kaggle_cli_upload/qwen8b_two_stage_dataset/qwen8b_two_stage_bundle.zip，包含训练参数、模型配置、tokenizer、265 个基础张量的 BF16 SHA256 清单；基础模型另行挂载。

已训练资产：[Kaggle Dataset](https://www.kaggle.com/datasets/yifeiorange/qwen3-8b-ntl-two-stage-20261119)。该 Dataset 为私有资产，需要所有者授权；仅克隆本仓库无法自动取得。权重不存入 GitHub。

在 Kaggle Notebook 添加比赛数据、bundle Dataset，以及基础模型 `qwen-lm/qwen-3/Transformers/8b-base/1`。选择 **T4 ×2**，关闭 Internet，运行 scripts/inference.ipynb 或 scripts/infer_kaggle.py。

实际通过验证的 Kaggle 环境：PyTorch 2.11.0+cu130、Transformers 5.20.2、safetensors 0.8.0；使用 Accelerate balanced 设备映射、FP32、batch size 2、最大长度 384。推理手动实现 LoRA，不依赖 PEFT。Kaggle 镜像可能变化，需先完整运行验证兼容性。

输出 /kaggle/working/submission.parquet，列为 id、model。完成 Save Version / Run All 后再提交成功的版本。Kaggle 服务器会替换隐藏数据并重跑，不是在 实验室服务器 直接计算官方分数。

CLI 入口：将 configs/kernel-metadata.json 复制至 Notebook 所在目录，修改 id 为自己的 Notebook，修改 Dataset ID，确保 code_file 与实际文件匹配。

```bash
kaggle kernels push -p ./your-notebook-directory
kaggle kernels status YOUR_USER/YOUR_NOTEBOOK
kaggle competitions submit name-that-llm \
  -k YOUR_USER/YOUR_NOTEBOOK -v YOUR_COMPLETED_VERSION \
  -f submission.parquet -m 'Qwen3-8B two-stage epoch2'
```

原实验使用 Kaggle CLI 2.2.5；其他版本请先核对 --help。

4,000 条规模检查使用官方训练样本及重复样本，双 T4 推理约 32.1 分钟。这是工程运行和一致性检查，不能作为独立验证集成绩。

## 整理与验证说明

scripts/train.py 只将根路径改为 NTL_ROOT，并以相同 GroupKFold 重新生成折索引，替代旧 UltraBERT OOF 文件校验；原始源码位于 archive。准备及导出脚本也支持 NTL_ROOT，模型下载固定实际 revision。正式推理代码直接复制自 实验室服务器 的已提交版本。

未在整理期间重新训练或新增 Kaggle 提交。模型、外部数据及参考代码遵循各自原许可，本仓库不替第三方资产重新授权。
