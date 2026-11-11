# 实际训练、推理、评价入口

## 三种核验不要混淆

1. `bash run_reproduce.sh audit`：从已存逐图测量重新计算表值和精确检验。任何具备 Python 3.10+ 的机器可运行，不需模型。
2. 模型复跑：重新读数据和权重、推理并测量。需要合法数据、模型以及原固定配置。下表列出原程序。
3. 完整从零训练：还包括抽帧、监督器训练和条件缓存构建。不能把已训练检查点的推理演示称为从零训练。

本次仓库的快速测试仅覆盖第1项；源码快照供第2、3项使用，不代表已经在一台全新 Linux / CUDA 机器上重跑全部实验。

## 原 Mac 工作区中的运行

以下以原正式论文工程为工作目录，并设
`R=实验以及中间产物/独立AU控制实验_20260914`，
`S=实验以及中间产物/盲审补充实验_20261023`。
不要在原实验输出目录覆盖文件，所有重跑使用新名称。

### 残差训练

```bash
python -B "$R/train_centered_adapter.py" \
  --config "$R/configs/v4_slot_2k_bidir_step200.json" \
  --resume "$R/runs/v4_slot_2k_batch1_smoke_202609141916/checkpoint_00000008.pt" \
  --output "$R/runs/reproduce_bidirectional_NEW"
```

原正式训练从8步父检查点续训至200步，记录分为1–8和9–200两段。若从随机初始化重新训练，应先用原 smoke 配置创建对应前8步，不把省略父检查点的新训练称为与原训练逐位一致。其他三个结构使用同目录的 `condition_reads_image_step200`、`image_reads_condition_step200`、`mlp_step200` 配置。

训练配置引用的源特征缓存、`pair_plan.json`、`dev12_manifest.json` 和 AU 监督器 qualification 需要通过数据准备、监督器训练和缓存脚本生成。它们不是仓库附带的数据集。

### 原正式环境生成入口

```bash
bash "$R/completion_20260915/delivery/run_demo.sh"
bash "$R/completion_20260915/delivery/run_demo.sh" --execute
bash "$R/completion_20260915/delivery/run_reproduce.sh" --help
```

第一条仅计划，第二条真实生成一个固定非零案例。完整480图生成的原始实现为 `completion_20260915/generation/run_grid.py`，它明确要求 `--freeze`、`--freeze-sha`、`--source-cache`、`--source-cache-sha`、`--source-measurements`、`--source-measurements-sha`、`--output` 和显式 `--execute`。原 `run_completion_queue.py` 的内部输出路径有固定绑定，不应换一个顶层输出参数便直接重跑。

### 外部方法

- MagicFace、Face-Adapter、LivePortrait：`completion_20260915/external/run_external.py`。使用 `--method` 选择方法，通过 `--freeze / --freeze-sha / --raw-manifest / --raw-sha` 绑定原计划。
- SwiftEdit：`recent_baselines_20260916/swiftedit/recovery_v3_20260916/run_formal.py`，32例 × 3种子。原输入批次与 `targeted_quality_20260918` 的对齐输入诊断分别保留。
- PerformRecast：`recent_baselines_20260916/performrecast/run_formal32.sh` 与 `formal_batch.py`。输入是每幅静态帧无损重复4帧，固定取第0帧的原生裁剪。

各方法采用各自公开权重，不把这些调用称为同数据重新训练。

### 288图约束消融

```bash
python -B "$S/generation/run_ablations.py" --preflight \
  --output "$S/generation/preflight_REPRO_NEW.json"
python -B "$S/generation/run_ablations.py" --execute "$S/generation/freeze_v1.json" \
  --output "$S/generation/runs/reproduce_NEW"
```

复跑需要原来源特征、噪声绑定、双向200步权重及公共骨干。四设置保留同一学习模块，只切换区域投影和末端局部平滑。原完整组72张的像素/SHA一致性门禁继续保留。

### 测量及统计

主比较：`completion_20260915/evaluation/evaluate.py` 和
`recent_baselines_20260916/evaluation_20260916_v1/`。
补充实验：`S/evaluation/evaluate.py` 的
`inventory / freeze / prepare / openface / facenet / export` 阶段。
主体统计：`R/statistics/paired_subject_statistics.py`。

这些阶段必须以实际新生成文件为输入，保留检测失败、缺测和真实驱动帧访问差异。重新运行时不要把旧测量CSV当作新模型输出。

## 迁移到新机器

`source/` 保留原目录结构与历史路径、哈希。新机器需要建立新的资源配置和运行回执；只替换字符串、删除门禁或照搬历史路径不能证明跨平台一致。Mac、Linux GPU 和 CPU 的浮点结果应通过相同案例实际比较，不能事先保证逐像素相同。

模型、数据及第三方源代码需按 docs/ASSETS.md 准备。本仓库当前不自动下载受限数据、不提供训练权重，不声称仅克隆代码就能重跑全部图像。
