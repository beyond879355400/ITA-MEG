# ITA-MEG 实验代码与结果复核

对应英文稿：**ITA-MEG: AU-Controlled Static Face Editing for Micro-Expression Analysis**。

本仓库保存论文实际使用的训练、生成、五种外部方法适配、评价和统计源码，以及便于独立复算的逐图测量记录。源码快照保持原文件字节，`MANIFEST.json` 记录文件 SHA-256 和原项目相对路径。提交日期不代表实验运行日期。

## 先运行这个：不需要 GPU、模型或 DISFA 图像

需要 Python 3.10+，使用标准库，不自动安装软件：

```bash
git clone https://github.com/beyond879355400/ITA-MEG.git
cd ITA-MEG
bash run_reproduce.sh test
bash run_reproduce.sh audit --output reports/check-01.json
```

私有仓库需使用获授权 GitHub 账号。输出文件必须是新文件，已有回执不会覆盖。

`audit` 从逐图测量记录重算种子、案例、主体等权均值，并复算五项外部与四项内部目标AU误差比较的精确符号翻转及 Holm 校正。当前包含 896 条主比较逻辑记录、288 条消融记录。LivePortrait 的种子别名不当作独立被试。SwiftEdit 的缺测按原记录保留，编辑比较使用 22 个有效案例，其余方法使用 24 个。

**这个命令是数值复核，不是重新训练、重新生成图像或重新运行 OpenFace / FaceNet。** 区间的原统计实现和已存结果都保留，快速核验不重新计算 bootstrap。

## 代码在哪里

为保持代码之间的相对路径和原校验关系，源码位于：

```text
source/实验以及中间产物/
├── 独立AU控制实验_20260914/
│   ├── train_centered_adapter.py       # 实际源条件残差训练
│   ├── modeling/                      # 双流模块、源条件差分、限幅、AU监督器
│   ├── data_prepare/                  # DISFA抽帧、标签及主体划分
│   ├── au_supervisor/                 # 监督器训练
│   ├── configs/                       # 四种结构的实际200步配置
│   ├── completion_20260915/
│   │   ├── generation/                # 正式480图生成与几何掩码
│   │   ├── external/                  # MagicFace、Face-Adapter、LivePortrait入口
│   │   ├── evaluation/                # 逐图评价、汇总
│   │   └── delivery/                  # 原Mac环境的演示、复跑及核验脚本
│   ├── external_baselines/runtime/    # Face-Adapter、LivePortrait适配器
│   ├── final_test/external/            # 固定原生计算内核
│   ├── improvements/                 # 正式V11采样所引用的历史实现
│   ├── statistics/                    # 原主体bootstrap与精确检验
│   └── recent_baselines_20260916/
│       ├── swiftedit/                 # SwiftEdit原批次及对齐输入诊断
│       ├── performrecast/             # 静态请求适配、正式32例入口
│       └── evaluation_20260916_v1/     # 五外部方法统一评价与导出
└── 盲审补充实验_20261023/
    ├── generation/                    # 区域投影/局部平滑四设置
    ├── evaluation/                    # 288图测量、统计、分层、真实参照
    └── validation/                    # 原始独立数值核算脚本
```

`records/` 中的数值来自原实验文件。主逐图CSV只删除了本机输出路径、执行ID、别名路径和自由文本错误信息列，测量数值、失败状态、分母和案例键不变。具体转换写在清单中。

## 重新跑模型

见 [实际训练、生成与评价入口](docs/REPRODUCING.md) 和 [外部资源及环境](docs/ASSETS.md)。

原实验的扩散推理使用 Mac MPS / FP32，评价以及 LivePortrait、PerformRecast 使用 CPU。历史程序保留了设备、输入文件、检查点和源码 SHA 校验。原路径、合法取得的数据、模型和条件缓存属于运行前置条件。本次没有把这些校验删除来制造“任意电脑一键通过”，也没有把零编辑演示当成整篇论文的复现。

## 论文与记录对应

| 英文稿内容 | 本仓库依据 |
|---|---|
| Table 1：五外部方法比较 | `records/main_per_image.csv`、`main_summary.csv` |
| Table 2：内部结构对照 | 同上，五种内部配置 |
| Table 3：主体配对检验 | `main_paired_statistics.json`、`internal_paired_statistics.json` |
| Table 4：四配置约束消融 | `ablation_per_image.csv`、`ablation_summary.csv`、`ablation_primary_DU.csv` |
| Table 5：请求分层 | `subgroup_summary.csv` |
| Table 6：零编辑检查 | `main_per_image.csv` 中 `no_edit` 组 |
| 真实标签参照检查 | `real_reference_calibration.csv/json` |

## AI 工具使用

AI 辅助了实验方案梳理、代码实现与调试、结果整理及英文稿编写。生成图来自模型推理，指标来自程序对图像的测量和统计。语言模型生成的说明文字不作为测量结果。原训练、推理、评价记录和失败记录均保留在研究工作区。

## 数据和权限

本仓库不含原始人脸数据集、生成图、模型权重、个人账号凭据、虚拟环境或聊天记录。第三方模型和仓库按其原许可证取得和使用，不因本仓库上传而获得重新分发授权。本次以私有仓库交付，没有擅自公开论文实验资产。
