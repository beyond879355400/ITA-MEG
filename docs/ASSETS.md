# 环境与外部资源

## 已记录的软件配置

- 原训练/ITA扩散推理：Python 3.10，PyTorch 2.1.1、TorchVision 0.16.1、Diffusers 0.26.0，MPS、FP32。
- 2026-10-23补充实验依赖：`source/实验以及中间产物/盲审补充实验_20261023/runtime-requirements.txt`。
- SwiftEdit：其 `recovery_v3_20260916/requirements.lock`。
- PerformRecast：`requirements-overlay.txt` 与 `formal_batch.py` 的版本检查。
- OpenFace 是独立可执行程序；FaceNet 评价使用与身份条件编码器不同的权重。仅安装PyTorch不能替代评价器及其模型。

这些是原实验记录的版本，不是对最新包的推荐。建议使用独立环境，不覆盖日常环境。

## 公开模型、仓库版本

原完整公共资源清单（含revision、文件名和SHA）位于：
`source/实验以及中间产物/独立AU控制实验_20260914/recent_baselines_20260916/preservation/bundle_v2_20260916/cloud_public_assets.json`。

固定基线版本从原程序直接保留：

| 组件 | 原记录版本位置或提交 |
|---|---|
| FineFace | `6e9cc8df1180092891c1f55e4d4a06110759dac6`，https://github.com/tvaranka/fineface |
| Face-Adapter | `6e313885188e05740a7c40e12c1b9015c748bf08`，https://github.com/FaceAdapter/Face-Adapter |
| MagicFace | `bcaca61620450681233529d472c452091c25c92b`，见 `final_test/external/contracts.py` |
| LivePortrait | `9b294b3d0536135442ea73cb01e6cb3ca7029dd3`，见同一contracts文件 |
| SwiftEdit | `230bff1aaff5d952210c717d55537c5b29d596ad`，https://github.com/Qualcomm-AI-research/SwiftEdit |
| PerformRecast | `0206abf3ba9e5b1eaa3cdc584879d5fa20d6a0b4`，https://github.com/youku-aigc/PerformRecast |

Realistic Vision、FineFace AU适配器、IP-Adapter-FaceID、身份编码器、FaceNet评价器等权重按原资源清单准备。各自的模型许可与第三方代码许可继续有效；本仓库没有复制这些权重，也不代替下载许可。

## 数据与训练产物

DISFA视频、原始人脸图像和标签按数据提供方授权取得。记录中的主体编号和固定请求用于区分实验单位，不能用生成图替换真实参照。

重新推理还需要训练出的200步适配器、AU监督器和来源条件缓存。参数文件不在本仓库中，应由作者在允许的范围内另行提供或通过所附训练程序获得。

## 发布边界

本仓库只新增私有代码空间，没有变更原论文仓库和旧实验产物。没有自动接受第三方许可、上传原始人脸、清除记录、购买资源或向导师发送GitHub邀请。
