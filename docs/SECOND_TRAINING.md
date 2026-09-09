# 二次训练：ScanNet 深度监督与轻量门控

基于 `scannet-orin-train` 的 `859d5b3`。该分支将 ScanNet 入口从
“冻结全部深度模块、拟合 oracle 门控标签”改为直接优化最终深度。
旧的 7-Scenes/oracle 入口仍保留，不能与本训练入口混用 checkpoint。

## 最小方案

- D-Net、F-Net 保持原 MaGNet 训练方式：固定预训练权重及 BatchNorm。
- 联合训练原 G-Net、上采样层，以及已有的一个 GeometryGate。
- 所有迭代共享同一个门控模块；没有额外可靠性预测头或新的 IRM。
- 沿用逐迭代 Gaussian NLL，`gamma=0.8`、默认训练/测试各 3 次迭代。
  当前代码的输出是 Gaussian sigma，不能直接把参考论文的 Laplace 损失
  当作同一参数化使用。
- 每个样本以 50% 概率保持准确位姿；其余样本的每个源视图独立采样
  `[0°, 5°]` 均匀角度和球面均匀轴，右乘旋转，平移和齐次行完全不变。
  batch size 1 时，此比例体现为连续训练步骤的混合，而非每批强制各占一半。
- 门控输入使用现有匹配熵、峰值、单目 sigma。为保持权重布局兼容，
  第四个旋转不确定性通道固定为零；训练和验证都传 `rot_unc=None`。
  注入噪声的真实角度仅用于日志，绝不作为模型输入。这是旋转扰动训练的
  匹配可靠性基线，不能宣称已经实现显式旋转协方差传播。
- 不使用门控 oracle 标签、辅助损失、课程学习或 U-ARE-ME 在线估计。

默认的 50% 和 5° 是可复现起点，并不是已经验证的最优超参数。
保留 MaGNet 原本逐迭代 detach 的方式，每一步通过对应深度损失训练；
不会反向穿过前一轮采样与匹配。不要把它描述为完整跨迭代反向传播。

## 数据与环境

复用 [SCANNET_ORIN.md](SCANNET_ORIN.md) 的导出和 split 构建流程。
继续使用已有 Jetson PyTorch/CUDA 环境，其他依赖见 `requirements_orin.txt`。
默认输入 480×640、代价体 120×160，窗口偏移 `[-20,-10,0,10,20]`。
训练/验证按物理场景分开；同一空间的不同扫描也不能跨集合。

## 启动二次训练

在仓库根目录执行，替换为实际 ScanNet 路径；默认 checkpoint 路径为
`ckpts/{DNET,FNET,MAGNET}_scannet.pt`：

```bash
python train_StructMaGNet_scannet.py \
  --dataset_root /path/to/ScanNet \
  --train_split ./data_split/scannet_train_stride5.txt \
  --val_split ./data_split/scannet_val_stride5.txt \
  --gate_mode learned \
  --clean_probability 0.5 --rotation_max_deg 5 \
  --batch_size 1 --num_workers 1 --epochs 1 --lr 1e-4 \
  --max_train_steps 10 --val_max_samples 8 \
  --output_dir ./exp/STRUCTMAGNET/scannet/second_smoke
```

先检查短跑损失和内存，再运行正式训练：将 `--max_train_steps` 改为 `0`，
设置所需 `--epochs`，并使用新的输出目录。`--amp` 可按设备情况开启；
这里没有预设它必然适用于所有 Jetson/PyTorch 组合。
训练 G-Net 与上采样层的内存高于旧的 gate-only 训练。

## A/B/C 对照

三组保持相同的初始 checkpoint、场景、seed、迭代数和训练步数，
只改变下表选项，并使用独立输出目录。

| 实验 | 参数 | 检验目标 |
|---|---|---|
| A：准确位姿，无门控 | `--gate_mode off --clean_probability 1` | 同预算微调基线 |
| B：扰动训练，无门控 | `--gate_mode off --clean_probability 0.5 --rotation_max_deg 5` | 数据增强收益 |
| C：扰动训练，有门控 | `--gate_mode learned --clean_probability 0.5 --rotation_max_deg 5` | 门控相对 B 的独立收益 |

`off` 使用完整 G-Net 提议，即有效几何下 `g=1`，并停止训练门控。
三组都训练 G-Net 和上采样层，不会出现 B 冻结而 C 微调的不公平比较。
若要记录未微调的原始 MaGNet，请另用原评估入口，不要与 A 混称。

默认验证 0°/5°/8°，8° 超出默认训练角度范围。每次验证使用独立且固定的
随机生成器，各组使用相同轴，不消耗训练噪声生成器。可使用
`--val_degrees 0 1 2 3 5 8 --val_max_samples 0` 做最终完整验证。
默认 32 个样本仅适合快速反馈，并非最终论文结果。

日志：`train.csv`、`val.csv`。除 RMSE、AbsRel、a1、原始预测 NLL 外，记录
门控均值及实际低分辨率深度更新的平均绝对幅度，检查“小门控、大残差”。
RMSE 按有效 GT 像素累计后开方；深度指标使用深度范围裁剪，NLL 使用未裁剪
预测。无有效 GT 的验证样本跳过；有效 GT 上非有限预测会报错而非被掩膜。

## Checkpoint 和恢复

`last.pt` 和 `best.pt` 包含 G-Net、上采样层、门控、优化器、AMP scaler、
epoch/step、历史 best score 和随机状态；固定 D/F 仍从原 checkpoint 加载。
best score 是所选验证角度 RMSE 的均值，准确位姿指标仍须单独检查。

```bash
# 其余训练参数必须与原任务一致，epochs 表示总目标 epoch 数。
python train_StructMaGNet_scannet.py \
  --dataset_root /path/to/ScanNet --epochs 5 \
  --output_dir ./exp/STRUCTMAGNET/scannet/second_training \
  --resume ./exp/STRUCTMAGNET/scannet/second_training/last.pt

python train_StructMaGNet_scannet.py \
  --dataset_root /path/to/ScanNet --gate_mode learned \
  --resume ./exp/STRUCTMAGNET/scannet/second_training/best.pt \
  --eval_only --val_max_samples 0 --val_degrees 0 1 2 3 5 8 \
  --output_dir ./exp/STRUCTMAGNET/scannet/second_eval
```

仅支持 epoch 边界恢复，不支持旧 `best_gate.pt`。恢复时继续使用同一组 D/F
预训练权重和数据。跨设备/不同 CUDA 算子不保证逐位复现。

## 本地回归验证

```bash
python -m unittest discover -s tests -p test_retraining.py -v
```

测试覆盖真实小尺寸 homography 前向、深度损失梯度、门控关闭行为、
无有效源视图回退、旋转扰动、混合精度统计及 checkpoint 恢复。
这些测试不替代 ScanNet/GPU 训练，也不证明已经取得深度精度提升。
