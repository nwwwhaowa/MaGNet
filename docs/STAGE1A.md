# StructMaGNet — Stage 1A oracle warm-up

本轮基线：`stage1a-gate-oracle`，提交 `ad52bb0`（2026-09-07）。
当前训练入口是 **`train_StructMaGNet.py`**。
`train_StructMaGNet_7scenes.py` 是保留的旧版深度 NLL 训练入口，不能用它复现本阶段的 oracle 监督。

## 当前实现与 RA-L 主线

| 研究主线 | 当前代码事实 | 下一步 |
| --- | --- | --- |
| Rotation-Aware Multi-View Matching | `homography_struct.py` 和 `rotation_utils.py` 保留了三假设实现；`874c3a3` 后当前模型 forward 已不调用它们 | 先完成门控验证，再以显式开关恢复匹配接口，补充协方差来源、切空间约定和权重定义 |
| Adaptive Monocular–Multi-View Fusion | GeometryGate 控制 MaGNet 均值残差和 sigma 更新；本阶段只监督第一步 gate | 验证第一步效用，再设计后续迭代监督和 gate-depth adaptation |
| Reliability-Aware Iterative Refinement | 当前是带门控的逐步高斯更新；每一步输入被 detach | 尚未实现论文中的能量梯度、深度解码器与 GRU；不能把当前循环称为该 IRM |

本阶段数据为 **7-Scenes Chess + Office**，不是稿件中的 ScanNet/TartanAir 完整实验。
D-Net、F-Net、G-Net、上采样头冻结，只有 GeometryGate 更新。
`rot_unc` 输入是合成扰动的已知角度（弧度），属于 **oracle rotation input**，不能声称是部署时估计出的旋转协方差。
当前 MaGNet volume 是“越大越匹配”的 similarity score，所以使用 `softmax(score)`；论文若使用负代价，应定义 `cost = -score`，不要直接反转现有实现的符号。

## 本轮修正

- 将训练和验证重复的 oracle 代码统一到 `utils/gate_oracle.py`。先转 FP32 再计算差值和除法，避免 FP16 溢出后再转 FP32。
- 在 `abs(delta) > gate_min_delta` 的可辨识像素上使用精确有界最小二乘系数：
  `g_star = clamp((gt - mu_mono) / (mu_mv - mu_mono), 0, 1)`。
  不再用分母的额外 `1e-6` 改变最优系数；GT 和两端预测均 detach。
- 只监督 `geometry_gate[0]`，损失仍是 Smooth-L1（beta=0.1）。`--gate_tau` 仅为旧命令兼容保留，不影响 oracle。
- 空监督批次跳过 optimizer step，包括 AdamW 的动量/weight decay；整轮无有效 step 时明确报错。
- cost entropy 在 FP32 的 log-softmax 上计算。没有有效源视图、非有限 matching 或非有限 proposal 时，gate 为零并保留上一轮高斯。若全程没有有效源视图，则低分辨率输出严格等于 monocular prior。
- `geometry_valid` 表示帧级 pose availability 加有限匹配/有效 proposal 检查，**尚不是完整的逐像素可见性或遮挡 mask**。
- 默认 32 个验证样本现在在 Chess/Office 间均分，并均匀覆盖各自 held-out 序列；以前按排序取前缀可能只评估 Chess。`--val_max_samples 0` 使用完整验证集。
- Oracle 统计按有效像素汇总，不再平均各图的均值/相关系数；空样本计数单独输出。深度 RMSE 仍是逐图 RMSE 的平均，验证 sample count 不再把 batch 当 image。
- 不完整或错配的 G-Net/上采样权重立即报错，避免“冻结随机 backbone”继续训练。
- checkpoint 保存历史 best score 和 AMP scaler；旧验证协议或改变验证设置时重置 best score，避免比较不同数据子集的成绩。
- eval-only 不构建训练集，使用独立 `eval_diagnostics.csv`；CSV 表头不一致时要求新的 output directory，避免旧日志混入新列。
- 相对位姿预处理直接使用 `torch.inverse`，修复 NumPy 2.x 下 `torch.from_numpy(np.linalg.inv(Tensor))` 的类型错误；同时处理非有限及不可逆 reference pose。

权重键名与 Gate 网络结构保持兼容。旧 Gate checkpoint 可用于重新评估；**旧版验证分数和本轮分数不可直接比较**。resume 会恢复 epoch/step、optimizer 和 scaler，但目前不保存全部 RNG 状态，因此不承诺逐步数值复现连续训练。

## 先运行 CPU 回归测试

从仓库根目录执行：

```bash
python -m unittest discover -s tests -v
```

本轮使用 Python 3.12、PyTorch 2.5.1+cpu、torchvision 0.20.1+cpu、NumPy 2.2.6。
测试不需要 checkpoint、7-Scenes 或模型下载。真实 forward/update 逻辑使用小型特征提供模块测试；原 homography、D-Net/F-Net 权重推理和 CUDA AMP 端到端训练仍需在你的训练机器验证。
原 `requirements.txt` 是上游 MaGNet 的旧环境，不能作为这些新增脚本的完整环境锁定文件。

## 在现有 GPU 环境运行 smoke training

准备已有的 `ckpts/DNET_scannet.pt`、`FNET_scannet.pt`、`MAGNET_scannet.pt`，以及含 Chess/Office 的 7-Scenes 根目录。使用新的输出目录，保留旧实验。

```bash
python train_StructMaGNet.py \
  --dataset_root /path/to/SevenScenes \
  --scenes chess,office \
  --epochs 1 --batch_size 1 --num_workers 1 \
  --max_train_steps 50 --val_max_samples 32 \
  --lr 1e-4 --lambda_gate 0.1 --gate_min_delta 0.01 \
  --num_train_iter 3 --num_test_iter 3 \
  --output_dir ./exp/STRUCTMAGNET/stage1a_oracle_v2_smoke
```

先不用 `--amp`，确认完整 smoke 流程与第一步诊断有效；再在另一个输出目录用 `--amp` 比较数值稳定性。
`--max_train_steps` 限制每轮尝试的 batch 数，空 oracle batch 会跳过，所以实际 optimizer step 可能更少。
验证默认 0/5/8 度固定轴扰动，训练的扰动分布和固定验证轴保持原设置；它是工程诊断，不替代论文中的多随机种子/随机轴鲁棒性评估。

只评估现有 Gate checkpoint：

```bash
python train_StructMaGNet.py \
  --dataset_root /path/to/SevenScenes --scenes chess,office \
  --eval_only \
  --resume ./exp/STRUCTMAGNET/stage1a_oracle_v2_smoke/checkpoints/last_gate.pt \
  --val_max_samples 0 \
  --output_dir ./exp/STRUCTMAGNET/stage1a_oracle_v2_eval
```

评估旧 checkpoint 也用同一命令，替换 `--resume` 即可。`--epochs` 在 resume 训练时表示总轮数，不是额外轮数。

## 如何判断 Stage 1A 是否值得继续

| 日志项 | 用途 |
| --- | --- |
| `oracle_valid_pixels` / `oracle_valid_fraction` | 监督是否覆盖足够多的可辨识像素；没有统一百分比阈值，应按场景和噪声级别比较 |
| `oracle_empty_samples` | 哪些验证条件没有可辨识 oracle；整组为空时 oracle 指标为 NaN，不应被当作零误差 |
| `gate_target_mean/std`、`target_closed_fraction/open_fraction` | 检查目标是否几乎全部为 0 或 1，确认不是退化的单值回归任务 |
| `gate_target_mae/corr` | 第一轮 gate 与 oracle 的对齐；目标或 gate 方差为零时 corr 按未定义处理 |
| `mono_rmse_low` / `mv_rmse_low` | 可辨识像素集合上，两端预测各自的误差 |
| `oracle_rmse_low` | 同一集合上逐像素最优插值的可达误差；它使用 GT，不能作为可部署方法成绩 |
| `fused_rmse_low` | 第一轮学习 gate 的实际插值误差，检查是否向 oracle 改善空间靠近 |
| `gate1_mean/gate2_mean/gate3_mean` | 共享 Gate 在各轮的行为；目前只有 gate1 被监督，后两轮不能只凭均值宣称有效 |
| `rmse/abs_rel/a1/nll` | 完整上采样深度的逐图平均指标；不能与上述 low-resolution 指标直接比较 |

优先比较同一验证集上的：未训练 Gate、训练后 Gate、无门控 MaGNet，以及 monocular 输出。
本轮已提供第一步两端/插值 oracle 的 low-resolution 诊断；完整无门控和 monocular 的高分辨率统一评估仍需后续接入。
历史 best checkpoint 的选择公式仍是 `RMSE_0 + RMSE_5 + RMSE_8`，没有擅自把训练目标换成最终深度 NLL。

如果 oracle 相比两个端点几乎没有收益，先检查原始 MV proposal、数据配准和监督范围；如果 oracle 有收益而学习 Gate 未利用，检查输入统计、初始化和监督覆盖，再决定是否修改特征或训练参数。
不能仅因 gate_mean 随噪声减小，就认定网络学到了几何可靠性——它也可能只使用已知噪声角度。

## 后续开发顺序

1. 收集本轮 smoke 的 `train.csv`、`val_diagnostics.csv`、完整集 `eval_diagnostics.csv` 和 checkpoint 配置，先确认 Stage 1A 的梯度、效用和 clean-pose 表现。
2. Stage 1B：明确后续迭代的 base prediction/target 定义，设计 gate-depth adaptation，对比固定 gate、无门控和训练 gate；此时再考虑释放 G-Net 或引入深度监督。
3. 恢复 Rotation-Aware Matching：提供显式开关与零不确定性回退；从真实估计器得到相对旋转 covariance，并先统一左右扰动坐标系。现有三点 `(-v,0,+v)` 的权重 `(0.25,0.5,0.25)` 只代表特定离散模型，不能冒充论文的完整三维 Gaussian Monte Carlo。
4. 单独实现 Reliability-Aware IRM：定义可求导的解码器、固定 target/weight 的能量梯度及 GRU 更新，再比较参数量/迭代次数匹配的普通递归基线。

这些是后续阶段，尚未在本轮宣称实现或取得数据集性能提升。
