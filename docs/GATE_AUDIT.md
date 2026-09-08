# Stage 1A: 门控归因与训练消融

## 已知实现与待检验假设

门控输入为 cost-volume entropy、peak、单目 sigma、注入旋转角度（弧度）。
旋转输入在图内恒定，是 oracle 噪声元信息，不是部署时估计的协方差；它不能直接区分同一图的像素。
entropy 和 peak 来自相同的五候选 softmax，可能重叠或接近常数，是否如此须看实际统计。
仅靠低 Pearson 相关不能否定非线性信息，也不能证明一个特征无用。

默认末层权重为零、bias=4，g=0.982014，Sigmoid 导数约0.01766。
bias=0 时 g=0.5，导数为0.25，约为前者14.2倍。这只是初始局部导数，不代表训练会快14.2倍。
末层零权重使第一步隐藏层梯度为零；末层开始更新后通常可以向前传播梯度，不应据此判为断梯度。
新参数不改变旧 checkpoint 的键或形状；加载时训练权重覆盖初始化。

默认 Smooth-L1(beta=0.1) 拟合截断的 oracle 系数。接近0/1的目标较多时，
它更偏向中位数式估计，预测均值不必匹配目标均值。新增两种可选监督：

- `--gate_loss mse`：系数 MSE，更直接惩罚较大的系数误差。
- `--gate_loss depth_mse`：冻结第一步候选后，直接优化融合深度与 GT 的 MSE，单位m²。
  系数误差对深度的影响与候选差值有关；GT超出两端点时，边界附近还存在交叉项，
  因而不能把该目标简单等同于“差值平方加权的截断系数 MSE”。

三种损失仍然仅监督第一步、仅更新 GeometryGate，并使用相同 identifiable mask。
深度 MSE 的尺度和梯度分布不同，数值不能与系数损失直接横向比较。

## 首先诊断已有检查点

在仓库根目录运行。检查点路径须对应实际三场景训练结果。

```bash
env -u LD_LIBRARY_PATH python train_StructMaGNet.py \
  --dataset_root /home/li/26workspace/datasets/SevenScenes \
  --scenes chess,office,redkitchen \
  --val_max_samples 96 \
  --resume ./exp/STRUCTMAGNET/stage1a_3scenes_full_epoch1/checkpoints/last_gate.pt \
  --eval_only --gate_audit --gate_audit_full \
  --output_dir ./exp/STRUCTMAGNET/stage1a_gate_audit
```

去掉 `--gate_audit_full` 可先快速检查第一步，共3个噪声档各一次模型前向遍历。
加上该选项会对8种策略分别完整运行3个噪声档，主干前向量约为普通验证的8倍。

生成 `logs/` 下的文件：

| 文件 | 内容 |
|---|---|
| gate_audit_first_step.csv | 固定两端点的系数/深度损失、RMSE、MAE、相关性 |
| gate_audit_per_image.csv | 同一image_index的配对结果，用于检查收益是否只来自少数图像 |
| gate_audit_inputs.csv | 特征范围、全局标准差、图内空间标准差、特征之间及与目标的Pearson相关 |
| gate_audit_full.csv | 每种策略重新运行全部迭代后的全分辨率指标；快速模式只有learned |
| gate_audit_config.json | 参数和评价口径 |

第一步策略包括：learned、固定 g=0/0.25/0.5/0.75/1、image_mean、shuffle，
以及逐个输入通道替换成其图内均值、将旋转输入置零的敏感性检查。
所有第一步结果共享相同 mono/MV 候选及有效GT像素集合。
image_mean 在几何有效像素上取每图均值；shuffle 在同一集合内打乱，保留权重分布。
两者均不使用GT选择权重，且几何无效像素保持g=0。
输入置均值/零属于推理干预，可能产生分布偏移；不能替代重新训练的特征消融。

完整迭代对照只运行learned、五种fixed、image_mean和shuffle。
后续采样、候选与sigma会随策略变化，因此使用完整迭代对照判断最终输出收益，
使用第一步对照隔离像素权重分配效果。g=0的全分辨率输出仍经过冻结的MaGNet上采样器，
不能称为原始D-Net独立推理结果；g=1仍保留无效几何回退，也不声称与原版MaGNet逐位一致。

## 判读

1. 若learned与image_mean表现接近，说明尚无明显证据支持像素级分配贡献。
2. 若learned优于image_mean，并优于shuffle，才支持空间对应关系有贡献。
   shuffle目前为固定种子的单次确定性对照，不能单独作为统计显著性证据。
3. 若最佳固定系数已达到learned效果，整体降权可能足以解释收益。
   固定系数只能在验证集选择并锁定，再用于最终测试；不能用测试集选系数。
4. 熵/峰值若图内波动极小且高度相关，需进一步检查候选分布与分数尺度。
   不能只为了拉开分布就调温度；应独立消融其对匹配与门控的影响。
5. rotation在每个固定噪声档内方差为零，相关系数NaN是合理结果。
6. 所有低分辨率指标限定在可辨识mask上，不可与全分辨率rmse直接比较。
   各噪声档mask不同，因此单目低分辨率RMSE也可能变化。

## 下一轮受控训练

先运行上述诊断，再按需运行以下2×2设计，每次从冻结主干和新门控开始，不使用resume。
保持seed、样本、噪声策略、训练步数、学习率、lambda_gate、验证集完全一致。
已经完成的bias=4/Smooth-L1一轮可以作为基准；不能用三轮模型与一轮消融比较。

| 实验 | gate_init_bias | gate_loss | 目的 |
|---|---:|---|---|
| A | 4 | smooth_l1 | 旧基准 |
| B | 0 | smooth_l1 | 单独检查初始化 |
| C | 4 | mse | 单独检查监督目标 |
| D | 0 | mse | 检查二者交互 |

以B为例，其余仅替换两项参数和输出目录：

```bash
env -u LD_LIBRARY_PATH python train_StructMaGNet.py \
  --dataset_root /home/li/26workspace/datasets/SevenScenes \
  --scenes chess,office,redkitchen \
  --epochs 1 --batch_size 1 --lr 0.0001 --lambda_gate 0.1 \
  --max_train_steps 0 --val_max_samples 96 --seed 1234 \
  --gate_init_bias 0 --gate_loss smooth_l1 \
  --output_dir ./exp/STRUCTMAGNET/stage1a_bias0_smoothl1
```

`depth_mse`作为后续独立目标消融，先不要与初始化、输入归一化等一起改动。
新旧CSV训练损失只有在同一目标下才可比较；以同一审计协议的RMSE、MAE和空间对照判读。
续训会恢复optimizer，并拒绝改变gate_loss或初始化标签；改变实验条件须新开训练。

本地测试覆盖toy主干上的完整forward、评价导出、屏蔽/打乱及监督梯度。
真实SevenScenes、预训练权重和GPU结果须在用户机器运行后才能判断，不预设改进。
