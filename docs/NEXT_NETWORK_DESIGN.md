# 下一阶段：对齐 main.pdf 的旋转退化可靠性网络

## 版本与本次交付范围

依据：用户提供的 `main.pdf`，5 页，方法位于第 2–4 页；代码基线
`bfe125992a332061a055698c6955267a468e7585`；归档分支
`archive/scannet-gate-baseline-20260912`。

本次新增：实验备份工具、独立的完整协方差高斯旋转采样函数及其测试。
本文定义其余模块的接口、数学口径和集成顺序。这些其余模块尚未实现；
现有 `STRUCTMAGNET`、训练入口、损失和归档分支均未被本次改动替换。
本分支不能被当作已经完成论文网络的版本直接启动长训练。

当前执行环境没有 PyTorch，也没有 Orin/ScanNet。备份工具使用模拟实验实际
验证；采样模块只有语法检查，随附数值测试需要在已有 structmagnet 环境执行。

## 1. 保留本次实验

本次实验是 C 组：扰动训练 + learned gate。10 epochs / 25,620 steps，
train split 10,248 references / 21 scans / 10 physical spaces，validation
108 references / 1 physical space，无训练/验证物理空间交集。
按已归档训练代码，第 4 轮应为 best.pt，三角度平均 RMSE 0.5175465 m；
第 10 轮平均 0.5597692 m。权重本体尚未读取，该 epoch 对应关系由日志和
保存逻辑推定。不能将第 1 轮当作原始 MaGNet 的未微调基线。

在 Orin 仓库根目录执行：

```bash
python tools/preserve_experiment.py \
  --project-root /home/li/26workspace/StructMaGNet/MaGNet \
  --run-dir /home/li/26workspace/StructMaGNet/MaGNet/exp/STRUCTMAGNET/scannet/orin_perf_xmns4us9/full_train \
  --backup-root /home/li/26workspace/experiment_backups \
  --code-ref bfe125992a332061a055698c6955267a468e7585
```

工具复制整个实验套件、实际 split、raw W/H metadata、DNET/FNET/MAGNET
初始权重，以 git archive 保存指定提交的源码，并逐文件比较 SHA-256。
完成后显示 VERIFIED，备份目录不再有 INCOMPLETE 标记。任何失败均保留源文件，
部分备份目录供诊断。没有训练数据；没有擅自将原路径改写为备份路径。
恢复到其他机器时依据 manifest 中的 original/copy 映射设置实际路径。
恢复旧训练还需注意旧 trainer 会严格比较配置字符串，不能直接换路径后冒充
无缝恢复；跨路径迁移应显式处理配置迁移并重新验证。

同一块磁盘上的副本只能防止误操作。复制完整备份目录到另一块磁盘或机器后运行：

```bash
python tools/preserve_experiment.py --verify /实际备份目录
```

保留原 `full_train` 目录，后续实验始终使用新输出目录。
Git 归档保存代码，实验备份保存权重与日志；二者通过完整 commit SHA 关联。
代码引用来自已知训练版本；运行时没有记录源码哈希，不能声称工具追溯证明了
训练过程中没有本地未提交修改。

## 2. 论文与当前实现的差距

| 论文部分 | 当前基线 | 下一阶段任务 |
|---|---|---|
| 式 1–2 Gaussian 单目先验与采样 | D/F 冻结，Gaussian sigma，5 个深度候选 | 保留并明确 sigma 单位 |
| 式 3–8 协方差感知旋转匹配 | 主模型调用原 homography；三点采样文件未接入 | 完整 3×3 covariance、Gaussian MC、有效性统计 |
| 式 10 独立可靠性 c_g | 只有熵/peak/sigma/固定零旋转通道的 GeometryGate | 独立 G_rel，加入多视图一致性与支持度 |
| 式 11 初始融合 g | 当前 g 控制每轮相对上一轮的更新 | 明确初始单目锚点和迭代回退语义 |
| 式 12–16 梯度驱动 ConvGRU IRM | 原 G-Net 多轮前馈，轮间 detach | 新 hidden state、decoder、objective gradient、ConvGRU |
| 式 17 Laplace 深度监督 | 原 Gaussian NLL | 独立 Laplace scale head，与采样 sigma 分开 |

旧 `rotation_utils.py` 是单个向量的 nominal/± 扰动，权重 0.25/0.5/0.25；
它不是论文中 iid Gaussian、1/Q 权重的完整协方差采样。
旧 `homography.py` 返回越大越好的相似度 S；论文 C 越小越好。
保持 G-Net 输入 S 的原语义，概率统一写成 softmax(S / tau)，或令
C=-S 后用 softmax(-C / tau)，不能误把现有 S 直接负号 softmax。

## 3. 旋转不确定性来源与接口

每条 reference-to-source 边提供：

| 字段 | 形状/约定 |
|---|---|
| `R_i0`, `t_i0` | `[B,V,3,3]`、`[B,V,3]`，参考到源 |
| `rot_cov` | `[B,V,3,3]`，右切空间，rad² |
| `rot_cov_valid` | `[B,V]`，区别缺失/不可观与真正零方差 |
| provenance | provider、标定版本、frame IDs、原始切空间约定 |

允许两种清晰标注的模式：

1. 固定协方差：所有测试角度使用同一预先指定的协方差，仅作为匹配模块开发
   和灵敏度实验。不能称作估计的旋转可靠性，更不能根据当前注入的真值角度改变它。
2. 估计协方差：由外部位姿估计器/结构旋转估计器输出并离线缓存。
   U-ARE-ME 依赖 Manhattan 假设及法线线索，不是所有 ScanNet 区域都满足；
   需要明确不可观、非 Manhattan 与估计失败时的处理。

论文引用的 U-ARE-ME 尚未在该仓库实现，不能把当前 `rot_unc=None` 宣称为
已完成协方差传播。禁止将训练噪声 labels/真实角度输入模型。
外部估计必须在可用观测和提供给模型的位姿条件下运行；不能直接套用一份
准确 GT 位姿的协方差，声称它刻画后来注入的人为偏差。

相对旋转协方差必须由两帧联合协方差经相对旋转函数的 Jacobian 推导：
Sigma_rel = J Sigma_joint J^T，保留跨帧交叉项；独立假设需明确记录。
左右切空间转换在同一旋转下为 Sigma_right = R^T Sigma_left R。
先用有限差分验证具体的 pose convention，禁止未经推导直接相加两帧协方差。
缺失不等于零，不可观也不等于很确定。固定模式必须显式打开并写入日志。

## 4. M1：旋转感知匹配

### 本次已写的基础函数

`sample_rotation_covariance(R, rot_cov, Q, generator=...)` 位于
`models/submodules/rotation_covariance.py`，返回 `[Q,B,V,3,3]` rotations、
`[Q,B,V,3]` tangent vectors 和 `[Q]` 等权重。

xi_q = L z_q；R_q = R Exp([xi_q]x)；z_q iid N(0,I)；alpha=1/Q。
严格验证形状、有限值、SO(3)、对称性和正定性。精确零协方差给出原旋转，
不静默添加 jitter；半正定但非零的退化矩阵暂时明确报错，由 provider 处理。
独立 generator 必须与 tensor 在同一设备，后续 checkpoint 要保存其状态。
Q=1 是一次随机采样，**不等于** nominal pose；确定性消融应显式使用零协方差
或关闭采样。验证按 frame IDs/seed 固定标准正态样本，避免 batch 改变结果。

### 下一步集成

新增匹配接口返回 `(S_bar, per_view_scores, support, consistency)`，而非仅聚合体积。
在 120×160 分辨率依次处理 Q 和源视图，累加统计，不保留所有 warped features。
先测 Q=4，比较 Q=1/4/8；这些是开发预算，不是已验证最优值。
保持 FP32 基线，单独报告采样增加的耗时/显存，再决定 batch。

有效性至少包括 pose-valid、正深度、有限投影、采样中心位于有效图像范围，以及
与原 CW 深度一致性筛选对应的支持标记。支持张量随 view/sample/depth 变化，
不能仅用每个样本 has_source 代替逐像素/候选有效性。

边界策略必须明确：若将越界样本重新归一化为 sum(m*C)/sum(m)，估计的是
有效投影条件下的代价，不再是式 7 原始的无条件 MC 期望。首版保留固定 1/Q，
定义越界样本的有限无效代价，并输出有效采样比例给可靠性头；所有无效候选在
softmax 前屏蔽，全部候选/视图无效则直接返回单目先验。无效代价的定义和
支持度归一化写入配置与论文。零协方差复现需区分旧 homography 和修正有效性后的
nominal matcher；将有效性修正作为单独开关，避免把收益全部归因于旋转采样。

跨视图一致性可先用有效视图的 depth posterior 间 JS divergence 和
CW 支持比例；不足两视图时 divergence 不应解释为强一致，额外输入 view count。

## 5. M2：可靠性与融合

在低分辨率上建立两个独立模块：

- `GeometricReliabilityHead`：输入 S_bar、normalized entropy、peak、
  view posterior disagreement、valid sample/view fractions、协方差统计。
  输出 c_g `[B,1,H,W]`；无支持像素强制为零。
- `AdaptiveFusion`：输入 S_bar、c_g、mono sigma、单目特征，输出融合门控 g、
  几何残差及 h0。首版 h0 设为 64 channels，参数需由实际 Orin 测试确定。

主融合遵循论文：mu0 = mu_m + g*delta_mu_mv。c_g 是证据可靠性，g 是实际融合量，
不能在报告中混称。开发中可选 g=valid*c_g*sigmoid(a) 强化回退，但这增加了
论文式 11 的显式约束，需要单独开关、消融并更新论文描述。

同时保留冻结单目 prior 作为整个 refinement 的锚点。旧代码的
mu_next=mu_prev+g*delta 在后续 gate=0 时只会停在上一轮，不会撤销既有错误更新。
为了检验论文的最终回退声明，新增测试在全无支持时从融合到 IRM 到上采样均保持
定义好的 monocular-only 路径。低分辨率 fallback 必须精确；高分辨率比较要使用
同一单目 upsampling 路径，避免 learned mask/边界 padding 导致假回退。

仅用深度监督先验证 c_g/g 梯度与作用，不新增 oracle gate 标签。
论文式 13 固定 c_g 对 h 求梯度，并不意味着 c_g 在所有计算路径全局 detach；
保留它经融合到深度监督的路径，否则可靠性头可能没有训练信号。
门控初始 bias=4 使旧 g≈0.982；它解释初始化，不足以证明最终门控失效。
新头的初始化作为明确实验因素，不用“强制 gate 很小”替代有效性检验。

## 6. M3：可靠性梯度驱动 IRM

先实现 K=0 的融合 decoder，再接 K=1，最后 K=3。
不要简单把旧 3 轮 G-Net 外再套 3 轮 ConvGRU，这改变计算预算也不对齐图 2/3。

1. p=softmax(S_bar/tau)，D_mv=sum(p*d_hypotheses)。支持不足处 D_mv=mu_m。
2. 固定 D_mv、mu_m、sigma_m 和当前 c_g 构造 robust depth objective：
   E=sum[c_g*rho(D(h)-D_mv)+lambda_m/(sigma_m²+eps)*rho(D(h)-mu_m)]。
   rho 首版选 smooth Charbonnier；lambda、eps、尺度归一化记录在配置。
   单目 inverse variance 可能很大，若做 clipping/normalization 必须声明为实现变体。
3. 经 decoder 自动微分求 dE/dh，ConvGRU 更新 h，再解码深度与不确定性。
4. 每步监督深度，记录 E_before/E_after，但不承诺 learned GRU 保证能量单调下降。
5. 图中的迭代是每个参考帧内部的迭代，不等于跨视频帧持久记忆；动态扰动实验
   不能将单帧内部 recurrence 称作显式时序建模。

推理仅冻结网络参数，不关闭计算 h 梯度的能力。旧 validate 外层
`@torch.no_grad()` 内，IRM 局部用 `torch.enable_grad()` 和
`autograd.grad(E,h)`；不能直接全局 inference_mode 后假设 enable_grad 能修复。
推理步间 detach h 防止计算图增长，不调用 optimizer，不修改参数。

首版采用显式 first-order 模式：梯度信号作为 detached input，其他 h 的递归
训练路径保留。它省去对梯度的二阶反传，不应宣称完整高阶可微优化。
若启用 create_graph=True，作为独立二阶实验报告内存、耗时与实际收益。
参考 PyTorch 2.8 的 enable_grad 和 autograd.grad 官方说明。

## 7. Gaussian prior 与 Laplace 输出

mono sigma_m：Gaussian 标准差，只用于原先验、深度候选和单目 objective 权重。
refinement b：独立 `softplus(raw_b)+eps` Laplace scale，用于式 17 的
abs(error)/b + log(b)。不能直接给旧 Gaussian sigma 换损失名称，也不能把 b
作为下一轮 Gaussian sigma 采样。若需要标准差，Laplace std=sqrt(2)*b。

首版 IRM 在固定 Gaussian prior 候选生成的 D_mv 上细化，不复用 Laplace b
生成搜索候选。需要重新搜索时必须增加独立 Gaussian 搜索尺度并明确论文修订。
切换到 Laplace 时保留匹配的 loss-only 对照，不能把损失函数差异算作 IRM 收益。

## 8. 实施顺序和验收

| 阶段 | 交付 | 必须通过的检查 |
|---|---|---|
| 0，本次 | 备份工具、Gaussian sampler、设计 | 文件哈希、SO(3)/协方差/seed 数值测试 |
| 1 | covariance provider contract + matcher + visibility | 零 covariance、无视图回退、有效性修正消融、40-step GPU |
| 2 | c_g + fusion、逐帧评估 | 每个 trainable head 梯度、0/5/8°、gate quantiles、风险覆盖 |
| 3 | K=0/1/3 decoder-gradient IRM | 推理梯度可用、参数不变、内存不累积、first/second-order 区别 |
| 4 | 完整网络短跑与正式实验 | 相同 split/初始化/预算，最佳模型选择规则，多个独立验证空间 |

无需先跑完 A/B 才能开展网络设计。A/B 仍是论文实验必须补齐的对照：
A 准确位姿无门控；B 扰动训练无门控；C 已完成的扰动+旧门控。
新增模块分阶段对照 matching、reliability、fusion、IRM、Gaussian/Laplace loss。
原论文 Table II 的 Fusion 输入 c_g：若禁用 learned reliability，要定义
c_g=valid-mask 或固定 heuristic，不能留空一个不可运行的组合。

所有结构对比从同一原始 backbone 权重和声明的新增头初始化开始。
可以用 C-best 做工程 warm-start，但不能把额外训练预算隐藏在与 A/B 的对照里。
保留预设的 0/5/8° 平均 RMSE 选模，再报告 0/1/2/3/5/8°；不要看 test 后改选模。

新的评估输出应包含 per-frame RMSE/AbsRel/delta1、c_g/g 分位数和直方图、
无支持比例、D_mv 误差、实际 depth update、K 次 refinement 指标与耗时。
风险覆盖分别按 output uncertainty、c_g 排序，在 100/90/80/70% coverage
报告误差；单个验证空间的曲线只作开发诊断。分开报告网络与协方差 provider 耗时。

## 9. 论文当前应同步修正之处

- 第 4 页 Fig.4 目前重复了 IRM 架构示意，并非 caption 所述动态旋转/可靠性/RMSE
  三条时序曲线，需要用真实实验图替换。
- Table I–III 仍为 XX，占位结果不能支撑摘要/结论中的已证明性能声明。
- 现有 C 组不能称为已实现 uncertainty propagation 或 reliability-aware IRM。
- 图中 sigma_R 标量与正文 3×3 Sigma_R 应统一符号，Laplace scale 建议用 b
  避免和 sigma_m 混淆。
- 固定协方差、估计协方差、假设匹配的合成协方差实验分别标注；任一 oracle
  upper bound 如单独研究都不能混入可部署主结果。

## 资料

- 用户 main.pdf：Fig.2/3，公式 3–17，Tables I–III。
- [基线源码](https://github.com/nwwwhaowa/MaGNet/tree/bfe125992a332061a055698c6955267a468e7585)
- [U-ARE-ME 原论文](https://arxiv.org/abs/2403.15583)
- [PyTorch 2.8 enable_grad](https://docs.pytorch.org/docs/2.8/generated/torch.enable_grad.html)
- [PyTorch 2.8 autograd.grad](https://docs.pytorch.org/docs/2.8/generated/torch.autograd.grad.html)
