# VIT-Diffusion

Multi-head diffusion model with mixture-of-experts style output.

## 1. Architecture

### HydraHead Output

模型使用 $x_0$-prediction 模式。HydraHead 输出三元组 $(w, \mu, U)$：

| Symbol | Dimensions | Description |
| :--- | :--- | :--- |
| $w$ | $(B, K)$ | 分支概率 (softmax normalized) |
| $\mu$ | $(B, K, C, H, W)$ | 各分支预测的 $x_0$ |
| $U$ | $(B, K, D, R)$ | 正交基矩阵 |

**$\mu$ 的计算**：HydraHead 输出系数 $c \in \mathbb{R}^{B \times K \times R}$，然后 $\mu = U \cdot c$，即 $\mu$ 是 $U$ 列向量的线性组合。这使得 $\mu$ 天然在低秩子空间内。

### 推理采样

给定 $x_t$，模型预测 $(w, \mu, U)$，按概率 $w$ 采样一个分支 $k$，使用 $\mu_k$ 作为 $x_0$ 预测。

---

## 2. Loss Function

$$ L_{total} = L_{GMM} + \lambda_{repul} L_{repul} + \lambda_{ortho} L_{ortho} + \lambda_{div} L_{div} $$

### 2.1 GMM Loss (主损失)
$$ L_{GMM} = -\log \sum_k w_k \exp\left(-\frac{\|x_0 - \mu_k\|^2}{2\sigma^2}\right) $$

Winner-takes-all 机制：只有最接近 ground truth 的分支获得梯度。$\sigma$ 从大到小退火（sigma_start → sigma_end）。

### 2.2 Repulsion Loss (防止分支坍缩)
$$ L_{repul} = \sum_{i \neq j} \sqrt{w_i w_j} \cdot \max(0, \cos(\mu_i, \mu_j)) $$

当两个高概率分支的 $\mu$ 相似时施加惩罚，防止多个分支预测相同内容。

### 2.3 Orthogonality Loss
$$ L_{ortho} = \|U^T U - I\|_F^2 $$

约束 $U$ 为正交矩阵。

### 2.4 Diversity Loss (可选)
Batch 级别的熵正则，鼓励不同样本使用不同分支。

---

## 3. Usage

### 3.1 Synthetic Experiments
2D 玩具数据集验证：

```bash
./train_synthetic.sh --synthetic_type 8gaussians
```

数据集选项: `swiss_roll`, `8gaussians`, `checkerboard`, `olympic`

### 3.2 Image Training
```bash
./run_train.sh --data_path /path/to/images --batch_size 32

# 断点续训
./run_train.sh --resume_from_checkpoint checkpoints/run_001/epoch_50
```

### 3.3 Docker Multi-GPU Training

```bash
# VIT 模式
./docker_train.sh --mode vit

# Standard 模式，指定 4 卡 + WandB
./docker_train.sh --mode standard --gpus 4 --wandb --wandb-key YOUR_KEY

# 自定义参数
./docker_train.sh --mode vit --batch-size 64 --epochs 1000 --data-dir /path/to/data

# 交互式 shell
./docker_train.sh --shell
```

| Option | Default | Description |
| :--- | :--- | :--- |
| `--mode` | `vit` | `vit` 或 `standard` |
| `--gpus` | auto | GPU 数量 |
| `--batch-size` | `32` | 每卡 batch size |
| `--epochs` | `500` | 训练轮数 |
| `--data-dir` | `~/.cache/kagglehub` | 数据集目录 |
| `--wandb` | off | 启用 WandB |
| `--shell` | - | 交互式 shell |

---

## 4. Hyperparameters

### 4.1 Training Configs
| Argument | Default | Description |
| :--- | :--- | :--- |
| `--mode` | `vit` | `vit` (multi-head) 或 `standard` (single-head) |
| `--dataset_type` | `image_folder` | `image_folder` 或 `synthetic` |
| `--batch_size` | `8` | 每卡 batch size |
| `--learning_rate` | `1e-4` | AdamW 学习率 |
| `--num_epochs` | `100` | 训练轮数 |

### 4.2 Architecture Configs
| Argument | Default | Description |
| :--- | :--- | :--- |
| `--hidden_size` | `384` | Transformer 隐藏维度 |
| `--depth` | `12` | DiT block 数量 |
| `--num_heads` | `6` | Attention heads |
| `--vit_num_heads` | `4` | **$K$**: HydraHead 分支数 |
| `--vit_rank` | `16` | **$R$**: 正交基矩阵的秩 |
| `--input_size` | `32` | Latent 尺寸 |

### 4.3 Loss Configs
| Argument | Default | Description |
| :--- | :--- | :--- |
| `--lambda_gmm` | `10.0` | GMM 重建损失权重 |
| `--lambda_ortho` | `0.01` | 正交约束权重 |
| `--lambda_div` | `1.0` | 多样性熵损失权重 |
| `--lambda_repul` | `1.0` | 分支排斥损失权重 |
| `--sigma_start` | `2.0` | GMM 初始温度 |
| `--sigma_end` | `0.1` | GMM 最终温度 |
| `--temp_anneal_steps`| `20000` | 温度退火步数 |

---

## 5. Visualization

训练时自动记录到 WandB：

- **Scatter Plots**: 采样结果 vs 均值结果
- **Vector Field**: 逆向扩散过程动画

---

## 6. Folder Structure
```
src/
├── models/     # DiT backbone, HydraHead, StandardHead
├── diffusion/  # Loss functions, schedulers
├── data/       # Dataset loaders
└── vis/        # Visualization utilities
train.py        # Main training script
eval.py         # Evaluation script
```
