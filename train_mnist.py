import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from diffusion import GaussianDiffusion
from UNet import Unet2D
from train import Trainer
import sys, os

# 设置当前工作目录为脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)
# 设置超参数
batch_size = 64
learning_rate = 1e-4
num_steps = 100000
image_size = 28
channels = 1

# 准备 MNIST 数据集
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
val_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 定义模型
model = Unet2D(
    dim=image_size, cond_emb=None, dim_mults=(1, 2), channels=channels, self_condition=False, resnet_block_groups=1
)

# 定义扩散过程
diffusion = GaussianDiffusion(
    model,
    seq_length=(channels, image_size, image_size),
    timesteps=1000,
    sampling_timesteps=100,
    objective="pred_noise",
    auto_normalize=False,
)


# 定义训练和验证函数
def train_function(model, batch, loss_fn):
    imgs, _ = batch
    return model(imgs)


def val_function(model, batch, loss_fn):
    imgs, _ = batch
    return model(imgs)


# 创建训练器
trainer = Trainer(
    model=diffusion,
    data_train=train_loader,
    data_val=val_loader,
    train_function=train_function,
    val_function=val_function,
    train_batch_size=batch_size,
    train_lr=learning_rate,
    train_num_steps=num_steps,
    save_every=2000,
    num_samples=25,
    results_folder="./results",
    amp=True,
)

# 开始训练
trainer.train()
