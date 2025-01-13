import math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader

from accelerate import Accelerator
from ema_pytorch import EMA
import os
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

__version__ = "1.0.0"


def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num


def cycle(dl):
    while True:
        for data in dl:
            yield data


def exists(x):
    return x is not None


class Trainer(object):

    def __init__(
        self,
        model,
        data_train,
        data_val,
        train_function,
        val_function,
        train_batch_size=16,
        gradient_accumulate_every=1,
        train_lr=1e-4,
        train_num_steps=100000,
        ema_update_every=10,
        ema_decay=0.995,
        adam_betas=(0.9, 0.99),
        save_every=1000,
        num_samples=25,
        results_folder="./results",
        amp=False,
        mixed_precision_type="fp16",
        split_batches=True,
        max_grad_norm=1.0,
        loss_fn=F.mse_loss,
    ):
        super().__init__()

        self.loss_fn = loss_fn
        self.train_function = train_function
        self.val_function = val_function
        # accelerator
        self.accelerator = Accelerator(
            split_batches=split_batches, mixed_precision=mixed_precision_type if amp else "no"
        )

        # model

        self.model = model
        # self.channels = diffusion_model.channels

        # sampling and training hyperparameters

        assert has_int_squareroot(num_samples), "number of samples must have an integer square root"
        self.num_samples = num_samples
        self.save_every = save_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.max_grad_norm = max_grad_norm

        self.train_num_steps = train_num_steps

        # dataset and dataloader

        dl = (
            DataLoader(data_train, batch_size=train_batch_size, shuffle=True)
            if not isinstance(data_train, DataLoader)
            else data_train
        )
        self.data_val = (
            DataLoader(data_val, batch_size=train_batch_size * 10, shuffle=True)
            if not isinstance(data_val, DataLoader)
            else data_val
        )
        # , pin_memory=True, num_workers=cpu_count())
        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)
        # optimizer

        self.opt = Adam(model.parameters(), lr=train_lr, betas=adam_betas)

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(model, beta=ema_decay, update_every=ema_update_every)
            self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)

        # step counter state

        self.step = 0
        self.record = []  # add: milestone, train loss, test loss per checkpoint
        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            "step": self.step,
            "model": self.accelerator.get_state_dict(self.model),
            "opt": self.opt.state_dict(),
            "ema": self.ema.state_dict(),
            "scaler": self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            "version": __version__,
        }

        torch.save(data, str(self.results_folder / f"model-{milestone}.pt"))
        np.save(str(self.results_folder / "record.npy"), np.array(self.record))

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f"model-{milestone}.pt"), map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data["model"])

        self.step = data["step"]
        self.opt.load_state_dict(data["opt"])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if "version" in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data["scaler"]):
            self.accelerator.scaler.load_state_dict(data["scaler"])

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:
                self.model.train()

                total_loss = 0.0

                for _ in range(self.gradient_accumulate_every):
                    batch = next(self.dl)
                    with self.accelerator.autocast():
                        loss = self.train_function(self.model, batch, self.loss_fn)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                pbar.set_description(f"loss: {total_loss:.6f}")

                accelerator.wait_for_everyone()
                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()

                    if self.step != 0 and self.step % self.save_every == 0:
                        self.ema.ema_model.eval()

                        with torch.no_grad():
                            milestone = self.step // self.save_every
                            # 采样生成一张图片
                            x = self.ema.ema_model.sample(batch_size=10, cond=None)
                            x = (x.clamp(-1, 1) + 1) / 2
                            save_samples(x, str(self.results_folder / f"pic-{milestone}.png"))

                        self.record.append([milestone, total_loss])
                        self.save(milestone)
                pbar.update(1)

        accelerator.print("training complete")
        self.record = torch.tensor(self.record)
        min_index = torch.argmin(self.record[:, 2])
        accelerator.print(
            "Minimum validation error is: ", self.record[min_index, 2], " in milestone: ", self.record[min_index, 0]
        )


from torchvision.utils import make_grid


def save_samples(samples, filename, nrow=10):
    """
    保存生成的MNIST图片
    :param samples: 生成的样本，shape为(n_samples, 1, 28, 28)
    :param filename: 保存的文件名
    :param nrow: 每行显示的图片数量
    """
    # 确保samples是CPU上的NumPy数组
    samples = samples.cpu().numpy()

    # 反归一化处理，将像素值范围从[-1, 1]转换回[0, 1]
    samples = (samples + 1) * 0.5
    samples = np.clip(samples, 0, 1)

    # 将单通道灰度图像转换为三通道RGB图像
    samples = np.repeat(samples, 3, axis=1)

    # 使用make_grid将多个图像拼接成一个网格
    grid = make_grid(torch.from_numpy(samples), nrow=nrow)

    # 转换为NumPy数组并调整维度顺序
    grid = grid.permute(1, 2, 0).numpy()

    # 保存图像
    plt.imsave(filename, grid)
    plt.close()
