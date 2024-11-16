import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

from tqdm import tqdm
from denoising_diffusion_pytorch import Unet
import data_preprocess
import utils.metods
from matplotlib.pyplot import figure

figure(figsize=(10, 10), dpi=80)
device = "cuda:0"
T = 1000

model = Unet(
    dim=64,
    dim_mults=(1, 2, 4, 8)
).to(device)  # 35.719.555 параметров


optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-4)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.99)

loss_train = []
loss_val = []


def train(size, batch_size, link):
    data_train, data_val, data_test = data_preprocess.get_data(photo_size=size, batch_size=batch_size, main_link=link)
    for epoch in tqdm(range(20)):
        memory_loss = []

        for batch in data_train:
            optimizer.zero_grad()
            x = batch[0].to(device)
            timestep = torch.randint(1, T, (x.shape[0],), device=device).long()

            xt, noise = utils.metods.get_xt_ddpm(x, timestep)
            xt = xt.clone().to(torch.float32)

            noise_predict = model(xt, timestep)
            loss = F.mse_loss(noise_predict, noise)
            memory_loss.append(loss.item())

            loss.backward()
            optimizer.step()
        loss_train.append(np.array(memory_loss).mean())

        memory_loss = []

        for batch in data_val:
            with torch.no_grad():
                x = batch[0].to(device)
                timestep = torch.randint(1, T, (x.shape[0],), device=device).long()

                xt, noise = utils.metods.get_xt_ddpm(x, timestep)
                xt = xt.clone().to(torch.float32)

                noise_predict = model(xt, timestep)
                loss = F.mse_loss(noise_predict, noise)
                memory_loss.append(loss.item())
        loss_val.append(np.array(memory_loss).mean())

        fig_samples, ax = plt.subplots(2, 2, figsize=(10, 10))
        imgs = utils.metods.generate_image(model, "DDPM", 3, 4, 64, 1, 1000)
        imgs[imgs > 1.0] = 1
        imgs[imgs < 0] = 0
        with torch.no_grad():
            for i in range(2):
                for j in range(2):
                    ax[i, j].imshow(imgs[i * 2 + j].to("cpu").permute(1, 2, 0))
        fig_samples.savefig(f"samples/classic/{epoch + 1}")

        fig_losses, ax = plt.subplots(1, figsize=(10, 10))

        ax.plot(loss_train, label='Обучение')
        ax.plot(loss_val, label='Валидация', linestyle='dashed')
        ax.legend()
        fig_losses.savefig(f"losses/classic/{epoch + 1}")

        if loss_val[-1] <= min(loss_val):
            torch.save(model.state_dict(), 'models/classic/model_shoes_1000.pth')

        scheduler.step()
