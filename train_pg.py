import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

from tqdm import tqdm
from denoising_diffusion_pytorch import Unet
import data_preprocess
import utils.metods

device = "cuda:0"
T = 1024

model = Unet(
    dim=64,
    dim_mults=(1, 2, 4, 8)
).to(device)  # 35.719.555 параметров


def train_base(size, batch_size, link):
    data_train, data_val, data_test = data_preprocess.get_data(photo_size=size, batch_size=batch_size, main_link=link)

    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-4)

    loss_train = []
    loss_val = []

    for epoch in tqdm(range(20)):
        memory_loss = []

        for batch in data_train:
            optimizer.zero_grad()
            x = batch[0].to(device)
            timestep = torch.randint(1, T, (x.shape[0],), device=device).long()

            xt, target = utils.metods.get_xt_pg(x, timestep)
            xt = xt.clone().to(torch.float32)

            image_predict = model(xt, timestep)

            loss = F.mse_loss(image_predict, target.to(torch.float32))
            memory_loss.append(loss.item())

            loss.backward()
            optimizer.step()

        loss_train.append(np.array(memory_loss).mean())
        memory_loss = []

        for batch in data_val:
            with torch.no_grad():
                x = batch[0].to(device)
                timestep = torch.randint(1, T, (x.shape[0],), device=device).long()

                xt, target = utils.metods.get_xt_pg(x, timestep)
                xt = xt.clone().to(torch.float32)

                image_predict = model(xt, timestep)

                loss = F.mse_loss(image_predict, target.to(torch.float32))
                memory_loss.append(loss.item())

        loss_val.append(np.array(memory_loss).mean())

        fig_samples, ax = plt.subplots(2, 2, figsize=(10, 10))
        imgs = utils.metods.generate_image(model, "PG", 3, 4, 64, 1, 1024)
        imgs[imgs > 1.0] = 1
        imgs[imgs < 0] = 0
        with torch.no_grad():
            for i in range(2):
                for j in range(2):
                    ax[i, j].imshow(imgs[i * 2 + j].to("cpu").permute(1, 2, 0))
        fig_samples.savefig(f"samples/prog_dist/1024/{epoch + 1}")

        fig_losses, ax = plt.subplots(1, figsize=(10, 10))

        ax.plot(loss_train, label='Обучение')
        ax.plot(loss_val, label='Валидация', linestyle='dashed')
        ax.legend()
        fig_losses.savefig(f"losses/prog_dist/1024/{epoch + 1}")

        if loss_val[-1] <= min(loss_val):
            torch.save(model.state_dict(), 'models/prog_dist/1024/model_shoes_1024.pth')


def train_distill(size, batch_size, link):
    data_train, data_val, data_test = data_preprocess.get_data(photo_size=size, batch_size=batch_size, main_link=link)

    model_teacher = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8)
    ).to(device)

    model_student = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8)
    ).to(device)

    for k in tqdm(range(5, 10)):
        step = 2 ** k

        N = 1024 // step

        model_teacher.load_state_dict(torch.load(f"models/prog_dist/{int(N * 2)}/model_shoes_{int(N * 2)}.pth"))
        model_teacher = model_teacher.to(device)
        model_student.load_state_dict(torch.load(f"models/prog_dist/{int(N * 2)}/model_shoes_{int(N * 2)}.pth"))
        model_student = model_student.to(device)

        optimizer = torch.optim.Adam(model_student.parameters(), lr=1.0e-4)

        loss_train = []
        loss_val = []

        for epoch in range(10):
            memory_loss = []
            for batch in data_train:
                optimizer.zero_grad()
                x = batch[0].to(device)
                timestep = (torch.randint(1, N + 1, (x.shape[0],)).long() / N * T).to(torch.int).to(device)
                timestep1 = (timestep - 0.5 / N * T).to(torch.int)
                timestep2 = (timestep - 1 / N * T).to(torch.int)

                target, predict_student = utils.metods.get_loss_distill_pg(x, model_student, model_teacher, timestep,
                                                                           timestep1, timestep2)

                loss = F.mse_loss(predict_student, target)
                memory_loss.append(loss.item())

                loss.backward()
                optimizer.step()

            loss_train.append(np.array(memory_loss).mean())
            memory_loss = []

            for batch in data_val:
                with torch.no_grad():
                    x = batch[0].to(device)
                    timestep = (torch.randint(1, N + 1, (x.shape[0],)).long() / N * T).to(torch.int).to(device)
                    timestep1 = (timestep - 0.5 / N * T).to(torch.int)
                    timestep2 = (timestep - 1 / N * T).to(torch.int)

                    target, predict_student = utils.metods.get_loss_distill_pg(x, model_student, model_teacher,
                                                                               timestep,
                                                                               timestep1, timestep2)

                    loss = F.mse_loss(predict_student, target)
                    memory_loss.append(loss.item())

            loss_val.append(np.array(memory_loss).mean())

            fig_samples, ax = plt.subplots(2, 2, figsize=(10, 10))
            imgs = utils.metods.generate_image(model_student, "PG", 3, 4, 64, step, 1024)
            imgs[imgs > 1.0] = 1
            imgs[imgs < 0] = 0
            with torch.no_grad():
                for i in range(2):
                    for j in range(2):
                        ax[i, j].imshow(imgs[i * 2 + j].to("cpu").permute(1, 2, 0))
            fig_samples.savefig(f"samples/prog_dist/{N}/{epoch + 1}")

            fig_losses, ax = plt.subplots(1, figsize=(10, 10))

            ax.plot(loss_train, label='Обучение')
            ax.plot(loss_val, label='Валидация', linestyle='dashed')
            ax.legend()
            fig_losses.savefig(f"losses/prog_dist/{N}/{epoch + 1}")

            if loss_val[-1] <= min(loss_val):
                torch.save(model_student.state_dict(), f'models/prog_dist/{N}/model_shoes_{N}.pth')
