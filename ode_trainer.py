import os
import time
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from torchdde import odesolve_adjoint


class ODETrainer:
    def __init__(self, model, lr_init, lr_final, saving_path):
        self.model = model
        self.lr_init = lr_init
        self.lr_final = lr_final
        self.saving_path = saving_path
        self.optimizers = torch.optim.Adam(
            self.model.parameters(), lr=lr_init, weight_decay=0
        )
        self.create_directories()

    def create_directories(self):
        os.makedirs(self.saving_path, exist_ok=True)
        os.makedirs(self.saving_path + "/training")
        os.makedirs(self.saving_path + "/delays_evolution")
        os.makedirs(self.saving_path + "/saved_data")

    def train(
        self,
        ts,
        train_loader,
        val_loader,
        init_ts_length=5,
        loss_func=nn.MSELoss(),
        max_epochs=5000,
        patience=10,
        validate_every=1,
        save_figure_every=50,
    ):
        assert init_ts_length < ts.shape[0]
        losses, eval_losses = [], torch.Tensor()
        best_val_counter = 0
        best_val_loss = torch.inf

        for epoch in range(1, max_epochs + 1):
            self.model.train()
            for p, data in enumerate(train_loader):
                self.optimizers.zero_grad()
                t = time.time()
                ret = odesolve_adjoint(data[:, 0], self.model, ts[: init_ts_length])
                loss = loss_func(ret, data[:, :init_ts_length])
                loss.backward()
                self.optimizers.step()
                print(
                    "Epoch : {}, Step {}/{}, Length {}, Loss : {:.3e}, Time {}".format(
                        epoch,
                        p,
                        len(train_loader),
                        init_ts_length,
                        loss.item(),
                        time.time() - t,
                    )
                )

                losses.append(loss.item())

                ### Visualization ###
                if epoch % save_figure_every == 0 or epoch == max_epochs - 1:
                    k = np.random.randint(0, data.shape[0])

                    training_prediction_example_path = (
                        self.saving_path + f"/training/epoch_{epoch}.png"
                    )
                    loss_path = self.saving_path + "/loss.png"
                    eval_loss_path = self.saving_path + "/eval_loss.png"

                    self.plot_training_prediction_example(
                        data[k, :init_ts_length], ret[k], training_prediction_example_path
                    )
                    self.plot_loss(losses, loss_path)
                    self.plot_loss(eval_losses, eval_loss_path)

                    torch.save(
                        losses, self.saving_path + "/saved_data/last_training_loss.pt"
                    )
                    torch.save(
                        self.model.state_dict(),
                        self.saving_path + "/saved_data/last_model.pt",
                    )

            if epoch % validate_every == 0:
                tmp_eval_loss = self.validate(ts, val_loader, init_ts_length, loss_func)
                eval_losses = torch.cat([eval_losses, tmp_eval_loss])
                mean_tmp_eval_loss = torch.mean(tmp_eval_loss).item()
                if mean_tmp_eval_loss < best_val_loss:
                    best_val_loss = mean_tmp_eval_loss
                    best_val_counter = 0
                    torch.save(
                        self.model.state_dict(),
                        self.saving_path + "/saved_data/best_model.pt",
                    )
                else:
                    best_val_counter += 1

            if best_val_counter > patience:
                if init_ts_length <= ts.shape[0]:
                    init_ts_length += 1
                    best_val_counter = 0 
                    for g in self.optimizers.param_groups:
                            ## lr descreases from lr_init to lr_final from 0 to max_epochs / step function
                            g["lr"] = (self.lr_final - g["lr"]) / max_epochs * epoch + self.lr_init
                else : 
                    print("Training done and saving models, data ...")
                    torch.save(
                        losses, self.saving_path + "/saved_data/last_training_loss.pt"
                    )
                    torch.save(
                        self.model.state_dict(),
                        self.saving_path + "/saved_data/last_model.pt",
                    )
                    torch.save(
                        delay_values, self.saving_path + "/saved_data/last_delay_values.pt"
                    )

        print("Finished {} epochs of training".format(max_epochs))
        
    def validate(self, ts, val_loader, init_ts_length, loss_func=nn.MSELoss()):
        eval_losses = []
        self.model.eval()
        for eval_data in val_loader:
            ret = odesolve_adjoint(eval_data[:, 0], self.model, ts[:init_ts_length])
            loss = loss_func(ret, eval_data[:, :init_ts_length])
            eval_losses.append(loss.item())
        return torch.tensor(eval_losses)

    @staticmethod
    def plot_training_prediction_example(y_truth, y_pred, saving_path):
        if y_truth.shape[-1] == 1 : 
            plt.plot(y_truth.cpu().detach().numpy(), label="Truth")
            plt.plot(y_pred.cpu().detach().numpy(), "--", label="Pred")
            plt.savefig(saving_path, bbox_inches="tight", dpi=100)
            plt.close()
        else : 
            plt.subplot(211)
            plt.title("Truth")
            plt.imshow(y_truth.cpu().detach().numpy())
            plt.colorbar()
            plt.subplot(212)
            plt.title("Pred")
            plt.imshow(y_pred.cpu().detach().numpy())
            plt.savefig(saving_path, bbox_inches="tight", dpi=100)
            plt.close()

    @staticmethod
    def plot_loss(losses, saving_path):
        plt.plot(range(len(losses)), losses)
        plt.xlabel("steps")
        plt.savefig(saving_path, bbox_inches="tight", dpi=100)
        plt.close()

