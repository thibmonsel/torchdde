import json
import os
import time
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from torchdde import TorchLinearInterpolator, ddesolve_adjoint


class DDETrainer:
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
        os.makedirs(self.saving_path)
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
        losses, eval_losses, delay_values = [], torch.Tensor(), []
        best_val_counter = 0
        best_val_loss = torch.inf

        for epoch in range(1, max_epochs + 1):
            self.model.train()
            for p, data in enumerate(train_loader):
                self.optimizers.zero_grad()
                t = time.time()
                idx = (ts >= max(self.model.delays)).nonzero().flatten()[0]
                ts_history_train, ts_train = (
                    ts[: idx + 1],
                    ts[idx : idx + init_ts_length],
                )
                ys_history, ys = data[:, : idx + 1], data[:, idx : idx + init_ts_length]
                history_interpolator = TorchLinearInterpolator(
                    ts_history_train, ys_history
                )
                history_function = lambda t: history_interpolator(t)
                ret = ddesolve_adjoint(history_function, self.model, ts_train)
                loss = loss_func(ret, ys)
                loss.backward()
                self.optimizers.step()
                exceeded_delays = self.model.delays.clone().detach() > torch.max(
                    self.model.delays
                )
                
                if torch.any(exceeded_delays) :
                    self.model.delays = torch.nn.Parameter(
                        torch.where(
                            exceeded_delays,
                            torch.max(self.model.delays),
                            self.model.delays,
                        )
                    )
                exceeded_delays2 = self.model.delays.clone().detach() < ts[1] - ts[0]
                if torch.any(exceeded_delays2):
                    self.model.delays = torch.nn.Parameter(
                        torch.where(
                            exceeded_delays2,
                            ts[1] - ts[0],
                            self.model.delays,
                        )
                    )
                    raise Warning(
                        "Gradient descent wants to increase the delay, we set it to the maximum delay"
                    )

                print(
                    "Epoch : {}, Step {}/{}, Length {}, Loss : {:.3e}, Tau {}, Time {}".format(
                        epoch,
                        p,
                        len(train_loader),
                        init_ts_length,
                        loss.item(),
                        [d.item() for d in self.model.delays],
                        time.time() - t,
                    )
                )

                losses.append(loss.item())
                delay_values.append(self.model.delays.clone().detach())
                ### Visualization ###
                if epoch % save_figure_every == 0 or epoch == max_epochs - 1:
                    k = np.random.randint(0, data.shape[0])

                    training_prediction_example_path = (
                        self.saving_path + f"/training/epoch_{epoch}.png"
                    )
                    loss_path = self.saving_path + "/loss.png"
                    eval_loss_path = self.saving_path + "/eval_loss.png"
                    delay_saving_path = self.saving_path + f"/delays_evolution/"

                    self.plot_training_prediction_example(
                        ys[k], ret[k], training_prediction_example_path
                    )
                    self.plot_loss(losses, loss_path)
                    self.plot_loss(eval_losses, eval_loss_path)
                    self.plot_delay_evolution(delay_values, delay_saving_path)

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
                if init_ts_length <= ts_train.shape[0]:
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
            idx = (ts >= max(self.model.delays)).nonzero().flatten()[0]
            ts_history_train, ts_train = ts[: idx + 1], ts[idx : idx + init_ts_length]
            ys_history, ys = (
                eval_data[:, : idx + 1],
                eval_data[:, idx : idx + init_ts_length],
            )
            history_interpolator = TorchLinearInterpolator(ts_history_train, ys_history)
            history_function = lambda t: history_interpolator(t)
            ret = ddesolve_adjoint(history_function, self.model, ts_train)
            loss = loss_func(ret, ys)
            eval_losses.append(loss.item())
        return torch.tensor(eval_losses)

    @staticmethod
    def plot_training_prediction_example(y_truth, y_pred, saving_path):
        plt.plot(y_truth.cpu().detach().numpy(), label="Truth")
        plt.plot(y_pred.cpu().detach().numpy(), "--", label="Pred")
        plt.savefig(saving_path, bbox_inches="tight", dpi=100)
        plt.close()

    @staticmethod
    def plot_loss(losses, saving_path):
        plt.plot(range(len(losses)), losses)
        plt.xlabel("steps")
        plt.savefig(saving_path, bbox_inches="tight", dpi=100)
        plt.close()

    @staticmethod
    def plot_delay_evolution(delays, saving_path):
        tmp_delays = torch.stack(delays)
        for i in range(tmp_delays.shape[1]):
            plt.plot(
                range(tmp_delays.shape[0]), tmp_delays[:, i].cpu().detach().numpy()
            )
            plt.xlabel("steps")
            plt.ylabel(f"Delay #{i} ")
            plt.savefig(saving_path + f"delays_{i}.png", bbox_inches="tight", dpi=100)
            plt.close()
