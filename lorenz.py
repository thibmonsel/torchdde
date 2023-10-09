
import argparse
import datetime
import os
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.integrate import solve_ivp
from torch.utils.data import DataLoader, Dataset, random_split

from dataset import MyDataset, lorenz
from model import NDDE, SimpleNDDE, SimpleNDDE2
from torchdde import (RK2, RK4, DDESolver, Euler, Ralston,
                      TorchLinearInterpolator, nddesolve_adjoint)

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description="Running node experiments")
    parser.add_argument("--seed", type=int, default=np.random.randint(0, 1000))
    parser.add_argument("--exp_path", default="")
    args = parser.parse_args()

    if args.exp_path == "":
        default_save_dir = "meta_data"
    else:
        default_save_dir = "meta_data/" + args.exp_path
    if not os.path.exists(default_save_dir):
        os.makedirs(default_save_dir)

    datestring = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    default_dir = default_save_dir + "/" + datestring
    os.makedirs(default_dir)
    os.makedirs(default_dir + "/training")
    os.makedirs(default_dir + "/delays_evolution")
    os.makedirs(default_dir + "/saved_data")

    #### GENERATING DATA #####
    dataset_size = 3
    device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ts = torch.linspace(0, 15, 601)
    y0 = np.random.uniform(0.0, 2.0, (dataset_size, 3))
    y0[:, 1:] = 0.0

    args = (10, 28, 8/3)
    ys = lorenz(y0, ts, args)
    ys = ys.to(torch.float32)
    ys, ts = ys.to(device), ts.to(device)

    for i in range(ys.shape[0]):
        plt.plot(ys[i].cpu().detach().numpy(), label="Truth")
    plt.savefig(default_dir + "/training_data.png",bbox_inches='tight',dpi=100)
    plt.close() 

    max_delay = torch.tensor([5.0])
    max_delay = max_delay.to(device)
    list_delays = torch.tensor([1.4, 2.0,  2.8])
    # list_delays = torch.abs(torch.rand((3,)))
    list_delays = torch.min(list_delays, max_delay.item() * torch.ones_like(list_delays))
    list_delays = list_delays.to(device)
    model = NDDE(ys.shape[-1], list_delays, width=128)

    model = model.to(device)
    lossfunc = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0)

    # computing history function 
    dataset = MyDataset(ys)
    train_len = int(len(dataset)*0.7)      
    train_set, test_set = random_split(dataset, [train_len, len(dataset)-train_len])
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False)

    max_epoch = 10000
    losses, eval_losses, delay_values = [], [], []
    # solver = Ralston()
    # dde_solver = DDESolver(solver, list_delays)
    for i in range(max_epoch):
        model.train()
        for p, data in enumerate(train_loader):
            idx = (ts >= max_delay).nonzero().flatten()[0]
            ts_history_train, ts_train = ts[:idx+1], ts[idx:]
            ys_history, ys = data[:, :idx+1], data[:, idx:]   
            history_interpolator = TorchLinearInterpolator(ts_history_train, ys_history)
            history_function = lambda t: history_interpolator(t)
            opt.zero_grad()
            t = time.time()
            ret = nddesolve_adjoint(history_function, model, ts_train)
            # ret, _ = dde_solver.integrate(model, ts_train, history_function)
            loss = lossfunc(ret, ys)
            loss.backward()
            opt.step()
            if i % 50 == 0:
                k = np.random.randint(0,ys.shape[0])
                plt.plot(ts_train.cpu(), ys[k].cpu().detach().numpy(), label="Truth")
                plt.plot(ts_train.cpu(),ret[k].cpu().detach().numpy(), "--")
                plt.xlabel("t")
                plt.ylabel("y(t)")
                plt.savefig(default_dir +  f'/training/step_{i}.png',bbox_inches='tight',dpi=100)
                plt.close()
            print("Epoch : {}, Step {}/{}, Loss : {:.3e}, tau : {}".format(i, p, int(train_len/train_loader.batch_size), loss.item(), [d.item() for d in model.delays]))

            losses.append(loss.item())
            delay_values.append(model.delays.clone().detach())
            
            j = np.random.randint(0, ys.shape[0])
            if losses[-1] < 1e-5 or i == max_epoch - 1:
                plt.plot(ts_train.cpu(),ys[j].cpu().detach().numpy())
                plt.plot(ts_train.cpu(),ret[j].cpu().detach().numpy(), "--")
                plt.xlabel("t")
                plt.ylabel("y(t)")
                plt.savefig(default_dir + "/training_example_pred.png",bbox_inches='tight',dpi=100)
                plt.close()
                break
        
        model.eval()
        for r, eval_data in enumerate(test_loader):
            idx2 = (ts >= max_delay).nonzero().flatten()[0]
            ts_history_eval, ts_eval = ts[:idx2+1], ts[idx2:]
            ys_history, ys = eval_data[:, :idx2+1], eval_data[:, idx2:]   
            history_interpolator = TorchLinearInterpolator(ts_history_eval, ys_history)
            history_function = lambda t: history_interpolator(t)
            ret = nddesolve_adjoint(history_function, model, ts_eval)
            loss = lossfunc(ret, ys)
            eval_losses.append(loss.item())
            
        plt.plot(range(len(losses)), losses)
        plt.xlabel("steps")
        plt.savefig( default_dir + '/loss.png',bbox_inches='tight',dpi=100)
        plt.close()
        
        plt.plot(range(len(eval_losses)), eval_losses)
        plt.xlabel("steps")
        plt.savefig( default_dir + '/eval_loss.png',bbox_inches='tight',dpi=100)
        plt.close()
        
        delay_values_2 = torch.stack(delay_values)
        for i in range(delay_values_2.shape[1]):
            plt.plot(range(len(losses)), delay_values_2[:, i].cpu().detach().numpy())
            plt.xlabel("steps")
            plt.ylabel(f"Delay #{i} : $\tau$")
            plt.savefig(default_dir + f'/delays_evolution/delays_{i}.png',bbox_inches='tight',dpi=100)
            plt.close()
        
        torch.save(losses, default_dir + "/saved_data/training_loss.pt")
        torch.save(delay_values_2, default_dir + "/saved_data/delay_values.pt")
        torch.save(model.state_dict(), default_dir + "/saved_data/model.pt")