
import argparse
import datetime
import json
import os
import time
import warnings

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from dataset import MyDataset
from model import NDDE, ConvNDDE, SimpleNDDE
from torchdde import (RK2, RK4, DDESolver, Euler, Ralston,
                      TorchLinearInterpolator, ddesolve_adjoint)

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

   
    ## For delays they need to be tau > dt and that max(tau) < max_delays defined in the pb 
    delays_init, delays_cvg, delays_min, delays_max = [], [], [], []
    dist = torch.distributions.uniform.Uniform(0.2, 1.9)
    # sample from the distribution
    list_delays_init = dist.sample((40, 1))
    plt.plot(list_delays_init.cpu().detach().numpy(), label="Init")
    plt.savefig(default_dir + "/delays_init.png",bbox_inches='tight',dpi=100)
    plt.close()
    
    for init_index, delays in enumerate(list_delays_init): 
         #### GENERATING DATA #####
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ts = torch.linspace(0, 100, 1000)
        dt = ts[1] - ts[0]
        ys = torch.sin(ts/2 *torch.pi) 
        ys = ys.to(torch.float32)
        ys = ys.view((10, int(ys.shape[0]/10), 1))
        ts_true = ts[:ys.shape[1]]
        ys, ts_true = ys.to(device), ts_true.to(device)
        print(ys.shape, ts_true.shape)

        i = np.random.randint(0, ys.shape[0])
        plt.plot(ts_true.cpu().detach().numpy(), ys[i].cpu().detach().numpy(), label="Truth")
        plt.savefig(default_dir + "/training_data.png",bbox_inches='tight',dpi=100)
        plt.close() 
        nb_delays = 1
        list_delays = delays
        tmp_delays_max, tmp_delays_max = list_delays, list_delays
        print("list_delays init",list_delays)
        list_delays = list_delays.to(device)
        delays_init.append(list_delays.clone().detach())
    
        model = NDDE(ys.shape[-1], list_delays, width=32)
        model = model.to(device)
        lossfunc = nn.MSELoss()
        lr = 0.01
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)

        # computing history function 
        dataset = MyDataset(ys)
        train_len = int(len(dataset)*0.7)      
        train_set, test_set = random_split(dataset, [train_len, len(dataset)-train_len])
        train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=128, shuffle=False)

        json_filename = "hyper_parameters.json"
    
        dic_data = {
            "id": datestring,
            "metadata": {
                "input_shape" : ys.shape,
                "nb_delays": nb_delays,
                "lr" : lr,
                "delays_init": list([l.item() for l in list_delays.cpu()]),
                "model_name" : model.__class__.__name__,
                "model_structure" : str(model).split("\n"), 
                "optimizer_state_dict" : opt.state_dict(),
            },
        }

        with open(default_dir + "/" + json_filename, "w") as file:
            json.dump([dic_data], file)
        
        max_epoch = 10000
        losses, eval_losses, delay_values = [], [], []
        done = False
        tmp_delays_min, tmp_delays_max = list_delays, list_delays
        for i in range(max_epoch):
            model.train()
            for p, data in enumerate(train_loader):
                max_delay = max(list_delays).to(device)
                tmp_delays_min, tmp_delays_max = min(list_delays, tmp_delays_min), max(list_delays, tmp_delays_max)
                idx = (ts_true > max_delay).nonzero().flatten()[0]
                ts_history_train, ts_train = ts_true[:idx+1], ts_true[ idx:]
                ys_history, ys = data[:, :idx+1], data[:, idx:]   
                history_interpolator = TorchLinearInterpolator(ts_history_train, ys_history)
                history_function = lambda t: history_interpolator(t)
                opt.zero_grad()
                t = time.time()
                ret = ddesolve_adjoint(history_function, model, ts_train)
                loss = lossfunc(ret, ys)
                loss.backward()
                opt.step()
                
                losses.append(loss.item())
                delay_values.append(model.delays.clone().detach()) 
                print("{} Epoch : {}, Step {}/{}, Loss : {:.3e}, tau : {}".format(init_index, i, p, int(train_len/train_loader.batch_size), loss.item(), [d.item() for d in model.delays])) 
                     
                if losses[-1] < 1e-4 or i == max_epoch - 1 :
                    delays_cvg.append(model.delays.clone().detach())
                    delays_max.append(tmp_delays_max.clone().detach())
                    delays_min.append(tmp_delays_max.clone().detach())
                    done = True
                    break
                
                if i > 2 : 
                    if losses[-1] == losses[-2]:
                        delays_cvg.append(model.delays.clone().detach())
                        delays_max.append(tmp_delays_max.clone().detach())
                        delays_min.append(tmp_delays_max.clone().detach())
                        done = True
                        break
            if done:
                break
    
    torch.save(torch.stack(delays_init) , default_dir + "/saved_data/delays_init.pt")
    torch.save(torch.stack(delays_cvg) , default_dir + "/saved_data/delays_cvg.pt")
    torch.save(torch.stack(delays_min) , default_dir + "/saved_data/delays_min.pt")
    torch.save(torch.stack(delays_max) , default_dir + "/saved_data/delays_max.pt")
