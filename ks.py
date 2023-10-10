
import argparse
import datetime
import os
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from dataset import MyDataset, ks
from model import NDDE, ConvNDDE, SimpleNDDE
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
    dataset_size = 16
    device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ts = torch.linspace(0, 40, 401)
    
    ys = ks(dataset_size, ts)
    ys = ys.to(torch.float32)
    ys, ts = ys.to(device), ts.to(device)
    ys, ts = ys[:, 50:], ts[:-50]
    print(ys.shape)

    j = np.random.randint(0, dataset_size)
    plt.imshow(ys[j].cpu().detach().numpy(), label="Truth")
    plt.savefig(default_dir + "/training_data.png",bbox_inches='tight',dpi=100)
    plt.close() 

    
    ## For delays they need to be tau > dt and that max(tau) < max_delays defined in the pb 
    nb_delays = 6
    max_delay = torch.tensor([5.0])
    list_delays = torch.arange(1, nb_delays+1)/2
    list_delays = torch.min(list_delays, max_delay.item() * torch.ones_like(list_delays))
    max_delay = max_delay.to(device)
    list_delays = list_delays.to(device)
    
    nb_features = 4
    features_idx = np.random.randint(0, ys.shape[-1], size=(nb_features, ))
    ys = ys[:, :, ::nb_features]
        
    model = ConvNDDE(ys.shape[-1], list_delays)
    model = model.to(device)
    lossfunc = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=10e-8)

    # computing history function 
    dataset = MyDataset(ys)
    train_len = int(len(dataset)*0.7)      
    train_set, test_set = random_split(dataset, [train_len, len(dataset)-train_len])
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False)

    max_epoch = 10000
    losses, eval_losses, delay_values = [], [], []
    length_init = 40 
    for i in range(max_epoch):
        model.train()
        for p, data in enumerate(train_loader):
            idx = (ts >= max_delay).nonzero().flatten()[0]
            ts_history_train, ts_train = ts[:idx+2], ts[idx:idx+length_init]
            ys_history, ys = data[:, :idx+2], data[:, idx:idx+length_init]   
            history_interpolator = TorchLinearInterpolator(ts_history_train, ys_history)
            history_function = lambda t: history_interpolator(t)
            opt.zero_grad()
            t = time.time()
            ret = nddesolve_adjoint(history_function, model, ts_train)
            loss = lossfunc(ret, ys)
            loss.backward()
            opt.step()
            tmp_delays = model.delays.clone().detach() > max_delay 
            if torch.any(tmp_delays) :
                model.delays = torch.nn.Parameter(torch.where(tmp_delays, max_delay, model.delays))
                
            if i % 50 == 0 or i == max_epoch - 1: 
                k = np.random.randint(0,ys.shape[0])
                plt.subplot(1,2,1)
                plt.imshow(ys[k].cpu().detach().numpy())
                plt.gca().set_title('Truth')
                plt.colorbar()
                plt.subplot(1,2,2)
                plt.imshow(ret[k].cpu().detach().numpy())
                plt.colorbar()
                plt.gca().set_title("Prediction")
                plt.savefig(default_dir +  f'/training/step_{i}.png',bbox_inches='tight',dpi=100)
                plt.close()
                
                plt.plot(range(len(losses)), losses)
                plt.xlabel("steps")
                plt.savefig( default_dir + '/loss.png',bbox_inches='tight',dpi=100)
                plt.close()
                
                plt.plot(range(len(eval_losses)), eval_losses)
                plt.xlabel("steps")
                plt.savefig( default_dir + '/eval_loss.png',bbox_inches='tight',dpi=100)
                plt.close()
                
                if delay_values != []:
                    delay_values2 = torch.stack(delay_values) 
                    for i in range(delay_values2.shape[1]):
                        plt.plot(range(len(losses)), delay_values2[:, i].cpu().detach().numpy())
                        plt.xlabel("steps")
                        plt.ylabel(f"Delay #{i} : $\tau$")
                        plt.savefig(default_dir + f'/delays_evolution/delays_{i}.png',bbox_inches='tight',dpi=100)
                        plt.close()
                    torch.save(delay_values2, default_dir + "/saved_data/delay_values.pt")
                
                torch.save(losses, default_dir + "/saved_data/training_loss.pt")
                torch.save(model.state_dict(), default_dir + "/saved_data/model.pt")
                
            print("Epoch : {}, Step {}/{}, Length {}, Loss : {:.3e}, tau : {}".format(i, p, int(train_len/train_loader.batch_size),length_init, loss.item(), [d.item() for d in model.delays]))

            losses.append(loss.item())
            delay_values.append(model.delays.clone().detach())
            
            if losses[-1] < 0.002 :
                length_init +=1
            if length_init == ys.shape[1] - idx -5 :
                break
            
            j = np.random.randint(0, ys.shape[0])
            if losses[-1] < 1e-5 or i == max_epoch - 1:
                plt.subplot(1,2,1)
                plt.imshow(ys[j].cpu().detach().numpy())
                plt.colorbar()
                plt.subplot(1,2,2)
                plt.imshow(ret[j].cpu().detach().numpy())
                plt.colorbar()
                plt.legend()
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
                   