
import argparse
import datetime
import json
import os
import time
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from dataset import MyDataset, ks
from model import MLP, NDDE, ConvNDDE, ConvODE
from torchdde import (TorchLinearInterpolator, nddesolve_adjoint,
                      odesolve_adjoint)

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
    default_dir_ode = default_save_dir + "/" + datestring + "/ode"
    default_dir_dde = default_save_dir + "/" + datestring + "/dde"
    
    os.makedirs(default_dir_dde)
    os.makedirs(default_dir_dde + "/training")
    os.makedirs(default_dir_dde + "/delays_evolution")
    os.makedirs(default_dir_dde + "/saved_data")
    
    os.makedirs(default_dir_ode)
    os.makedirs(default_dir_ode + "/training")
    os.makedirs(default_dir_ode + "/delays_evolution")
    os.makedirs(default_dir_ode + "/saved_data")

    matplotlib.use('Agg')
    #### GENERATING DATA #####
    dataset_size = 16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ts = torch.linspace(0, 40, 401)
    
    ys = ks(dataset_size, ts)
    ys = ys.to(torch.float32)
    ys, ts = ys.to(device), ts.to(device)
    ys, ts = ys[:, 200:], ts[:-200]
    print(ys.shape)

    j = np.random.randint(0, dataset_size)
    plt.imshow(ys[j].cpu().detach().numpy(), label="Truth")
    plt.savefig(default_dir + "/training_data.png",bbox_inches='tight',dpi=100)
    plt.close() 

    
    ## For delays they need to be tau > dt and that max(tau) < max_delays defined in the pb 
    nb_delays = 6
    list_delays = torch.rand((nb_delays,))
    list_delays = list_delays.to(device)
    
    nb_features = 32
    ys = ys[:, :, ::nb_features] # 4 features
        
    model = ConvNDDE(ys.shape[-1], list_delays)
    model = model.to(device)
    ode_model = MLP(ys.shape[-1]) #ConvODE(ys.shape[-1])
    ode_model = ode_model.to(device)
    
    lossfunc = nn.MSELoss()
    lr = 0.001
    ode_opt = torch.optim.Adam(ode_model.parameters(), lr=lr, weight_decay=0)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)

    # computing history function 
    dataset = MyDataset(ys)
    train_len = int(len(dataset)*0.7)      
    train_set, test_set = random_split(dataset, [train_len, len(dataset)-train_len])
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False)

    length_init = 40 
    max_epoch = 5000
    losses, eval_losses, delay_values = [], [], []
    
    json_filename = "hyper_parameters.json"
   
    dic_data = {
        "id": datestring,
        "metadata": {
            "input_shape" : ys.shape,
            "dataset_size": dataset_size,
            "nb_delays": nb_delays,
            "delays_init": list([l.item() for l in list_delays.cpu()]),
            "features_every": nb_features,
            "length_init" : length_init,
            "max_epoch" : max_epoch,
            "model_name" : model.__class__.__name__,
            "model_structure" : str(ode_model).split("\n"), 
            "optimizer_state_dict" : opt.state_dict(),
        },
    }

    with open(default_dir_ode + "/" + json_filename, "w") as file:
        json.dump([dic_data], file)
    
    
    ### ODE fitting
    for i in range(max_epoch):
        ode_model.train()
        for p, data in enumerate(train_loader):  
            ode_opt.zero_grad()
            t = time.time()
            ret = odesolve_adjoint(data[:, 0], ode_model, ts[:length_init])
            loss = lossfunc(ret, data[:, :length_init])
            loss.backward()
            ode_opt.step()
            
            if i % 50 == 0 or i == max_epoch - 1: 
                k = np.random.randint(0, data.shape[0])
                plt.plot(ys[k, :length_init].cpu().detach().numpy(), label="Truth")
                plt.plot(ret[k].cpu().detach().numpy(), '--', label="Pred")
                plt.savefig(default_dir_ode +  f'/training/step_{i}.png',bbox_inches='tight',dpi=100)
                plt.close()
                
                plt.plot(range(len(losses)), losses)
                plt.xlabel("steps")
                plt.savefig( default_dir_ode + '/loss.png',bbox_inches='tight',dpi=100)
                plt.close()
                
                plt.plot(range(len(eval_losses)), eval_losses)
                plt.xlabel("steps")
                plt.savefig( default_dir_ode + '/eval_loss.png',bbox_inches='tight',dpi=100)
                plt.close()
    
                torch.save(losses, default_dir_ode + "/saved_data/training_loss.pt")
                torch.save(ode_model.state_dict(), default_dir_ode + "/saved_data/model.pt")
                
            print("Epoch : {}, Step {}/{}, Length {}, Loss : {:.3e}".format(i, p, int(train_len/train_loader.batch_size),length_init, loss.item()))

            losses.append(loss.item())
            
            if losses[-1] < 0.005 :
                length_init +=1
            if length_init == ys.shape[1] :
                break
            
            j = np.random.randint(0, ys.shape[0])
            if losses[-1] < 1e-5 or i == max_epoch - 1:
                plt.plot(ys[j, :length_init].cpu().detach().numpy(), label="Truth")
                plt.plot(ret[j].cpu().detach().numpy(), '--', label="Pred")
                plt.legend()
                plt.savefig(default_dir_ode + "/training_example_pred.png",bbox_inches='tight',dpi=100)
                plt.close()
                break
        
        ode_model.eval()
        for r, eval_data in enumerate(test_loader):
            ret = odesolve_adjoint(eval_data[:, 0], ode_model, ts)
            loss = lossfunc(ret, eval_data)
            eval_losses.append(loss.item())
    
    
    # computing history function 
    dataset = MyDataset(ys)
    train_len = int(len(dataset)*0.7)      
    train_set, test_set = random_split(dataset, [train_len, len(dataset)-train_len])
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False)

    length_init = 40 
    max_epoch = 5000
    losses, eval_losses, delay_values = [], [], []
    
                
    ### DDE fitting 
    for i in range(max_epoch):
        model.train()
        for p, data in enumerate(train_loader):
            idx = (ts >= max(list_delays)).nonzero().flatten()[0]
            ts_history_train, ts_train = ts[:idx+1], ts[idx:idx+length_init]
            ys_history, ys = data[:, :idx+1], data[:, idx:idx+length_init]   
            history_interpolator = TorchLinearInterpolator(ts_history_train, ys_history)
            history_function = lambda t: history_interpolator(t)
            opt.zero_grad()
            t = time.time()
            ret = nddesolve_adjoint(history_function, model, ts_train)
            loss = lossfunc(ret, ys)
            loss.backward()
            opt.step()
            tmp_delays = model.delays.clone().detach() > torch.max(model.delays) 
            if torch.any(tmp_delays) :
                model.delays = torch.nn.Parameter(torch.where(tmp_delays, torch.max(model.delays), model.delays))
                
            if i % 50 == 0 or i == max_epoch - 1: 
                k = np.random.randint(0,data.shape[0])
                plt.plot(ys[k].cpu().detach().numpy(), label= "Truth")
                plt.plot(ret[k].cpu().detach().numpy(), '--', label="Pred")
                plt.savefig(default_dir_dde +  f'/training/step_{i}.png',bbox_inches='tight',dpi=100)
                plt.close()
                
                plt.plot(range(len(losses)), losses)
                plt.xlabel("steps")
                plt.savefig( default_dir_dde + '/loss.png',bbox_inches='tight',dpi=100)
                plt.close()
                
                plt.plot(range(len(eval_losses)), eval_losses)
                plt.xlabel("steps")
                plt.savefig( default_dir_dde + '/eval_loss.png',bbox_inches='tight',dpi=100)
                plt.close()
                
                if delay_values != []:
                    delay_values2 = torch.stack(delay_values) 
                    for i in range(delay_values2.shape[1]):
                        plt.plot(range(len(losses)), delay_values2[:, i].cpu().detach().numpy())
                        plt.xlabel("steps")
                        plt.ylabel(f"Delay #{i} : $\tau$")
                        plt.savefig(default_dir_dde + f'/delays_evolution/delays_{i}.png',bbox_inches='tight',dpi=100)
                        plt.close()
                    torch.save(delay_values2, default_dir_dde + "/saved_data/delay_values.pt")
                
                torch.save(losses, default_dir_dde + "/saved_data/training_loss.pt")
                torch.save(model.state_dict(), default_dir_dde + "/saved_data/model.pt")
                
            print("Epoch : {}, Step {}/{}, Length {}, Loss : {:.3e}, tau : {}".format(i, p, int(train_len/train_loader.batch_size),length_init, loss.item(), [d.item() for d in model.delays]))

            losses.append(loss.item())
            delay_values.append(model.delays.clone().detach())
            
            if losses[-1] < 0.005 :
                length_init +=1
            if length_init == ys.shape[1] - idx -5 :
                break
            
            j = np.random.randint(0, data.shape[0])
            if losses[-1] < 1e-5 or i == max_epoch - 1:
                plt.plot(ys[j].cpu().detach().numpy(), label= "Truth")
                plt.plot(ret[j].cpu().detach().numpy(), '--', label="Pred")
                plt.colorbar()
                plt.legend()
                plt.savefig(default_dir_dde + "/training_example_pred.png",bbox_inches='tight',dpi=100)
                plt.close()
                break
        
        model.eval()
        for r, eval_data in enumerate(test_loader):
            idx2 = (ts >= torch.max(model.delays)).nonzero().flatten()[0]
            ts_history_eval, ts_eval = ts[:idx2+1], ts[idx2:]
            ys_history, ys = eval_data[:, :idx2+1], eval_data[:, idx2:]   
            history_interpolator = TorchLinearInterpolator(ts_history_eval, ys_history)
            history_function = lambda t: history_interpolator(t)
            ret = nddesolve_adjoint(history_function, model, ts_eval)
            loss = lossfunc(ret, ys)
            eval_losses.append(loss.item())
                   