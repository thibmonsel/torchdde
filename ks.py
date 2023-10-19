
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
from scipy.integrate import solve_ivp
from torch.utils.data import DataLoader, Dataset, random_split

from dataset import MyDataset, brusellator, ks
from dde_trainer import DDETrainer
from model import NDDE

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    matplotlib.use("Agg")

    parser = argparse.ArgumentParser(description="Running node experiments")
    parser.add_argument("--seed", type=int, default=np.random.randint(0, 1000))
    parser.add_argument("--exp_path", default="")
    parser.add_argument("--delays", type=int, required=True)
    parser.add_argument("--nb_features", type=int, required=True)

    args = parser.parse_args()

    default_dir_dde = os.environ["default_dir"]
    print("default_dir_dde", default_dir_dde)
    
    #### GENERATING DATA #####
    dataset_size = 16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ts = torch.linspace(0, 80, 801)
    
    ys = ks(dataset_size, ts)
    ys = ys.to(torch.float32)
    ys, ts = ys.to(device), ts.to(device)
    ys, ts = ys[:, 300:], ts[:-300]
    j = np.random.randint(0, dataset_size)

    plt.imshow(ys[j].cpu().detach().numpy(), label="Truth")
    plt.savefig(default_dir_dde + "/data_example.png",bbox_inches='tight',dpi=100)
    plt.close() 
    
    feat_idx = torch.randint(0, 10, (args.nb_features,))
    feat_idx = torch.arange(0, ys.shape[-1], step = ys.shape[-1]  // args.features ) + feat_idx
    ys = ys[:, :, feat_idx]
    
    plt.plot(ys[j].cpu().detach().numpy(), label="Truth")
    plt.savefig(default_dir_dde + "/training_data.png",bbox_inches='tight',dpi=100)
    plt.close() 

    nb_delay = args.delays
    list_delays = torch.abs(torch.rand((nb_delay,)))
    list_delays = list_delays.to(device)
    
    model = NDDE(ys.shape[-1], list_delays, width=64)
    model = model.to(device)

    dataset = MyDataset(ys)
    train_len = int(len(dataset) * 0.7)
    train_set, test_set = random_split(dataset, [train_len, len(dataset) - train_len])
    train_loader = DataLoader(train_set, batch_size=512, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=512, shuffle=False)

    lr, init_ts_length, max_epochs, validate_every, patience = 0.001, 50, 10000, 1, 30
    trainer = DDETrainer(model, lr_init=lr, lr_final=lr/100, saving_path=default_dir_dde)

    dic_data = {
        "metadata": {
            "input_shape": ys.shape,
            "dataset_size": dataset_size,
            "batch_size " : train_loader.batch_size,
            "nb_delays": args.delays,
            "feat_idx" : feat_idx.tolist(),
            "delays_init": list([l.item() for l in list_delays.cpu()]),
            "init_ts_length": init_ts_length,
            "validate_every" : validate_every,
            "patience" : patience,
            "lr_init": trainer.lr_init,
            "lr_final" : trainer.lr_final,
            "max_epochs": max_epochs,
            "dde_model_name": model.__class__.__name__,
            "dde_model_structure": str(model).split("\n"),
            "optimizer_state_dict": trainer.optimizers.state_dict(),
        }}

    with open(default_dir_dde + "/hyper_parameters.json", "w") as file:
        json.dump([dic_data], file)

    trainer.train(
        ts,
        train_loader,
        test_loader,
        init_ts_length=init_ts_length,
        max_epochs=max_epochs,
        validate_every=validate_every,
        patience=patience,
    )