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

from dataset import MyDataset, brusellator
from dde_trainer import DDETrainer
from model import MLP, NDDE, SimpleNDDE, SimpleNDDE2

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    matplotlib.use("Agg")

    parser = argparse.ArgumentParser(description="Running node experiments")
    parser.add_argument("--seed", type=int, default=np.random.randint(0, 1000))
    parser.add_argument("--exp_path", default="")
    parser.add_argument("--delays", type=int, required=True)
    args = parser.parse_args()

    default_dir_dde = os.environ["default_dir"]
    print("default_dir_dde", default_dir_dde)
    
    #### GENERATING DATA #####
    dataset_size = 1024
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ts = torch.linspace(0, 25, 501)
    y0 = np.random.uniform(0.0, 2.0, (dataset_size, 2))
    y0[:, 1] = 0.0

    ys = brusellator(y0, ts, args=(1.0, 3.0))
    ys = ys[:, :, 0][..., None]
    ys = ys.to(torch.float32)
    ys, ts = ys.to(device), ts.to(device)
    print(ys.shape)

    nb_delay = args.delays
    list_delays = torch.abs(torch.rand((nb_delay,)))
    list_delays = list_delays.to(device)
    
    model = NDDE(ys.shape[-1], list_delays, width=32)
    model = model.to(device)

    dataset = MyDataset(ys)
    train_len = int(len(dataset) * 0.7)
    train_set, test_set = random_split(dataset, [train_len, len(dataset) - train_len])
    train_loader = DataLoader(train_set, batch_size=512, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=512, shuffle=False)

    lr, init_ts_length, max_epochs, validate_every, patience = 0.001, 80, 10000, 1, 20
    trainer = DDETrainer(model, lr_init=lr, lr_final=lr/100, saving_path=default_dir_dde)

    dic_data = {
        "metadata": {
            "input_shape": ys.shape,
            "dataset_size": dataset_size,
            "batch_size " : train_loader.batch_size,
            "nb_delays": args.delays,
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

    for i in range(ys.shape[0]):
        plt.plot(ys[i].cpu().detach().numpy(), label="Truth")
    plt.savefig(default_dir_dde + "/training_data.png", bbox_inches="tight", dpi=100)
    plt.close()

    trainer.train(
        ts,
        train_loader,
        test_loader,
        init_ts_length=init_ts_length,
        max_epochs=max_epochs,
        validate_every=validate_every,
        patience=patience,
    )
