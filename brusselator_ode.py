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
from torch.utils.data import DataLoader, random_split

from dataset import MyDataset, brusellator
from model import MLP
from ode_trainer import ODETrainer

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    matplotlib.use("Agg")

    parser = argparse.ArgumentParser(description="Running node experiments")
    parser.add_argument("--seed", type=int, default=np.random.randint(0, 1000))
    parser.add_argument("--exp_path", default="")
    args = parser.parse_args()

    if args.exp_path == "":
        default_save_dir = "meta_data"
    else:
        default_save_dir = "meta_data/" + args.exp_path
    
    os.makedirs(default_save_dir, exist_ok=True)

    datestring = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    default_dir = default_save_dir + "/" + datestring
    default_dir_ode = default_save_dir + "/" + datestring + "/ode"
    print("default_dir_dde", default_dir_ode)
    
    #### GENERATING DATA #####
    dataset_size = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ts = torch.linspace(0, 25, 501)
    y0 = np.random.uniform(0.0, 2.0, (dataset_size, 2))
    y0[:, 1] = 0.0

    ys = brusellator(y0, ts, args=(1.0, 3.0))
    ys = ys[:, :, 0][..., None]
    ys = ys.to(torch.float32)
    ys, ts = ys.to(device), ts.to(device)
    print(ys.shape)

    model = MLP(ys.shape[-1], width=32)
    model = model.to(device)

    dataset = MyDataset(ys)
    train_len = int(len(dataset) * 0.7)
    train_set, test_set = random_split(dataset, [train_len, len(dataset) - train_len])
    train_loader = DataLoader(train_set, batch_size=512, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=512, shuffle=False)

    lr, init_ts_length, max_epochs, validate_every, patience = 0.001, 40, 10000, 1, 20
    trainer = ODETrainer(model, lr_init=lr, lr_final=lr/100, saving_path=default_dir_ode)

    dic_data = {
        "id": datestring,
        "metadata": {
            "input_shape": ys.shape,
            "dataset_size": dataset_size,
            "batch_size " : train_loader.batch_size,
            "init_ts_length": init_ts_length,
            "validate_every" : validate_every,
            "patience" : patience,
            "lr_init": trainer.lr_init,
            "lr_final" : trainer.lr_final,
            "max_epochs": max_epochs,
            "ode_model_name": model.__class__.__name__,
            "ode_model_structure": str(model).split("\n"),
            "optimizer_state_dict": trainer.optimizers.state_dict(),
        },
    }

    with open(default_dir_ode + "/hyper_parameters.json", "w") as file:
        json.dump([dic_data], file)

    for i in range(ys.shape[0]):
        plt.plot(ys[i].cpu().detach().numpy(), label="Truth")
    plt.savefig(default_dir + "/training_data.png", bbox_inches="tight", dpi=100)
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
