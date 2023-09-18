"""
程序入口
"""
import pickle
import argparse
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

from dataset import TrajDataset, get_iterator, batch2device, plot_record
from model import cal_loss, get_record

from model_ode import GRUODEBayes, Seq2SeqMulti

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", "-e", type=int, default=100)
    parser.add_argument("--batch_size", "-b", type=int, default=128)
    parser.add_argument("--lr", "-l", type=float, default=0.001)
    parser.add_argument("--rate_weight", "-r", type=float, default=10)
    parser.add_argument("--keep_ratio", "-k", type=float, default=0.5)
    parser.add_argument("--gpu", "-g", type=int, default=0)
    parser.add_argument("--load", action="store_false", help="load model")
    parser.add_argument("--ode", action="store_false", help="use ode")
    args = parser.parse_args()

    # DEVICE
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # DATASETS & DATALOADERS
    trajectory = pickle.load(open("./data/c2/traj_c2.pkl", "rb"))[
        :10000
    ]  # 130153个轨迹, 5912767个轨迹点

    # train:valid:test = 7:2:1
    train_data = trajectory[: int(len(trajectory) * 0.7)]
    valid_data = trajectory[int(len(trajectory) * 0.7) : int(len(trajectory) * 0.9)]
    test_data = trajectory[int(len(trajectory) * 0.9) :]

    train_dataset, valid_dataset, test_dataset = [
        TrajDataset(data, keep_ratio=args.keep_ratio, token=not args.ode)
        for data in [train_data, valid_data, test_data]
    ]

    train_iterator, valid_iterator, test_iterator = [
        get_iterator(data, batch_size=args.batch_size, ode=args.ode)
        for data in [train_dataset, valid_data, test_data]
    ]

    # batch = next(iter(train_iterator))
    # import pickle

    # pickle.dump(batch, open("./data/batch_ode.pkl", "wb"))
    # sys.exit()

    # MODEL & OPTIMIZER & CRITERION
    if args.ode:
        params = json.load(open("./params_ode.json", "r"))
        model = GRUODEBayes(params).to(device)
    else:
        params = json.load(open("./params.json", "r"))
        model = Seq2SeqMulti(params).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    criterion_eid = nn.NLLLoss()
    criterion_rate = nn.MSELoss()

    epoch_start = 0
    # train, valid, test; recall, precision, mse, rmse, loss
    record = np.zeros((3, 5, args.epochs))

    if args.load:
        for model_file in os.listdir("./model"):
            if model_file.startswith(f"seq2seq_multi_{args.keep_ratio}"):
                epoch_start_ = int(model_file.split("_")[-1].split(".")[0])
                if epoch_start_ > epoch_start:
                    epoch_start = epoch_start_
        if epoch_start > 0:  # load model
            model.load_state_dict(
                torch.load(f"./model/seq2seq_multi_{args.keep_ratio}_{epoch_start}.pt")
            )
            record = np.zeros((3, 5, epoch_start + args.epochs))
            record[:, :, :epoch_start] = np.load(
                f"./data/record/seq2seq_multi_{args.keep_ratio}_{epoch_start}.npy"
            )

    # TRAINING & EVALUATING & TESTING
    try:
        for epoch in range(epoch_start, epoch_start + args.epochs):
            model.train()
            with torch.autograd.set_detect_anomaly(False):
                for batch in tqdm(train_iterator, desc=f"Epoch {epoch + 1}"):
                    (
                        src,
                        src_gps,
                        src_len,
                        trg_gps,
                        trg_eid,
                        trg_rate,
                        trg_len,
                    ) = batch2device(batch, device)

                    optimizer.zero_grad()
                    with torch.cuda.device(device):
                        if args.ode:
                            (
                                eid_result,
                                rate_result,
                            ) = model(src, src_len, trg_len)
                            loss = cal_loss(
                                eid_result,
                                rate_result,
                                trg_eid,
                                trg_rate,
                                criterion_eid,
                                criterion_rate,
                                args.rate_weight,
                            )

                        else:
                            eid_result, rate_result = model(
                                src, src_len, trg_eid, trg_rate
                            )
                            loss = cal_loss(
                                eid_result,
                                rate_result,
                                trg_eid,
                                trg_rate,
                                criterion_eid,
                                criterion_rate,
                                args.rate_weight,
                            )

                    loss.backward(retain_graph=True)
                    optimizer.step()

                    record[0, :, epoch] = get_record(
                        eid_result, rate_result, trg_eid, trg_rate, trg_len
                    )

            model.eval()
            with torch.no_grad():
                for batch in valid_iterator:
                    (
                        src,
                        src_gps,
                        src_len,
                        trg_gps,
                        trg_eid,
                        trg_rate,
                        trg_len,
                    ) = batch2device(batch, device)
                    with torch.cuda.device(device):
                        if args.ode:
                            eid_result, rate_result = model(src, src_len, trg_len)
                            loss = cal_loss(
                                eid_result,
                                rate_result,
                                trg_eid,
                                trg_rate,
                                criterion_eid,
                                criterion_rate,
                                args.rate_weight,
                            )
                        else:
                            eid_result, rate_result = model(
                                src, src_len, trg_eid, trg_rate
                            )
                            loss = cal_loss(
                                eid_result,
                                rate_result,
                                trg_eid,
                                trg_rate,
                                criterion_eid,
                                criterion_rate,
                                args.rate_weight,
                            )
                    record[1, :, epoch] = get_record(
                        eid_result, rate_result, trg_eid, trg_rate, trg_len
                    )

                for batch in test_iterator:
                    (
                        src,
                        src_gps,
                        src_len,
                        trg_gps,
                        trg_eid,
                        trg_rate,
                        trg_len,
                    ) = batch2device(batch, device)

                    with torch.cuda.device(device):
                        if args.ode:
                            eid_result, rate_result = model(src, src_len, trg_len)
                            loss = cal_loss(
                                eid_result,
                                rate_result,
                                trg_eid,
                                trg_rate,
                                criterion_eid,
                                criterion_rate,
                                args.rate_weight,
                            )
                        else:
                            eid_result, rate_result = model(
                                src, src_len, trg_eid, trg_rate
                            )
                            loss = cal_loss(
                                eid_result,
                                rate_result,
                                trg_eid,
                                trg_rate,
                                criterion_eid,
                                criterion_rate,
                                args.rate_weight,
                            )

                    record[2, :, epoch] = get_record(
                        eid_result, rate_result, trg_eid, trg_rate, trg_len
                    )

    except KeyboardInterrupt:
        print(f"Training Interrupted on Epoch {epoch}!")
        # SAVE MODEL & RECORD
        if args.ode:
            torch.save(
                model.state_dict(),
                f"./model/gruode_multi_{args.keep_ratio}_{epoch}_ode.pt",
            )
            np.save(
                f"./data/record/gruode_multi_{args.keep_ratio}_{epoch}_ode.npy",
                record,
            )
        else:
            torch.save(
                model.state_dict(),
                f"./model/seq2seq_multi_{args.keep_ratio}_{epoch}.pt",
            )
            np.save(
                f"./data/record/seq2seq_multi_{args.keep_ratio}_{epoch}.npy", record
            )

        # PLOT RECORD
        plot_record(record, args.keep_ratio, "GRUODE" if args.ode else "Seq2Seq")

    else:
        print("Training Finished!")
        # SAVE MODEL & RECORD
        if args.ode:
            torch.save(
                model.state_dict(),
                f"./model/gruode_multi_{args.keep_ratio}_{epoch}_ode.pt",
            )
            np.save(
                f"./data/record/gruode_multi_{args.keep_ratio}_{epoch}_ode.npy",
                record,
            )
        else:
            torch.save(
                model.state_dict(),
                f"./model/seq2seq_multi_{args.keep_ratio}_{epoch}.pt",
            )
            np.save(
                f"./data/record/seq2seq_multi_{args.keep_ratio}_{epoch}.npy", record
            )

        # PLOT RECORD
        plot_record(record, args.keep_ratio, "GRUODE" if args.ode else "Seq2Seq")
