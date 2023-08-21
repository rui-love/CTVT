"""
程序入口
"""
import pickle
import argparse

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

from dataset import TrajDataset, get_iterator
from model import Seq2SeqMulti

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", "-e", type=int, default=10)
    parser.add_argument("--batch_size", "-b", type=int, default=64)
    parser.add_argument("--lr", "-l", type=float, default=0.001)
    parser.add_argument("--rate_weight", "-r", type=float, default=0.1)
    args = parser.parse_args()

    # DEVICE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # DATASETS & DATALOADERS
    trajectory = pickle.load(open("./data/traj_final.pkl", "rb"))[
        :1000
    ]  # 130153个轨迹, 5912767个轨迹点

    # train:valid:test = 7:2:1
    train_data = trajectory[: int(len(trajectory) * 0.7)]
    valid_data = trajectory[int(len(trajectory) * 0.7) : int(len(trajectory) * 0.9)]
    test_data = trajectory[int(len(trajectory) * 0.9) :]

    train_dataset = TrajDataset(train_data)
    valid_dataset = TrajDataset(valid_data)
    test_dataset = TrajDataset(test_data)

    train_iterator = get_iterator(train_dataset, batch_size=args.batch_size)
    valid_iterator = get_iterator(valid_dataset, batch_size=args.batch_size)
    test_iterator = get_iterator(test_dataset, batch_size=args.batch_size)

    batch = next(iter(train_iterator))

    # MODEL & OPTIMIZER & CRITERION
    params = {
        "Encoder": {"input_dim": 3, "hid_dim": 512, "dropout": 0.5},
        "Decoder": {
            "eid_size": 30000,
            "rate_size": 1,
            "embedding_size": 128,
            "hidden_size": 512,
            "dropout_rate": 0.5,
        },
    }
    model = Seq2SeqMulti(params).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    criterion_eid = nn.NLLLoss()
    criterion_rate = nn.MSELoss()

    train_loss = np.zeros(args.epochs)
    valid_loss = np.zeros(args.epochs)
    test_loss = np.zeros(args.epochs)

    # TRAINING & EVALUATING & TESTING
    for epoch in range(args.epochs):
        model.train()
        for batch in tqdm(train_iterator, desc=f"Epoch {epoch + 1}"):
            src, src_gps, src_len, trg_gps, trg_eid, trg_rate, trg_len = batch

            src = src.to(device)
            src_gps = src_gps.to(device)
            trg_gps = trg_gps.to(device)
            trg_eid = trg_eid.to(device)
            trg_rate = trg_rate.to(device)

            # src = [src len, batch size, 3]
            # src_gps = [src len, batch size, 2]
            # src_len = tuple(int)[batch size]
            # trg_gps = [trg len, batch size, 2]
            # trg_eid = [trg len, batch size]
            # trg_rate = [trg len, batch size]
            # trg_len = tuple(int)[batch size]

            optimizer.zero_grad()
            eid_result, rate_result = model(src, src_len, trg_eid, trg_rate)
            # break
            # eid_result = [trg len, batch size, id one hot output dim]
            # rate_result = [trg len, batch size]
            mask = trg_eid != 0
            loss = criterion_rate(rate_result[mask], trg_rate[mask]) * args.rate_weight
            eid_result = eid_result.reshape(-1, eid_result.shape[-1])
            trg_eid = trg_eid.reshape(-1)
            mask = trg_eid != 0
            loss += criterion_eid(eid_result[mask], trg_eid[mask])  # 进入到loss计算中的数不能是负数
            loss.backward()
            optimizer.step()

            train_loss[epoch] += loss.item()

        model.eval()
        with torch.no_grad():
            for batch in valid_iterator:
                src, src_gps, src_len, trg_gps, trg_eid, trg_rate, trg_len = batch

                src = src.to(device)
                src_gps = src_gps.to(device)
                trg_gps = trg_gps.to(device)
                trg_eid = trg_eid.to(device)
                trg_rate = trg_rate.to(device)

                # src = [src len, batch size, 3]
                # src_gps = [src len, batch size, 2]
                # src_len = tuple(int)[batch size]
                # trg_gps = [trg len, batch size, 2]
                # trg_eid = [trg len, batch size]
                # trg_rate = [trg len, batch size]
                # trg_len = tuple(int)[batch size]

                eid_result, rate_result = model(src, src_len, trg_eid, trg_rate)
                # break
                # eid_result = [trg len, batch size, id one hot output dim]
                # rate_result = [trg len, batch size]
                mask = trg_eid != 0
                loss = (
                    criterion_rate(rate_result[mask], trg_rate[mask]) * args.rate_weight
                )
                eid_result = eid_result.reshape(-1, eid_result.shape[-1])
                trg_eid = trg_eid.reshape(-1)
                mask = trg_eid != 0
                loss += criterion_eid(
                    eid_result[mask], trg_eid[mask]
                )  # 进入到loss计算中的数不能是负数

                valid_loss[epoch] += loss.item()

            for batch in test_iterator:
                src, src_gps, src_len, trg_gps, trg_eid, trg_rate, trg_len = batch

                src = src.to(device)
                src_gps = src_gps.to(device)
                trg_gps = trg_gps.to(device)
                trg_eid = trg_eid.to(device)
                trg_rate = trg_rate.to(device)

                # src = [src len, batch size, 3]
                # src_gps = [src len, batch size, 2]
                # src_len = tuple(int)[batch size]
                # trg_gps = [trg len, batch size, 2]
                # trg_eid = [trg len, batch size]
                # trg_rate = [trg len, batch size]
                # trg_len = tuple(int)[batch size]

                eid_result, rate_result = model(src, src_len, trg_eid, trg_rate)
                # break
                # eid_result = [trg len, batch size, id one hot output dim]
                # rate_result = [trg len, batch size]
                mask = trg_eid != 0
                loss = (
                    criterion_rate(rate_result[mask], trg_rate[mask]) * args.rate_weight
                )
                eid_result = eid_result.reshape(-1, eid_result.shape[-1])
                trg_eid = trg_eid.reshape(-1)
                mask = trg_eid != 0
                loss += criterion_eid(
                    eid_result[mask], trg_eid[mask]
                )  # 进入到loss计算中的数不能是负数

                test_loss[epoch] += loss.item()

        train_loss[epoch] /= len(train_iterator)
        valid_loss[epoch] /= len(valid_iterator)
        test_loss[epoch] /= len(test_iterator)

        print(
            f"Epoch {epoch + 1}:\n\tTrain Loss: {train_loss[epoch]:.3f} | Train PPL: {np.exp(train_loss[epoch]):7.3f}"
        )
        print(
            f"\t Val. Loss: {valid_loss[epoch]:.3f} |  Val. PPL: {np.exp(valid_loss[epoch]):7.3f}"
        )
        print(
            f"\tTest Loss: {test_loss[epoch]:.3f} | Test PPL: {np.exp(test_loss[epoch]):7.3f}"
        )
    # SAVE MODEL
    torch.save(model.state_dict(), "./model/seq2seq_multi.pt")

    # PLOT
    plt.plot(train_loss, label="train loss")
    plt.plot(valid_loss, label="valid loss")
    plt.plot(test_loss, label="test loss")
    plt.legend()
    plt.savefig(f"./figure/{args.batch_size}_{args.lr}.png")
