import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import numpy as np
from torch.utils.data import DataLoader


class TrajDataset(torch.utils.data.Dataset):
    def __init__(self, traj, keep_ratio=0.5, ds_type="random", token=True):
        # init函数的作用：输入原始数据

        self.traj = traj

        self.src = []
        self.src_gps = []
        self.trg_gps = []
        self.trg_eid = []
        self.trg_rate = []
        self.src_len = []
        self.trg_len = []

        self.keep_ratio = keep_ratio
        self.ds_type = ds_type
        self.token = token

        self.get_data()

    def __len__(self):
        # 返回batch大小
        return len(self.traj)

    def __getitem__(self, index):
        # 返回一个batch的数据
        return (
            self.src[index],
            self.src_gps[index],
            self.src_len[index],
            self.trg_gps[index],
            self.trg_eid[index],
            self.trg_rate[index],
            self.trg_len[index],
        )

    def get_data(self):
        for traj in tqdm(self.traj):
            traj_sample = self.downsample_traj(
                traj[:, [0, 1, 8, 2, 3, 7]], self.ds_type, self.keep_ratio
            )  # 加入7是为了保证src中不含eid为空的点
            # grid_x, grid_y, src_lat, src_lng, trg_lat, trg_lng, ratio, edge_id, tid

            if self.token:
                traj = np.concatenate((np.zeros((1, 9)), traj))
            self.trg_gps.append(torch.Tensor(traj[:, [4, 5]]))
            self.trg_eid.append(torch.LongTensor(traj[:, 7]))
            self.trg_rate.append(torch.Tensor(traj[:, 6]))
            self.trg_len.append(len(traj))

            if self.token:
                traj_sample = np.concatenate((np.zeros((1, 5)), traj_sample))
            self.src.append(torch.Tensor(traj_sample[:, :3]))
            self.src_gps.append(torch.Tensor(traj_sample[:, 3:]))
            self.src_len.append(len(traj_sample))

    @staticmethod
    def downsample_traj(traj, ds_type="random", keep_ratio=0.5):
        """
        Down sample trajectory
        Args:
        -----
        traj:
            list of Point()
        ds_type:
            ['uniform', 'random']
            uniform: sample GPS point every down_stepth element.
                     the down_step is calculated by 1/remove_ratio
            random: randomly sample (1-down_ratio)*len(old_traj) points by ascending.
        keep_ratio:
            float. range in (0,1). The ratio that keep GPS points to total points.
        Returns:
        -------
        traj:
            new Trajectory()
        """
        assert ds_type in [
            "uniform",
            "random",
        ], "only `uniform` or `random` is supported"

        old_traj = traj.copy()
        old_traj = traj[np.where(traj[:, -1] != 0)]
        keep_ratio = traj.shape[0] * keep_ratio / (old_traj.shape[0])
        start_pt = old_traj[0].reshape(1, -1)
        end_pt = old_traj[-1].reshape(1, -1)

        if ds_type == "uniform":
            new_traj_ = old_traj[:: int(1 / keep_ratio)].reshape(1, -1)
            if (len(old_traj) - 1) % int(1 / keep_ratio) == 0:
                new_traj = new_traj_
            else:
                new_traj = np.concatenate((new_traj_, end_pt))
        elif ds_type == "random":
            sampled_inds = sorted(
                random.sample(
                    range(1, len(old_traj) - 1),
                    int((len(old_traj) - 2) * keep_ratio),
                )
            )
            new_traj = np.concatenate((start_pt, old_traj[sampled_inds], end_pt))

        return new_traj[:, :-1]


def batch2device(batch, device):
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

    return src, src_gps, src_len, trg_gps, trg_eid, trg_rate, trg_len


def collate_fn_ode(data):
    """
    Collate function for DataLoader
    """
    data.sort(key=lambda x: x[2], reverse=True)
    src, src_gps, src_len, trg_gps, trg_eid, trg_rate, trg_len = zip(*data)
    batch = len(src)
    src_ = torch.zeros(trg_len[0], len(src), src[0].shape[-1]) - 1
    src_gps_ = torch.zeros(trg_len[0], len(src), src_gps[0].shape[-1])
    src_len = torch.zeros(batch, src_len[0])
    for i in range(batch):
        idx = (src[i][:, -1] - src[i][0, -1]).long()
        src_[idx, i, :] = src[i]
        src_gps_[idx, i, :] = src_gps[i]
        src_len[i, : len(idx)] = idx
    src_len = torch.unique(
        src_len,
    ).long()
    trg_gps = torch.nn.utils.rnn.pad_sequence(trg_gps, padding_value=0)
    trg_eid = torch.nn.utils.rnn.pad_sequence(trg_eid, padding_value=0)
    trg_rate = torch.nn.utils.rnn.pad_sequence(trg_rate, padding_value=0)

    return src_, src_gps_, src_len, trg_gps, trg_eid, trg_rate, trg_len


def collate_fn(data):
    """
    Collate function for DataLoader
    """
    data.sort(key=lambda x: x[2], reverse=True)
    src, src_gps, src_len, trg_gps, trg_eid, trg_rate, trg_len = zip(*data)
    src = torch.nn.utils.rnn.pad_sequence(src, padding_value=0)
    src_gps = torch.nn.utils.rnn.pad_sequence(src_gps, padding_value=0)
    trg_gps = torch.nn.utils.rnn.pad_sequence(trg_gps, padding_value=0)
    trg_eid = torch.nn.utils.rnn.pad_sequence(trg_eid, padding_value=0)
    trg_rate = torch.nn.utils.rnn.pad_sequence(trg_rate, padding_value=0)

    return src, src_gps, src_len, trg_gps, trg_eid, trg_rate, trg_len


def get_iterator(dataset, batch_size, ode=False):
    """
    获得dataloader
    """
    iterator = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_ode if ode else collate_fn,
        pin_memory=True,
    )

    return iterator


def plot_record(record: np.ndarray, keep_ratio: float, model_name: str):
    """
    根据record画图

    Input:
        record: np.ndarray, shape=(3, 5, epoch)
        keep_ratio: float, keep ratio

    Output:
        saved figure
    """
    data_name = ["train", "valid", "test"]
    record_name = ["Recall", "Precision", "MSE", "RMSE", "Loss"]
    for j in range(5):
        plt.figure()
        for i in range(3):
            plt.plot(record[i, j, :], label=data_name[i])
            plt.title(f"{model_name} with Keep Ratio: {keep_ratio:.2%}", fontsize=18)
            plt.ylabel(record_name[j], fontsize=16)
            plt.xlabel("Epoch", fontsize=16)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.legend(fontsize=16)

        plt.tight_layout()
        plt.savefig(f"./data/figure/{record_name[j]}_{keep_ratio}.png")
        plt.close()
