import random
import pickle
from tqdm import tqdm
import torch
import numpy as np
from torch.utils.data import DataLoader


class TrajDataset(torch.utils.data.Dataset):
    def __init__(self, traj, keep_ratio=0.5, ds_type="random"):
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
                traj[:, [0, 1, 4, 2, 3]], self.ds_type, self.keep_ratio
            )
            # grid_x, grid_y, src_lat, src_lng, trg_lat, trg_lng, ratio, edge_id, tid

            traj = np.concatenate((np.zeros((1, 9)), traj))
            self.trg_gps.append(torch.Tensor(traj[:, [4, 5]]))
            self.trg_eid.append(torch.LongTensor(traj[:, 7]))
            self.trg_rate.append(torch.Tensor(traj[:, 6]))
            self.trg_len.append(len(traj))

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

        return new_traj


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


def get_iterator(dataset, batch_size):
    """
    获得dataloader
    """
    iterator = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return iterator
