import random

import torch
import torch.nn as nn
from torchdiffeq import odeint
import torch.nn.functional as F


class MutiTaskBlock(nn.Module):
    """
    hidden -> eid, rate
    """

    def __init__(self, params):
        super(MutiTaskBlock, self).__init__()

        self.eid_size = params["eid_size"]
        self.rate_size = params["rate_size"]
        self.emb_dim = params["embedding_size"]
        self.hidden_dim = params["hidden_size"]

        self.embedding = nn.Embedding(self.eid_size, self.emb_dim)

        self.dropout = nn.Dropout(params["dropout_rate"])

        self.pre_eid = nn.Linear(self.hidden_dim, self.eid_size)
        self.pre_rate_1 = nn.Linear(self.hidden_dim + self.emb_dim, self.hidden_dim)
        self.pre_rate_2 = nn.Linear(self.hidden_dim, self.rate_size)

    def forward(self, hidden):
        """
        forward
        """
        # eid = [batch_size]
        # rate = [batch_size]
        # hidden = [batch_size, hidden_dim]
        pre_eid_1 = F.log_softmax(self.pre_eid(hidden), dim=1)  # [batch_size, eid_size]
        pre_eid_2 = pre_eid_1.argmax(dim=1).long()  # [batch_size]

        pre_rate_1 = self.pre_rate_1(
            torch.cat((self.dropout(self.embedding(pre_eid_2)), hidden), dim=1)
        )  # [batch_size, hidden_size]
        pre_rate_2 = self.pre_rate_2(F.relu(pre_rate_1)).squeeze(
            1
        )  # [batch_size, rate_size]
        pre_rate_3 = F.sigmoid(pre_rate_2)  # [batch_size, rate_size]

        return pre_eid_1, pre_rate_3


class GRUODEFunc(nn.Module):
    """
    Func of GRUODE
    """

    def __init__(self, params):
        super().__init__()

        hidden_size = params["hidden_size"]
        self.zeros_x = params["zeros_x"]

        if self.zeros_x:
            self.lin_hh = torch.nn.Linear(hidden_size, hidden_size, bias=False)
            self.lin_hz = torch.nn.Linear(hidden_size, hidden_size, bias=False)
            self.lin_hr = torch.nn.Linear(hidden_size, hidden_size, bias=False)

        else:
            self.net = nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 2),
                nn.Tanh(),
                nn.Linear(hidden_size * 2, hidden_size),
            )

    def forward(self, t, h):
        if self.zeros_x:
            z = torch.sigmoid(self.lin_hz(h))
            r = torch.sigmoid(self.lin_hr(h))
            h_tilde = torch.tanh(self.lin_hh(r * h))
            dh = (1 - z) * (h_tilde - h)
        else:
            dh = self.net(h)

        return dh


class Encoder(nn.Module):
    """
    Encoder
    """

    def __init__(self, params):
        super(Encoder, self).__init__()

        self.hid_dim = params["hid_dim"]

        self.input_dim = params["input_dim"]

        self.rnn = nn.GRU(self.input_dim, self.hid_dim)  # 单层GRU

        self.dropout = nn.Dropout(params["dropout"])

    def forward(self, src, src_len):
        """
        forward
        """
        # src = [src len, batch size, 3]
        # src_len = tuple(int)[batch size]

        # pack操作，将长度不一的序列打包，使得RNN只对实际的序列长度进行计算，忽略pad加的0部分，src指定序列，src_len指定每个序列的长度
        packed_embedded = nn.utils.rnn.pack_padded_sequence(src, src_len)

        packed_outputs, hidden = self.rnn(packed_embedded)

        # pad_packed_sequence将packed sequence解包，输出和输入是一样的，只是长度不一样
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs)

        # outputs = [src len, batch size, hidden_dim * num directions]
        # hidden = [1, batch size, hidden_dim]

        return outputs, hidden


class Decoder(nn.Module):
    """
    Decoder
    """

    def __init__(self, params):
        super(Decoder, self).__init__()

        self.eid_size = params["eid_size"]
        self.rate_size = params["rate_size"]
        self.emb_dim = params["embedding_size"]
        self.hidden_dim = params["hidden_size"]

        self.embedding = nn.Embedding(self.eid_size, self.emb_dim)

        self.rnn = nn.GRUCell(self.emb_dim + self.rate_size, self.hidden_dim)

        self.dropout = nn.Dropout(params["dropout_rate"])

    def forward(self, eid, rate, hidden):
        """
        forward
        """
        # eid = [batch_size]
        # rate = [batch_size]
        # hidden = [batch_size, hidden_dim]

        eid_embbeded = self.dropout(self.embedding(eid))  # [batch_size, emb_dim]
        rnn_input = torch.cat(
            (eid_embbeded, rate.unsqueeze(1)), dim=1
        )  # [batch_size, emb_dim + rate_size]
        hidden = self.rnn(rnn_input, hidden)  # output = [batch_size, hidden_dim]

        return hidden


class GRUODEBayes(nn.Module):
    def __init__(self, params) -> None:
        super().__init__()

        self.params = params
        self.gru_obs = nn.GRUCell(
            input_size=self.params["input_size"], hidden_size=self.params["hidden_size"]
        )
        self.gru_ode = GRUODEFunc(params=params)
        self.muti_task = MutiTaskBlock(params=params)

    def ode_h(self, t, h):
        index = list(range(0, len(t) - 1, self.params["ode_step"])) + [len(t) - 2]
        h = odeint(self.gru_ode, h, t)[index]
        return h

    def obs_x(self, x, h):
        h = self.gru_obs(x, h)
        return h

    def forward(self, src, src_len, trg_len):
        """
        return result, loss
        """
        trg_len_max = trg_len[0]
        batch_size = src.shape[1]
        device = src.device
        eid_result = torch.zeros(trg_len_max, batch_size, self.params["eid_size"]).to(
            device
        )
        rate_result = torch.zeros(trg_len_max, batch_size).to(device)

        min_len_idx = len(trg_len)
        h = self.gru_obs(src[0])
        for i in range(1, src_len.shape[0]):
            while src_len[i] >= trg_len[min_len_idx - 1]:
                min_len_idx -= 1
            # t[k] -> t[k+1], 使用ode_h来计算
            h_ode = self.ode_h(
                torch.linspace(
                    0,
                    src_len[i] - src_len[i - 1],
                    steps=self.params["ode_step"] * (src_len[i] - src_len[i - 1]) + 1,
                ).to(device),
                h[:min_len_idx],
            )

            h = h_ode[-1]

            # obs_idx 找到该时刻有obs的idx
            obs_idx = torch.where(src[src_len[i], :min_len_idx, -1] != -1)[0]
            # obs_x 用于计算obs_idx对应的h，bayes更新
            h[obs_idx] = self.gru_obs(src[src_len[i], obs_idx, :], h[obs_idx])
            # h_post_result[src_len[i], :min_len_idx] = h

            # muti_task 用于计算eid和ratio
            for j in range(h_ode.shape[0] - 1):
                (
                    eid_result[src_len[i], :min_len_idx],
                    rate_result[src_len[i], :min_len_idx],
                ) = self.muti_task(h_ode[j])
            if i == src_len.shape[0] - 1:
                (
                    eid_result[src_len[i], :min_len_idx],
                    rate_result[src_len[i], :min_len_idx],
                ) = self.muti_task(h)

        return eid_result, rate_result


class Seq2SeqMulti(nn.Module):
    """
    Seq2Seq
    """

    def __init__(self, params):
        super(Seq2SeqMulti, self).__init__()

        self.params = params
        self.encoder = Encoder(params["Encoder"])
        self.decoder = Decoder(params["Decoder"])
        self.muti_task = MutiTaskBlock(params=params["Decoder"])

    def forward(self, src, src_len, trg_eid, trg_rate, teacher_forcing_ratio=0):
        """
        forward
        """
        # src = [src len, batch size, 3]
        # src_len = tuple(int)[batch size]
        # trg_eid = [trg len, batch size]
        # trg_rate = [trg len, batch size]

        _, hidden = self.encoder(src, src_len)
        # hidden = [1, batch size, hidden_dim]
        hidden = hidden.squeeze(0)

        max_trg_len = trg_eid.shape[0]
        batch_size = trg_eid.shape[1]

        eid_result = torch.zeros(
            max_trg_len, batch_size, self.params["Decoder"]["eid_size"]
        ).to(src.device)
        rate_result = torch.zeros(max_trg_len, batch_size).to(src.device)

        eid = trg_eid[0]
        rate = trg_rate[0]
        # eid = [batch_size]
        # rate = [batch_size]
        for i in range(1, max_trg_len):
            hidden = self.decoder(eid, rate, hidden)
            pre_eid, pre_rate = self.muti_task(hidden)
            eid_result[i] = pre_eid
            rate_result[i] = pre_rate

            # teacher_force = random.random() < teacher_forcing_ratio

            # eid = trg_eid[i] if teacher_force else pre_eid.argmax(dim=1).long()
            # rate = trg_rate[i] if teacher_force else pre_rate

            eid = pre_eid.argmax(dim=1).long()
            rate = pre_rate

        return eid_result, rate_result


def cal_loss(
    eid_result,
    rate_result,
    trg_eid,
    trg_rate,
    criterion_eid,
    criterion_rate,
    rate_weight,
):
    mask = trg_eid != 0
    loss = criterion_rate(rate_result[mask], trg_rate[mask]) * rate_weight
    eid_result = eid_result.reshape(-1, eid_result.shape[-1])
    trg_eid = trg_eid.reshape(-1)
    mask = trg_eid != 0
    loss += criterion_eid(eid_result[mask], trg_eid[mask])  # 进入到loss计算中的数不能是负数
    return loss


def memoize(fn):
    """Return a memoized version of the input function.

    The returned function caches the results of previous calls.
    Useful if a function call is expensive, and the function
    is called repeatedly with the same arguments.
    """
    cache = dict()

    def wrapped(*v):
        key = tuple(v)  # tuples are hashable, and can be used as dict keys
        if key not in cache:
            cache[key] = fn(*v)
        return cache[key]

    return wrapped


def lcs(xs, ys):
    """Return the longest subsequence common to xs and ys.

    Example
    >>> lcs("HUMAN", "CHIMPANZEE")
    ['H', 'M', 'A', 'N']
    """

    @memoize
    def lcs_(i, j):
        if i and j:
            xe, ye = xs[i - 1], ys[j - 1]
            if xe == ye:
                return lcs_(i - 1, j - 1) + [xe]
            else:
                return max(lcs_(i, j - 1), lcs_(i - 1, j), key=len)
        else:
            return []

    return lcs_(len(xs), len(ys))


def shrink_seq(seq):
    """remove repeated ids"""
    s0 = seq[0]
    new_seq = [s0]
    for s in seq[1:]:
        if s == s0:
            continue
        else:
            new_seq.append(s)
        s0 = s

    return new_seq


def cal_id_acc(predict, target, trg_len):
    """
    Calculate RID accuracy between predicted and targeted RID sequence.
    1. no repeated rid for two consecutive road segments
    2. longest common subsequence
    http://wordaligned.org/articles/longest-common-subsequence
    Args:
    -----
        predict = [seq len, batch size, id one hot output dim]
        target = [seq len, batch size, 1]
        predict and target have been removed sos
    Returns:
    -------
        mean matched RID accuracy.
    """
    predict = predict.permute(1, 0, 2)  # [batch size, seq len, id dim]
    target = target.permute(1, 0)  # [batch size, seq len, 1]
    bs = predict.size(0)

    correct_id_num = 0
    ttl_trg_id_num = 0
    ttl_pre_id_num = 0
    ttl = 0
    cnt = 0
    for bs_i in range(bs):
        pre_ids = []
        trg_ids = []
        # -1 because predict and target are removed sos.
        for len_i in range(trg_len[bs_i] - 1):
            pre_id = predict[bs_i][len_i].argmax()
            trg_id = target[bs_i][len_i]
            if trg_id == 0:
                continue
            pre_ids.append(pre_id)
            trg_ids.append(trg_id)
            if pre_id == trg_id:
                cnt += 1
            ttl += 1

        # compute average rid accuracy
        shr_trg_ids = shrink_seq(trg_ids)
        shr_pre_ids = shrink_seq(pre_ids)
        correct_id_num += len(lcs(shr_trg_ids, shr_pre_ids))
        ttl_trg_id_num += len(shr_trg_ids)
        ttl_pre_id_num += len(shr_pre_ids)

    rid_acc = cnt / ttl
    rid_recall = correct_id_num / ttl_trg_id_num
    rid_precision = correct_id_num / ttl_pre_id_num
    return rid_acc, rid_recall, rid_precision


if __name__ == "__main__":
    """
    加载数据
    实例化模型
    测试模型
    """
    import pickle
    import json
    import sys

    from dataset import batch2device
    from model import cal_loss

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch = pickle.load(open("./data/batch_ode.pkl", "rb"))
    params = json.load(open("./params_ode.json", "r"))

    (
        src,
        src_gps,
        src_len,
        trg_gps,
        trg_eid,
        trg_rate,
        trg_len,
    ) = batch2device(batch, device)

    model = GRUODEBayes(params=params).to(device)

    eid_result, rate_result = model(batch[0].to(device), batch[2], batch[6])

    print(torch.max(trg_eid))
    # print(trg_eid[0])

    # sys.exit(0)
    """
    src_len_max = 201, min = 0
    src.shape = [202, 32, 3]
    trg_len_max = 202
    h_0.shape = [32, 512]
    h_all.shape = [202, 32, 512]
    eid_result.shape = [202, 32, 30000]
    """
    eid_true = batch[4]
    rate_true = batch[5]

    criterion_eid = nn.NLLLoss()
    criterion_rate = nn.MSELoss()
    loss = cal_loss(
        eid_result,
        rate_result,
        trg_eid,
        trg_rate,
        criterion_eid,
        criterion_rate,
        0.1,
    )
    loss.backward()
