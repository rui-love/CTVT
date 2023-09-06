"""
Seq2Seq模型
"""
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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


class DecoderMulti(nn.Module):
    """
    Decoder
    """

    def __init__(self, params):
        super(DecoderMulti, self).__init__()

        self.eid_size = params["eid_size"]
        self.rate_size = params["rate_size"]
        self.emb_dim = params["embedding_size"]
        self.hidden_dim = params["hidden_size"]

        self.embedding = nn.Embedding(self.eid_size, self.emb_dim)

        self.rnn = nn.GRU(self.emb_dim + self.rate_size, self.hidden_dim)

        self.dropout = nn.Dropout(params["dropout_rate"])

        self.pre_eid = nn.Linear(self.hidden_dim, self.eid_size)
        self.pre_rate_1 = nn.Linear(self.hidden_dim + self.emb_dim, self.hidden_dim)
        self.pre_rate_2 = nn.Linear(self.hidden_dim, self.rate_size)

    def forward(self, eid, rate, hidden):
        """
        forward
        """
        # eid = [batch_size]
        # rate = [batch_size]
        # hidden = [1, batch_size, hidden_dim]

        eid_embbeded = self.embedding(eid.unsqueeze(0))  # [1, batch_size, emb_dim]
        rnn_input = torch.cat(
            (eid_embbeded, rate.reshape(1, -1, 1)), dim=2
        )  # [1, batch_size, emb_dim + rate_size]
        output, hidden = self.rnn(
            rnn_input, hidden
        )  # output = [1, batch_size, hidden_dim]

        pre_eid_1 = F.log_softmax(
            self.pre_eid(output.squeeze(0)), dim=1
        )  # [batch_size, eid_size]
        pre_eid_2 = pre_eid_1.argmax(dim=1).long()  # [batch_size]

        pre_rate_1 = self.pre_rate_1(
            torch.cat(
                (self.dropout(self.embedding(pre_eid_2)), output.squeeze(0)), dim=1
            )
        )  # [batch_size, hidden_size]
        pre_rate_2 = self.pre_rate_2(F.relu(pre_rate_1)).squeeze(
            1
        )  # [batch_size, rate_size]
        pre_rate_3 = F.sigmoid(pre_rate_2)  # [batch_size, rate_size]

        return pre_eid_1, pre_rate_3, hidden


class Seq2SeqMulti(nn.Module):
    """
    Seq2Seq
    """

    def __init__(self, params):
        super(Seq2SeqMulti, self).__init__()

        self.params = params
        self.encoder = Encoder(params["Encoder"])
        self.decoder = DecoderMulti(params["Decoder"])

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

        max_trg_len = trg_eid.shape[0]
        batch_size = trg_eid.shape[1]

        eid_result = torch.zeros(
            max_trg_len, batch_size, self.params["Decoder"]["eid_size"]
        ).to(device)
        rate_result = torch.zeros(max_trg_len, batch_size).to(device)

        eid = trg_eid[0]
        rate = trg_rate[0]
        # eid = [batch_size]
        # rate = [batch_size]

        for i in range(1, max_trg_len):
            pre_eid, pre_rate, hidden = self.decoder(eid, rate, hidden)

            eid_result[i] = pre_eid
            rate_result[i] = pre_rate

            teacher_force = random.random() < teacher_forcing_ratio

            # eid = trg_eid[i] if teacher_force else pre_eid.argmax(dim=1).long()
            # rate = trg_rate[i] if teacher_force else pre_rate

            eid = pre_eid.argmax(dim=1).long()
            rate = pre_rate

        return eid_result, rate_result


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
