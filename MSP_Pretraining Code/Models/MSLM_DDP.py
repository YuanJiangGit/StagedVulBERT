import os


import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
import random
import numpy as np
from torch.nn.utils.rnn import pad_sequence


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 104)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = x.reshape(-1, x.size(-1))
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class Decoder(nn.Module):
    def __init__(self, config, embedding, dropout):
        super().__init__()
        self.hid_dim = config.hidden_size
        self.output_dim = config.vocab_size
        # self.embedding = nn.Embedding(self.output_dim, emb_dim)
        self.embedding = embedding
        emb_dim = embedding.embedding_dim
        self.rnn = nn.GRU(emb_dim + self.hid_dim, self.hid_dim)
        self.fc_out = nn.Linear(emb_dim + self.hid_dim * 2, self.output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, context):
        # input = [batch size]
        # hidden = [n layers * n directions, batch size, hid dim]
        # context = [n layers * n directions, batch size, hid dim]

        # n layers and n directions in the decoder will both always be 1, therefore:
        # hidden = [1, batch size, hid dim]
        # context = [1, batch size, hid dim]
        input = input.unsqueeze(0)
        # input = [1, batch size]
        embedded = self.dropout(self.embedding(input))
        # embedded = [1, batch size, emb dim]
        emb_con = torch.cat((embedded, context), dim=2)
        # emb_con = [1, batch size, emb dim + hid dim]
        output, hidden = self.rnn(emb_con, hidden)
        # output = [seq len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # seq len, n layers and n directions will always be 1 in the decoder, therefore:
        # output = [1, batch size, hid dim]
        # hidden = [1, batch size, hid dim]
        output = torch.cat((embedded.squeeze(0), hidden.squeeze(0), context.squeeze(0)), dim=1)
        # output = [batch size, emb dim + hid dim * 2]
        prediction = self.fc_out(output)
        # prediction = [batch size, output dim]
        return prediction, hidden


class MSLMTask(nn.Module):
    def __init__(self, decoder, tokenizer, args):
        super().__init__()
        self.tokenizer = tokenizer
        self.teacher_forcing_ratio = 0.5
        self.decoder = decoder
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        self.args = args

    def forward(self, se_outputs, inputs_ids, mask_row_pos, first_stoken_mask):
        # 获取True元素(语句第一个token)对应的位置索引，first_stoken_mask = torch.tensor([True, False, True, False, True])
        row_indices = torch.nonzero(first_stoken_mask) # row_indices = [[0,0],[0,10],[1,30]]represents the first token index of the second statement in the first sample is 10
        # merge the first token indexes of statements in a sample into a list, e.g., [[0, 10],[30]]
        row_indices = [row_indices[row_indices[:, 0] == i][:, 1] for i in range(inputs_ids.shape[0])]

        # 获取了非 padding 部分的索引和对应的值, -1是pad
        masked_pos = (torch.where(mask_row_pos != -1)[0], mask_row_pos[torch.where(mask_row_pos != -1)])
        # 取出inputs_ids中被掩码的source_ids
        true_token_ids = [inputs_ids[i, row_indices[i][j]:row_indices[i][j + 1]] for i, j in zip(*masked_pos)]

        # 计算masked语句的最大长度
        max_trg_len = self.args.max_trg_len  # 80
        trg_len = min(max(len(x) for x in true_token_ids), max_trg_len)
        # print("max_row_len: " + str(trg_len))
        # padding true_token_ids，并转化为 PyTorch Tensor 矩阵
        padded_true_token_ids = [[self.tokenizer.cls_token_id] + (x.tolist())[:trg_len] +
                                 [self.tokenizer.pad_token_id] * (trg_len - len(x)) + [self.tokenizer.sep_token_id]
                                 for x in true_token_ids]
        trg = torch.tensor(padded_true_token_ids).to(self.args.device)
        # +2是（CLS和SEP）
        trg_len += 2

        # 提取了与非padding位置相对应的特征，masked_se_features [masked row length, hid dim]
        masked_se_features = se_outputs[masked_pos].to(self.args.device)
        batch_size = masked_se_features.size(0)
        # 对masked_se_feature进行预测 [masked row length, hid dim]
        masked_se_features = masked_se_features.unsqueeze(0)

        # src = [src len, batch size]
        # trg = [trg len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        trg_vocab_size = self.decoder.output_dim
        # tensor to store decoder outputs
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.args.device)

        # context also used as the initial hidden state of the decoder
        hidden = masked_se_features
        context = hidden

        # first input to the decoder is the <sos> tokens
        input = trg[:, 0]
        for t in range(1, trg_len):
            # insert input token embedding, previous hidden state and the context state
            # receive output tensor (predictions) and new hidden state
            output, hidden = self.decoder(input, hidden, context)
            # place predictions in a tensor holding predictions for each token
            outputs[:, t, :] = output
            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < self.teacher_forcing_ratio
            # get the highest predicted token from our predictions
            top1 = output.argmax(1)
            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = trg[:, t] if teacher_force else top1

        # trg = [trg len, batch size]
        # output = [trg len, batch size, output dim]
        output_dim = outputs.shape[-1]
        output = outputs[:, 1:, :].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)

        # trg = [(trg len - 1) * batch size]
        # output = [(trg len - 1) * batch size, output dim]
        loss = self.criterion(output, trg)
        return loss


class Model(nn.Module):
    def __init__(self, TEtransformer, SETransformer, decoder, tokenizer, args, config):
        super(Model, self).__init__()
        self.TEtransformer = TEtransformer
        self.SETransformer = SETransformer
        self.tokenizer = tokenizer
        self.mslmTask = MSLMTask(decoder, self.tokenizer, args)
        self.Q = nn.Parameter(torch.randn(config.hidden_size, config.hidden_size))
        self.K = nn.Parameter(torch.randn(config.hidden_size, config.hidden_size))
        self.args = args

        nn.init.xavier_uniform_(self.Q, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self.K, gain=nn.init.calculate_gain("relu"))

    def forward(self, inputs_ids, first_stoken_mask, row2row_mask, attn_mask, masked_source_ids, mask_row_pos,
                output_attentions=False):
        # (1) 计算每个Token的特征表示
        batch, vob = masked_source_ids.shape
        te_outputs = self.TEtransformer.roberta(masked_source_ids, attention_mask=attn_mask,
                                                output_attentions=output_attentions)[0]
        # （2） 计算每个token的注意力分数
        # 提取CLS token的表示
        cls_rep = te_outputs[:, 0, :]
        # 计算Q、K矩阵和QK的点积(注意力得分)
        q = torch.einsum("ac,cc->ac", cls_rep, self.Q)
        k = torch.einsum("abc,cc->abc", te_outputs, self.K)
        attn_score = torch.einsum("ac,abc->ab", q, k)

        # 每个语句第一个token和所在语句其他token之间的连接关系

        # （纵向）将每个语句第一个token位置之外的位置所在的整行都置为0
        row2row_mask[~first_stoken_mask] = 0
        # None 表示在该位置插入一个新的维度, 广播操作, 确保两个张量的维度能够对齐
        row2row_mask = row2row_mask * attn_score[:, None, :]

        # （3）构造语句的特征表示
        # 将为0的位置设置为一个极小的负数
        row2row_mask[row2row_mask == 0] = float('-1e9')
        row2row_mask = F.softmax(row2row_mask, dim=2).clone()
        row2row_mask[~first_stoken_mask] = 0
        # cls_mask = cls_mask / (cls_mask.sum(-1) + 1e-10)[:, :, None]
        avg_outputs = torch.einsum("abc,acd->abd", row2row_mask, te_outputs)
        # 提取每个样本中非零行(每个statement第一个token)的表示，该表示是同一个statement中的所有token加权求和
        rows_rep = [avg_outputs[i, first_stoken_mask[i]] for i in range(batch)]

        max_row_num = self.args.max_row_size  # 80
        max_row_num = min(max(len(x) for x in rows_rep), max_row_num)
        rows_rep = [rows[:max_row_num] for rows in rows_rep]
        mask_row_pos[mask_row_pos >= max_row_num] = -1
        # 每个样本的行数不同，得到的每个样本的行特征向量不同，将所有样本的行特征向量个数统一
        padded_rows_rep = pad_sequence(rows_rep, batch_first=True, padding_value=0)

        # （4）构造语句编码SETransfomer的attention mask
        row_num = padded_rows_rep.shape[1]
        # print("max_row_num: " + str(row_num))
        attn_mask = [[1] * rows.shape[0] + [0] * (row_num - rows.shape[0]) for rows in rows_rep]
        attn_mask = torch.tensor(attn_mask).to(self.args.device)

        se_outputs = self.SETransformer.roberta(inputs_embeds=padded_rows_rep, attention_mask=attn_mask,
                                                output_attentions=output_attentions)[0]

        end_pos = [torch.where(inputs_ids[i] == 1)[0][0] if len(torch.where(inputs_ids[i] == 1)[0]) > 0
                   else torch.tensor(inputs_ids.shape[1]-1) for i in range(inputs_ids.shape[0])]
        first_stoken_mask_t = first_stoken_mask.clone()
        first_stoken_mask_t[(torch.arange(inputs_ids.shape[0]), torch.tensor(end_pos))] = True
        lossMSLM = self.mslmTask(se_outputs, inputs_ids, mask_row_pos, first_stoken_mask_t)
        return se_outputs, lossMSLM
