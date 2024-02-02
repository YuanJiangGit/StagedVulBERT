import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_sequence
from transformers import RobertaForSequenceClassification


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 2)

    def forward(self, features):
        x = features  
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class Model(RobertaForSequenceClassification):
    def __init__(self, TEtransformer, SETransformer, config, tokenizer, args):
        super(Model, self).__init__(config=config)
        self.TEtransformer = TEtransformer
        self.SETransformer = SETransformer
        self.tokenizer = tokenizer
        self.classifier = RobertaClassificationHead(config)
        # self.Q = nn.Parameter(torch.randn(config.hidden_size, config.hidden_size))
        # self.K = nn.Parameter(torch.randn(config.hidden_size, config.hidden_size))
        self.args = args
        self.config = config

        # nn.init.xavier_uniform_(self.Q, gain=nn.init.calculate_gain("relu"))
        # nn.init.xavier_uniform_(self.K, gain=nn.init.calculate_gain("relu"))

    def get_row_rep(self, tokens_features):
        ori_batch = tokens_features[0].shape[0]
        is_split = tokens_features[-1].squeeze()
        # only the split_codes need to use the second TETransformer(to reduce computation)
        input_ids, attn_mask, row2row_mask, first_stoken_mask = [x[is_split] for x in tokens_features[:-1]]
        # do not use the second TETransformer if no split_code (all codes in the batch length <  512)
        if input_ids.shape[0] == 0:
            return None, None
        # record the indices of split_codes in the batch
        row_map = torch.zeros(ori_batch).long()
        row_map[is_split == 0] = 0
        row_map[is_split == 1] = torch.tensor(list(range(input_ids.shape[0]))) + 1
        batch, vob = input_ids.shape
        # （1） 计算每个Token的特征表示
        te_outputs = self.TEtransformer.roberta(input_ids, attention_mask=attn_mask)[0]
        # （2） 计算每个token的注意力分数
        # 提取CLS token的表示
        # cls_rep = te_outputs[:, 0, :]
        # 计算Q、K矩阵和QK的点积(注意力得分)
        # q = torch.einsum("ac,cc->ac", cls_rep, self.Q)
        # k = torch.einsum("abc,cc->abc", te_outputs, self.K)
        # attn_score = torch.einsum("ac,abc->ab", q, k)

        # 每个语句第一个token和所在语句其他token之间的连接关系
        ftoken_to_other = row2row_mask.clone()
        # （纵向）将每个语句第一个token位置之外的位置所在的整行都置为0
        ftoken_to_other[~first_stoken_mask] = 0
        # None 表示在该位置插入一个新的维度, 广播操作, 确保两个张量的维度能够对齐
        # ftoken_to_other = ftoken_to_other * attn_score[:, None, :]
        ftoken_to_other = ftoken_to_other / (ftoken_to_other.sum(-1) + 1e-10)[:, :, None]

        # （3）构造语句的特征表示
        ftoken_to_other_t = ftoken_to_other.clone()
        # 将为0的位置设置为一个极小的负数
        ftoken_to_other_t[ftoken_to_other_t == 0] = float('-1e9')
        ftoken_to_other_t = F.softmax(ftoken_to_other_t, dim=2).clone()
        ftoken_to_other_t[~first_stoken_mask] = 0
        # cls_mask = cls_mask / (cls_mask.sum(-1) + 1e-10)[:, :, None]
        avg_outputs = torch.einsum("abc,acd->abd", ftoken_to_other_t, te_outputs)
        # 提取每个样本中非零行(每个statement第一个token)的表示，该表示是同一个statement中的所有token加权求和
        rows_reps = [avg_outputs[i, first_stoken_mask[i]][1:] for i in range(batch)]
        cls_reps = [avg_outputs[i, first_stoken_mask[i]][0].unsqueeze(dim=0) for i in range(batch)]

        # alignment: non_split_code's row_rep is tensor[0, 768], split_code's row_rep is tensor[row_num, 768]
        rows_reps = [torch.zeros(0, 768).to(self.args.device)] + rows_reps
        rows_reps = [rows_reps[row_map[i]] for i in range(ori_batch)]

        cls_reps = [torch.zeros(0, 768).to(self.args.device)] + cls_reps
        cls_reps = [cls_reps[row_map[i]] for i in range(ori_batch)]
        return rows_reps, cls_reps

    def forward(self, tokens_features, labels=None):
        rows_reps, cls_reps = self.get_row_rep(tokens_features[0])
        for i in range(1, len(tokens_features)):
            rows_rep, cls_rep = self.get_row_rep(tokens_features[i])
            if rows_rep is None:
                break
            rows_reps = [torch.cat((tensor0, tensor1), dim=0) for tensor0, tensor1 in zip(rows_reps, rows_rep)]
            cls_reps = [torch.cat((tensor0, tensor1), dim=0) for tensor0, tensor1 in zip(cls_reps, cls_rep)]

        cls_reps = [torch.mean(cls_rep, dim=0).unsqueeze(dim=0) for cls_rep in cls_reps]
        rows_reps = [torch.cat((tensor0, tensor1), dim=0) for tensor0, tensor1 in zip(cls_reps, rows_reps)]
        # 每个样本的行数不同，得到的每个样本的行特征向量不同，将所有样本的行特征向量个数统一
        padded_rows_rep = pad_sequence(rows_reps, batch_first=True, padding_value=0)

        # （4）构造语句编码SETransfomer的attention mask
        max_row_num = padded_rows_rep.shape[1]
        attn_mask = [[1] * rows.shape[0] + [0] * (max_row_num - rows.shape[0]) for rows in rows_reps]

        # ignoring the <cls> at the start
        row_mask = [[0] + [1] * (rows.shape[0] - 1) + [0] * (max_row_num - rows.shape[0]) for rows in rows_reps]
        attn_mask = torch.tensor(attn_mask).to(self.args.device)
        row_mask = torch.tensor(row_mask).to(self.args.device)

        x = self.SETransformer.roberta(inputs_embeds=padded_rows_rep, attention_mask=attn_mask)[0]

        logits = self.classifier(x)
        logits_cls = logits[:, 0, :]
        prob = torch.softmax(logits, dim=-1)
        loss = torch.zeros(1).to(self.args.device)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits_cls, labels)
        return loss, prob, row_mask
