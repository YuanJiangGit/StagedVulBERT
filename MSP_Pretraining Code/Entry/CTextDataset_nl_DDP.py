# -*- coding: utf-8 -*-
# @Author  : Jiang Yuan
# @Time    : 2023/11/3 10:20
# @Function:

import logging
import multiprocessing
import os
import pickle
import random
from collections import Counter
from datetime import datetime
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import copy
import re

logger = logging.getLogger(__name__)


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 input_tokens,
                 input_ids,
                 token_row_indices,
                 doc,
                 code):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.row_idx = token_row_indices
        self.doc = doc
        self.code = code


def convert_examples_to_features(data):
    df, tokenizer, args = data
    doc = df['docstring']
    func = df['function']
    # 将函数代码按行拆分为列表
    doc_rows = re.split(r'[.;\n]', str(doc))
    code_rows = str(func).split('\n')
    # 使用tokenizer对每一行的代码进行分词，并过滤掉空列表
    doc_tokens = [tokenizer.tokenize(x) for x in doc_rows if tokenizer.tokenize(x) != []]
    code_tokens = [tokenizer.tokenize(x) for x in code_rows if tokenizer.tokenize(x) != []]
    source_tokens = [[tokenizer.cls_token]] + doc_tokens + [[tokenizer.sep_token]] + code_tokens
    # 创建每个分词后的“Token所在的行”的索引列表
    token_row_indices = [[idx] * len(row_token) for idx, row_token in enumerate(source_tokens)]

    # 将分词后的列表展平，并展平索引列表
    source_tokens = [y for x in source_tokens for y in x]  # 平铺token_list
    token_row_indices = [y for x in token_row_indices for y in x]

    # 对token列表和行索引列表进行截断(only truncate the code for the doc-parts are commonly short)，保留前block_size-1个元素
    source_tokens = source_tokens[:args.block_size - 1]
    token_row_indices = token_row_indices[:args.block_size - 1]

    # add <sep> as a single row at the end
    source_tokens += [tokenizer.sep_token]
    token_row_indices += [token_row_indices[-1] + 1]

    # 将tokens转换为对应的token_id
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)

    # 对token_id进行填充，使其达到block_size长度
    padding_length = args.block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id] * padding_length

    return InputFeatures(source_tokens, source_ids, token_row_indices, doc, func)


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_type="train"):
        self.args = args
        self.tokenizer = tokenizer
        if file_type == "all":
            file_path = "../resources/data/code_search_net/all.pkl"
        elif file_type == "train":
            file_path = args.train_data_file
        elif file_type == "eval":
            file_path = args.eval_data_file
        elif file_type == "test":
            file_path = args.test_data_file
        elif file_type == "vul":
            file_path = "../resources/data/big-vul_dataset/vul_input.pkl"
        else:
            exit(0)

        data_path = "/".join(file_path.split('/')[:-1] + [file_type + "_input.pkl"])
        if os.path.exists(data_path):
            logger.info("data exists")
            with open(data_path, 'rb') as f:
                self.examples = pickle.load(f)
        else:
            logger.info("no data")
            self.examples = []
            df = pd.read_pickle(file_path)

            # 创建与函数数量相同的tokenizer列表，每个函数对应一个tokenizer
            tokenizers = [tokenizer] * len(df)
            arg = [args] * len(df)
            # 将funcs、labels、tokenizers和arg列表打包为元组列表source
            source = list(zip(df, tokenizers, arg))

            # 创建一个拥有CPU核心数量个进程的进程池
            pool = multiprocessing.Pool(multiprocessing.cpu_count())
            # 使用进程池并行地将source中的每个元组传递给convert_examples_to_features函数，得到一个InputFeatures，并赋值给self.examples
            self.examples = pool.map(convert_examples_to_features, tqdm(source, total=len(source)))
            print("parse done!")

            with open(data_path, 'wb') as f:
                pickle.dump(self.examples, f)
            print("saved at %s", data_path)

        # 删除特殊字符
        self.non_special_ids = [id for id in self.tokenizer.get_vocab().values() if
                                id not in self.tokenizer.all_special_ids]

    def __len__(self):
        return len(self.examples)

    def gen_mslm_mask(self, source_ids, row_indices):
        row_num = len(row_indices)  # 验证row_indices和row_num的关系
        masked_source_ids = copy.deepcopy(source_ids)
        # 确定掩码的语句的行数
        num_rows_to_mask = int(0.20 * row_num)
        mask_row_pos = random.sample(range(1, row_num - 1), num_rows_to_mask)
        for pos in mask_row_pos:
            prob = random.randint(1, 100)
            # 概率小于20，则将该句子替换成其他句子
            if prob <= 20:
                # 得到要替换的语句中token的字数
                replaced_state_len = row_indices[pos + 1] - row_indices[pos]
                # 生成替换字符序列
                new_state_ids = random.sample([id for id in self.non_special_ids], replaced_state_len)
                # 将原始的语句替换成新生成的随机字符序列
                masked_source_ids[row_indices[pos]: row_indices[pos + 1]] = new_state_ids
            else:
                masked_source_ids[row_indices[pos]: row_indices[pos + 1]] = [self.tokenizer.mask_token_id] * (
                            row_indices[pos + 1] - row_indices[pos])

        # prevent the mask_row_pos large 20
        mask_row_pos = mask_row_pos[:round(self.args.max_row_size * 0.2)]
        # 对齐
        padding = round(self.args.max_row_size * 0.2) - len(mask_row_pos)
        mask_row_pos += [-1] * padding  # (模型中识别)

        return masked_source_ids, mask_row_pos

    def gen_mlm_mask(self, source_ids, token_length):
        masked_source_ids = copy.deepcopy(source_ids)
        max_length = len(source_ids)
        num_tokens_to_mask = int(0.15 * token_length)
        mask_token_pos = random.sample(range(1, token_length), num_tokens_to_mask)

        # 实现mask过程，在抽取的15%的token中，选择80%mask掉，10%保持不变，10%随机替换
        for pos in mask_token_pos:
            prob = random.randint(1, 100)
            if prob <= 10:
                continue
            elif prob <= 20:
                ori_id = masked_source_ids[pos]
                if ori_id in self.tokenizer.all_special_ids:
                    masked_source_ids[pos] = random.choice(self.non_special_ids)  # 特殊字符随机替换为非特殊字符
                else:
                    masked_source_ids[pos] = random.choice([id for id in self.non_special_ids if id != ori_id])
            else:
                masked_source_ids[pos] = self.tokenizer.mask_token_id

        # 对齐
        padding = round(max_length * 0.15) - len(mask_token_pos)
        mask_token_pos += [-1] * padding  # (模型中识别)

        return masked_source_ids, mask_token_pos

    def __getitem__(self, i):
        max_length = self.args.block_size

        token_row_indices = self.examples[i].row_idx
        # 使用Counter统计token_row_indices 中每个元素的出现次数，生成字典（）
        row_token_counts = Counter(token_row_indices)
        # 获取每个元素在token_row_indices中第一次出现的索引，构建行索引列表row_indices，最后添加token_row_indices的长度
        row_indices = [np.where(np.array(token_row_indices) == x)[0][0] for x in row_token_counts.keys()] + [
            len(token_row_indices)]
        # 获取行的数量
        num_rows = len(row_token_counts)

        # self-attention mask
        attn_mask = torch.zeros(max_length, max_length)
        attn_mask[:len(token_row_indices), :len(token_row_indices)] = 1
        attn_mask = attn_mask.bool()

        # 创建row2row_mask，表示同一行内的token有连接，
        row2row_mask = torch.zeros(max_length, max_length)
        for idx in range(num_rows):
            row2row_mask[row_indices[idx]:row_indices[idx + 1], row_indices[idx]:row_indices[idx + 1]] = 1
        # 同时将第0行和最后一行的前len(token_row_indices)列设置为1 (第0行：cls，最后一行：sep)
        row2row_mask[[0, len(token_row_indices) - 1], :len(token_row_indices)] = 1

        # 创建表示每行第一个token位置的mask，其中每行的第一个token所在的列设置为1，最后转换为bool类型
        first_stoken_mask = torch.zeros(max_length)
        first_stoken_mask[row_indices[:-1]] = 1
        first_stoken_mask = first_stoken_mask.bool()
        if self.args.do_MSLM_train:
            masked_source_ids_MLM, mask_row_pos = self.gen_mslm_mask(self.examples[i].input_ids, row_indices[:-1])
            # inputs_ids, first_stoken_mask, row2row_mask, attn_mask, masked_source_ids, mask_row_pos
            return (torch.tensor(self.examples[i].input_ids), first_stoken_mask, row2row_mask,
                    attn_mask, torch.tensor(masked_source_ids_MLM), torch.tensor(mask_row_pos))

        elif self.args.do_MLM_train:
            masked_input_ids, mask_token_pos = self.gen_mlm_mask(self.examples[i].input_ids, row_indices)
            return (torch.tensor(self.examples[i].input_ids), torch.tensor(self.examples[i].label),
                    attn_mask, row2row_mask, first_stoken_mask,
                    torch.tensor(masked_input_ids), torch.tensor(mask_token_pos))
        else:
            return (torch.tensor(self.examples[i].input_ids), torch.tensor(self.examples[i].label),
                    attn_mask, row2row_mask, first_stoken_mask)