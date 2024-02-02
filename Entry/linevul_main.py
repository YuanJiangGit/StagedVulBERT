# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import, division, print_function

import math
import os

from clang import cindex

import argparse
from datetime import datetime
import glob
import logging
import os
import pickle
import random
import re
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (WEIGHTS_NAME, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
from tqdm import tqdm
import multiprocessing
from Models.linevul_model import Model
import pandas as pd
# metrics
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import auc
# model reasoning
from captum.attr import LayerIntegratedGradients, DeepLift, DeepLiftShap, GradientShap, Saliency
# word-level tokenizer
from tokenizers import NormalizedString, PreTokenizedString
from tokenizers import Tokenizer
from tokenizers import normalizers, decoders
from tokenizers.normalizers import StripAccents, Replace
from tokenizers.processors import TemplateProcessing
from tokenizers.pre_tokenizers import PreTokenizer
from tokenizers import processors, pre_tokenizers
from tokenizers.models import BPE
from typing import List

logger = logging.getLogger(__name__)


class MyTokenizer:
    cidx = cindex.Index.create()

    def clang_split(self, i: int, normalized_string: NormalizedString) -> List[NormalizedString]:
        ## Tokkenize using clang
        tok = []
        tu = self.cidx.parse('tmp.c',
                             args=[''],
                             unsaved_files=[('tmp.c', str(normalized_string.original))],
                             options=0)
        for t in tu.get_tokens(extent=tu.cursor.extent):
            spelling = t.spelling.strip()

            if spelling == '':
                continue

            ## Keyword no need

            ## Punctuations no need

            ## Literal all to BPE

            # spelling = spelling.replace(' ', '')
            tok.append(NormalizedString(spelling))

        return (tok)

    def pre_tokenize(self, pretok: PreTokenizedString):
        pretok.split(self.clang_split)


vocab, merges = BPE.read_file(vocab="../resource/tokenizer/drapgh-vocab.json",
                              merges="../resource/tokenizer/drapgh-merges.txt")
vul_tokenizer = Tokenizer(BPE(vocab, merges, unk_token="<unk>"))

vul_tokenizer.normalizer = normalizers.Sequence([StripAccents(), Replace(" ", "Ä")])
vul_tokenizer.pre_tokenizer = PreTokenizer.custom(MyTokenizer())
vul_tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
vul_tokenizer.post_processor = TemplateProcessing(
    single="<s> $A </s>",
    special_tokens=[
        ("<s>", 0),
        ("<pad>", 1),
        ("</s>", 2),
        ("<unk>", 3),
        ("<mask>", 4)
    ]
)


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 input_tokens,
                 input_ids,
                 label):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.label = label


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path='', file_type="train"):
        if file_type == "train":
            file_path = args.train_data_file
        elif file_type == "eval":
            file_path = args.eval_data_file
        elif file_type == "test":
            # file_path = args.test_data_file
            file_path = file_path
        else:
            exit()

        data_path = "/".join(file_path.split('/')[:-1] + [file_type + "_v.pkl"])  # _c2.pkl

        if file_path.endswith(".csv"):
            data_path = file_path[:-4]
        data_path += "_vulBERTa.pkl"

        # data_path = os.path.join(os.path.dirname(file_path), file_type + "_codeBERT.pkl")
        # data_path = os.path.join(os.path.dirname(file_path), file_type + "_lp.pkl")

        # if file_path.endswith(".csv"):
        #     data_path = file_path[:-4]
        # data_path += "_codeBERT.pkl"
        if os.path.exists(data_path):
            with open(data_path, 'rb') as f:
                self.examples = pickle.load(f)
        else:
            self.examples = []
            df = pd.read_csv(file_path)
            funcs = df["func_before"].tolist()
            labels = df["target"].tolist()
            tokenizers = [tokenizer] * len(funcs)
            arg = [args] * len(funcs)
            source = list(zip(funcs, labels, tokenizers, arg))
            # for i in range(100):
            #     convert_examples_to_features(source[i])
            # exit()
            pool = multiprocessing.Pool(32)
            self.examples = pool.map(convert_examples_to_features, tqdm(source, total=len(source)))
            print("parse done!")

            with open(data_path, 'wb') as f:
                pickle.dump(self.examples, f)
            print("saved at %s", data_path)

        # for i in tqdm(range(len(funcs))):
        #     self.examples.append(convert_examples_to_features(funcs[i], labels[i], tokenizer, args))
        if file_type == "train":
            for example in self.examples[:3]:
                logger.info("*** Example ***")
                logger.info("label: {}".format(example.label))
                logger.info("input_tokens: {}".format([x.replace('\u0120', '_') for x in example.input_tokens]))
                logger.info("input_ids: {}".format(' '.join(map(str, example.input_ids))))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i].input_ids), torch.tensor(self.examples[i].label)


def convert_examples_to_features(data):
    func, label, tokenizer, args = data
    if tokenizer is None:
        vul_tokenizer.enable_truncation(max_length=1024)
        vul_tokenizer.enable_padding(direction='right', pad_id=1, pad_type_id=0, pad_token='<pad>', length=1024,
                                     pad_to_multiple_of=None)

        pat = re.compile(r'(/\*([^*]|(\*+[^*/]))*\*+/)|(//.*)')
        func = re.sub(pat, '', func)
        func = re.sub('\n', '', func)
        func = re.sub('\t', '', func)

        code_encode = vul_tokenizer.encode_batch([func])[0]
        source_ids = code_encode.ids
        source_tokens = code_encode.tokens
        return InputFeatures(source_tokens, source_ids, label)
    if args.use_word_level_tokenizer:
        encoded = tokenizer.encode(func)
        encoded = encoded.ids
        if len(encoded) > 510:
            encoded = encoded[:510]
        encoded.insert(0, 0)
        encoded.append(2)
        if len(encoded) < 512:
            padding = 512 - len(encoded)
            for _ in range(padding):
                encoded.append(1)
        source_ids = encoded
        source_tokens = []
        return InputFeatures(source_tokens, source_ids, label)
    # source
    code_tokens = tokenizer.tokenize(str(func))[:args.block_size - 2]
    source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id] * padding_length
    return InputFeatures(source_tokens, source_ids, label)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer, eval_dataset):
    """ Train the model """
    # build dataloader
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=0)

    args.max_steps = args.epochs * len(train_dataloader)
    # evaluate the model per epoch
    args.save_steps = len(train_dataloader) // 2
    args.warmup_steps = args.max_steps // 5
    model.to(args.device)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.max_steps)

    # multi-gpu training
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size // max(args.n_gpu, 1))
    logger.info("  Total train batch size = %d", args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)

    global_step = 0
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0
    best_f1 = 0

    model.zero_grad()

    for idx in range(args.epochs):
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        tr_num = 0
        train_loss = 0
        train_acc = 0
        for step, batch in enumerate(bar):
            (inputs_ids, labels) = [x.to(args.device) for x in batch]
            model.train()
            loss, logits = model(input_ids=inputs_ids, labels=labels)
            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            tr_num += 1
            train_loss += loss.item()
            if avg_loss == 0:
                avg_loss = tr_loss

            _, predicted = torch.max(logits.data, 1)
            predicted = predicted.cpu().numpy()
            labels = labels.cpu().numpy()
            train_acc += len(labels[predicted == labels]) / len(labels)

            avg_acc = round(train_acc / tr_num, 5)

            avg_loss = round(train_loss / tr_num, 5)
            bar.set_description("epoch {} loss {} acc {}".format(idx, avg_loss, avg_acc))

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                output_flag = True
                avg_loss = round(np.exp((tr_loss - logging_loss) / (global_step - tr_nb)), 4)

                if global_step % args.save_steps == 0:
                    logger.info("***** Running epoch {} evaluation *****".format(idx))
                    results = evaluate(args, model, tokenizer, eval_dataset, eval_when_training=True)

                    # Save model checkpoint
                    if results['eval_f1'] > best_f1:
                        best_f1 = results['eval_f1']
                        logger.info("  " + "*" * 20)
                        logger.info("  Best f1:%s", round(best_f1, 4))
                        logger.info("  " + "*" * 20)

                        checkpoint_prefix = 'checkpoint-best-f1'
                        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = model.module if hasattr(model, 'module') else model
                        output_dir = os.path.join(output_dir, '{}'.format(args.model_name))
                        torch.save(model_to_save.state_dict(), output_dir)
                        logger.info("Saving model checkpoint to %s", output_dir)


def evaluate(args, model, tokenizer, eval_dataset, eval_when_training=False):
    # build dataloader
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=0)

    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    y_trues = []
    for batch in eval_dataloader:
        (inputs_ids, labels) = [x.to(args.device) for x in batch]
        with torch.no_grad():
            lm_loss, logit = model(input_ids=inputs_ids, labels=labels)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            y_trues.append(labels.cpu().numpy())
        nb_eval_steps += 1

    # calculate scores
    logits = np.concatenate(logits, 0)
    y_trues = np.concatenate(y_trues, 0)
    best_threshold = 0.5
    best_f1 = 0
    y_preds = logits[:, 1] > best_threshold
    recall = recall_score(y_trues, y_preds)
    precision = precision_score(y_trues, y_preds)
    f1 = f1_score(y_trues, y_preds)
    result = {
        "eval_recall": float(recall),
        "eval_precision": float(precision),
        "eval_f1": float(f1),
        "eval_threshold": best_threshold,
    }

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))

    return result


def test(args, model, tokenizer, test_dataset, best_threshold=0.5):
    # build dataloader
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.eval_batch_size, num_workers=0)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(test_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    y_trues = []
    bar = tqdm(test_dataloader, total=len(test_dataloader))
    for batch in bar:
        (inputs_ids, labels) = [x.to(args.device) for x in batch]
        with torch.no_grad():
            lm_loss, logit = model(input_ids=inputs_ids, labels=labels)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            y_trues.append(labels.cpu().numpy())
        nb_eval_steps += 1
    # calculate scores
    logits = np.concatenate(logits, 0)
    y_trues = np.concatenate(y_trues, 0)
    y_preds = logits[:, 1] > best_threshold
    acc = accuracy_score(y_trues, y_preds)
    recall = recall_score(y_trues, y_preds)
    precision = precision_score(y_trues, y_preds)
    f1 = f1_score(y_trues, y_preds)
    result = {
        "test_accuracy": float(acc),
        "test_recall": float(recall),
        "test_precision": float(precision),
        "test_f1": float(f1),
        "test_threshold": best_threshold,
    }

    logger.info("***** Test results *****")
    logger.info("  TPR = {} / {}".format(str((y_preds & y_trues).sum()), str(y_trues.sum())))
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))
    return

    logits = [l[1] for l in logits]
    result_df = generate_result_df(logits, y_trues, y_preds, args)

    # define reasoning method
    reasoning_method = "attention"

    dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=1, num_workers=0)
    progress_bar = tqdm(dataloader, total=len(dataloader))
    all_pos_score_label = []
    all_neg_score_label = []
    index = 0
    total_pred_as_vul = 0
    for mini_batch in progress_bar:
        # if predicted as vulnerable
        if result_df["logits"][index] > 0.5:
            total_pred_as_vul += 1
            all_lines_score_with_label = \
                line_level_localization(flaw_lines=result_df["flaw_line"][index],
                                        tokenizer=tokenizer,
                                        model=model,
                                        mini_batch=mini_batch,
                                        original_func=result_df["processed_func"][index],
                                        args=args,
                                        top_k_loc=None,
                                        top_k_constant=None,
                                        reasoning_method=reasoning_method,
                                        index=index)
            all_pos_score_label.append(all_lines_score_with_label)
        # else predicted as non vulnerable
        else:
            all_lines_score_with_label = \
                line_level_localization(flaw_lines=result_df["flaw_line"][index],
                                        tokenizer=tokenizer,
                                        model=model,
                                        mini_batch=mini_batch,
                                        original_func=result_df["processed_func"][index],
                                        args=args,
                                        top_k_loc=None,
                                        top_k_constant=None,
                                        reasoning_method=reasoning_method,
                                        index=index)
            all_neg_score_label.append(all_lines_score_with_label)
        index += 1

    my_res = all_pos_score_label + all_neg_score_label

    for i in my_res:
        for j in i:
            j[0] = j[0].item()

    k_pers = [0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2]
    all_res = []

    for k_per in k_pers:
        good = 0
        for d in my_res:
            d.sort(key=lambda x: x[0], reverse=True)

            k = math.ceil(k_per * len(d))
            flag = 0
            for i in d[:k]:
                flag += i[1]

            if flag != 0:
                good += 1

        all_res.append(100 * (good / len(my_res)))
    for k_per in k_pers:
        logger.info("  %s = %s", key, str(round(result[key], 4)))



def generate_result_df(logits, y_trues, y_preds, args):
    df = pd.read_csv(args.test_data_file)
    all_num_lines = []
    all_processed_func = df["processed_func"].tolist()
    for func in all_processed_func:
        all_num_lines.append(get_num_lines(func))
    flaw_line_indices = df["flaw_line_index"].tolist()
    all_num_flaw_lines = []
    total_flaw_lines = 0
    for indices in flaw_line_indices:
        if isinstance(indices, str):
            indices = indices.split(",")
            num_flaw_lines = len(indices)
            total_flaw_lines += num_flaw_lines
        else:
            num_flaw_lines = 0
        all_num_flaw_lines.append(num_flaw_lines)
    assert len(logits) == len(y_trues) == len(y_preds) == len(all_num_flaw_lines)
    return pd.DataFrame({"logits": logits, "y_trues": y_trues, "y_preds": y_preds,
                         "index": list(range(len(logits))), "num_flaw_lines": all_num_flaw_lines,
                         "num_lines": all_num_lines,
                         "flaw_line": df["flaw_line"], "processed_func": df["processed_func"]})


def write_raw_preds_csv(args, y_preds):
    df = pd.read_csv(args.test_data_file)
    df["raw_preds"] = y_preds
    df.to_csv("../results/raw_preds.csv", index=False)


def get_num_lines(func):
    func = func.split("\n")
    func = [line for line in func if len(line) > 0]
    return len(func)


def get_line_statistics(result_df):
    total_lines = sum(result_df["num_lines"].tolist())
    total_flaw_lines = sum(result_df["num_flaw_lines"].tolist())
    return total_lines, total_flaw_lines


def rank_lines(all_lines_score_with_label, is_attention, ascending_ranking):
    # flatten the list
    all_lines_score_with_label = [line for lines in all_lines_score_with_label for line in lines]
    if is_attention:
        all_scores = [line[0].item() for line in all_lines_score_with_label]
    else:
        all_scores = [line[0] for line in all_lines_score_with_label]
    all_labels = [line[1] for line in all_lines_score_with_label]
    rank_df = pd.DataFrame({"score": all_scores, "label": all_labels})
    rank_df = rank_dataframe(rank_df, "score", ascending_ranking)
    return len(rank_df), rank_df


def rank_dataframe(df, rank_by: str, ascending: bool):
    df = df.sort_values(by=[rank_by], ascending=ascending)
    df = df.reset_index(drop=True)
    return df


def top_k_effort(rank_df, sum_lines, sum_flaw_lines, top_k_loc, label_col_name="label"):
    target_flaw_line = int(sum_flaw_lines * top_k_loc)
    caught_flaw_line = 0
    inspected_line = 0
    for i in range(len(rank_df)):
        inspected_line += 1
        if rank_df[label_col_name][i] == 1:
            caught_flaw_line += 1
        if target_flaw_line == caught_flaw_line:
            break
    effort = round(inspected_line / sum_lines, 4)
    return effort, inspected_line


def top_k_effort_pred_prob(rank_df, sum_lines, sum_flaw_lines, top_k_loc, label_col_name="y_preds"):
    target_flaw_line = int(sum_flaw_lines * top_k_loc)
    caught_flaw_line = 0
    inspected_line = 0
    for i in range(len(rank_df)):
        inspected_line += rank_df["num_lines"][i]
        if rank_df[label_col_name][i] == 1 or rank_df[label_col_name][i] is True:
            caught_flaw_line += rank_df["num_flaw_lines"][i]
        if caught_flaw_line >= target_flaw_line:
            break
    effort = round(inspected_line / sum_lines, 4)
    return effort, inspected_line


def top_k_recall(pos_rank_df, neg_rank_df, sum_lines, sum_flaw_lines, top_k_loc):
    target_inspected_line = int(sum_lines * top_k_loc)
    caught_flaw_line = 0
    inspected_line = 0
    inspect_neg_lines = True
    for i in range(len(pos_rank_df)):
        inspected_line += 1
        if inspected_line > target_inspected_line:
            inspect_neg_lines = False
            break
        if pos_rank_df["label"][i] == 1 or pos_rank_df["label"][i] is True:
            caught_flaw_line += 1
    if inspect_neg_lines:
        for i in range(len(neg_rank_df)):
            inspected_line += 1
            if inspected_line > target_inspected_line:
                break
            if neg_rank_df["label"][i] == 1 or neg_rank_df["label"][i] is True:
                caught_flaw_line += 1
    return round(caught_flaw_line / sum_flaw_lines, 4)


def top_k_recall_pred_prob(rank_df, sum_lines: int, sum_flaw_lines: int, top_k_loc: float, label_col_name="y_preds"):
    target_inspected_line = int(sum_lines * top_k_loc)
    caught_flaw_line = 0
    inspected_line = 0
    for i in range(len(rank_df)):
        inspected_line += rank_df["num_lines"][i]
        if inspected_line > target_inspected_line:
            break
        if rank_df[label_col_name][i] == 1 or rank_df[label_col_name][i] is True:
            caught_flaw_line += rank_df["num_flaw_lines"][i]
    return round(caught_flaw_line / sum_flaw_lines, 4)


def summarize_attributions(attributions):
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    return attributions


def create_ref_input_ids(input_ids, ref_token_id, sep_token_id, cls_token_id):
    seq_length = input_ids.size(1)
    ref_input_ids = [cls_token_id] + [ref_token_id] * (seq_length - 2) + [sep_token_id]
    return torch.tensor([ref_input_ids])


def line_level_localization_tp(flaw_lines: str, tokenizer, model, mini_batch, original_func: str, args, top_k_loc: list,
                               top_k_constant: list, reasoning_method: str, index: int, write_invalid_data: bool):
    # function for captum LIG.
    def predict(input_ids):
        return model(input_ids=input_ids)[0]

    def lig_forward(input_ids):
        logits = model(input_ids=input_ids)[0]
        y_pred = 1  # for positive attribution, y_pred = 0 for negative attribution
        pred_prob = logits[y_pred].unsqueeze(-1)
        return pred_prob

    flaw_line_seperator = "/~/"
    (input_ids, labels) = mini_batch
    ids = input_ids[0].detach().tolist()
    all_tokens = tokenizer.convert_ids_to_tokens(ids)
    all_tokens = [token.replace("Ġ", "") for token in all_tokens]
    all_tokens = [token.replace("ĉ", "Ċ") for token in all_tokens]
    original_lines = ''.join(all_tokens).split("Ċ")

    # flaw line verification
    # get flaw tokens ground truth
    flaw_lines = get_all_flaw_lines(flaw_lines=flaw_lines, flaw_line_seperator=flaw_line_seperator)
    flaw_tokens_encoded = encode_all_lines(all_lines=flaw_lines, tokenizer=tokenizer)
    verified_flaw_lines = []
    do_explanation = False
    for i in range(len(flaw_tokens_encoded)):
        encoded_flaw = ''.join(flaw_tokens_encoded[i])
        encoded_all = ''.join(all_tokens)
        if encoded_flaw in encoded_all:
            verified_flaw_lines.append(flaw_tokens_encoded[i])
            do_explanation = True

    # do explanation if at least one flaw line exist in the encoded input
    if do_explanation:
        if reasoning_method == "attention":
            # attentions: a tuple with of one Tensor with 4D shape (batch_size, num_heads, sequence_length, sequence_length)
            input_ids = input_ids.to(args.device)
            prob, attentions = model(input_ids=input_ids, output_attentions=True)
            # take from tuple then take out mini-batch attention values
            attentions = attentions[0][0]
            attention = None
            # go into the layer
            for i in range(len(attentions)):
                layer_attention = attentions[i]
                # summerize the values of each token dot other tokens
                layer_attention = sum(layer_attention)
                if attention is None:
                    attention = layer_attention
                else:
                    attention += layer_attention
            # clean att score for <s> and </s>
            attention = clean_special_token_values(attention, padding=True)
            # attention should be 1D tensor with seq length representing each token's attention value
            word_att_scores = get_word_att_scores(all_tokens=all_tokens, att_scores=attention)
            all_lines_score, flaw_line_indices = get_all_lines_score(word_att_scores, verified_flaw_lines)
            # return if no flaw lines exist
            if len(flaw_line_indices) == 0:
                return "NA"
            total_lines, num_of_flaw_lines, all_correctly_predicted_flaw_lines, min_clean_lines_inspected, max_clean_lines_inspected, all_correctly_localized_func, top_10_correct_idx, top_10_not_correct_idx \
                = \
                line_level_evaluation(all_lines_score=all_lines_score, flaw_line_indices=flaw_line_indices,
                                      top_k_loc=top_k_loc, top_k_constant=top_k_constant, true_positive_only=True,
                                      index=index)
        elif reasoning_method == "lig":
            ref_token_id, sep_token_id, cls_token_id = tokenizer.pad_token_id, tokenizer.sep_token_id, tokenizer.cls_token_id
            ref_input_ids = create_ref_input_ids(input_ids, ref_token_id, sep_token_id, cls_token_id)
            # send data to device
            input_ids = input_ids.to(args.device)
            labels = labels.to(args.device)
            ref_input_ids = ref_input_ids.to(args.device)
            lig = LayerIntegratedGradients(lig_forward, model.encoder.roberta.embeddings)
            attributions, delta = lig.attribute(inputs=input_ids,
                                                baselines=ref_input_ids,
                                                internal_batch_size=32,
                                                return_convergence_delta=True)
            score = predict(input_ids)
            pred_idx = torch.argmax(score).cpu().numpy()
            pred_prob = score[pred_idx]
            attributions_sum = summarize_attributions(attributions)
            attr_scores = attributions_sum.tolist()
            # each token should have one score
            assert len(all_tokens) == len(attr_scores)
            # store tokens and attr scores together in a list of tuple [(token, attr_score)]
            word_attr_scores = get_word_att_scores(all_tokens=all_tokens, att_scores=attr_scores)
            # remove <s>, </s>, <unk>, <pad>
            word_attr_scores = clean_word_attr_scores(word_attr_scores=word_attr_scores)
            all_lines_score, flaw_line_indices = get_all_lines_score(word_attr_scores, verified_flaw_lines)
            # return if no flaw lines exist
            if len(flaw_line_indices) == 0:
                return "NA"
            total_lines, num_of_flaw_lines, all_correctly_predicted_flaw_lines, min_clean_lines_inspected, max_clean_lines_inspected, all_correctly_localized_func, top_10_correct_idx, top_10_not_correct_idx \
                = \
                line_level_evaluation(all_lines_score=all_lines_score, flaw_line_indices=flaw_line_indices,
                                      top_k_loc=top_k_loc, top_k_constant=top_k_constant, true_positive_only=True,
                                      index=index)
        elif reasoning_method == "deeplift" or \
                reasoning_method == "deeplift_shap" or \
                reasoning_method == "gradient_shap" or \
                reasoning_method == "saliency":
            # send data to device
            input_ids = input_ids.to(args.device)
            input_embed = model.encoder.roberta.embeddings(input_ids).to(args.device)
            if reasoning_method == "deeplift":
                # baselines = torch.randn(1, 512, 768, requires_grad=True).to(args.device)
                baselines = torch.zeros(1, 512, 768, requires_grad=True).to(args.device)
                reasoning_model = DeepLift(model)
            elif reasoning_method == "deeplift_shap":
                # baselines = torch.randn(16, 512, 768, requires_grad=True).to(args.device)
                baselines = torch.zeros(16, 512, 768, requires_grad=True).to(args.device)
                reasoning_model = DeepLiftShap(model)
            elif reasoning_method == "gradient_shap":
                # baselines = torch.randn(16, 512, 768, requires_grad=True).to(args.device)
                baselines = torch.zeros(16, 512, 768, requires_grad=True).to(args.device)
                reasoning_model = GradientShap(model)
            elif reasoning_method == "saliency":
                reasoning_model = Saliency(model)
            # attributions -> [1, 512, 768]
            if reasoning_method == "saliency":
                attributions = reasoning_model.attribute(input_embed, target=1)
            else:
                attributions = reasoning_model.attribute(input_embed, baselines=baselines, target=1)
            attributions_sum = summarize_attributions(attributions)
            attr_scores = attributions_sum.tolist()
            # each token should have one score
            assert len(all_tokens) == len(attr_scores)
            # store tokens and attr scores together in a list of tuple [(token, attr_score)]
            word_attr_scores = get_word_att_scores(all_tokens=all_tokens, att_scores=attr_scores)
            # remove <s>, </s>, <unk>, <pad>
            word_attr_scores = clean_word_attr_scores(word_attr_scores=word_attr_scores)
            all_lines_score, flaw_line_indices = get_all_lines_score(word_attr_scores, verified_flaw_lines)
            # return if no flaw lines exist
            if len(flaw_line_indices) == 0:
                return "NA"
            total_lines, num_of_flaw_lines, all_correctly_predicted_flaw_lines, min_clean_lines_inspected, max_clean_lines_inspected, all_correctly_localized_func, top_10_correct_idx, top_10_not_correct_idx \
                = \
                line_level_evaluation(all_lines_score=all_lines_score, flaw_line_indices=flaw_line_indices,
                                      top_k_loc=top_k_loc, top_k_constant=top_k_constant, true_positive_only=True,
                                      index=index)

        results = {"total_lines": total_lines,
                   "num_of_flaw_lines": num_of_flaw_lines,
                   "all_correctly_predicted_flaw_lines": all_correctly_predicted_flaw_lines,
                   "all_correctly_localized_function": all_correctly_localized_func,
                   "min_clean_lines_inspected": min_clean_lines_inspected,
                   "max_clean_lines_inspected": max_clean_lines_inspected,
                   "top_10_correct_idx": top_10_correct_idx,
                   "top_10_not_correct_idx": top_10_not_correct_idx}
        return results
    else:
        if write_invalid_data:
            with open("../invalid_data/invalid_line_lev_data.txt", "a") as f:
                f.writelines("--- ALL TOKENS ---")
                f.writelines("\n")
                alltok = ''.join(all_tokens)
                alltok = alltok.split("Ċ")
                for tok in alltok:
                    f.writelines(tok)
                    f.writelines("\n")
                f.writelines("--- FLAW ---")
                f.writelines("\n")
                for i in range(len(flaw_tokens_encoded)):
                    f.writelines(''.join(flaw_tokens_encoded[i]))
                    f.writelines("\n")
                f.writelines("\n")
                f.writelines("\n")
    # if no flaw line exist in the encoded input
    return "NA"


def line_level_localization(flaw_lines: str, tokenizer, model, mini_batch, original_func: str, args,
                            top_k_loc: list, top_k_constant: list, reasoning_method: str, index: int):
    # function for captum LIG.
    def predict(input_ids):
        return model(input_ids=input_ids)[0]

    def lig_forward(input_ids):
        logits = model(input_ids=input_ids)[0]
        y_pred = 1  # for positive attribution, y_pred = 0 for negative attribution
        pred_prob = logits[y_pred].unsqueeze(-1)
        return pred_prob

    flaw_line_seperator = "/~/"
    (input_ids, labels) = mini_batch
    ids = input_ids[0].detach().tolist()
    all_tokens = tokenizer.convert_ids_to_tokens(ids)
    all_tokens = [token.replace("Ġ", "") for token in all_tokens]
    all_tokens = [token.replace("ĉ", "Ċ") for token in all_tokens]
    original_lines = ''.join(all_tokens).split("Ċ")

    # flaw line verification
    # get flaw tokens ground truth
    flaw_lines = get_all_flaw_lines(flaw_lines=flaw_lines, flaw_line_seperator=flaw_line_seperator)
    flaw_tokens_encoded = encode_all_lines(all_lines=flaw_lines, tokenizer=tokenizer)
    verified_flaw_lines = []
    for i in range(len(flaw_tokens_encoded)):
        encoded_flaw = ''.join(flaw_tokens_encoded[i])
        encoded_all = ''.join(all_tokens)
        if encoded_flaw in encoded_all:
            verified_flaw_lines.append(flaw_tokens_encoded[i])

    if reasoning_method == "attention":
        # attentions: a tuple with of one Tensor with 4D shape (batch_size, num_heads, sequence_length, sequence_length)
        input_ids = input_ids.to(args.device)
        model.eval()
        model.to(args.device)
        with torch.no_grad():
            prob, attentions = model(input_ids=input_ids, output_attentions=True)
        # take from tuple then take out mini-batch attention values
        attentions = attentions[0][0]
        attention = None
        # go into the layer
        for i in range(len(attentions)):
            layer_attention = attentions[i]
            # summerize the values of each token dot other tokens
            layer_attention = sum(layer_attention)
            if attention is None:
                attention = layer_attention
            else:
                attention += layer_attention
        # clean att score for <s> and </s>
        attention = clean_special_token_values(attention, padding=True)
        # attention should be 1D tensor with seq length representing each token's attention value
        word_att_scores = get_word_att_scores(all_tokens=all_tokens, att_scores=attention)
        all_lines_score, flaw_line_indices = get_all_lines_score(word_att_scores, verified_flaw_lines)
        all_lines_score_with_label = \
            line_level_evaluation(all_lines_score=all_lines_score, flaw_line_indices=flaw_line_indices,
                                  top_k_loc=top_k_loc, top_k_constant=top_k_constant, true_positive_only=False)
    elif reasoning_method == "lig":
        ref_token_id, sep_token_id, cls_token_id = tokenizer.pad_token_id, tokenizer.sep_token_id, tokenizer.cls_token_id
        ref_input_ids = create_ref_input_ids(input_ids, ref_token_id, sep_token_id, cls_token_id)
        # send data to device
        input_ids = input_ids.to(args.device)
        labels = labels.to(args.device)
        ref_input_ids = ref_input_ids.to(args.device)

        lig = LayerIntegratedGradients(lig_forward, model.encoder.roberta.embeddings)

        attributions, delta = lig.attribute(inputs=input_ids,
                                            baselines=ref_input_ids,
                                            internal_batch_size=32,
                                            return_convergence_delta=True)
        score = predict(input_ids)
        pred_idx = torch.argmax(score).cpu().numpy()
        pred_prob = score[pred_idx]
        attributions_sum = summarize_attributions(attributions)
        attr_scores = attributions_sum.tolist()
        # each token should have one score
        assert len(all_tokens) == len(attr_scores)
        # store tokens and attr scores together in a list of tuple [(token, attr_score)]
        word_attr_scores = get_word_att_scores(all_tokens=all_tokens, att_scores=attr_scores)
        # remove <s>, </s>, <unk>, <pad>
        word_attr_scores = clean_word_attr_scores(word_attr_scores=word_attr_scores)
        all_lines_score, flaw_line_indices = get_all_lines_score(word_attr_scores, verified_flaw_lines)
        all_lines_score_with_label = \
            line_level_evaluation(all_lines_score=all_lines_score, flaw_line_indices=flaw_line_indices,
                                  top_k_loc=top_k_loc, top_k_constant=top_k_constant, true_positive_only=False)
    elif reasoning_method == "deeplift" or \
            reasoning_method == "deeplift_shap" or \
            reasoning_method == "gradient_shap" or \
            reasoning_method == "saliency":
        # send data to device
        input_ids = input_ids.to(args.device)
        input_embed = model.encoder.roberta.embeddings(input_ids).to(args.device)
        if reasoning_method == "deeplift":
            # baselines = torch.randn(1, 512, 768, requires_grad=True).to(args.device)
            baselines = torch.zeros(1, 512, 768, requires_grad=True).to(args.device)
            reasoning_model = DeepLift(model)
        elif reasoning_method == "deeplift_shap":
            # baselines = torch.randn(16, 512, 768, requires_grad=True).to(args.device)
            baselines = torch.zeros(16, 512, 768, requires_grad=True).to(args.device)
            reasoning_model = DeepLiftShap(model)
        elif reasoning_method == "gradient_shap":
            # baselines = torch.randn(16, 512, 768, requires_grad=True).to(args.device)
            baselines = torch.zeros(16, 512, 768, requires_grad=True).to(args.device)
            reasoning_model = GradientShap(model)
        elif reasoning_method == "saliency":
            reasoning_model = Saliency(model)
        # attributions -> [1, 512, 768]
        if reasoning_method == "saliency":
            attributions = reasoning_model.attribute(input_embed, target=1)
        else:
            attributions = reasoning_model.attribute(input_embed, baselines=baselines, target=1)
        attributions_sum = summarize_attributions(attributions)
        attr_scores = attributions_sum.tolist()
        # each token should have one score
        assert len(all_tokens) == len(attr_scores)
        # store tokens and attr scores together in a list of tuple [(token, attr_score)]
        word_attr_scores = get_word_att_scores(all_tokens=all_tokens, att_scores=attr_scores)
        # remove <s>, </s>, <unk>, <pad>
        word_attr_scores = clean_word_attr_scores(word_attr_scores=word_attr_scores)
        all_lines_score, flaw_line_indices = get_all_lines_score(word_attr_scores, verified_flaw_lines)

        all_lines_score_with_label = \
            line_level_evaluation(all_lines_score=all_lines_score, flaw_line_indices=flaw_line_indices,
                                  top_k_loc=top_k_loc, top_k_constant=top_k_constant, true_positive_only=False)
    return all_lines_score_with_label


def line_level_evaluation(all_lines_score: list, flaw_line_indices: list, top_k_loc: list, top_k_constant: list,
                          true_positive_only: bool, index=None):
    if true_positive_only:
        # line indices ranking based on attr values 
        ranking = sorted(range(len(all_lines_score)), key=lambda i: all_lines_score[i], reverse=True)
        # total flaw lines
        num_of_flaw_lines = len(flaw_line_indices)
        # clean lines + flaw lines
        total_lines = len(all_lines_score)
        ### TopK% Recall ###
        all_correctly_predicted_flaw_lines = []
        ### IFA ###
        ifa = True
        all_clean_lines_inspected = []
        for top_k in top_k_loc:
            correctly_predicted_flaw_lines = 0
            for indice in flaw_line_indices:
                # if within top-k
                k = int(len(all_lines_score) * top_k)
                # if detecting any flaw lines
                if indice in ranking[: k]:
                    correctly_predicted_flaw_lines += 1
                if ifa:
                    # calculate Initial False Alarm
                    # IFA counts how many clean lines are inspected until the first vulnerable line is found when inspecting the lines ranked by the approaches.
                    flaw_line_idx_in_ranking = ranking.index(indice)
                    # e.g. flaw_line_idx_in_ranking = 3 will include 1 vulnerable line and 3 clean lines
                    all_clean_lines_inspected.append(flaw_line_idx_in_ranking)
                    # for IFA
            min_clean_lines_inspected = min(all_clean_lines_inspected)
            # for All Effort
            max_clean_lines_inspected = max(all_clean_lines_inspected)
            # only do IFA and All Effort once
            ifa = False
            # append result for one top-k value
            all_correctly_predicted_flaw_lines.append(correctly_predicted_flaw_lines)

        ### Top10 Accuracy ###
        all_correctly_localized_func = []
        top_10_correct_idx = []
        top_10_not_correct_idx = []
        correctly_located = False
        for k in top_k_constant:
            for indice in flaw_line_indices:
                # if detecting any flaw lines
                if indice in ranking[: k]:
                    """
                    # extract example for the paper
                    if index == 2797:
                        print("2797")
                        print("ground truth flaw line index: ", indice)
                        print("ranked line")
                        print(ranking)
                        print("original score")
                        print(all_lines_score)
                    """
                    # append result for one top-k value
                    all_correctly_localized_func.append(1)
                    correctly_located = True
                else:
                    all_correctly_localized_func.append(0)
            if correctly_located:
                top_10_correct_idx.append(index)
            else:
                top_10_not_correct_idx.append(index)
        return total_lines, num_of_flaw_lines, all_correctly_predicted_flaw_lines, min_clean_lines_inspected, max_clean_lines_inspected, all_correctly_localized_func, \
            top_10_correct_idx, top_10_not_correct_idx
    else:
        # all_lines_score_with_label: [[line score, line level label], [line score, line level label], ...]
        all_lines_score_with_label = []
        for i in range(len(all_lines_score)):
            if i in flaw_line_indices:
                all_lines_score_with_label.append([all_lines_score[i], 1])
            else:
                all_lines_score_with_label.append([all_lines_score[i], 0])
        return all_lines_score_with_label


def clean_special_token_values(all_values, padding=False):
    # special token in the beginning of the seq 
    all_values[0] = 0
    if padding:
        # get the last non-zero value which represents the att score for </s> token
        idx = [index for index, item in enumerate(all_values) if item != 0][-1]
        all_values[idx] = 0
    else:
        # special token in the end of the seq 
        all_values[-1] = 0
    return all_values


def clean_shap_tokens(all_tokens):
    for i in range(len(all_tokens)):
        all_tokens[i] = all_tokens[i].replace('Ġ', '')
    return all_tokens


def get_all_lines_score(word_att_scores: list, verified_flaw_lines: list):
    verified_flaw_lines = [''.join(l) for l in verified_flaw_lines]
    # word_att_scores -> [[token, att_value], [token, att_value], ...]
    separator = ["Ċ", " Ċ", "ĊĊ", " ĊĊ"]
    # to return
    all_lines_score = []
    score_sum = 0
    line_idx = 0
    flaw_line_indices = []
    line = ""
    for i in range(len(word_att_scores)):
        # summerize if meet line separator or the last token
        if ((word_att_scores[i][0] in separator) or (i == (len(word_att_scores) - 1))) and score_sum != 0:
            score_sum += word_att_scores[i][1]
            all_lines_score.append(score_sum)
            is_flaw_line = False
            for l in verified_flaw_lines:
                if l == line:
                    is_flaw_line = True
            if is_flaw_line:
                flaw_line_indices.append(line_idx)
            line = ""
            score_sum = 0
            line_idx += 1
        # else accumulate score
        elif word_att_scores[i][0] not in separator:
            line += word_att_scores[i][0]
            score_sum += word_att_scores[i][1]
    return all_lines_score, flaw_line_indices


def get_all_flaw_lines(flaw_lines: str, flaw_line_seperator: str) -> list:
    if isinstance(flaw_lines, str):
        flaw_lines = flaw_lines.strip(flaw_line_seperator)
        flaw_lines = flaw_lines.split(flaw_line_seperator)
        flaw_lines = [line.strip() for line in flaw_lines]
    else:
        flaw_lines = []
    return flaw_lines


def encode_all_lines(all_lines: list, tokenizer) -> list:
    encoded = []
    for line in all_lines:
        encoded.append(encode_one_line(line=line, tokenizer=tokenizer))
    return encoded


def get_word_att_scores(all_tokens: list, att_scores: list) -> list:
    word_att_scores = []
    for i in range(len(all_tokens)):
        token, att_score = all_tokens[i], att_scores[i]
        word_att_scores.append([token, att_score])
    return word_att_scores


def clean_word_attr_scores(word_attr_scores: list) -> list:
    to_be_cleaned = ['<s>', '</s>', '<unk>', '<pad>']
    cleaned = []
    for word_attr_score in word_attr_scores:
        if word_attr_score[0] not in to_be_cleaned:
            cleaned.append(word_attr_score)
    return cleaned


def encode_one_line(line, tokenizer):
    # add "@ " at the beginning to ensure the encoding consistency, i.e., previous -> previous, not previous > pre + vious
    code_tokens = tokenizer.tokenize("@ " + line)
    return [token.replace("Ġ", "") for token in code_tokens if token != "@"]


def main():
    parser = argparse.ArgumentParser()
    ## parameters
    parser.add_argument("--train_data_file", default="../resource/dataset/train.csv", type=str, required=False,
                        help="The input training data file (a csv file).")
    parser.add_argument("--output_dir", default="../new_model/saved_models", type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--model_type", default="roberta", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--block_size", default=512, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs "
                             "(take into account special tokens).")
    parser.add_argument("--eval_data_file", default="../resource/dataset/val.csv", type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", default="../resource/dataset/val.csv", type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--model_name", default="model_vul.bin", type=str,
                        help="Saved model name.")
    parser.add_argument("--model_name_or_path", default="../resource/VulBERTa", type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default="../resource/VulBERTa", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--use_non_pretrained_model", action='store_true', default=False,
                        help="Whether to use non-pretrained model.")
    parser.add_argument("--tokenizer_name", default="../resource/VulBERTa", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--code_length", default=256, type=int,
                        help="Optional Code input sequence length after tokenization.")

    parser.add_argument("--do_train", action='store_true', default=False,
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', default=False,
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true', default=False,
                        help="Whether to run eval on the dev set.")

    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_local_explanation", default=False, action='store_true',
                        help="Whether to do local explanation. ")
    parser.add_argument("--reasoning_method", default='attention', type=str,
                        help="Should be one of 'attention', 'shap', 'lime', 'lig'")

    parser.add_argument("--train_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=2e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--seed', type=int, default=123456,
                        help="random seed for initialization")
    parser.add_argument('--epochs', type=int, default=10,
                        help="training epochs")
    # RQ2
    parser.add_argument("--effort_at_top_k", default=0.2, type=float,
                        help="Effort@TopK%Recall: effort at catching top k percent of vulnerable lines")
    parser.add_argument("--top_k_recall_by_lines", default=0.01, type=float,
                        help="Recall@TopK percent, sorted by line scores")
    parser.add_argument("--top_k_recall_by_pred_prob", default=0.2, type=float,
                        help="Recall@TopK percent, sorted by prediction probabilities")

    parser.add_argument("--do_sorting_by_line_scores", default=False, action='store_true',
                        help="Whether to do sorting by line scores.")
    parser.add_argument("--do_sorting_by_pred_prob", default=False, action='store_true',
                        help="Whether to do sorting by prediction probabilities.")
    # RQ3 - line-level evaluation
    parser.add_argument('--top_k_constant', type=int, default=10,
                        help="Top-K Accuracy constant")
    # num of attention heads
    parser.add_argument('--num_attention_heads', type=int, default=12,
                        help="number of attention heads used in CodeBERT")
    # raw predictions
    parser.add_argument("--write_raw_preds", default=False, action='store_true',
                        help="Whether to write raw predictions on test data.")
    # word-level tokenizer
    parser.add_argument("--use_word_level_tokenizer", default=False, action='store_true',
                        help="Whether to use word-level tokenizer.")
    # bpe non-pretrained tokenizer
    parser.add_argument("--use_non_pretrained_tokenizer", default=False, action='store_true',
                        help="Whether to use non-pretrained bpe tokenizer.")
    # initialize the model
    parser.add_argument("--initialize_model_name", default="../resource/VulBERTa/pytorch_model.bin",
                        type=str,
                        help="initialize the model")
    args = parser.parse_args()
    # Setup CUDA, GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = 1  # torch.cuda.device_count()
    args.device = device
    # Setup logging
    log_name = "log_" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_codebert.txt"
    logging.basicConfig(filename="../log/vulBERTa/" + log_name,
                        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    logger.warning("device: %s, n_gpu: %s", device, args.n_gpu, )
    # Set seed
    set_seed(args)
    config = RobertaConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    config.num_labels = 1
    config.num_attention_heads = args.num_attention_heads
    if args.use_word_level_tokenizer:
        print('using wordlevel tokenizer!')
        tokenizer = Tokenizer.from_file('../resource/word_level_tokenizer/wordlevel.json')
    elif args.use_non_pretrained_tokenizer:
        tokenizer = RobertaTokenizer(vocab_file="../resource/bpe_tokenizer/bpe_tokenizer-vocab.json",
                                     merges_file="../resource/bpe_tokenizer/bpe_tokenizer-merges.txt")
    elif args.tokenizer_name == '../resource/VulBERTa':
        ## Load pre-trained tokenizers
        tokenizer = None
    else:
        tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
    if args.use_non_pretrained_model:
        model = RobertaForSequenceClassification(config=config)
    else:
        model = RobertaForSequenceClassification.from_pretrained(args.model_name_or_path,  config=config,
                                                                 ignore_mismatched_sizes=True)
    model = Model(model, config, tokenizer, args)
    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        nums = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info("The number of parameter is {}".format(nums))
        logger.info("Training/evaluation parameters %s", args)

        training_file_path = args.train_data_file
        eval_file_path = args.eval_data_file
        train_dataset = TextDataset(tokenizer, args, file_path=training_file_path, file_type='train')
        eval_dataset = TextDataset(tokenizer, args, file_path=eval_file_path, file_type='eval')

        train(args, train_dataset, model, tokenizer, eval_dataset)
    # Evaluation
    results = {}
    if args.do_test:
        checkpoint_prefix = f'checkpoint-best-f1/{args.model_name}'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        model.load_state_dict(torch.load(output_dir, map_location=args.device), strict=False)
        model.to(args.device)
        test_file_path = args.test_data_file
        # test_dataset = TextDataset(tokenizer, args, file_type='test')
        test_dataset = TextDataset(tokenizer, args, file_path=test_file_path,
                                   file_type='test')
        test(args, model, tokenizer, test_dataset, best_threshold=0.5)
    return results


if __name__ == "__main__":
    main()
