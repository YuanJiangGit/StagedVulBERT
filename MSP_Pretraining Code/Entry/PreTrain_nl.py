# coding=utf-8

from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import pickle
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
# model reasoning
from captum.attr import LayerIntegratedGradients, DeepLift, DeepLiftShap, GradientShap, Saliency
# metrics
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import auc
# word-level tokenizer
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from tqdm import tqdm
from transformers import (get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
from CTextDataset_nl import TextDataset

from Models.MLM import Model as MLModel
from Models.MSLM import Model as MSLModel
from Models.MSLM import Decoder

logger = logging.getLogger(__name__)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 input_tokens,
                 input_ids,
                 token_row_indices,
                 label,
                 code):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.row_idx = token_row_indices
        self.label = label
        self.code = code


def train(args, train_dataset, MLM_model, MSLM_model, tokenizer, eval_dataset):
    """ Train the model """
    # build dataloader
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=0)

    args.max_steps = args.epochs * len(train_dataloader)

    # evaluate the model per epoch
    args.save_steps = len(train_dataloader) // 20
    args.warmup_steps = args.max_steps // 5
    MSLM_model.to(args.device)
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in MSLM_model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in MSLM_model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.max_steps)
    # multi-gpu training
    if args.n_gpu > 1:
        MSLM_model = torch.nn.DataParallel(MSLM_model)

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
    best_loss = float('inf')

    MSLM_model.zero_grad()
    torch.autograd.set_detect_anomaly(True)

    for idx in range(args.epochs):
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        tr_num = 0
        train_loss = 0
        train_acc = 0
        for step, batch in enumerate(bar):
            (inputs_ids, first_stoken_mask, row2row_mask, attn_mask, masked_source_ids, mask_row_pos) \
                = [x.to(args.device) for x in batch]
            MSLM_model.train()
            # inputs_ids, position_idx, attn_mask, masked_inputs_pos, masked_inputs_ids
            # inputs_ids, first_stoken_mask, row2row_mask, attn_mask, masked_source_ids, mask_row_pos
            logits, loss = MSLM_model(inputs_ids, first_stoken_mask, row2row_mask,
                                      attn_mask, masked_source_ids, mask_row_pos)
            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(MSLM_model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            tr_num += 1
            train_loss += loss.item()
            if avg_loss == 0:
                avg_loss = tr_loss

            # _, predicted = torch.max(logits.data, 1)
            # predicted = predicted.cpu().numpy()
            # labels = labels.cpu().numpy()
            # train_acc += len(labels[predicted == labels]) / len(labels)
            #
            # avg_acc = round(train_acc / tr_num, 5)
            #
            # avg_loss = round(train_loss / tr_num, 5)
            # bar.set_description("epoch {} loss {} acc {}".format(idx, avg_loss, avg_acc))
            bar.set_description("epoch {} loss {}".format(idx, avg_loss))

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                output_flag = True
                avg_loss = round(np.exp((tr_loss - logging_loss) / (global_step - tr_nb)), 4)

                if global_step % args.save_steps == 0:
                    logger.info("***** Epoch {} Running evaluation *****".format(idx))
                    results = evaluate(args, MSLM_model, tokenizer, eval_dataset, eval_when_training=True)

                    # Save model checkpoint
                    if results['eval_loss'] < best_loss:
                        best_loss = results['eval_loss']
                        logger.info("  " + "*" * 20)
                        logger.info("  Best loss:%s", round(best_loss, 4))
                        logger.info("  " + "*" * 20)

                        checkpoint_prefix = 'checkpoint-best-f1'
                        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = MSLM_model.module if hasattr(MSLM_model, 'module') else MSLM_model
                        model_name = "model_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".bin"
                        output_dir = os.path.join(output_dir, '{}'.format(model_name))
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
    # logits = []
    # y_trues = []
    for batch in eval_dataloader:
        (inputs_ids, first_stoken_mask, row2row_mask, attn_mask, masked_source_ids, mask_row_pos) \
            = [x.to(args.device) for x in batch]
        with torch.no_grad():
            logits, loss = model(inputs_ids, first_stoken_mask, row2row_mask,
                                 attn_mask, masked_source_ids, mask_row_pos)
            eval_loss += loss.item()
            # logits.append(logit.cpu().numpy())
            # y_trues.append(labels.cpu().numpy())
        nb_eval_steps += 1

    # calculate scores
    # logits = np.concatenate(logits, 0)
    # y_trues = np.concatenate(y_trues, 0)
    best_threshold = 0.5
    best_loss = 0
    # y_preds = logits[:, 1] > best_threshold
    # recall = recall_score(y_trues, y_preds)
    # precision = precision_score(y_trues, y_preds)
    # f1 = f1_score(y_trues, y_preds)
    result = {
        # "eval_recall": float(recall),
        # "eval_precision": float(precision),
        # "eval_f1": float(f1),
        # "eval_threshold": best_threshold,
        "eval_loss": round(eval_loss / nb_eval_steps, 5)
    }

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))

    return result


def main():
    parser = argparse.ArgumentParser()
    ## parameters
    parser.add_argument("--train_data_file", default="../data/big-vul_dataset/train.csv", type=str, required=False,
                        help="The input training data file (a csv file).")
    parser.add_argument("--output_dir", default="../new_model/saved_models", type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--model_type", default="bert", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--block_size", default=512, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs "
                             "(take into account special tokens).")
    parser.add_argument("--max_row_size", default=100, type=int)
    parser.add_argument("--eval_data_file", default="../data/big-vul_dataset/eval.csv", type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", default="../data/big-vul_dataset/val.csv", type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--model_name", default="model_c.bin", type=str,
                        help="Saved model name.")
    parser.add_argument("--model_name_or_path", default="./codebert-base", type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--use_non_pretrained_model", action='store_true', default=False,
                        help="Whether to use non-pretrained model.")
    parser.add_argument("--tokenizer_name", default="./codebert-base", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--doc_length", default=128, type=int,
                        help="Optional Code input sequence length after tokenization.")
    parser.add_argument("--code_length", default=384, type=int,
                        help="Optional Code input sequence length after tokenization.")

    parser.add_argument("--do_train", action='store_true', default=True,
                        help="Whether to run training.")
    parser.add_argument("--do_MLM_train", action='store_true', default=False,
                        help="Whether to run training.")
    parser.add_argument("--do_MSLM_train", action='store_true', default=True,
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")

    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_local_explanation", default=False, action='store_true',
                        help="Whether to do local explanation. ")
    parser.add_argument("--reasoning_method", default=None, type=str,
                        help="Should be one of 'attention', 'shap', 'lime', 'lig'")

    parser.add_argument("--train_batch_size", default=24, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=24, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
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
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--epochs', type=int, default=5,
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
    args = parser.parse_args()
    # Setup CUDA, GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    args.n_gpu = 1  # torch.cuda.device_count()
    args.device = device
    # Setup logging
    log_name = "log_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".txt"
    logging.basicConfig(filename="../resources/log/" + log_name,
                        format='%(asctime)s - %(levelname)s - %(name)s -  %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    logger.warning("device: %s, n_gpu: %s", device, args.n_gpu, )
    # Set seed
    set_seed(args)
    config = RobertaConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    config.num_labels = 1
    config.num_attention_heads = args.num_attention_heads
    config.num_hidden_layers = 6
    if args.use_word_level_tokenizer:
        print('using wordlevel tokenizer!')
        tokenizer = Tokenizer.from_file('./word_level_tokenizer/wordlevel.json')
    elif args.use_non_pretrained_tokenizer:
        tokenizer = RobertaTokenizer(vocab_file="bpe_tokenizer/bpe_tokenizer-vocab.json",
                                     merges_file="bpe_tokenizer/bpe_tokenizer-merges.txt")
    else:
        tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
    if args.use_non_pretrained_model:
        TEtransformer = RobertaForSequenceClassification(config=config)
    else:
        TEtransformer = RobertaForSequenceClassification.from_pretrained(args.model_name_or_path, config=config,
                                                                         ignore_mismatched_sizes=True)
    SETransformer = RobertaForSequenceClassification(config=config)

    MLM_model, MSLM_model = None, None
    if args.do_MLM_train:
        MLM_model = MLModel(TEtransformer, SETransformer, config, tokenizer, args)  # encoder, config, tokenizer, args
    if args.do_MSLM_train:
        embedding = TEtransformer.roberta.embeddings.word_embeddings
        decoder = Decoder(config, embedding, dropout=0.5)
        MSLM_model = MSLModel(TEtransformer, SETransformer, decoder, tokenizer, args, config)

    # print the number of parameters in Model
    # MLM_model_nums = sum(p.numel() for p in MLM_model.parameters() if p.requires_grad)
    # logger.info("The number of parameter is {}".format(nums))
    # logger.info("Training/evaluation parameters %s", args)

    # load the dataset
    dataset = TextDataset(tokenizer, args, file_type='all')
    ratio = '99:1'
    data_num = len(dataset)
    ratios = [int(r) for r in ratio.split(':')]
    train_size = int(ratios[0] / sum(ratios) * data_num)
    validate_size = data_num - train_size
    # test_size = data_num - train_size - validate_size
    train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_size, validate_size])

    # Training
    if args.do_MSLM_train:
        train(args, train_dataset, MLM_model, MSLM_model, tokenizer, eval_dataset)
    # Evaluation
    results = {}
    # if args.do_test:
    #     checkpoint_prefix = f'checkpoint-best-f1/{args.model_name}'
    #     output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
    #     model.load_state_dict(torch.load(output_dir, map_location=args.device), strict=False)
    #     model.to(args.device)
    #     # test(args, model, tokenizer, test_dataset, best_threshold=0.5)
    # return results


if __name__ == "__main__":
    main()
