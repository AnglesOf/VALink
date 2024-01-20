from __future__ import absolute_import, division, print_function

import datetime
import gc
import os
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, \
    roc_auc_score, matthews_corrcoef, brier_score_loss, confusion_matrix
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from tqdm import tqdm
from transformers import (AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaTokenizer, AutoModel)

from adv_model import ADVModel
from utils import getargs, set_seed, TextDataset

warnings.filterwarnings(action='ignore')

best_f = -1.0


def train(args, model, train_dataset, eval_dataset):
    dfScores = pd.DataFrame(columns=['Epoch', 'Metrics', 'Score'])
    torch.set_grad_enabled(True)
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    """ Train the model """
    if args.do_dynSample:
        train_dataset.DynamicSubSampler(args, model)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                  batch_size=args.train_batch_size, num_workers=4, pin_memory=True)

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    max_steps = len(train_dataloader) * args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=max_steps * 0.1,
                                                num_training_steps=max_steps)

    # Train!
    print("********** Running training **********")
    print("  Num examples = {}".format(len(train_dataset)))
    print("  Num Epochs = {}".format(args.num_train_epochs))
    print("  batch size = {}".format(args.train_batch_size))
    print("  Total optimization steps = {}".format(max_steps))
    global best_f
    # best_f = 0.0
    model.zero_grad()

    model.train()
    # no_improve = 0
    for idx in range(args.num_train_epochs):
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        losses = []

        for step, batch in enumerate(bar):
            loss, logits, lossCLS, lossCE, weight_CLS, weight_CE = model(batch[0].to(args.device),
                                                                         batch[1].to(args.device),
                                                                         batch[2].to(args.device),
                                                                         adv_flag=False)
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            losses.append(loss.item())
            bar.set_description(
                "epoch {} loss {} cls {} ce {} cls_w {} ce_w {}".format(idx, round(float(np.mean(losses)), 3),
                                                                        round(float(lossCLS.mean()), 3),
                                                                        round(float(lossCE.mean()), 3),
                                                                        round(float(weight_CLS.mean()), 3),
                                                                        round(float(weight_CE.mean()), 3)))

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            del loss
            del logits
            torch.cuda.empty_cache()
            gc.collect()
        results = evaluate(args, model, eval_dataset)
        for key, value in results.items():
            print('-' * 10 + "  {} = {}".format(key, round(value, 4)))
        for key in sorted(results.keys()):
            print('-' * 10 + "  {} = {}".format(key, str(round(results[key], 4))))
            dfScores.loc[len(dfScores)] = [idx, key, str(round(results[key], 4))]

        if results['eval_f1'] > best_f:
            best_f = results['eval_f1']
            print("  " + "*" * 20)
            print("  Best f1: {}".format(round(best_f, 4)))
            print("  " + "*" * 20)
            checkpoint_prefix = args.pro + '_checkpoint-best'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            output_dir = os.path.join(output_dir, '{}'.format('model.bin'))
            torch.save(model, output_dir)
            print("Saving model checkpoint to {}".format(output_dir))
            dfScores.loc[len(dfScores)] = [idx, '___best___', '___best___']

        if not os.path.exists(args.result_dir):
            os.makedirs(args.result_dir)
        dfScores.to_csv(os.path.join(args.result_dir, args.pro + "Epoch_Metrics.csv"), index=False)
        if args.do_dynSample:
            train_dataset.DynamicSubSampler(args, model)
            train_sampler = RandomSampler(train_dataset)
            train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                          batch_size=args.train_batch_size, num_workers=4, pin_memory=True)
        model.train()


def advTrain(args, train_dataset, eval_dataset):
    # load model
    checkpoint_prefix = args.pro + '_checkpoint-best/model.bin'
    output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
    model = torch.load(output_dir)
    model.to(args.device)

    dfScores = pd.DataFrame(columns=['Epoch', 'Metrics', 'Score'])
    torch.set_grad_enabled(True)
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    """ Train the model """
    if args.do_dynSample:
        train_dataset.DynamicSubSampler(args, model)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                  batch_size=args.train_batch_size, num_workers=4, pin_memory=True)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    max_steps = len(train_dataloader) * args.num_train_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=max_steps * 0.1,
                                                num_training_steps=max_steps)

    # Train!
    print("********** Running ADV training **********")
    print("  Num examples = {}".format(len(train_dataset)))
    print("  Num Epochs = {}".format(args.num_train_epochs))
    print("  batch size = {}".format(args.train_batch_size))
    print("  Total optimization steps = {}".format(max_steps))
    global best_f
    model.zero_grad()

    model.train()
    for idx in range(args.num_train_epochs):
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        losses = []
        for step, batch in enumerate(bar):
            loss, logits, lossCLS, lossCE, weight_CLS, weight_CE = model(batch[0].to(args.device),
                                                                         batch[1].to(args.device),
                                                                         batch[2].to(args.device),
                                                                         adv_flag=False)
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            loss.backward()

            loss_adv, logits = model(batch[0].to(args.device), batch[1].to(args.device), batch[2].to(args.device),
                                     adv_flag=True)
            if args.n_gpu > 1:
                loss_adv = loss_adv.mean()  # mean() to average on multi-gpu parallel training
            loss_adv.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            losses.append(loss.item())
            bar.set_description(
                "ADV epoch:{} loss:{} cls_loss:{} adv_loss:{}".format(idx, round(float(np.mean(losses)), 3),
                                                                      round(float(loss.item()), 3),
                                                                      round(float(loss_adv.item()), 3)))
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            del loss
            del logits
            torch.cuda.empty_cache()
            gc.collect()  # 清理内存
        results = evaluate(args, model, eval_dataset)
        for key, value in results.items():
            print('-' * 10 + "  {} = {}".format(key, round(value, 4)))
        for key in sorted(results.keys()):
            print('-' * 10 + "  {} = {}".format(key, str(round(results[key], 4))))
            dfScores.loc[len(dfScores)] = [idx, key, str(round(results[key], 4))]

        if results['eval_f1'] > best_f:
            best_f = results['eval_f1']
            print("  " + "*" * 20)
            print("  Best f1: {}".format(round(best_f, 4)))
            print("  " + "*" * 20)
            checkpoint_prefix = args.pro + '_checkpoint-best'
            output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            output_dir = os.path.join(output_dir, '{}'.format('advmodel.bin'))
            torch.save(model, output_dir)
            print("Saving model checkpoint to {}".format(output_dir))
            dfScores.loc[len(dfScores)] = [idx, '___best___', '___best___']
        if not os.path.exists(args.result_dir):
            os.makedirs(args.result_dir)
        dfScores.to_csv(os.path.join(args.result_dir, args.pro + "advEpoch_Metrics.csv"), index=False)
        if args.do_dynSample:
            train_dataset.DynamicSubSampler(args, model)
            train_sampler = RandomSampler(train_dataset)
            train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                          batch_size=args.train_batch_size, num_workers=4, pin_memory=True)
        model.train()


def evaluate(args, model, eval_dataset):
    stime = datetime.datetime.now()
    eval_output_dir = args.output_dir

    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=4,
                                 pin_memory=True)

    # Eval!
    print("***** Running evaluation *****")
    print("  Num examples = {}".format(len(eval_dataset)))
    print("  Batch size = {}".format(args.eval_batch_size))
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    labels = []
    for batch in eval_dataloader:
        label = batch[2].to(args.device)
        with torch.no_grad():
            lm_loss, logit, lossCLS, lossCE, weight_CLS, weight_CE = model(batch[0].to(args.device),
                                                                           batch[1].to(args.device), label,
                                                                           adv_flag=False)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            labels.append(label.cpu().numpy())
        nb_eval_steps += 1
    logits = np.concatenate(logits, 0)
    labels = np.concatenate(labels, 0)
    preds = logits.argmax(-1)
    print('Predictions', preds[:25])
    print('Labels:', labels[:25])
    etime = datetime.datetime.now()
    eval_time = (etime - stime).seconds
    eval_acc = np.mean(labels == preds)
    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.tensor(eval_loss)
    eval_precision = precision_score(labels, preds)
    eval_recall = recall_score(labels, preds)
    eval_f1 = f1_score(labels, preds)
    eval_auc = roc_auc_score(labels, preds)
    eval_mcc = matthews_corrcoef(labels, preds)
    tn, fp, fn, tp = confusion_matrix(y_true=labels, y_pred=preds).ravel()
    eval_pf = fp / (fp + tn)
    eval_brier = brier_score_loss(labels, preds)

    result = {
        "eval_loss": float(perplexity),
        "eval_time": float(eval_time),
        "eval_acc": round(float(eval_acc), 4),
        "eval_precision": round(eval_precision, 4),
        "eval_recall": round(eval_recall, 4),
        "eval_f1": round(eval_f1, 4),
        "eval_auc": round(eval_auc, 4),
        "eval_mcc": round(eval_mcc, 4),
        "eval_brier": round(eval_brier, 4),
        "eval_pf": round(eval_pf, 4),
    }
    return result


def model_test(args, model, eval_dataset):
    # Note that DistributedSampler samples randomly
    stime = datetime.datetime.now()
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    print("********** Running Test **********")
    print("  Num examples = {}".format(len(eval_dataset)))
    print("  Batch size = {}".format(args.eval_batch_size))
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    labels = []
    issues = []
    commits = []
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
        label = batch[2].to(args.device)
        with torch.no_grad():
            lm_loss, logit, lossCLS, lossCE, weight_CLS, weight_CE = model(batch[0].to(args.device),
                                                                           batch[1].to(args.device), label,
                                                                           adv_flag=False)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            labels.append(label.cpu().numpy())
            for iss, com in zip(batch[3][0], batch[4][0]):
                issues.append(iss)
                commits.append(com)
        nb_eval_steps += 1

    logits = np.concatenate(logits, 0)
    labels = np.concatenate(labels, 0)
    preds = logits.argmax(-1)
    etime = datetime.datetime.now()

    eval_time = (etime - stime).seconds
    eval_acc = np.mean(labels == preds)
    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.tensor(eval_loss)
    eval_precision = precision_score(labels, preds)
    eval_recall = recall_score(labels, preds)
    eval_f1 = f1_score(labels, preds)
    eval_auc = roc_auc_score(labels, preds)
    eval_mcc = matthews_corrcoef(labels, preds)
    tn, fp, fn, tp = confusion_matrix(y_true=labels, y_pred=preds).ravel()
    eval_pf = fp / (fp + tn)
    eval_brier = brier_score_loss(labels, preds)

    result = {
        "eval_loss": float(perplexity),
        "eval_time": float(eval_time),
        "eval_acc": round(float(eval_acc), 4),
        "eval_precision": round(eval_precision, 4),
        "eval_recall": round(eval_recall, 4),
        "eval_f1": round(eval_f1, 4),
        "eval_auc": round(eval_auc, 4),
        "eval_mcc": round(eval_mcc, 4),
        "eval_brier": round(eval_brier, 4),
        "eval_pf": round(eval_pf, 4),
    }
    print(preds[:25], labels[:25])
    print("********** Test results **********")
    dfScores = pd.DataFrame(columns=['Metrics', 'Score'])
    for key in sorted(result.keys()):
        print('-' * 10 + "  {} = {}".format(key, str(round(result[key], 4))))
        dfScores.loc[len(dfScores)] = [key, str(round(result[key], 4))]
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    dfScores.to_csv(os.path.join(args.result_dir, args.pro + "_Metrics.csv"), index=False)
    assert len(logits) == len(preds) and len(logits) == len(labels), 'error'
    logits4class0, logits4class1 = \
        [logits[iclass][0] for iclass in range(len(logits))], \
        [logits[iclass][1] for iclass in range(len(logits))]
    df = pd.DataFrame(np.transpose([issues, commits, logits4class0, logits4class1, preds, labels]),
                      columns=['Issue_Key', 'Commit_SHA', '0_logit', '1_logit', 'preds', 'labels'])
    df.to_csv(os.path.join(args.result_dir, args.pro + "_Prediction.csv"), index=False)


def main():
    print("======BTLink BEGIN...======" * 5)
    args = getargs()
    print(args.key)
    print(f'===Project:{args.pro}---{args.pro}===' * 2)
    print("device: {}, n_gpu: {}".format(args.device, args.n_gpu))
    # Set seed
    set_seed(args.seed)

    # Roberta
    config = RobertaConfig.from_pretrained(args.text_model_path)
    config.num_labels = 2
    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
    textEncoder = AutoModel.from_pretrained(args.text_model_path, config=config)

    # CodeBERT
    config4Code = RobertaConfig.from_pretrained(args.code_model_path)
    config4Code.num_labels = 2
    codeTokenizer = tokenizer
    codeEncoder = AutoModel.from_pretrained(args.code_model_path, config=config4Code)

    model = ADVModel(textEncoder, codeEncoder, config.hidden_size, config4Code.hidden_size, args.num_class)
    model.to(args.device)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    print("Training/evaluation parameters {}".format(args))
    # Training
    args.do_train = False
    if args.do_train:
        args.seed += 3
        train_dataset = TextDataset(tokenizer, codeTokenizer, args,
                                    os.path.join(args.data_dir, args.pro + '_TRAIN.csv'))
        eval_dataset = TextDataset(tokenizer, codeTokenizer, args, os.path.join(args.data_dir, args.pro + '_TEST.csv'))
        train(args, model, train_dataset, eval_dataset)
        advTrain(args, train_dataset, eval_dataset)

    args.do_test = True
    if args.do_test:
        checkpoint_prefix = args.pro + '_checkpoint-best/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
        model = torch.load(output_dir)
        model.to(args.device)
        eval_dataset = TextDataset(tokenizer, codeTokenizer, args, os.path.join(args.data_dir, args.pro + '_TRAIN.csv'))
        model_test(args, model, eval_dataset)


if __name__ == "__main__":
    main()
