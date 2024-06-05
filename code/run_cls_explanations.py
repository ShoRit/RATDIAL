'''
A script to run a classification task using the transformers library on transfer evaluation on dialogue data.
'''
import argparse
import logging
import os
import random
import sys
import copy
import torch
import csv
import json
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from collections import defaultdict as ddict
import pandas as pd

import sys; sys.path.append('./')
from dataloader import *
# from models import *
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset#, load_metric
import wandb

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)

def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.cuda.manual_seed_all(seed)
    transformers.set_seed(seed)



def get_labels_and_model_predictions(model, loader, device):
    y_true, y_pred = [], []

    for data in tqdm(loader):
        utterance_ids       = data["input_ids"].to(device)
        utterance_mask      = data["attention_mask"].to(device)
        labels              = data["label_ids"].to(device)

        with torch.no_grad():
                output_dict = model(
                    labels = labels,
                    input_ids = utterance_ids,
                    attention_mask = utterance_mask
                )

        _, pred = torch.max(output_dict['logits'], 1)
        y_pred.extend(list(np.array(pred.cpu().detach())))
        y_true.extend(list(np.array(labels.cpu().detach())))

    return y_true, y_pred

def seen_eval(model, loader, device, tokenizer, lbl2idx, args):
    model.eval()
    if args.instruct_flag:
        y_true, y_pred = get_generated_labels(model, loader, device, tokenizer, lbl2idx)
    else:
        y_true, y_pred = get_labels_and_model_predictions(model, loader, device)

    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")
    return p, r, f1


def get_predictions(model, loader, device, tokenizer, lbl2idx, args):
    model.eval()
    
    if args.instruct_flag:
        y_true, y_pred = get_generated_labels(model, loader, device, tokenizer, lbl2idx)
        lbl2idx['UNK'] = len(lbl2idx)
    else:
        y_true, y_pred = get_labels_and_model_predictions(model, loader, device)

    idx2lbl = {v:k for k,v in lbl2idx.items()}

    y_true = [idx2lbl[i] for i in y_true]
    y_pred = [idx2lbl[i] for i in y_pred]

    return y_true, y_pred


def get_generated_labels(model, loader, device, tokenizer, lbl2idx):
    y_true, y_pred = [], []
    lbl2idx['UNK'] = len(lbl2idx)

    for data in tqdm(loader):
        utterance_ids       = data["input_ids"].to(device)
        utterance_mask      = data["attention_mask"].to(device)
        labels              = data["label_ids"].to(device)

        with torch.no_grad():
            output_sequences = model.generate(
                input_ids = utterance_ids,
                attention_mask = utterance_mask,
                max_length = 128,
                num_beams = 5,
                do_sample = False
            )
        
        preds = [tokenizer.decode(ids,  skip_special_tokens=True) for ids in output_sequences]
        for pred in preds:
            if pred in lbl2idx:
                y_pred.append(lbl2idx[pred])
            else:
                y_pred.append(lbl2idx['UNK'])

        y_true.extend(list(np.array(labels.cpu().detach())))

    return y_true, y_pred

def map_model_name(model_name):
    if 'flan-t5' in model_name:
        return f'google/{model_name}'
    return model_name



def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task',           type=str, default='ERC', help='The task to run')
    parser.add_argument('--input_dir',      type=str, default='../data/', help='The input directory')
    parser.add_argument('--info',           type=str, default='utterance', help='Choice of information to use')
    parser.add_argument('--fewshot',        type=float, default=-1, help='The fraction of data to use')
    parser.add_argument('--mode',           type=str, default='ID', help='The mode to run in')
    parser.add_argument('--src_dataset',    type=str, default='friends', help='source dataset')
    parser.add_argument('--tgt_dataset',    type=str, default='friends', help='target dataset')
    parser.add_argument('--org_fewshot',    type=float, default=-1, help='The fraction of data to use')
    parser.add_argument('--turns',          type=int,   default=5, help='Past context length')
    parser.add_argument('--wandb_project',  type = str, default = 'TLCONV', help = 'The wandb project name')
    parser.add_argument('--wandb_entity',   type = str, default = 'flow-graphs-cmu', help = 'The wandb entity name')
    parser.add_argument('--model_name',     type=str, default='bert-base-uncased', help='The model to use')
    parser.add_argument('--dataset',        type=str, default='friends', help='The input directory')
    parser.add_argument('--do_train',       type=int, default=1, help='Whether to train the model')
    parser.add_argument('--do_test',        type=int, default=1, help='Whether to test the model')
    parser.add_argument('--max_seq_len',    type=int, default=512, help='The maximum sequence length')
    parser.add_argument('--batch_size',     type=int, default=4, help='The batch size')
    parser.add_argument('--gpu',            type=str, default='0', help='The gpu to use')
    parser.add_argument('--epochs',         type=int, default=15, help='The number of training epochs')
    parser.add_argument('--seed',           type=int, default=0, help='The random seed')
    parser.add_argument('--learning_rate',  type=float, default=2e-5, help='The learning rate')
    parser.add_argument('--grad_accumulation_steps', type=int, default=1, help='The number of gradient accumulation steps')
    parser.add_argument('--patience',       type=int, default=5, help='The number of patience steps')
    parser.add_argument('--instruct_flag',  type=int, default=1, help='The model to use for the instruction encoder')

    args = parser.parse_args()
    return args

if __name__ =='__main__':
    
    args        = get_arguments()
    
    if args.mode == 'ID':
        args.src_dataset = args.dataset
        args.tgt_dataset = args.dataset
    else:
        args.dataset = args.tgt_dataset
    
    if 't5' in args.model_name:
        args.instruct_flag = 1
    else:
        args.instruct_flag = 0

    seed_everything(args.seed)
    #### WANDB INIT ####

    if args.turns == 0:
        identifiable_name = f'{args.task}-{args.mode}-{args.src_dataset}-{args.tgt_dataset}-{args.model_name}-{args.info}-{args.fewshot}-{args.seed}'
    else:
        identifiable_name = f'{args.task}-{args.mode}-{args.src_dataset}-{args.tgt_dataset}-{args.model_name}-{args.info}-{args.fewshot}-{args.turns}-{args.seed}'

    wandb.login()
    wandb.init(
        project=args.wandb_project,
        entity =args.wandb_entity,
        name   = identifiable_name,
    )
    wandb.config.update(args)
    
    print('WANDB INITIALIZED')

    os.environ['CUDA_VISIBLE_DEVICES']= args.gpu
    device      = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

 
    train_data  = json.load(open(f'{args.input_dir}/{args.task}-{args.dataset}-train.json'))
    dev_data    = json.load(open(f'{args.input_dir}/{args.task}-{args.dataset}-valid.json'))
    test_data   = json.load(open(f'{args.input_dir}/{args.task}-{args.dataset}-test.json'))
    lbl2idx     = json.load(open(f'{args.input_dir}/{args.task}-labels.json'))
    idx2lbl     = {v:k for k,v in lbl2idx.items()}


    if args.instruct_flag == 0:
        assert args.model_name in ['bert-base-uncased', 'bert-large-uncased', 'gpt2']
        tokenizer   = AutoTokenizer.from_pretrained(args.model_name)
        model       = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=len(lbl2idx))
    else:
        tokenizer   = AutoTokenizer.from_pretrained(map_model_name(args.model_name))
        model       = AutoModelForSeq2SeqLM.from_pretrained(map_model_name(args.model_name))


    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id    
    
    if tokenizer.sep_token is None:
        tokenizer.sep_token = tokenizer.eos_token
        model.config.sep_token_id = model.config.eos_token_id

    print("Loading model...")

    train_loader, dev_loader, test_loader = get_data_loaders_info(train_data, dev_data, test_data, lbl2idx, tokenizer, args)
    
    if args.turns == 0:
        loaded_checkpoint_file  = f"../ckpts/{args.task}-ID-{args.src_dataset}-{args.src_dataset}-{args.model_name}-{args.info}--1.0-{args.seed}-response.pt"
    else:
        loaded_checkpoint_file = f"../ckpts/{args.task}-ID-{args.src_dataset}-{args.src_dataset}-{args.model_name}-{args.info}--1.0-{args.turns}-{args.seed}-response.pt"
    
    if args.mode == 'TF':
        # assert the loaded_checkpoint_file exists
        assert os.path.exists(loaded_checkpoint_file)
        model.load_state_dict(torch.load(loaded_checkpoint_file))
    
    model.to(device)
    
    checkpoint_file         = f"../ckpts/{identifiable_name}-response.pt"
    best_model              = model

    if args.fewshot == 0:
        model.load_state_dict(torch.load(loaded_checkpoint_file))
        model.to(device)
        model.eval()
        pt, rt, test_f1 = seen_eval(model, test_loader, device=device, args=args, lbl2idx=lbl2idx, tokenizer=tokenizer)
        wandb.log({"test_f1": test_f1, "test_precision": pt, "test_recall": rt})
        wandb.run.finish()
        
        # model, loader, device, tokenizer, lbl2idx, idx2lbl, args
        dev_labels, dev_preds   = get_predictions(model, dev_loader, device=device, tokenizer=tokenizer, lbl2idx=lbl2idx,  args=args)
        test_labels, test_preds = get_predictions(model, test_loader, device=device, tokenizer=tokenizer, lbl2idx=lbl2idx,  args=args)

        dev_dict, test_dict = ddict(list), ddict(list)

        for i, (dev_pred, dev_label, dev_inst) in enumerate(zip(dev_preds, dev_labels, dev_data)):
            dev_dict['pred'].append(dev_pred)
            dev_dict['label'].append(dev_label)
            dev_dict['text'].append(dev_inst['utterance'])
            assert dev_label == dev_inst[f'{args.task}_label']
                

        ## carry out the same for the test set
        for i, (test_pred, test_label, test_inst) in enumerate(zip(test_preds, test_labels, test_data)):
            test_dict['pred'].append(test_pred)
            test_dict['label'].append(test_label)
            test_dict['text'].append(test_inst['utterance'])
            assert test_label == test_inst[f'{args.task}_label']
        
        dev_df = pd.DataFrame(dev_dict)
        test_df = pd.DataFrame(test_dict)

        dev_df.to_csv(f'../csv_files/{identifiable_name}-dev.csv')
        test_df.to_csv(f'../csv_files/{identifiable_name}-test.csv')
        
        exit()

    if args.do_train == 1:

        #######################################
        # TRAINING LOOP                       #
        #######################################
        print("Setting up training loop...")
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        best_p, best_r, best_f1 = 0, 0, 0
        kill_cnt = 0

        for epoch in range(args.epochs):
            print(f"============== TRAINING ON EPOCH {epoch} ==============")
            running_loss = 0.0
            model.train()

            for i, data in enumerate(tqdm(train_loader)):
                # load the data from the data loader
                input_ids           = data["input_ids"].to(device)
                attention_mask      = data["attention_mask"].to(device)    
                labels              = data["label_ids"].to(device)

                optimizer.zero_grad()
                
                if args.instruct_flag == 0:
                    labels              = data["label_ids"].to(device)
                    output_dict = model(
                    labels = labels,
                    input_ids = input_ids,
                    attention_mask = attention_mask
                )

                else:
                    labels              = data["labels"].to(device)
                    labels[labels[:, :] == tokenizer.pad_token_id] = -100
                    output_dict = model(
                    labels = labels,
                    input_ids = input_ids,
                    attention_mask = attention_mask,
                )

                # forward pass
                
                loss = output_dict['loss']
                loss.backward()
                running_loss += loss.item()
                if (i + 1) % args.grad_accumulation_steps == 0:
                    optimizer.step()

            # final gradient step if we hadn't handled it already
            if (i + 1) % args.grad_accumulation_steps != 0:
                optimizer.step()
                wandb.log({"batch_loss": loss.item()})

            wandb.log({"loss": running_loss})

            print("============== EVALUATION ==============")

            p_dev, r_dev, f1_dev = seen_eval(model, dev_loader, device=device, args=args, lbl2idx=lbl2idx, tokenizer=tokenizer)
            wandb.log({"dev_f1": f1_dev})
            print(f"Eval data F1: {f1_dev} \t Precision: {p_dev} \t Recall: {r_dev}")

            if f1_dev > best_f1:
                kill_cnt = 0
                best_p, best_r, best_f1 = p_dev, r_dev, f1_dev
                best_model = model
                torch.save(best_model.state_dict(), checkpoint_file)
            else:
                kill_cnt += 1
                if kill_cnt >= args.patience:
                    torch.save(best_model.state_dict(), checkpoint_file)
                    break
            
            wandb.log({"running_best_f1": best_f1})
            print(f"[best val] precision: {best_p:.4f}, recall: {best_r:.4f}, f1 score: {best_f1:.4f}")

        wandb.log({"best_f1": best_f1, "best_precision": best_p, "best_recall": best_r})
        torch.save(best_model.state_dict(), checkpoint_file)


        print("============== EVALUATION ON TEST DATA ==============")
        best_model.to(device)
        best_model.eval()
        pt, rt, test_f1 = seen_eval(best_model, test_loader, device=device, args=args, lbl2idx=lbl2idx, tokenizer=tokenizer)
        wandb.log({"test_f1": test_f1, "test_precision": pt, "test_recall": rt})
        wandb.run.finish()

    if args.do_test == 1:
        print("============== EVALUATION ON TEST DATA ==============")
        model.load_state_dict(torch.load(checkpoint_file))
        model.to(device)
        model.eval()
        pt, rt, test_f1 = seen_eval(model, test_loader, device=device, args=args, lbl2idx=lbl2idx, tokenizer=tokenizer)
        print({"test_f1": test_f1, "test_precision": pt, "test_recall": rt, "dataset": args.dataset, "model_name": args.model_name, "fewshot": args.fewshot, "seed": args.seed, "info": args.info})
        
        
        
        dev_labels, dev_preds   = get_predictions(model, dev_loader, device=device, tokenizer=tokenizer, lbl2idx=lbl2idx,  args=args)
        test_labels, test_preds = get_predictions(model, test_loader, device=device, tokenizer=tokenizer, lbl2idx=lbl2idx,  args=args)

        dev_dict, test_dict = ddict(list), ddict(list)

        for i, (dev_pred, dev_label, dev_inst) in enumerate(zip(dev_preds, dev_labels, dev_data)):
            dev_dict['pred'].append(dev_pred)
            dev_dict['label'].append(dev_label)
            dev_dict['text'].append(dev_inst['utterance'])
            assert dev_label == dev_inst[f'{args.task}_label']
                

        ## carry out the same for the test set
        for i, (test_pred, test_label, test_inst) in enumerate(zip(test_preds, test_labels, test_data)):
            test_dict['pred'].append(test_pred)
            test_dict['label'].append(test_label)
            test_dict['text'].append(test_inst['utterance'])
            assert test_label == test_inst[f'{args.task}_label']
        
        dev_df = pd.DataFrame(dev_dict)
        test_df = pd.DataFrame(test_dict)

        dev_df.to_csv(f'../csv_files/{identifiable_name}-dev.csv')
        test_df.to_csv(f'../csv_files/{identifiable_name}-test.csv')
    
    
    
    
    
        
        
        
