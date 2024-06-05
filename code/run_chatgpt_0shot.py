'''
A script to carry out utterance classification using gpt-3.5
'''

import os
import openai
import asyncio
from typing import Any
import argparse
import logging
import random
import sys
import copy
import wandb
import csv
import json
import time
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import sys; sys.path.append('./')
import torch
from dataloader import chat_LLM_info

openai.organization = ""
openai.api_key = ""
MODEL="gpt-3.5-turbo-16k"
max_tokens = 10

async def dispatch_openai_requests(
    messages_list: list[list[dict[str,Any]]],
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
) -> list[str]:
    """Dispatches requests to OpenAI API asynchronously.
    
    Args:
        messages_list: List of messages to be sent to OpenAI ChatCompletion API.
        model: OpenAI model to use.
        temperature: Temperature to use for the model.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use for the model.

    Returns:
        List of responses from OpenAI API.
    """
    async_responses = [
        openai.ChatCompletion.acreate(
            model=MODEL,
            messages=x,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
        )
        for x in messages_list
    ]
    return await asyncio.gather(*async_responses)


def gen_predictions(messages_list):
    predictions = asyncio.run(
        dispatch_openai_requests(
            # messages_list=[
            #     [{"role": "user", "content": "Write a poem about asynchronous execution."}],
            #     [{"role": "user", "content": "Write a poem about asynchronous pirates."}],
            # ],
            messages_list=messages_list,
            model=MODEL,
            temperature=0,
            max_tokens=10,
            top_p=1.0,
        )
    )    
    return predictions




def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
 

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


def seen_eval(model, loader, device, tokenizer, lbl2idx, args):
    model.eval()
    if args.instruct_model:
        y_true, y_pred = get_generated_labels(model, loader, device, tokenizer, lbl2idx)
    else:
        y_true, y_pred = get_labels_and_model_predictions(model, loader, device)

    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")
    return p, r, f1



def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task',       type=str, default='res', help='The task to run')
    parser.add_argument('--input_dir',  type=str, default='../data', help='The input directory')
    parser.add_argument('--info',       type=str, default='all', help='Choice of information to use')
    parser.add_argument('--wandb_project', type = str, default = 'TLCONV', help = 'The wandb project name')
    parser.add_argument('--wandb_entity', type = str, default = 'flow-graphs-cmu', help = 'The wandb entity name')
    parser.add_argument('--model_name', type=str, default='bert-base-uncased', help='The model to use')
    parser.add_argument('--dataset',    type=str, default='P4G', help='The input directory')
    parser.add_argument('--do_test',    type=int, default=1, help='Whether to test the model')
    parser.add_argument('--seed',       type=int, default=0, help='The random seed')
    parser.add_argument('--turns',      type=int, default=5, help='number of turns ')
    parser.add_argument('--gpu',        type=str, default='0', help='The gpu to use')
    parser.add_argument('--sync',       type=str, default='sync', help='Whether to use sync training')
    parser.add_argument('--buffer_size', type=int, default=100, help='The max buffer size')
    args = parser.parse_args()
    return args


if __name__ =='__main__':
    args        = get_arguments()

    seed_everything(args.seed)
    
    os.environ['CUDA_VISIBLE_DEVICES']= args.gpu
    device      = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    train_data  = json.load(open(f'{args.input_dir}/{args.task}-{args.dataset}-train.json'))
    dev_data    = json.load(open(f'{args.input_dir}/{args.task}-{args.dataset}-valid.json'))
    test_data   = json.load(open(f'{args.input_dir}/{args.task}-{args.dataset}-test.json'))
    
    lbl2idx     = json.load(open(f'{args.input_dir}/{args.task}-labels.json'))
    idx2lbl     = {v:k for k,v in lbl2idx.items()} 
    lbl2desc    = json.load(open(f'{args.input_dir}/{args.task}-{args.dataset}-description-labels.json'))
    
    desc        = []
    for lbl in lbl2desc:
        desc.append(f'{lbl}: {lbl2desc[lbl]}')
    desc_str    = "\n".join(desc)
    

    results_file = f"../chatgpt_results/{args.task}-{args.dataset}-chatgpt-{args.info}-0shot-{args.seed}.json"
    # check if results file exists
    if os.path.exists(results_file):
        # load data from results file
        results_dict = json.load(open(results_file))
    else:
        # create results dict
        results_dict = []

    dialog_dataset = []
    
    completed_dataset = [elem['id'] for elem in results_dict]
       
    if args.sync == 'async':
        for idx, item in enumerate(test_data):
            print(f'Done for {idx}', end='\r')        
            dialog_text, label_name = chat_LLM_info(item, args.task, args.dataset, item[f'{args.task}_label'], args.info , desc_str, args.turns)
            dialog_dataset.append(dialog_text)


        print('Async processing')
        num_batches         = (len(dialog_dataset) // args.buffer_size) +1
        
        for batch_idx in range(num_batches):
            messages_list       = []
            predictions_list    = []
        
            for idx in range(0, len(dialog_dataset)):
                messages_list.append([{"role": "user", "content": f"{dialog_dataset[idx]}"}])
                # small_dataset = dialog_dataset[:50]
            
        predictions = gen_predictions(messages_list)
        print(predictions)
        
        import pdb; pdb.set_trace()
    
    
    elif args.sync == 'sync':
        for idx, item in enumerate(test_data):
            try:
                print(f'Done for {idx}', end='\r')
                if item['id'] in completed_dataset:
                    continue
                else:
                    
                    
                    dialog_text, label_name = chat_LLM_info(item, args.task, args.dataset, item[f'{args.task}_label'], args.info , desc_str, args.turns)
                    

                    response = openai.ChatCompletion.create(
                        model=MODEL,
                        messages=[
                            {"role": "user", "content": f"{dialog_text}"},
                        ],
                        max_tokens=max_tokens,
                        temperature=0,
                    )
                    
                    pred_label = response["choices"][0]["message"]["content"]
                    item['pred_label'] = pred_label
                    print(f'Label: {label_name}\t Pred: {pred_label}')
                    results_dict.append(item)

                    json.dump(results_dict, open(results_file, 'w'), indent=4)
                    time.sleep(1)
                    
                    if idx % 50 ==0:
                        time.sleep(5)
                    
            except Exception as e:
                print(f'Error: {e}')
                json.dump(results_dict, open(results_file, 'w'), indent=4)
                print(f'Done for {len(results_dict)}/{len(test_data)}')
                exit()
        
            

# def asyn():
#     import pdb; pdb.set_trace()

#     response = openai.ChatCompletion.create(
#         model=MODEL,
#         messages=[
#             {"role": "user", "content": f"{dialog_text}"},
#         ],
#         temperature=0,
#     )
#     pred_label = response["choices"][0]["message"]["content"]
#     item['pred_label'] = pred_label
#     print(f'Label: {label_name}\t Pred: {pred_label}')
#     results_dict.append(item)

#     json.dump(results_dict, open(results_file, 'w'), indent=4)
#     time.sleep(6)
    
#     if idx % 10 ==0:
#         time.sleep(10)
                        
    
            