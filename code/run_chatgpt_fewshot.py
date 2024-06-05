import os
import time
import json
import pandas as pd
import numpy as np
from collections import defaultdict as ddict
import random
import wandb
from pprint import pprint
import uuid
import argparse
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, AutoConfig, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq, pipeline, Conversation
from datasets import load_dataset, Dataset, load_metric, disable_caching
from sklearn.metrics import f1_score, accuracy_score
import deepspeed
import transformers
import openai
import asyncio
from typing import Any
import argparse


openai.organization = ""
openai.api_key = ""
MODEL="gpt-3.5-turbo-16k"
max_tokens = 10

disable_caching()

def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.cuda.manual_seed_all(seed)
    transformers.set_seed(seed)



def get_dialogue_instance(item, info, num_turns=5):
    
    utterance                           = item['utterance']
    speaker                             = item['speaker']
    context                             = item['context']
    
    rationales                          = item['parsed_response']

    conversational_context              = ""
    ctx_len                             = min(len(context), num_turns)
    conversational_context              = ""    

    for idx in range(0, ctx_len):
        utt                             = context[ctx_len - idx - 1]
        curr_conversational_context     = f"[{utt[0]}]:{utt[1]}\n"
        conversational_context          = f'{curr_conversational_context}{conversational_context}'

    dialog_text                         = f"[CONTEXT]\n{conversational_context}\n[RESPONSE]\n[{speaker}]:{utterance}\n"

    if item['task']                     == 'ERC':
        rationales                      = [rationales]
    
    if len(rationales) > 0:
        try:
            if info == 'intention':
                dialog_text             = f"{dialog_text}[INTENTION] {rationales[-1][0]}\n"
            elif info == 'assumption':
                dialog_text             = f"{dialog_text}[ASSUMPTION] {rationales[-1][1]}\n"
            elif info == 'implicit_info':
                dialog_text             = f"{dialog_text}[IMPLICIT INFORMATION] {rationales[-1][2]}\n"
            elif info == 'all':
                dialog_text             = f"{dialog_text}[INTENTION] {rationales[-1][0]}\n[ASSUMPTION] {rationales[-1][1]}\n[IMPLICIT INFORMATION] {rationales[-1][2]}\n"
            elif info == 'utterance':
                dialog_text             = f'{dialog_text}'
        except Exception as e:
            import pdb; pdb.set_trace()

    return dialog_text



prompt_templates = {
    "context": "Given a response for a particular speaker and recent dialogue context containing the past utterances (wherever available), output 'Yes' if the utterance contains the above strategy, otherwise output 'No'. Your output should contain only 'Yes' or 'No', and no other text.\n",
    "no_context":"Given a response for a particular speaker output 'Yes' if the utterance contains the strategy, otherwise output 'No'. "
}

dataset_strategies = {
    "friends":{
            "other": "An emotion or feeling which does not include anger, surprise, sadness, joy, fear, or disgust.",
            "anger": "Anger is characterized by antagonism toward someone or something you feel has deliberately done you wrong.",
            "surprise": "Surprsie is an emotion typically resulting from the violation of an expectation or the detection of novelty in the environment",
            "neutral": "Neutral emotion is characterized by the absence of strong feelings or emotions.",
            "sadness": "Sadness is an emotional state of unhappiness, ranging in intensity from mild to extreme and usually aroused by the loss of something that is highly valued ",
            "joy": "Joy is a feeling of extreme gladness, delight, or exultation of the spirit arising from a sense of well-being or satisfaction. ",
            "fear": "Fear is a basic, intense emotion aroused by the detection of imminent threat, involving an immediate alarm reaction that mobilizes the organism by triggering a set of physiological changes.",
            "disgust": "Disgust is characterized by strong aversion to something deemed revolting, or toward a person or behavior deemed morally repugnant."
        },
    
    "iemocap":{
        "other": "An emotion or feeling which does not include anger, surprise, sadness, joy, fear, or disgust.",
        "anger": "Anger is characterized by antagonism toward someone or something you feel has deliberately done you wrong.",
        "surprise": "Surprsie is an emotion typically resulting from the violation of an expectation or the detection of novelty in the environment",
        "neutral": "Neutral emotion is characterized by the absence of strong feelings or emotions.",
        "sadness": "Sadness is an emotional state of unhappiness, ranging in intensity from mild to extreme and usually aroused by the loss of something that is highly valued ",
        "joy": "Joy is a feeling of extreme gladness, delight, or exultation of the spirit arising from a sense of well-being or satisfaction. ",
        "fear": "Fear is a basic, intense emotion aroused by the detection of imminent threat, involving an immediate alarm reaction that mobilizes the organism by triggering a set of physiological changes.",
        "disgust": "Disgust is characterized by strong aversion to something deemed revolting, or toward a person or behavior deemed morally repugnant."
        },
    
    "P4G":{
            "Information Inquiry": "Ask for factual information about the organisation for clarification or as an attempt to stall.",
            "Source Derogation": "Attacks/doubts the organisation's credibility",
            "Hesitance": "Attempts to stall the conversation by either stating they would donate later or is currently unsure about donating.",
            "Personal Choice": "Attempts to saves face by asserting their personal preference such as their choice of charity and their choice of donation.",
            "Self Pity": "Provides a self-centred reason for not being able/willing to donate at the moment",
            "Self Assertion": "Explicitly refuses to donate without even providing a factual/personal reason",
            "Counter Argumentation": "Argues that the responsibility of donation is not on them or refutes a previous statement.",
            "Not a resistance strategy": "Does not conform to any resistance strategy"
        },
    
    "CB":{
        "Information Inquiry": "Requests for clarification or asks additional information about the item or situation.",
        "Source Derogation": "Attacks the other party or questions the item",
        "Hesitance": "Stalls for time and is hesitant to commit; specifically, they seek to further the conversation and provide a chance for the other party to make a better offer",
        "Personal Choice": "Provides a personal reason for disagreeing with the current situation or chooses to agree with the situation provided some specific condition is met.",
        "Self Pity": "Provides a reason (meant to elicit sympathy) for disagreeing with the current terms.",
        "Self Assertion": "Asserts a new claim or refutes a previous claim with an air of finality/ confidence.",
        "Counter Argumentation": "Provides a non-personal argument/factual response to refute a previous claim or to justify a new claim.",
        "Not a resistance strategy": "Does not conform to any resistance strategy"
    }    
}



def get_prompt_initial(dataset, label, num_turns=5):
    label_description = dataset_strategies[dataset][label]
    
    # prompt            = f"The given strategy is {label}. For the dataset {dataset}, the description of {label} is as follows:\n {label_description}\n"
    prompt            = f"These examples pertains to the {label} strategy. For the dataset {dataset}, the description of {label} is as follows:\n {label_description}\n."
    
    if num_turns == 0: 
        prompt       = f"{prompt}{prompt_templates['no_context']}"
    else:
        prompt       = f"{prompt}{prompt_templates['context']}"
        
    return prompt


def update_prompt_fewshot(pos_fshot_instances, inst2dict, label_now):
    
    # print(f"\nPrompt Initial: {prompt_initial}\n")
    pos_fshot_prompts, neg_fshot_prompts = [], []
    
    if args.fshot_sampling_strategy == 'random':
        num_pos_ids                 = len(pos_fshot_instances[label_now])
        pos_ids                     = []
        
        if args.fshot_samples <     num_pos_ids:
            pos_ids                 = random.sample(pos_fshot_instances[label_now], args.fshot_samples)
        else:
            pos_ids                 = random.choices(pos_fshot_instances[label_now], k= args.fshot_samples)
        
        for pos_id in pos_ids:
            pos_fshot_prompts.append(inst2dict[pos_id])
            
        if args.fshot_positive_negative:
            all_neg_ids                 = list(set(inst2dict.keys()) - set(pos_fshot_instances[label_now]))
            neg_ids                     = []
            
            if args.fshot_samples <     len(all_neg_ids):
                neg_ids                 = random.sample(all_neg_ids, args.fshot_samples)
            else:
                neg_ids                 = random.choices(all_neg_ids, k= args.fshot_samples)
            
            for neg_id in neg_ids:
                neg_fshot_prompts.append(inst2dict[neg_id])
    
        ##### Generate the prompt for the datapoint #####
        
        if args.fshot_samples > 0:
            # prompt_fshot_text = f"Below are {args.fshot_samples} positive examples for {label_now}\n"
            prompt_fshot_text = ""
            for prompt in pos_fshot_prompts:
                prompt_fshot_text += f"{prompt}\n[OUTPUT]\nYes\n\n"
            
            # prompt_fshot_text += f"\nBelow are {args.fshot_samples} negative examples for {label_now}\n"
            # prompt_fshot_text = ""
        
            for prompt in neg_fshot_prompts:
                prompt_fshot_text += f"{prompt}\n[OUTPUT]\nNo\n\n"
        
        else:
            prompt_fshot_text = ""
        
    return prompt_fshot_text



def get_fewshot_samples(dataset, task, label, fshot_samples, fshot_positive_negative):
    selected_fewshot = []
    if fshot_samples != 0:
        if fshot_positive_negative == True:
            positive_examples = []
            negative_examples = []
            for i in range(len(dataset)):
                if dataset[i][f'{task}_label'] == label:
                    positive_examples.append(dataset[i])
                else:
                    negative_examples.append(dataset[i])
            selected_fewshot = positive_examples[:fshot_samples] + negative_examples[:fshot_samples]
        else:
            selected_fewshot = dataset[:fshot_samples]
    return selected_fewshot

def get_dialogue_history_string(datapoint, dialogue_context_length):
    dialogue_history = "\nDialogue history:\n"
    context_here = datapoint['context'][-dialogue_context_length:]
    for i in range(len(context_here)):
        dialogue_history += f"{context_here[i][0]}: {context_here[i][1]}\n"
    return dialogue_history

def get_prompt_fewshotpart(label, dialogue_context_length, selected_fewshot):
    prompt_part2 = ""
    for fs_sample in selected_fewshot:
        if fs_sample['res_label'] == label:
            answer = 'Yes'
        else:
            answer = 'No'
        utterance_final_part = f"\nUtterance:\n{fs_sample['speaker']}: {fs_sample['utterance']}\n\nOutput:\n{answer}\n"
        if dialogue_context_length == 0:
            prompt_part2 += f"\n\nInput: \n" + utterance_final_part
        else:
            dialogue_history = get_dialogue_history_string(fs_sample, dialogue_context_length)
            prompt_part2 += f"\n\nInput: \n" + dialogue_history + utterance_final_part
    return prompt_part2

def get_prompt_datapoint(datapoint, label, dialogue_context_length):
    utterance_final_part = f"\nUtterance:\n{datapoint['speaker']}: {datapoint['utterance']}\n\nOutput:\n"
    if dialogue_context_length == 0:
        return f"\n\nInput: \n" + utterance_final_part
    else:
        dialogue_history = get_dialogue_history_string(datapoint, dialogue_context_length)
        return f"\n\nInput: \n" + dialogue_history + utterance_final_part
    
def sample_random_fewshot(datapoint_index, positive_examples, negative_examples, fshot_samples, fshot_positive_negative):
    if fshot_positive_negative:
        selected_positive = random.sample(list(positive_examples), fshot_samples)
        while datapoint_index in selected_positive:
            selected_positive = random.sample(list(positive_examples), fshot_samples)
        
        selected_negative = random.sample(list(negative_examples), fshot_samples)
        while datapoint_index in selected_negative:
            selected_negative = random.sample(list(negative_examples), fshot_samples)
        
        selected_positive = [positive_examples[k] for k in selected_positive]
        selected_negative = [negative_examples[k] for k in selected_negative]
        return selected_positive + selected_negative
    else:
        all_examples = {}
        all_examples.update(positive_examples)
        all_examples.update(negative_examples)
        selected_fewshot = random.sample(list(all_examples), fshot_samples)
        while datapoint_index in selected_fewshot:
            selected_fewshot = random.sample(list(all_examples), fshot_samples)
        selected_fewshot = [all_examples[k] for k in selected_fewshot]
        return selected_fewshot







### Add the huggingface dataset generation function here

def generate_huggingface_dataset(train_dataset, test_dataset):
    
    pos_fshot_instances     = ddict(list)
    inst2dict               = {}
       
    for idx, elem in enumerate(train_dataset):
        dialog_text           =  get_dialogue_instance(elem, args.info, num_turns=args.turns)
        label                 =  elem[f'{args.task}_label']
        inst2dict[elem['id']] = dialog_text
        pos_fshot_instances[label].append(elem['id'])
    
    if args.label == 'all':
        labels_to_use = list(dataset_strategies[args.dataset].keys())
    else:
        labels_to_use = [args.label]
    
    
    for idx, elem in enumerate(test_dataset):
        
        for label_now in labels_to_use:
            # print("LABEL NOW = ", label_now)
            init_prompt         = get_prompt_initial(args.dataset, label_now, args.turns)
            
            prompt_fshot_text   = update_prompt_fewshot(pos_fshot_instances, inst2dict, label_now)
            
            init_prompt         = f"{init_prompt}\n{prompt_fshot_text}"
            
            tst_prompt          = get_dialogue_instance(elem, args.info, num_turns=args.turns)
            
            # final_prompt        = f"{init_prompt}\n Test example:\n {tst_prompt}. \n\nIs the above example {label_now}? Answer only in 'Yes' or 'No'.\n "
            final_prompt        = f"{init_prompt}\n{tst_prompt}\n[OUTPUT]\n"
            
            yield_dictionary = {
                "id"                        : elem["id"],
                "dialogue_id"               : elem["dialogue_id"], 
                "dataset_name"              : elem["dataset_name"],
                "task"                      : elem['task'],
                f"{elem['task']}_label"     : elem[f"{elem['task']}_label"],
                "utterance"                 : elem["utterance"],
                "context"                   : elem["context"],
                "speaker"                   : elem["speaker"],
                "task"                      : elem["task"],
                "split"                     : elem["split"],
                "label_for_prompt"          : label_now,
                "gold_answer"               : 1 if label_now == elem[f"{elem['task']}_label"] else 0,   
                "prompt"                    : final_prompt,
            }
            
            yield yield_dictionary


def preprocess_tokenize(datapoints, tokenizer):#, max_input_length, max_target_length):
    inputs = [datapoint for datapoint in datapoints["prompt"]]
    model_inputs = tokenizer(inputs)#, max_length=max_input_length, truncation=True)
    labels = tokenizer(datapoints["answer"])#, max_length=max_target_length, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
        

def calculate_scores(results):
    dataset_labels = list(dataset_strategies[results[0]['dataset_name']].keys())
    scores = {}
    for label in dataset_labels:
        scores[label] = {"accuracy": None,
                         "f1-score": None,
                         "answers": [],
                         "predictions": [],
                        }
    for i in range(len(results)):
        result_here = results[i]
        label_here = result_here['label_for_prompt']
        answer_here = "yes" if result_here[f'{args.task}_label'] == label_here else "no"
        prediction_here = result_here['answer_generated'].lstrip().rstrip().lower()
        scores[label_here]['answers'].append(answer_here)
        scores[label_here]['predictions'].append(prediction_here)
        # results[i]['label_']

    for label in dataset_labels:
        answers_num = []
        predictions_num = []
        for each in scores[label]['answers']:
            if each == 'yes':
                answers_num.append(1)
            else:
                answers_num.append(0)
        for each in scores[label]['predictions']:
            if each == 'yes':
                predictions_num.append(1)
            else:
                predictions_num.append(0)
        scores[label]['predictions_num'] = predictions_num
        scores[label]['answers_num'] = answers_num
    
    for label in dataset_labels:
        scores[label]['total_count'] = len(scores[label]['answers'])
        scores[label]['total_correct'] = sum([1 for i in range(len(scores[label]['predictions'])) if scores[label]['predictions'][i] == scores[label]['answers'][i]])
        scores[label]['total_positive'] = scores[label]['answers'].count("yes")
        scores[label]['total_negative'] = scores[label]['answers'].count("no")
        scores[label]['predicted_positive'] = scores[label]['predictions'].count('yes')
        scores[label]['predicted_negative'] = scores[label]['predictions'].count('no')
        scores[label]['accuracy'] = accuracy_score(scores[label]['answers_num'], scores[label]['predictions_num'])
        scores[label]['f1_score'] = f1_score(scores[label]['answers_num'], scores[label]['predictions_num'], average='weighted')
        # del scores[label]['answers']
        # del scores[label]['predictions']
        # del scores[label]['answers_num']
        # del scores[label]['predictions_num']
    return scores


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task',           type=str, default='res', help='The task to run')
    parser.add_argument('--input_dir',      type=str, default='../data', help='The input directory')
    parser.add_argument('--output_dir',     type=str, default='../ICL_results', help='The input directory')
    parser.add_argument('--info',           type=str, default='all', help='Choice of information to use')
    parser.add_argument('--wandb_project',  type = str, default = 'TLCONV', help = 'The wandb project name')
    parser.add_argument('--wandb_entity',   type = str, default = 'flow-graphs-cmu', help = 'The wandb entity name')
    parser.add_argument('--dataset',        type=str, default='P4G', help='The input directory')
    parser.add_argument('--do_test',        type=int, default=1, help='Whether to test the model')
    parser.add_argument('--seed',           type=int, default=0, help='The random seed')
    parser.add_argument('--turns',          type=int, default=5, help='number of turns ')
    parser.add_argument('--gpu',            type=str, default='0', help='The gpu to use')
    parser.add_argument('--buffer_size',    type=int, default=100, help='The max buffer size')
    parser.add_argument('--label',          type=str, default='all', help='The label to use')
    parser.add_argument('--fshot_samples',  default=5, type=int, help='the number of few-shot examples to be used')
    parser.add_argument("--fshot_sampling_strategy", default='random', help="strategy to select the few-shot samples to be used -- either 'random' or 'fixed'")
    parser.add_argument("--fshot_positive_negative", action='store_true', default=True, help='whether to have an equal number of positive and negative samples provided with the prompt')
    parser.add_argument('--model_checkpoint', default='/data/datasets/models/huggingface/meta-llama/Llama-2-7b-chat-hf/', help='the huggingface model checkpoint to be used')
    parser.add_argument('--eval_batch_size', default=1, type=int, help='batch size to be used during evaluation')
    
    args = parser.parse_args()
    return args



        

if __name__ == '__main__':
    args = get_arguments()
    
    print("Arguments:\n", args)
    seed_everything(args.seed)
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    os.environ['CUDA_VISIBLE_DEVICES']= args.gpu
    device      = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    train_data  = json.load(open(f'{args.input_dir}/{args.task}-{args.dataset}-train.json'))
    dev_data    = json.load(open(f'{args.input_dir}/{args.task}-{args.dataset}-valid.json'))
    test_data   = json.load(open(f'{args.input_dir}/{args.task}-{args.dataset}-test.json'))
    
    # test_data   = test_data[:10]
    
    lbl2idx     = json.load(open(f'{args.input_dir}/{args.task}-labels.json'))
    idx2lbl     = {v:k for k,v in lbl2idx.items()} 
    lbl2desc    = json.load(open(f'{args.input_dir}/{args.task}-{args.dataset}-description-labels.json')) 
    
        
    target_dataset_hf = Dataset.from_generator(generate_huggingface_dataset, gen_kwargs={'train_dataset': train_data, 'test_dataset': test_data})
    
    
    results_file = f"../chatgpt_results/{args.task}-{args.dataset}-chatgpt-{args.info}-{args.fshot_samples}_shot-{args.seed}.json"
    # check if results file exists
    if os.path.exists(results_file):
        # load data from results file
        results_dict = json.load(open(results_file))
    else:
        # create results dict
        results_dict = []

    dialog_dataset = []
    
    completed_dataset = [f"{elem['id']}_{elem['label_for_prompt']}" for elem in results_dict]
    
    for idx, item in enumerate(target_dataset_hf):
        try:
            print(f'Done for {idx}/{len(target_dataset_hf)}', end='\r')
            if f"{item['id']}_{item['label_for_prompt']}" in completed_dataset:
                continue
            else:
                
                dialog_text = item['prompt']
                
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
                gold_label         = item[f'{args.task}_label']
                
                print(f'Gold label: {gold_label}\tLabel: {item["label_for_prompt"]}\t Pred: {pred_label}')
                results_dict.append(item)

                json.dump(results_dict, open(results_file, 'w'), indent=4)
                
                if idx % 50 ==0:
                    time.sleep(1)
                
        except Exception as e:
            print(f'Error: {e}')
            json.dump(results_dict, open(results_file, 'w'), indent=4)
            print(f'Done for {len(results_dict)}/{len(test_data)}')
            exit()
    
        
    