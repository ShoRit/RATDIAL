from collections import defaultdict
import random
from typing import Union
import json
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from pprint import pprint
# create a Dialogue Dataset class and the corresponding DataLoader

def find_label_list(task, dataset):
    lbl_dict    = json.load(open(f'../data/{task}-{dataset}-description-labels.json', 'r'))
    lbl_names   = []
    for label in lbl_dict:
        lbl_names.append(label)

    return f': {", ".join(lbl_names)}'


def create_mini_batch(samples):
    utterance_ids   = [s["input_ids"] for s in samples]
    utterance_mask  = [s["attention_mask"] for s in samples]
    labels          = [s["labels"] for s in samples]
    labels_mask     = [s["label_mask"] for s in samples]

    utterance_ids   = pad_sequence(utterance_ids, batch_first=True)
    utterance_mask  = pad_sequence(utterance_mask, batch_first=True)
    labels          = pad_sequence(labels, batch_first=True)

    if samples[0]["label_ids"] is not None:
        label_ids = torch.stack([s["label_ids"] for s in samples])
    else:
        label_ids = None

    return {
        "input_ids": utterance_ids,
        "attention_mask": utterance_mask,
        "label_ids": label_ids,
        "labels": labels,
        "label_mask": labels_mask,
        
    }


def sample_fraction(dataset, fraction, task):
    sampled_instances = random.sample(list(enumerate(dataset)), int(fraction * len(dataset)))
    sampled_indices, sampled_dataset = zip(*sampled_instances)
    return sampled_indices, sampled_dataset


def stratified_sample(dataset, n_per_class, task):
    instances_by_label = defaultdict(list)
    for index, instance in enumerate(dataset):
        lbl              = instance[f'{task}_label']
        instances_by_label[lbl].append((index, instance))

    all_sampled_instances = []
    for label, instance_list in instances_by_label.items():
        
        if n_per_class > len(instance_list):
            print(
                f"Requested more labels of class {label} ({n_per_class}) than exist ({len(instance_list)}). Using all examples."
            )
            all_sampled_instances.extend(instance_list)
        else:
            sampled_instances = random.sample(instance_list, int(n_per_class))
            all_sampled_instances.extend(sampled_instances)

    random.shuffle(all_sampled_instances)
    sampled_indices, sampled_dataset = zip(*all_sampled_instances)
    return sampled_indices, sampled_dataset

# decide the format of the dialogue dataset based on whether it is a simple encoder model or a text-generation model.

def encoder_LLM(item, tokenizer, max_len, label_name, ):
    utterance                       = item['utterance']
    speaker                         = item['speaker']
    context                         = item['context']
    ctx_len                         = len(context)
    
    conversational_context          = ""
    dialog_text                     = f"[{speaker}]:{utterance}"
    dial_encoding                   = tokenizer.encode_plus(
            dialog_text,
            add_special_tokens=True,
            max_length=max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

    dial_len                        = len(dial_encoding['input_ids'][0].nonzero()) 
    conversational_context          = ""
    upd_utt_len                     = dial_len

    for idx in range(0, ctx_len):
        utt                         = context[ctx_len - idx - 1]
        curr_conversational_context = f"[{utt[0]}]:{utt[1]}{tokenizer.sep_token}"

        curr_utt_encoding            = tokenizer.encode_plus(
            curr_conversational_context,
            add_special_tokens=True,
            max_length= max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        curr_utt_len                = len(curr_utt_encoding['input_ids'][0].nonzero()) 
        
        if upd_utt_len + curr_utt_len  >= max_len: 
            break
        else:
            upd_utt_len             = upd_utt_len + curr_utt_len
            conversational_context  = f'{curr_conversational_context}{conversational_context}'
    
    dialog_text                      = f"{conversational_context}{dialog_text}"
    dial_encoding                    = tokenizer.encode_plus(
            dialog_text,
            add_special_tokens=True,
            max_length=max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

    labels                          = tokenizer.encode_plus(
            label_name,
            add_special_tokens=True,
            max_length=max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

    return dial_encoding, dialog_text, labels
    

def instruction_LLM(item, tokenizer, task_name, max_len, label_name):

    utterance               = item['utterance']
    speaker                 = item['speaker']
    context                 = item['context']
    response_lbls           = find_label_list(task_name)

    conversational_context   = ""
    dialog_text              = f"Input:[CONTEXT]{conversational_context}[RESPONSE][{speaker}]:{utterance}[end_of_dialogue] Classify the response into one of {response_lbls}"

    dial_encoding            = tokenizer.encode_plus(
        dialog_text,
        add_special_tokens=True,
        max_length=max_len,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    dial_len                        = len(dial_encoding['input_ids'][0].nonzero())
    ctx_len                         = len(context)
    conversational_context          = ""
    upd_utt_len                     = dial_len

    for idx in range(0, ctx_len):
        utt                         = context[ctx_len - idx - 1]
        curr_conversational_context = f"[{utt[0]}]:{utt[1]}[end_of_utterance]"

        curr_utt_encoding            = tokenizer.encode_plus(
            curr_conversational_context,
            add_special_tokens=True,
            max_length= max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        curr_utt_len                = len(curr_utt_encoding['input_ids'][0].nonzero()) 
        
        if upd_utt_len + curr_utt_len  >= max_len: 
            break
        else:
            upd_utt_len             = upd_utt_len + curr_utt_len
            conversational_context  = f'{curr_conversational_context}{conversational_context}'
    
    dialog_text                       = f"Input:[CONTEXT]{conversational_context}[RESPONSE][{speaker}]:{utterance}[end_of_dialogue] Classify the response into one of {response_lbls}"

    dial_encoding                   = tokenizer.encode_plus(
        dialog_text + tokenizer.sep_token,
        add_special_tokens=True,
        max_length=max_len,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    labels                          = tokenizer.encode_plus(
        label_name + tokenizer.sep_token,
        add_special_tokens=True,
        max_length=10,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )


    return dial_encoding, dialog_text, labels


def chat_LLM(item, task_name, dataset_name, label_name, label_description):
    utterance                       = item['utterance']
    speaker                         = item['speaker']
    context                         = item['context']
    response_lbls                   = find_label_list(task_name, dataset_name)

    conversational_context          = ""
    dialog_text                     = f"Given below are the following descriptions of the labels for {task_name}.\n{label_description}\nInput:[CONTEXT]{conversational_context}[RESPONSE][{speaker}]:{utterance}[end_of_dialogue]\nClassify the response into one of {response_lbls}"
    ctx_len                         = len(context)
    conversational_context          = ""
    

    for idx in range(0, ctx_len):
        utt                         = context[ctx_len - idx - 1]
        curr_conversational_context = f"[{utt[0]}]:{utt[1]}[end_of_utterance]\n"
        conversational_context      = f'{curr_conversational_context}{conversational_context}'
    
    dialog_text                     =f"Given below are the following descriptions of the labels for {task_name}.\n{label_description}\nInput:[CONTEXT]{conversational_context}[RESPONSE][{speaker}]:{utterance}[end_of_dialogue]\nClassify the response into one of {response_lbls}"

    return dialog_text, label_name

# process the dialogue dataset with a common format during processing, that has both input as 
# the conversational context and the response which needs to be predicted.

class DialogueDataset(Dataset):
    # needs to be modified later to include the fewshot case as well

    def __init__(self, data, tokenizer, max_len, lbl2idx, task, fewshot, instruct_flag=False, turns=5):
        
        if fewshot == -1 or fewshot == 0:
            self.data = data
        elif fewshot < 1.0 and fewshot >0:
            sampled_indices, sampled_dataset = sample_fraction(data, fewshot)
            self.data = sampled_dataset
            self.sampled_indices = tuple(sampled_indices)
            self.sampled_index_hash = hash(sampled_indices)

        elif fewshot > 1.0:
            sampled_indices, sampled_dataset = stratified_sample(data, n_per_class=fewshot, task=task)
            self.data = sampled_dataset
            self.sampled_indices = tuple(sampled_indices)
            self.sampled_index_hash = hash(sampled_indices)
        else:
            raise AssertionError(
                f"Unexpected value for parameter 'fewshot': {fewshot}. Parameter should be a float in the range (0, 1.0] or an int > 0"
            )
        
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.lbl2idx = lbl2idx

        if task =='ERC':
            self.task_name = 'emotion'
        elif task == 'fact':
            self.task_name = 'face act'
        elif task == 'res':
            self.task_name = 'resisting strategy'

        self.task           = task
        self.turns          = turns
        self.fewshot        = fewshot
        self.instruct_flag  = instruct_flag


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item        = self.data[idx]
        text        = item['utterance']
        label       = self.lbl2idx[item[f'{self.task}_label']]
        label_name  = item[f'{self.task}_label']

        if self.instruct_flag == True:
            dial_encoding, dial_text, labels = instruction_LLM(item, self.tokenizer, self.task_name, self.max_len, label_name)
        else:
            dial_encoding, dial_text, labels = encoder_LLM(item, self.tokenizer, self.max_len, label_name)
        

        # present the text as Instruction followed by input and context and response
        return {
            'text':             text,
            'dialog_text':      dial_text,
            'input_ids':        dial_encoding['input_ids'].flatten(),
            'attention_mask':   dial_encoding['attention_mask'].flatten(),
            'label_ids':        torch.tensor(label, dtype=torch.long),
            'labels':           labels['input_ids'].flatten(),
            'label_mask':       labels['attention_mask'].flatten(),
        }
    
def get_data_loaders(
    train_data,
    dev_data,
    test_data,
    lbl2id,
    tokenizer,
    args,
    shuffle_train=True,
):
    train_set = DialogueDataset(
        train_data, lbl2idx= lbl2id, max_len= args.max_seq_len, tokenizer= tokenizer, task =args.task, fewshot = args.fewshot
    )
    train_loader = DataLoader(
        train_set, batch_size= args.batch_size, collate_fn=create_mini_batch, shuffle=shuffle_train
    )

    dev_set = DialogueDataset(
        dev_data, lbl2idx= lbl2id, max_len= args.max_seq_len, tokenizer= tokenizer, task = args.task, fewshot = -1,
    )
    dev_loader = DataLoader(
        dev_set, batch_size=args.batch_size, collate_fn=create_mini_batch, shuffle=False
    )

    test_set = DialogueDataset(
        test_data, lbl2idx= lbl2id, max_len= args.max_seq_len, tokenizer= tokenizer, task = args.task, fewshot = -1,
    )
    test_loader = DataLoader(
        test_set, batch_size= args.batch_size, collate_fn=create_mini_batch, shuffle=False
    )
    

    return train_loader, dev_loader, test_loader


def encoder_LLM_info(item, tokenizer, max_len, info, turns):
    utterance                           = item['utterance']
    speaker                             = item['speaker']
    context                             = item['context']
    ctx_len                             = min(len(context), turns)
    rationales                          = item['parsed_response']
    label_name                          = item[f"{item['task']}_label"]
    conversational_context              = ""
    dialog_text                         = f"[{speaker}]:{utterance}"
    
    if len(rationales) > 0:
        try:
            if info == 'intention' or info == 'topic':
                # print(info)
                # print(f"all_rationales = {rationales}")
                # print(f"rationales[-1] = {rationales[-1]}")
                # print(f"rationales[0] = {rationales[0]}")
                # print(f"rationales now = {rationales[-1][0]}")
                dialog_text             = f"{dialog_text}{tokenizer.sep_token}{rationales[0]}"
                # print(f"dialog_text = {dialog_text}")
            elif info == 'assumption' or info == 'sentiments':
                # print(info)
                # print(f"rationales = {rationales[-1][1]}")
                dialog_text             = f"{dialog_text}{tokenizer.sep_token}{rationales[1]}"
                # print(f"dialog_text = {dialog_text}")
            elif info == 'implicit_info' or info == 'sarcasm':
                # print(info)
                # print(f"rationales = {rationales[-1][2]}")
                dialog_text             = f"{dialog_text}{tokenizer.sep_token}{rationales[2]}"
                # print(f"dialog_text = {dialog_text}")
            elif info == 'all' or info == 'all_emotion':
                dialog_text             = f"{dialog_text}{tokenizer.sep_token}{rationales[0]}{tokenizer.sep_token}{rationales[1]}{tokenizer.sep_token}{rationales[2]}"
                # print(f"dialog_text = {dialog_text}")
            elif info == 'utterance':
                dialog_text             = f'{dialog_text}'
                # print(f"dialog_text = {dialog_text}")
        except Exception as e:
            # print("exception!")
            import pdb; pdb.set_trace()
        
    dial_encoding                   = tokenizer.encode_plus(
            dialog_text,
            add_special_tokens=True,
            max_length=max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

    dial_len                        = len(dial_encoding['input_ids'][0].nonzero()) 
    conversational_context          = ""
    upd_utt_len                     = dial_len

    for idx in range(0, ctx_len):
        utt                         = context[ctx_len - idx - 1]
        curr_conversational_context = f"[{utt[0]}]:{utt[1]}{tokenizer.sep_token}"

        curr_utt_encoding            = tokenizer.encode_plus(
            curr_conversational_context,
            add_special_tokens=True,
            max_length= max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        curr_utt_len                = len(curr_utt_encoding['input_ids'][0].nonzero()) 
        
        if upd_utt_len + curr_utt_len  >= max_len: 
            break
        else:
            upd_utt_len             = upd_utt_len + curr_utt_len
            conversational_context  = f'{curr_conversational_context}{conversational_context}'
    
    dialog_text                      = f"{conversational_context}{dialog_text}"
    dial_encoding                    = tokenizer.encode_plus(
            dialog_text,
            add_special_tokens=True,
            max_length=max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

    labels                          = tokenizer.encode_plus(
            label_name,
            add_special_tokens=True,
            max_length=max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )


    return dial_encoding, dialog_text, labels
    

def instruction_LLM_info(item, tokenizer, max_len, info, turns):

    utterance               = item['utterance']
    speaker                 = item['speaker']
    context                 = item['context']
    response_lbls           = find_label_list(item['task'], item['dataset_name'])
    ctx_len                 = min(len(context), turns)
    rationales              = item['parsed_response']
    label_name              = item[f"{item['task']}_label"]
    conversational_context              = ""
    dialog_text                         = f"[CONTEXT]{conversational_context}[RESPONSE][{speaker}]:{utterance}"
    
    
    if len(rationales) > 0:
        try:
            if info == 'topic':
                dialog_text             = f"{dialog_text}[TOPIC]{rationales[0]}"
            elif info == 'sentiments':
                dialog_text             = f"{dialog_text}[SENTIMENTS]{rationales[1]}"
            elif info == 'sarcasm':
                dialog_text             = f"{dialog_text}[SARCASM]{rationales[2]}"
            elif info == 'intention':
                dialog_text             = f"{dialog_text}[INTENTION]{rationales[0]}"
            elif info == 'assumption':
                dialog_text             = f"{dialog_text}[ASSUMPTION]{rationales[1]}"
            elif info == 'implicit_info':
                dialog_text             = f"{dialog_text}[IMPLICIT INFORMATION]{rationales[2]}"
            elif info == "all_emotion":
                dialog_text             = f"{dialog_text}[TOPIC]{rationales[0]}[SENTIMENTS]{rationales[1]}[SARCASM]{rationales[2]}"
            elif info == 'all':
                dialog_text             = f"{dialog_text}[INTENTION]{rationales[0]}[ASSUMPTION]{rationales[1]}[IMPLICIT INFORMATION]{rationales[2]}"
            elif info == 'utterance':
                dialog_text             = f'{dialog_text}'
        except Exception as e:
            import pdb; pdb.set_trace()

    dialog_text = f"{dialog_text}[end_of_dialogue] Classify the response into one of {response_lbls}"


    dial_encoding            = tokenizer.encode_plus(
        dialog_text,
        add_special_tokens=True,
        max_length=max_len,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    dial_len                        = len(dial_encoding['input_ids'][0].nonzero())
    upd_utt_len                     = dial_len

    for idx in range(0, ctx_len):
        utt                         = context[ctx_len - idx - 1]
        curr_conversational_context = f"[{utt[0]}]:{utt[1]}[end_of_utterance]"

        curr_utt_encoding            = tokenizer.encode_plus(
            curr_conversational_context,
            add_special_tokens=True,
            max_length= max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        curr_utt_len                = len(curr_utt_encoding['input_ids'][0].nonzero()) 
        
        if upd_utt_len + curr_utt_len  >= max_len: 
            break
        else:
            upd_utt_len             = upd_utt_len + curr_utt_len
            conversational_context  = f'{curr_conversational_context}{conversational_context}'
    
    
    dialog_text                         = f"[CONTEXT]{conversational_context}[RESPONSE][{speaker}]:{utterance}"
    
    
    if len(rationales) > 0:
        try:
            if info == 'topic':
                dialog_text             = f"{dialog_text}[TOPIC]{rationales[0]}"
            elif info == 'sentiments':
                dialog_text             = f"{dialog_text}[SENTIMENTS]{rationales[1]}"
            elif info == 'sarcasm':
                dialog_text             = f"{dialog_text}[SARCASM]{rationales[2]}"
            if info == 'intention':
                dialog_text             = f"{dialog_text}[INTENTION]{rationales[0]}"
            elif info == 'assumption':
                dialog_text             = f"{dialog_text}[ASSUMPTION]{rationales[1]}"
            elif info == 'implicit_info':
                dialog_text             = f"{dialog_text}[IMPLICIT INFORMATION]{rationales[2]}"
            elif info == 'all_emotion':
                dialog_text             = f"{dialog_text}[TOPIC]{rationales[0]}[SENTIMENTS]{rationales[1]}[SARCASM]{rationales[2]}"
            elif info == 'all':
                dialog_text             = f"{dialog_text}[INTENTION]{rationales[0]}[ASSUMPTION]{rationales[1]}[IMPLICIT INFORMATION]{rationales[2]}"
            elif info == 'utterance':
                dialog_text             = f'{dialog_text}'
        except Exception as e:
            import pdb; pdb.set_trace()

    dialog_text                       = f"{dialog_text}[end_of_dialogue] Classify the response into one of {response_lbls}"
    
    dial_encoding                   = tokenizer.encode_plus(
        dialog_text + tokenizer.sep_token,
        add_special_tokens=True,
        max_length=max_len,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    labels                          = tokenizer.encode_plus(
        label_name + tokenizer.sep_token,
        add_special_tokens=True,
        max_length=20,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )


    return dial_encoding, dialog_text, labels



class DialogueExplanationsDataset(Dataset):
    # needs to be modified later to include the fewshot case as well

    def __init__(self, data, tokenizer, max_len, lbl2idx, task, fewshot, info="utterance", instruct_flag=False, turns=5):
        
        if fewshot == -1 or fewshot == 0:
            self.data = data
        elif fewshot < 1.0 and fewshot >0:
            sampled_indices, sampled_dataset = sample_fraction(data, fewshot)
            self.data = sampled_dataset
            self.sampled_indices = tuple(sampled_indices)
            self.sampled_index_hash = hash(sampled_indices)

        elif fewshot > 1.0:
            sampled_indices, sampled_dataset = stratified_sample(data, n_per_class=fewshot, task=task)
            self.data = sampled_dataset
            self.sampled_indices = tuple(sampled_indices)
            self.sampled_index_hash = hash(sampled_indices)
        else:
            raise AssertionError(
                f"Unexpected value for parameter 'fewshot': {fewshot}. Parameter should be a float in the range (0, 1.0] or an int > 0"
            )
        
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.lbl2idx = lbl2idx

        if task =='ERC':
            self.task_name = 'emotion'
        elif task == 'fact':
            self.task_name = 'face act'
        elif task == 'res':
            self.task_name = 'resisting strategy'

        self.task           = task
        self.turns          = turns
        self.fewshot        = fewshot
        self.info           = info
        self.instruct_flag  = instruct_flag


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item        = self.data[idx]
        text        = item['utterance']
        label       = self.lbl2idx[item[f'{self.task}_label']]
        
        if self.instruct_flag == 1:
            dial_encoding, dial_text, labels = instruction_LLM_info(item, self.tokenizer, self.max_len, self.info, self.turns)
        else:
            dial_encoding, dial_text, labels = encoder_LLM_info(item, self.tokenizer, self.max_len, self.info, self.turns)
        

        # present the text as Instruction followed by input and context and response
        return {
            'text':             text,
            'dialog_text':      dial_text,
            'input_ids':        dial_encoding['input_ids'].flatten(),
            'attention_mask':   dial_encoding['attention_mask'].flatten(),
            'label_ids':        torch.tensor(label, dtype=torch.long),
            'labels':           labels['input_ids'].flatten(),
            'label_mask':       labels['attention_mask'].flatten(),
        }
    
def get_data_loaders_info(
    train_data,
    dev_data,
    test_data,
    lbl2id,
    tokenizer,
    args,
    shuffle_train=True,
):
    train_set = DialogueExplanationsDataset(
        train_data, lbl2idx= lbl2id, max_len= args.max_seq_len, tokenizer= tokenizer, task =args.task, fewshot = args.fewshot, info=args.info, turns=args.turns, instruct_flag=args.instruct_flag
    )
    train_loader = DataLoader(
        train_set, batch_size= args.batch_size, collate_fn=create_mini_batch, shuffle=shuffle_train
    )

    dev_set = DialogueExplanationsDataset(
        dev_data, lbl2idx= lbl2id, max_len= args.max_seq_len, tokenizer= tokenizer, task = args.task, fewshot = -1, info=args.info, turns=args.turns, instruct_flag=args.instruct_flag
    )
    dev_loader = DataLoader(
        dev_set, batch_size=args.batch_size, collate_fn=create_mini_batch, shuffle=False
    )

    test_set = DialogueExplanationsDataset(
        test_data, lbl2idx= lbl2id, max_len= args.max_seq_len, tokenizer= tokenizer, task = args.task, fewshot = -1, info=args.info, turns=args.turns, instruct_flag=args.instruct_flag
    )
    test_loader = DataLoader(
        test_set, batch_size= args.batch_size, collate_fn=create_mini_batch, shuffle=False
    )
    

    return train_loader, dev_loader, test_loader



