import os
from glob import glob
import numpy as np
import pandas as pd
import argparse
from collections import defaultdict as ddict, Counter
import json
from pprint import pprint
from tqdm import tqdm
from datasets import Dataset
import random

def validate_labels():

    # Read the files according to the task
    root_path   = f'{args.input_dir}/{args.task}/*'
    folders     = glob(root_path)
    label_dict  = ddict(lambda: ddict(int))
    
    all_labels  = ddict(int)

    for folder in folders:
        dataset_name = folder.split('/')[-1]
        if dataset_name == 'all':
            continue

        # Read the json files
        json_files = glob(f'{folder}/*.json')
        for json_file in tqdm(json_files):
            with open(json_file, 'r') as f:
                data = json.load(f)
                # Check if the labels are valid
                for dialogue in data:
                    for utterance in dialogue['dialogue']:
                        label_dict[dataset_name][utterance['label']] += 1
                        all_labels[utterance['label']] += 1

    lbl2idx = {}
    for idx, lbl in enumerate(all_labels.keys()):
        lbl2idx[lbl] = idx

    pprint(all_labels)
    pprint(label_dict)

    for dataset in label_dict.keys():
        for lbl in label_dict[dataset].keys():
            assert lbl in all_labels.keys(), f'{lbl} not in {dataset}'

    with open(f'{args.output_dir}/{args.task}-labels.json', 'w') as f:
        json.dump(lbl2idx, f, indent=4)
                   
def dump_json():
    root_path   = f'{args.input_dir}/{args.task}/*'
    folders     = glob(root_path)

    # Define the structure of a huggingface dataset
    '''
    Each entry in the dataset is a snippet of a dialogue, with the following fields:

    dialogue_id (str): the id of the dialogue the entry belongs to.
    dataset_name (str): the name of the dataset the entry belongs to.
    split (str): the split of the dataset the entry belongs to (train, test or valid).
    labels: the label of the current utterance for each task.
    utterance: the current utterance
    context: the previous utterances in the dialogue
    speaker: the speaker of the current utterance
    '''

    for folder in folders:
        dataset_name = folder.split('/')[-1]
        # Read the json files
        all_splits = {'train': [], 'valid': [], 'test': []}

        for split in ['train', 'valid', 'test']:
            print(dataset_name, split)
            if dataset_name == 'all':
                continue

            data     = json.load(open(f'{folder}/{split}.json'))
            curr_data = []
            for idx, dialogue in enumerate(data):
                try:
                    domain_name,  dialogue_id = dialogue['domain'], dialogue['id']
                except Exception as e:
                    domain_name = dataset_name
                    dialogue_id = idx

                context = []

                for utterance in dialogue['dialogue']:
                    temp_dict = {}
                    temp_dict['dialogue_id']        = dialogue_id
                    temp_dict['dataset_name']       = domain_name
                    #name emotion, DAP, fact, res_strategy
                    temp_dict['DA_label']           = ''
                    temp_dict['ERC_label']          = ''
                    temp_dict['fact_label']         = ''
                    temp_dict['res_label']          = ''
                    temp_dict[f'{args.task}_label'] = utterance['label']

                    temp_dict['utterance']          = utterance['text']
                    temp_dict['context']            = list(context)
                    temp_dict['speaker']            = utterance['speaker']
                    temp_dict['task']               = args.task
                    temp_dict['split']              = split

                    curr_data.append(temp_dict)
                    all_splits[split].append(temp_dict)
                    context.append((utterance["speaker"], utterance["text"]))


            with open(f'{args.output_dir}/{args.task}-{dataset_name}-{split}.json', 'w') as f:
                json.dump(curr_data, f, indent=4)

        for split in ['train', 'valid', 'test']:
            with open(f'{args.output_dir}/{args.task}-all-{split}.json', 'w') as f:
                    json.dump(all_splits[split], f, indent=4)


def dump_res():

    label_map = {
    'information-inquiry' : 'Information Inquiry',
    'source-degradation'   : 'Source Derogation',
    'selective-avoidance'  : 'Hesitance',
    'personal-choice'   : 'Personal Choice',
    'not-a-strategy'    :   'Not a resistance strategy',
    'self-pity'         : 'Self Pity',
    'self-assertion'    : 'Self Assertion',
    'counter-argumentation':    'Counter Argumentation',
    'Not a resistance strategy' : 'Not a resistance strategy',
    'Hesitance-Hesitance': 'Hesitance',
    'Self Assertion-Empowerment': 'Self Assertion',
    'Source Derogation-Contesting': 'Source Derogation',
    'Counter Argumentation-Contesting': 'Counter Argumentation',
    'Self Pity-Empowement': 'Self Pity',
    'Information Inquiry-Contesting': 'Information Inquiry',
    'Personal Choice-Empowerment':  'Personal Choice',
    'nas': 'Not a resistance strategy',
    }


    data_dir   = '/projects/NO-BACKUP/persuasionforgood-master/Face_acts/dialogue_act_prediction/resisting-persuasion/data'
    P4G_df    = pd.read_csv(f'{data_dir}/P4G.csv')
    P4G_res_labels = list(set(list(P4G_df['resistance_labels'])))

    all_splits  = {'train': [], 'valid': [], 'test': []}
    P4G_splits  = {'train': [], 'valid': [], 'test': []}
    CB_splits   = {'train': [], 'valid': [], 'test': []}

    # currently working on P4G dataset

    P4G_ids        =  list(set(list(P4G_df['B2'])))
    # shuffle the ids randomly 
    random.shuffle(P4G_ids)

    train_ids, valid_ids, test_ids = P4G_ids[:int(0.8*len(P4G_ids))], P4G_ids[int(0.8*len(P4G_ids)):int(0.9*len(P4G_ids))], P4G_ids[int(0.9*len(P4G_ids)):]
    print(len(train_ids), len(valid_ids), len(test_ids))

    P4G_split_map = {}
    for id in P4G_ids:
        if id in train_ids:
            P4G_split_map[id] = 'train'
        elif id in valid_ids:
            P4G_split_map[id] = 'valid'
        elif id in test_ids:
            P4G_split_map[id] = 'test'

    context        =  []
    curr_dial_id   =  P4G_df['B2'][0]
    temp_dict      =  {}

    for index, row in P4G_df.iterrows():
        '''
        Each entry in the dataset is a snippet of a dialogue, with the following fields:

        dialogue_id (str): the id of the dialogue the entry belongs to.
        dataset_name (str): the name of the dataset the entry belongs to.
        split (str): the split of the dataset the entry belongs to (train, test or valid).
        labels: the label of the current utterance for each task.
        utterance: the current utterance
        context: the previous utterances in the dialogue
        speaker: the speaker of the current utterance
        '''
        temp_dict = {}

        if row['B2'] != curr_dial_id:
            # dump the current dialogue
            context     = []

        curr_dial_id                    = row['B2']
        # populate the temp_dict
        temp_dict['dialogue_id']        = row['B2']
        temp_dict['dataset_name']       = 'P4G'
        #name emotion, DAP, fact, res_strategy
        temp_dict['DA_label']           = ''
        temp_dict['ERC_label']          = ''
        temp_dict['fact_label']         = ''
        temp_dict['res_label']          = ''
        temp_dict[f'{args.task}_label'] = label_map[row['resistance_labels']]

        temp_dict['utterance']          = row['Unit']
        temp_dict['context']            = list(context)
        temp_dict['speaker']            = 'Persuader' if row['B4'] ==0 else 'Persuadee'
        temp_dict['task']               = args.task
        temp_dict['split']              = P4G_split_map[row['B2']]

        if temp_dict['speaker'] == 'Persuadee':    
            P4G_splits[P4G_split_map[row['B2']]].append(temp_dict)
            all_splits[P4G_split_map[row['B2']]].append(temp_dict)

        context.append((temp_dict["speaker"], temp_dict["utterance"]))

    ## Now work on the CB dataset

    CB_df     = pd.read_csv(f'{data_dir}/CB.csv')
    CB_df.fillna('nas', inplace=True)

    # filter out the rows according to the trues and dupes columns
    CB_df   = CB_df[(CB_df['Trues'] == 'nas') & (CB_df['Dupes'] == 'nas')]
    
    # get the CB dialogue ids
    CB_ids        =  list(set(list(CB_df['Index'])))
    # shuffle the ids randomly
    random.shuffle(CB_ids)
    train_ids, valid_ids, test_ids = CB_ids[:int(0.8*len(CB_ids))], CB_ids[int(0.8*len(CB_ids)):int(0.9*len(CB_ids))], CB_ids[int(0.9*len(CB_ids)):]
    print(len(train_ids), len(valid_ids), len(test_ids))

    CB_split_map = {}
    for id in CB_ids:
        if id in train_ids:
            CB_split_map[id] = 'train'
        elif id in valid_ids:
            CB_split_map[id] = 'valid'
        elif id in test_ids:
            CB_split_map[id] = 'test'

    context        =  []
    curr_dial_id   =  CB_df['Index'][0]
    temp_dict      =  {}

    for index, row in CB_df.iterrows():
        '''
        Each entry in the dataset is a snippet of a dialogue, with the following fields:

        dialogue_id (str): the id of the dialogue the entry belongs to.
        dataset_name (str): the name of the dataset the entry belongs to.
        split (str): the split of the dataset the entry belongs to (train, test or valid).
        labels: the label of the current utterance for each task.
        utterance: the current utterance
        context: the previous utterances in the dialogue
        speaker: the speaker of the current utterance
        '''
        temp_dict = {}

        if row['Index'] != curr_dial_id:
            # dump the current dialogue
            context     = []

        curr_dial_id                    = row['Index']
        # populate the temp_dict
        temp_dict['dialogue_id']        = row['Index']
        temp_dict['dataset_name']       = 'CB'
        #name emotion, DAP, fact, res_strategy
        temp_dict['DA_label']           = ''
        temp_dict['ERC_label']          = ''
        temp_dict['fact_label']         = ''
        temp_dict['res_label']          = ''
        temp_dict[f'{args.task}_label'] = label_map[row['Resistance Labels']]

        temp_dict['utterance']          = row['Text']
        temp_dict['context']            = list(context)
        temp_dict['speaker']            = 'Buyer' if row['User'] ==0 else 'Seller'
        temp_dict['task']               = args.task
        temp_dict['split']              = CB_split_map[row['Index']]

        CB_splits[temp_dict['split']].append(temp_dict)
        all_splits[temp_dict['split']].append(temp_dict)

        context.append((temp_dict["speaker"], temp_dict["utterance"]))

    # Dump the resisting stratgies files in json format

    for split in ['train', 'valid', 'test']:
        with open(f'{args.output_dir}/{args.task}-CB-{split}.json', 'w') as f:
            json.dump(CB_splits[split], f, indent=4)
        with open(f'{args.output_dir}/{args.task}-P4G-{split}.json', 'w') as f:
            json.dump(P4G_splits[split], f, indent=4)
        with open(f'{args.output_dir}/{args.task}-all-{split}.json', 'w') as f:
            json.dump(all_splits[split], f, indent=4)

    # dump the resisting strategies labels

    res_labels = {}
    for lbl in label_map:
        if label_map[lbl] not in res_labels:
            res_labels[label_map[lbl]] = len(res_labels)
        

    with open(f'{args.output_dir}/{args.task}-labels.json', 'w') as f:
        json.dump(res_labels, f, indent=4)
    

if __name__ =='__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='/projects/NO-BACKUP/DA_DIAL/TL_DIAL/data/')
    parser.add_argument('--output_dir', type=str, default='/projects/SiCON/data')
    parser.add_argument('--task', type=str, default='DA')
    parser.add_argument('--step', type=str, required=True)

    args = parser.parse_args()

    if args.step == 'validate_labels':    
        validate_labels()
    
    elif args.step == 'dump_json':
        dump_json()
    
    elif args.step == 'dump_res':
        dump_res()

    

