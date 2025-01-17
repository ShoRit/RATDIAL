import os
from glob import glob
import numpy as np
import pandas as pd
import argparse
from collections import defaultdict as ddict, Counter
import json
from pprint import pprint
from tqdm import tqdm
import random
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, classification_report

sns.set_theme(style="darkgrid", font_scale=1.5)

def compute_dataset_statistics():
    stats_dict = ddict(lambda: ddict(list))
    dataset_dict    = {'ERC': ['meld', 'iemocap','friends','emotionpush'], 'res': ['P4G','CB']}

    for task in ['ERC', 'res']:
        for data in dataset_dict[task]:
            for split in ['train','test', 'valid']:
                curr_dataset = json.load(open(f'../data/{task}-{data}-{split}.json'))
                for elem in curr_dataset:
                    dataset = elem['dataset_name']
                    text    = elem['utterance']
                    label   = elem[f'{task}_label']
                    words   = text.split()
                    dialogue_id = elem['dialogue_id']

                    stats_dict[dataset]['num_words'].append(len(words))
                    stats_dict[dataset]['num_dialogues'].append(dialogue_id)
                    stats_dict[dataset]['num_labels'].append(label)
            
    for dataset in stats_dict:

        mean_words          = np.mean(stats_dict[dataset]['num_words'])
        tot_dialogues       = Counter(stats_dict[dataset]['num_dialogues'])
        utt_per_dialogues   = np.mean(list(tot_dialogues.values()))
        num_dialogues       = len(tot_dialogues)
        label_dist          = Counter(stats_dict[dataset]['num_labels'])

        print(f"Dataset: {dataset}")
        print(f"Mean words per utterance: {mean_words}")
        print(f"Mean utterances per dialogue: {utt_per_dialogues}")
        print(f"Number of dialogues: {num_dialogues}")
        print(f"Label distribution: {label_dist}")
        print()

    # load dataset

### Analyse the csv file generated by the wandb and use it to plot the graphs


def analyse_performance():

    df = pd.read_csv('../csv_files/wandb.csv')
    print(df.columns)
    df              = df[(df['State'] == 'finished') & (df['test_f1'].isna() == False) & (df['seed'].isin([0, 1,2]))]

    fshot_dict      = ddict(lambda : ddict(lambda : ddict(lambda : ddict(lambda: ddict(list)))))
    perf_dict       = ddict(list)

    model_map       = {'bert-base-uncased': 'BERT', 'gpt2': 'GPT2', 't5-base': 'T5', 'flan-t5-base': 'FLAN-T5'}

    for i, row in df.iterrows():
        dataset, src_dataset, tgt_dataset, test_f1, fewshot, model_name, seed, mode = row['dataset'], row['src_dataset'], row['tgt_dataset'], row['test_f1'], row['fewshot'], row['model_name'], row['seed'], row['mode']

        
        fshot_dict[model_name][mode][tgt_dataset][fewshot][seed].append((row['Created'], test_f1))

    for model_name in ['bert-base-uncased', 'gpt2','t5-base','flan-t5-base']:
        for mode in ['ID', 'TF']:
            perf_dict['mode'].append(mode)
            perf_dict['model'].append(model_map[model_name])

            for dataset in ['P4G', 'CB']:
                for fshot in [5, 10, 20, 50, 100, -1]:
                    
                    fshot_vals = []
                    for seed in [0,1,2]:
                        fshot_arr = sorted(fshot_dict[model_name][mode][dataset][fshot][seed], reverse=True)
                        try:
                            fshot_vals.append(fshot_arr[0][1])
                        except:
                            print(f'{model_name}-{mode}-{dataset}-{fshot}-{seed}')
                            
                        
                    try:
                        f1_mean = round(100* np.mean(fshot_vals),1)
                        f1_std  = round(100* np.std(fshot_vals),1)
                    except:
                        f1  = '-'
                        perf_dict[f'{dataset}-{fshot}'].append(f1)
                        continue

                    if f1_mean != f1_mean:
                        f1 = '-'
                    else:
                        f1 = f"{f1_mean}$\pm${f1_std}"
                    perf_dict[f'{dataset}-{fshot}'].append(f1)
                    

    res_df = pd.DataFrame(perf_dict)
    res_df.to_csv('../csv_files/res_fshot.csv', index=False, sep='&')

    perf_dict       = ddict(list)

    for model_name in ['bert-base-uncased', 'gpt2','t5-base','flan-t5-base']:
        for mode in ['ID', 'TF']:
            perf_dict['mode'].append(mode)
            perf_dict['model'].append(model_map[model_name])

            for dataset in ['iemocap','meld']:
                for fshot in [5, 10, 20, 50, 100, -1]:
                    
                    fshot_vals = []
                    for seed in [0,1,2]:
                        fshot_arr = sorted(fshot_dict[model_name][mode][dataset][fshot][seed], reverse=True)
                        try:
                            fshot_vals.append(fshot_arr[0][1])
                        except:
                            print(f'{model_name}-{mode}-{dataset}-{fshot}-{seed}')
                            
                    try:
                        f1_mean = round(100* np.mean(fshot_vals),1)
                        f1_std  = round(100* np.std(fshot_vals),1)
                    except:
                        f1  = '-'
                        perf_dict[f'{dataset}-{fshot}'].append(f1)
                        continue
                        
                    if f1_mean != f1_mean:
                        f1 = '-'
                    else:
                        f1 = f"{f1_mean}$\pm${f1_std}"
                    perf_dict[f'{dataset}-{fshot}'].append(f1)
                    

    emo_df = pd.DataFrame(perf_dict)
    emo_df.to_csv('../csv_files/emo_fshot.csv', index=False, sep='&')





def create_dfs():
    df = pd.read_csv('../csv_files/wandb.csv')
    print(df.columns)
    df              = df[(df['State'] == 'finished') & (df['test_f1'].isna() == False) & (df['seed'].isin([0, 1,2]))]

    fshot_dict      = ddict(lambda : ddict(lambda : ddict(lambda : ddict(lambda: ddict(list)))))
    perf_dict       = ddict(list)

    model_map       = {'bert-base-uncased': 'BERT', 'gpt2': 'GPT2', 't5-base': 'T5', 'flan-t5-base': 'FLAN-T5'}

    model_mapped    = ['BERT', 'GPT2', 'T5', 'FLAN-T5']

    for i, row in df.iterrows():
        dataset, src_dataset, tgt_dataset, test_f1, fewshot, model_name, seed, mode = row['dataset'], row['src_dataset'], row['tgt_dataset'], row['test_f1'], row['fewshot'], row['model_name'], row['seed'], row['mode']

        fshot_dict[model_name][mode][dataset][fewshot][seed].append((row['Created'], test_f1))

    for model_name in ['bert-base-uncased', 'gpt2','t5-base','flan-t5-base']:
        for mode in ['TF','ID']:
    
            for dataset in ['P4G', 'CB']:
                for fshot in [5, 10, 20, 50, 100]:
                    
                    fshot_vals = []
                    for seed in [0,1,2]:
                        fshot_arr = sorted(fshot_dict[model_name][mode][dataset][fshot][seed], reverse=True)
                        try:
                            fshot_vals.append(fshot_arr[0][1])
                        except:
                            print(f'{model_name}-{mode}-{dataset}-{fshot}-{seed}')
                    
                    for seed in [0,1,2]:    
                        perf_dict['f1'].append(fshot_vals[seed])
                        perf_dict['task'].append('Res')
                        # perf_dict['setting'].append(f'{model_map[model_name]}-{mode}')
                        perf_dict['mode'].append(mode)
                        perf_dict['model'].append(model_map[model_name])
                        if fshot == -1:
                            fshot = 500
                        perf_dict['fshot'].append(fshot)
                        perf_dict['dataset'].append(dataset)
                    
                    
    for model_name in ['bert-base-uncased', 'gpt2','t5-base','flan-t5-base']:
        for mode in ['TF', 'ID']:
            
            for dataset in ['iemocap','meld']:
                for fshot in [5, 10, 20, 50, 100]:
                    
                    fshot_vals = []
                    for seed in [0,1,2]:
                        fshot_arr = sorted(fshot_dict[model_name][mode][dataset][fshot][seed], reverse=True)
                        try:
                            fshot_vals.append(fshot_arr[0][1])
                        except:
                            print(f'{model_name}-{mode}-{dataset}-{fshot}-{seed}')
    
                    
                    for seed in [0,1,2]:    
                        perf_dict['f1'].append(fshot_vals[seed])
                        perf_dict['task'].append('ERC')
                        perf_dict['mode'].append(mode)
                        perf_dict['model'].append(model_map[model_name])
                        # perf_dict['setting'].append(f'{model_map[model_name]}-{mode}')
                        if fshot == -1:
                            fshot = 500
                        perf_dict['fshot'].append(fshot)
                        perf_dict['dataset'].append(dataset)
                

    perf_df = pd.DataFrame(perf_dict)
    fticks  = [5, 10, 20, 50, 100]
    labels  = ['5', '10', '20', '50', '100']
    markers = {"TF": "X", "ID": "o"}
    # make a seaborn line plot with error bars in a FacetGrid
    
    #### Res ####
    
    # temp_df = perf_df[(perf_df['task'] == 'Res')]
    # g = sns.FacetGrid(temp_df, col="dataset", hue='model', height=5, aspect=1.5)
    # # plot the lineplot
    # g.map_dataframe(sns.lineplot, "fshot", "f1", errorbar='sd', style='mode', hue='model', hue_order=model_mapped, style_order=['ID','TF'], markers=markers, \
    #     err_style="bars", legend='full').set(xscale="log", xticks=fticks, xticklabels=labels)
    # # add the legend
    # g.add_legend()
    # # save the figure
    # g.savefig("../figures/lineplot_RES.pdf")
    
    # # destroy the FacetGrid
    # plt.clf()
    # plt.close()
    
    # temp_df = perf_df[(perf_df['task'] == 'ERC')]
    # g = sns.FacetGrid(temp_df, col="dataset", hue='model', height=5, aspect=1.5)
    # # plot the lineplot
    # g.map_dataframe(sns.lineplot, "fshot", "f1", errorbar='sd', style='mode', hue='model', hue_order=model_mapped, style_order=['ID','TF'], markers= markers, \
    #     err_style="bars", legend='full').set(xscale="log", xticks=fticks, xticklabels=labels)
    # # add the legend
    # g.add_legend()
    # # save the figure
    # g.savefig("../figures/lineplot_ERC.pdf")
    
    # plt.clf()
    # plt.close()
    
    #### Relplot ####
    
    temp_df = perf_df[(perf_df['task'] == 'Res')]
    g = sns.relplot(data=temp_df, x="fshot", y="f1", col='dataset', hue="model", style="mode", hue_order=model_mapped, style_order=['ID','TF'], kind="line", \
        height=5, aspect=1.5,  dashes= True, legend='full', errorbar= None, facet_kws=dict(sharex=False, sharey=True)).set(xscale="log", xticks=fticks, xticklabels=labels)
    # add the legend
    # g.add_legend()
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

    # save the figure
    g.savefig("../figures/relplot_RES.pdf")
    
    # destroy the FacetGrid
    plt.clf()
    plt.close()
    
    temp_df = perf_df[(perf_df['task'] == 'ERC')]
    g = sns.relplot(data=temp_df, x="fshot", y="f1", col='dataset', hue="model", style="mode", hue_order=model_mapped, style_order=['ID','TF'], kind="line",\
        height=5, aspect=1.5, dashes= True, legend='full', errorbar= None,  facet_kws=dict(sharex=False, sharey=True)).set(xscale="log", xticks=fticks, xticklabels=labels)

    # g.add_legend()
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

    # save the figure
    g.savefig("../figures/relplot_ERC.pdf")
    
    plt.clf()
    plt.close()
    
    
def compute_chatgpt():
    
    results_dir     = f'/data/shire/projects/SiCON/chatgpt-results'
    chatgpt_dict    = ddict(lambda: ddict(list))
    results_dict    = ddict(list)
    
    for fname in ['res-P4G','res-CB', 'ERC-meld','ERC-iemocap']:
        data        = json.load(open(f'{results_dir}/{fname}-chatgpt-0-indomain.json'))
        task_name   = fname.split('-')[0]
        valid_preds = [elem.lower().strip()  for elem in json.load(open(f'../data/{task_name}-labels.json'))]
        
        for elem in data:
            tweak       = 0 
            pred_label  = elem['pred_label'].lower()
            pred_flag   = False
            
            for val_pred in valid_preds:
                if val_pred in pred_label:
                    pred_label = val_pred
                    pred_flag = True
                    break
            
            if pred_flag == False:
                tweak           = 1
                if task_name == 'res':
                    pred_label = 'not a resistance strategy'
                else:
                    pred_label = 'neutral'
            
            chatgpt_dict[fname]['pred'].append(pred_label.lower())
            chatgpt_dict[fname]['gold'].append(elem[f'{elem["task"]}_label'].lower())
            chatgpt_dict[fname]['tweaks'].append(tweak)
            
    model_map = {'bert-base-uncased': 'BERT', 'gpt2': 'GPT2', 't5-base': 'T5', 'flan-t5-base': 'FLAN-T5', 'chatgpt': 'GPT-3.5 ZS'}
    
    for fname in chatgpt_dict:
        task_name, dataset_name   = fname.split('-')
        results_dict['task'].append(task_name)
        results_dict['dataset'].append(dataset_name)
        results_dict['model'].append(model_map['chatgpt'])
        f1 =  f1_score(chatgpt_dict[fname]['gold'], chatgpt_dict[fname]['pred'], average='macro')
        results_dict['f1'].append(f1)                        
        
    
    tasks = {'res': ['P4G', 'CB'], 'ERC': ['iemocap', 'meld']}
    
    
    
    for task in tasks:
        for dataset in tasks[task]:
            for model in ['bert-base-uncased', 'gpt2','t5-base','flan-t5-base']:            
                f1_arr = []
                for seed in [0,1,2]:
                    df = pd.read_csv(f'../csv_files/{task}-ID-{dataset}-{dataset}-{model}--1.0-{seed}.csv')
                    pred = list(df['pred'])
                    gold = list(df['label'])
                    f1_arr.append(f1_score(gold, pred, average='macro'))
                    
                
                results_dict['task'].append(task)
                results_dict['dataset'].append(dataset)
                results_dict['model'].append(model_map[model])
                results_dict['f1'].append(np.mean(f1_arr))
                print(f"{task}-{dataset}-{model}-{np.mean(f1_arr)}")
    
    results_df = pd.DataFrame(results_dict)
    
    perf_df    = results_df[(results_df['task'] == 'ERC')]
    g          = sns.catplot(data=perf_df, x="dataset", y="f1", hue="model", kind="bar", hue_order=['BERT','GPT2', 'T5', 'FLAN-T5', 'GPT-3.5 ZS'], \
                height=5, aspect=1.5, legend='full', errorbar= None)
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    g.savefig("../figures/catplot_ERC.pdf")
    plt.clf()
    plt.close()
    
    
    perf_df    = results_df[(results_df['task'] == 'res')]
    g          = sns.catplot(data=perf_df, x="dataset", y="f1", hue="model", kind="bar",  hue_order=['BERT','GPT2', 'T5', 'FLAN-T5', 'GPT-3.5 ZS'],  \
        height=5, aspect=1.5,  legend='full', errorbar= None)
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    g.savefig("../figures/catplot_res.pdf")
    plt.clf()
    plt.close()
    
    g          = sns.catplot(data=results_df, x="dataset", y="f1", col='task', hue="model", kind="bar", \
        height=5, aspect=1.5,  legend='full', errorbar= None, facet_kws=dict(sharex=False, sharey=True))
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    g.savefig("../figures/catplot_all.pdf")
    plt.clf()
    plt.close()
    
    results_df.to_csv('../csv_files/indomain.csv', index=False, sep=',')


    
def compute_zeroshot():
    
    results_dir     = f'/data/shire/projects/SiCON/chatgpt-results'
    chatgpt_dict    = ddict(lambda: ddict(list))
    results_dict    = ddict(list)
    
    for fname in ['res-P4G','res-CB', 'ERC-meld','ERC-iemocap']:
        data        = json.load(open(f'{results_dir}/{fname}-chatgpt-0-indomain.json'))
        task_name   = fname.split('-')[0]
        valid_preds = [elem.lower().strip()  for elem in json.load(open(f'../data/{task_name}-labels.json'))]
        
        for elem in data:
            tweak       = 0 
            pred_label  = elem['pred_label'].lower()
            pred_flag   = False
            
            for val_pred in valid_preds:
                if val_pred in pred_label:
                    pred_label = val_pred
                    pred_flag = True
                    break
            
            if pred_flag == False:
                tweak           = 1
                if task_name == 'res':
                    pred_label = 'not a resistance strategy'
                else:
                    pred_label = 'neutral'
            
            chatgpt_dict[fname]['pred'].append(pred_label.lower())
            chatgpt_dict[fname]['gold'].append(elem[f'{elem["task"]}_label'].lower())
            chatgpt_dict[fname]['tweaks'].append(tweak)
            
    model_map = {'bert-base-uncased': 'BERT', 'gpt2': 'GPT2', 't5-base': 'T5', 'flan-t5-base': 'FLAN-T5', 'chatgpt': 'GPT-3.5 ZS'}
    
    for fname in chatgpt_dict:
        task_name, dataset_name   = fname.split('-')
        results_dict['task'].append(task_name)
        results_dict['dataset'].append(dataset_name)
        results_dict['model'].append(model_map['chatgpt'])
        f1 =  f1_score(chatgpt_dict[fname]['gold'], chatgpt_dict[fname]['pred'], average='macro')
        results_dict['f1'].append(f1)                        
        
    
    tasks = {'res': ['P4G', 'CB'], 'ERC': ['iemocap', 'meld']}
    
    rev_dataset = {'iemocap':'meld', 'meld':'iemocap', 'P4G': 'CB', 'CB':'P4G'}
    
    for task in tasks:
        for dataset in tasks[task]:
            for model in ['bert-base-uncased', 'gpt2','t5-base','flan-t5-base']:            
                f1_arr = []
                for seed in [0,1,2]:
                    df = pd.read_csv(f'../csv_files/{task}-TF-{dataset}-{rev_dataset[dataset]}-{model}-0.0-{seed}.csv')
                    pred = list(df['pred'])
                    gold = list(df['label'])
                    f1_arr.append(f1_score(gold, pred, average='macro'))
                    
                
                results_dict['task'].append(task)
                results_dict['dataset'].append(dataset)
                results_dict['model'].append(model_map[model])
                results_dict['f1'].append(np.mean(f1_arr))
                print(f"{task}-{dataset}-{model}-{np.mean(f1_arr)}")
    
    
    
    
    
    results_df = pd.DataFrame(results_dict)
    
    perf_df    = results_df[(results_df['task'] == 'ERC')]
    g          = sns.catplot(data=perf_df, x="dataset", y="f1", hue="model", kind="bar", hue_order=['BERT','GPT2', 'T5', 'FLAN-T5', 'GPT-3.5 ZS'], \
                height=5, aspect=1.5, legend='full', errorbar= None)
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    g.savefig("../figures/catplot_zshot_ERC.pdf")
    plt.clf()
    plt.close()
    
    
    perf_df    = results_df[(results_df['task'] == 'res')]
    g          = sns.catplot(data=perf_df, x="dataset", y="f1", hue="model", kind="bar",  hue_order=['BERT','GPT2', 'T5', 'FLAN-T5', 'GPT-3.5 ZS'],  \
        height=5, aspect=1.5,  legend='full', errorbar= None)
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    g.savefig("../figures/catplot_zshot_res.pdf")
    plt.clf()
    plt.close()
    
    g          = sns.catplot(data=results_df, x="dataset", y="f1", col='task', hue="model", kind="bar", \
        height=5, aspect=1.5,  legend='full', errorbar= None, facet_kws=dict(sharex=False, sharey=True))
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    g.savefig("../figures/catplot_zshot_all.pdf")
    plt.clf()
    plt.close()
    
    results_df.to_csv('../csv_files/indomain.csv', index=False, sep=',')





def label_desc():
    tasks = {'res': ['P4G', 'CB'], 'ERC': ['iemocap', 'meld']}
    label_dict = ddict(lambda: ddict(lambda: ddict(int)))
    for task in tasks:
        for dataset in tasks[task]:
            
            data = json.load(open(f'../data/{task}-{dataset}-train.json'))
            for elem in data:
                label_dict[task][dataset][elem[f'{task}_label']]+=1
                label_dict[task][dataset]['total']+=1
    
    tot_dict = ddict(list)
    
    res_mapping = {
    "Information Inquiry": "Info Inq",
    "Source Derogation": "Src Derog",
    "Hesitance": "Hesitance",
    "Personal Choice": "Pers Choice",
    "Not a resistance strategy": "Not a res strat",
    "Self Pity": "Self Pity",
    "Self Assertion": "Self Assertion",
    "Counter Argumentation": "Counter Arg",
    }
    
    for task in label_dict:
        for dataset in label_dict[task]:
            for label in label_dict[task][dataset]:
                if label !='total':
                    tot_dict['task'].append(task)
                    tot_dict['dataset'].append(dataset)
                    if task == 'res':
                        tot_dict['label'].append(res_mapping[label])
                    else:
                        tot_dict['label'].append(label)
                    tot_dict['Dist'].append(100*label_dict[task][dataset][label]/label_dict[task][dataset]['total'])
    
    tot_df = pd.DataFrame(tot_dict)
    task_df = tot_df[(tot_df['task'] == 'ERC')]
    g = sns.catplot(data=task_df, x="Dist", y="label", hue="dataset", kind="bar", \
        height=5, aspect=1.5,  legend='full', errorbar= None).set_ylabels("Emotion")
    
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    g.savefig("../figures/label_dist_ERC.pdf")
    plt.clf()
    
    
    plt.close()
    
    task_df = tot_df[(tot_df['task'] == 'res')]
    g = sns.catplot(data=task_df, x="Dist", y="label", hue="dataset", kind="bar", \
        height=5, aspect=1.5,  legend='full', errorbar= None).set_ylabels("Resisting Strategies")
    
    
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

    g.savefig("../figures/label_dist_res.pdf")
    plt.clf()
    plt.close()


# main function

if __name__ == "__main__":
    # get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",     type=str, default="ERC", help="dataset name")
    parser.add_argument("--data_dir", type=str, default="../data", help="data directory")
    parser.add_argument("--step",     type=str, required=True, help="output directory")
    args = parser.parse_args()

    if args.step == 'compute_stats':
        compute_dataset_statistics()
    
    elif args.step == 'analyse':
        analyse_performance()
    
    elif args.step == 'create_df':
        create_dfs()
    
    elif args.step == 'chatgpt':
        compute_chatgpt()

    elif args.step =='label_desc':
        label_desc()
    
    elif args.step =='zshot':
        compute_zeroshot()