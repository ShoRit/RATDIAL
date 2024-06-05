import json
import os
from collections import defaultdict as ddict
from tqdm import tqdm
from pprint import pprint
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.metrics import f1_score
import math
import textwrap
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
from sklearn.metrics import f1_score, classification_report, confusion_matrix

sns.set_style('darkgrid')

# convert a jsonl file to a json file

def convert_jsonl_to_json(jsonl_file, json_file):
    with open(jsonl_file, 'r') as f:
        lines = f.readlines()
    
    data = []
    for line in lines:
        data.append(json.loads(line))
        
    # dump json data
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=4)


## analysis of the parsed responses for the dataset

def parse_response_distribution():
    dist_dict       = ddict(lambda: ddict(int))
    speaker_dict    = ddict(lambda: ddict(int))
    
    task, dataset = 'res', 'CB'
    
    for split in ['train', 'valid', 'test']:
        
        try:
            curr_data       = json.load(open(f'../data/{task}-{dataset}-{split}-explained.json', 'r'))
        except:
            convert_jsonl_to_json(f'../data/{task}-{dataset}-{split}-explained.jsonl', f'../data/{task}-{dataset}-{split}-explained.json')
            curr_data       = json.load(open(f'../data/{task}-{dataset}-{split}-explained.json', 'r'))
            
        updated_data    = []
        
        for elem in tqdm(curr_data):
            ctx_len             = len(elem['context'])
            exp_len             = len(elem['parsed_response'])
            speaker             = elem['speaker'].lower()
                    
            if exp_len == 0:
                dist_dict[split]['empty']   += 1
                elem['parsed_response']     = []
            
            else:
                
                last_turn = elem['parsed_response'][-1]
                ## observe whether the speaker is in the explanation
                if speaker not in last_turn[0][:15].lower():    
                    dist_dict[split][f'speaker_False'] += 1
                    elem['parsed_response']             = []
                    
                else:
                    
                    info_len = len(last_turn)
                    if info_len == 3:    
                        if ctx_len +1 == exp_len:
                            dist_dict[split][f'speaker_True_ctx_equal'] += 1    
                        elif ctx_len +1 > exp_len:
                            dist_dict[split][f'speaker_True_ctx_greater'] += 1
                        elif ctx_len +1 < exp_len:
                            dist_dict[split][f'speaker_True_ctx_less'] += 1
                    else:
                        dist_dict[split][f'speaker_True_insuff_data'] += 1
                        elem['parsed_response'] = []
                
            updated_data.append(elem)

        with open(f'../data/{task}-{dataset}-{split}-explained-updated.json', 'w') as f:
            json.dump(updated_data, f, indent=4)
        
    pprint(dist_dict)


### code to generate the catplots and relplots for the ERC and the RES tasks
def analyse_exp_results():
    
    import wandb
    api = wandb.Api()
    
    runs = api.runs(path="flow-graphs-cmu/TLCONV")

    exp_dict = ddict(list)
    summary_list, config_list, name_list = [], [], []
    for run in tqdm(runs):
        try:
            # .summary contains output keys/values for
            # metrics such as accuracy.
            #  We call ._json_dict to omit large files
            exp_dict['test_f1'].append(run.summary._json_dict['test_f1'])
            # summary_list.append(run.summary._json_dict)

            exp_dict['model_name'].append(run.config['model_name'])
            exp_dict['tgt_dataset'].append(run.config['tgt_dataset'])
            exp_dict['info'].append(run.config['info'])
            exp_dict['mode'].append(run.config['mode'])
            exp_dict['fewshot'].append(run.config['fewshot'])
            exp_dict['seed'].append(run.config['seed'])
            exp_dict['turns'].append(run.config['turns'])
            exp_dict['task'].append(run.config['task'])
            # .config contains the hyperparameters.
            #  We remove special values that start with _.
            # config_list.append(
            #     {k: v for k,v in run.config.items()
            #     if not k.startswith('_')})

            # .name is the human-readable name of the run.
            # name_list.append(run.name)
        except Exception as e:
            print(run.summary._json_dict, run.config )
            continue
    
    df = pd.DataFrame(exp_dict)
    print(len(df))                        
    
    df     = df.replace({'model_name': {'bert-base-uncased': 'BERT', 't5-base': 'T5', 'gpt2': 'GPT2'}})
    # choose model_name amongst the ['BERT', 'GPT2', 'T5']
    df     = df[df['model_name'].isin(['BERT', 'GPT2', 'T5'])]
    
    ### all_lineplot_df
    
    perf_df = df[df['turns']==5]
    # display the TF results for all shot
    sns.set(style="darkgrid")
    sns.set(font_scale=2.0)
    
    fticks  = [2, 5, 10, 20, 50, 100, 500]
    labels  = ['0', '5', '10', '20', '50', '100', 'all']
    
    perf_df = perf_df.replace({'fewshot': {-1: 500, 0: 2}})
    # replace the name of the column from info to rationale
    perf_df = perf_df.rename(columns={'info': 'rationale', 
                                      'model_name': 'model', 
                                      'tgt_dataset': 'dataset'})
        
    g = sns.relplot(data=perf_df, x="fewshot", y="test_f1", col='dataset', hue="rationale", style="mode", row = 'model',\
        hue_order=['utterance', 'intention', 'assumption', 'implicit_info' ,'all'], style_order=['ID','TF'], kind="line", \
        height=5, aspect=1.5,  dashes= True, errorbar= None, legend='full', lw=3, facet_kws=dict(sharex=False, sharey=True)\
        ).set(xscale="log", xticks=fticks, xticklabels=labels)
    
    sns.move_legend(g, "lower center", bbox_to_anchor=(0.5, 1.0), ncol=9, title=None, frameon=False)
        
    g.savefig("../figures/lineplot_all_info.pdf")
    plt.clf()
    plt.close()
    
    
    g = sns.relplot(data=perf_df[perf_df['rationale'].isin(['utterance', 'intention', 'all'])], x="fewshot", y="test_f1", col='dataset',\
        hue="rationale", style="mode", row = 'model',\
        hue_order=['utterance', 'intention', 'all'], style_order=['ID','TF'], kind="line", \
        height=5, aspect=1.5,  dashes= True, errorbar= None, legend='full', lw=3, facet_kws=dict(sharex=False, sharey=True)\
        ).set(xscale="log", xticks=fticks, xticklabels=labels)
    
    sns.move_legend(g, "lower center", bbox_to_anchor=(0.5, 1.0), ncol=7, title=None, frameon=False)
        
    g.savefig("../figures/lineplot_info.pdf")
    plt.clf()
    plt.close()
    
    #########################################
    
    ## observe for the res tasks
    perf_df = df[(df['fewshot'] != 0) & (df['task']=='res') & (df['turns']==5)]
    zero_df = df[(df['fewshot'] == 0) & (df['task']=='res') & (df['turns']==5)]
    TF_df   = df[(df['mode'] == 'TF')]
    ID_df   = perf_df[perf_df['mode'] == 'ID']
    
    # display the TF results for 0 shot
    sns.set(style="darkgrid")
    sns.set(font_scale=1.5)
    
    all_ID_df = ID_df[(ID_df['fewshot']==-1)]
    
    g     = sns.catplot(data=all_ID_df, x="model_name", y="test_f1", hue="info", kind="bar", col='tgt_dataset',\
        hue_order= ['utterance', 'intention', 'assumption', 'implicit_info' ,'all'], height=5, aspect=1.5, \
        legend='full')
    g.savefig("../figures/catplot_ID_all_res_info.png")
    plt.clf()
    plt.close()
    
    g     = sns.catplot(data=TF_df[TF_df['fewshot']==-1], x="model_name", y="test_f1", hue="info", kind="bar", col='tgt_dataset',\
        hue_order= ['utterance', 'intention', 'assumption', 'implicit_info' ,'all'], height=5, aspect=1.5, \
        legend='full')
    g.savefig("../figures/catplot_TF_res_info.png")
    plt.clf()
    plt.close()
    
    g     = sns.catplot(data=zero_df, x="model_name", y="test_f1", hue="info", kind="bar", col='tgt_dataset',\
        hue_order= ['utterance', 'intention', 'assumption', 'implicit_info' ,'all'], height=5, aspect=1.5, \
        legend='top')
    g.savefig("../figures/catplot_ZS_res_info.png")
    plt.clf()
    plt.close()
    
    # display the ID results across different shots
    sns.set(style="darkgrid")
    sns.set(font_scale=1.5)
    
    fticks  = [5, 10, 20, 50, 100, 500]
    labels  = ['5', '10', '20', '50', '100', 'all']
    
    ID_df   = ID_df.replace({'fewshot': {-1: 500}})
    perf_df = perf_df.replace({'fewshot': {-1: 500}})
        
    g = sns.relplot(data=perf_df, x="fewshot", y="test_f1", col='tgt_dataset', hue="info", style="mode", row = 'model_name',\
        hue_order=['utterance', 'intention', 'assumption', 'implicit_info' ,'all'], style_order=['ID','TF'], kind="line", \
        height=5, aspect=1.5,  dashes= True, legend='full', errorbar= None, facet_kws=dict(sharex=False, sharey=True)).set(xscale="log", xticks=fticks, xticklabels=labels)
    
    # add the legend
    # g.add_legend()
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    g.savefig("../figures/lineplot_res_info_all.png")
    plt.clf()
    plt.close()
    
    
    g = sns.relplot(data=perf_df[(perf_df['info'] != 'implicit_info') & (perf_df['info']!='assumption')], x="fewshot", y="test_f1", col='tgt_dataset', hue="info", style="mode", row = 'model_name',\
        hue_order=['utterance', 'intention','all'], style_order=['ID','TF'], kind="line", \
        height=5, aspect=1.5,  dashes= True, legend='full', errorbar= None, facet_kws=dict(sharex=False, sharey=True)).set(xscale="log", xticks=fticks, xticklabels=labels)
    
    # add the legend
    # g.add_legend()
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    g.savefig("../figures/lineplot_res_info.png")
    plt.clf()
    plt.close()
    
    
    ### observe for the ERC tasks
    
    perf_df = df[(df['fewshot'] != 0) & (df['task']=='ERC') & (df['turns']==5)]
    zero_df = df[(df['fewshot'] == 0) & (df['task']=='ERC') & (df['turns']==5)]
    TF_df   = df[(df['mode'] == 'TF')]
    ID_df   = perf_df[perf_df['mode'] == 'ID']
    
    # display the TF results for 0 shot
    sns.set(style="darkgrid")
    sns.set(font_scale=1.5)
    
    all_ID_df = ID_df[(ID_df['fewshot']==-1)]
    g     = sns.catplot(data=all_ID_df, x="model_name", y="test_f1", hue="info", kind="bar", col='tgt_dataset',\
        hue_order= ['utterance', 'intention', 'assumption', 'implicit_info' ,'all'], height=5, aspect=1.5, \
        legend='full')
    g.savefig("../figures/catplot_ID_all_ERC_info.png")
    plt.clf()
    plt.close()
    
    g     = sns.catplot(data=TF_df[TF_df['fewshot']==-1], x="model_name", y="test_f1", hue="info", kind="bar", col='tgt_dataset',\
        hue_order= ['utterance', 'intention', 'assumption', 'implicit_info' ,'all'], height=5, aspect=1.5, \
        legend='full')
    g.savefig("../figures/catplot_TF_ERC_info.png")
    plt.clf()
    plt.close()
    
    g     = sns.catplot(data=zero_df, x="model_name", y="test_f1", hue="info", kind="bar", col='tgt_dataset',\
        hue_order= ['utterance', 'intention', 'assumption', 'implicit_info' ,'all'], height=5, aspect=1.5, \
        legend='bottom')
    g.savefig("../figures/catplot_ZS_ERC_info.png")
    plt.clf()
    plt.close()
    
    
    # display the ID results across different shots
    sns.set(style="darkgrid")
    sns.set(font_scale=1.5)
    
    fticks  = [5, 10, 20, 50, 100, 500]
    labels  = ['5', '10', '20', '50', '100', 'all']
    
    ID_df   = ID_df.replace({'fewshot': {-1: 500}})
    perf_df = perf_df.replace({'fewshot': {-1: 500}})
        
    g = sns.relplot(data=perf_df, x="fewshot", y="test_f1", col='tgt_dataset', hue="info", style="mode", row = 'model_name',\
        hue_order=['utterance', 'intention', 'assumption', 'implicit_info' ,'all'], style_order=['ID','TF'], kind="line", \
        height=5, aspect=1.5,  dashes= True, legend='full', errorbar= None, facet_kws=dict(sharex=False, sharey=True)).set(xscale="log", xticks=fticks, xticklabels=labels)
    
    # add the legend
    # g.add_legend()
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    g.savefig("../figures/lineplot_ERC_info_all.png")
    plt.clf()
    plt.close()
    
    
    g = sns.relplot(data=perf_df[(perf_df['info'] != 'implicit_info') & (perf_df['info']!='assumption')], x="fewshot", y="test_f1", col='tgt_dataset', hue="info", style="mode", row = 'model_name',\
        hue_order=['utterance', 'intention','all'], style_order=['ID','TF'], kind="line", \
        height=5, aspect=1.5,  dashes= True, legend='full', errorbar= None, facet_kws=dict(sharex=False, sharey=True)).set(xscale="log", xticks=fticks, xticklabels=labels)
    
    # add the legend
    # g.add_legend()
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    g.savefig("../figures/lineplot_ERC_info.png")
    plt.clf()
    plt.close()
    
    
    zero_df     = df[(df['fewshot'] == 0) & (df['turns']==5) & (df['info'].isin(['utterance', 'intention', 'all']))]
 
    g     = sns.catplot(data=zero_df, x="model_name", y="test_f1", hue="info", kind="bar", col='tgt_dataset',\
        hue_order= ['utterance', 'intention','all'], height=5, aspect=1.5, legend='center')
    g.savefig("../figures/catplot_ZS_info.pdf")
    plt.clf()
    plt.close()
    
    zero_df     = df[(df['fewshot'] == 0) & (df['turns']==5)]
 
    g     = sns.catplot(data=zero_df, x="model_name", y="test_f1", hue="info", kind="bar", col='tgt_dataset',\
        hue_order= ['utterance', 'intention', 'assumption', 'implicit_info' ,'all'], height=5, aspect=1.5, legend='center')
    
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='center', borderaxespad=0)
    g.savefig("../figures/catplot_ZS_all_info.pdf")
    plt.clf()
    plt.close()
    

   
### code to generate the catplots and relplots for the RES tasks in the judgement setting
def analyse_judgement_results():
    
    import wandb
    api = wandb.Api()
    
    runs = api.runs(path="kellyshiiii/TLDIAL-judgement")

    exp_dict = ddict(list)
    summary_list, config_list, name_list = [], [], []
    for run in tqdm(runs):
        try:
            # .summary contains output keys/values for
            # metrics such as accuracy.
            #  We call ._json_dict to omit large files
            exp_dict['test_f1'].append(run.summary._json_dict['test_f1'])
            # summary_list.append(run.summary._json_dict)

            exp_dict['model_name'].append(run.config['model_name'])
            exp_dict['tgt_dataset'].append(run.config['tgt_dataset'])
            exp_dict['info'].append(run.config['info'])
            exp_dict['mode'].append(run.config['mode'])
            exp_dict['fewshot'].append(run.config['fewshot'])
            exp_dict['seed'].append(run.config['seed'])
            exp_dict['turns'].append(run.config['turns'])
            exp_dict['task'].append(run.config['task'])
            # .config contains the hyperparameters.
            #  We remove special values that start with _.
            # config_list.append(
            #     {k: v for k,v in run.config.items()
            #     if not k.startswith('_')})

            # .name is the human-readable name of the run.
            # name_list.append(run.name)
        except Exception as e:
            print(run.summary._json_dict, run.config )
            continue
    
    df = pd.DataFrame(exp_dict)
    print(len(df))                        
    
    
    df = df.replace({'info': {'cognitive':'inclination', 
                              'affective': 'obligation',
                              'motivational': 'capacity'}})
    
    ## observe for the res tasks
    zero_df = df[(df['fewshot'] == 0) & (df['task']=='res') & (df['turns']==5)]
    perf_df = df[(df['fewshot'] != 0) & (df['task']=='res') & (df['turns']==5)]
    TF_df   = df[(df['mode'] == 'TF')]
    ID_df   = perf_df[perf_df['mode'] == 'ID']
    
    # display the TF results for 0 shot
    sns.set(style="darkgrid")
    sns.set(font_scale=1.5)
    
    all_ID_df = ID_df[(ID_df['fewshot']==-1)]
    g     = sns.catplot(data=all_ID_df, x="model_name", y="test_f1", hue="info", kind="bar", col='tgt_dataset',\
        hue_order= ['utterance', 'inclination', 'obligation', 'capacity' ,'all'], height=5, aspect=1.5, \
        legend='full')
    g.savefig("../figures/catplot_ID_res_judgment.pdf")
    plt.clf()
    plt.close()
    
    g     = sns.catplot(data=TF_df[TF_df['fewshot']==-1], x="model_name", y="test_f1", hue="info", kind="bar", col='tgt_dataset',\
        hue_order= ['utterance', 'inclination', 'obligation', 'capacity' ,'all'], height=5, aspect=1.5, \
        legend='full')
    g.savefig("../figures/catplot_TF_res_judgment.pdf")
    plt.clf()
    plt.close()
    
    g     = sns.catplot(data=zero_df, x="model_name", y="test_f1", hue="info", kind="bar", col='tgt_dataset',\
        hue_order= ['utterance', 'inclination', 'obligation', 'capacity' ,'all'], height=5, aspect=1.5, \
        legend='full')
    g.savefig("../figures/catplot_ZS_res_judgment.pdf")
    plt.clf()
    plt.close()
    
    
    
    
    # display the ID results across different shots
    sns.set(style="darkgrid")
    sns.set(font_scale=1.5)
    
    fticks  = [5, 10, 20, 50, 100, 500]
    labels  = ['5', '10', '20', '50', '100', 'all']
    
    ID_df   = ID_df.replace({'fewshot': {-1: 500}})
    perf_df = perf_df.replace({'fewshot': {-1: 500}})
        
    g = sns.relplot(data=perf_df, x="fewshot", y="test_f1", col='tgt_dataset', hue="info", style="mode", row = 'model_name',\
        hue_order=['utterance', 'inclination', 'obligation', 'capacity' ,'all'], style_order=['ID','TF'], kind="line", \
        height=5, aspect=1.5,  dashes= True, legend='full', errorbar= None, facet_kws=dict(sharex=False, sharey=True)).set(xscale="log", xticks=fticks, xticklabels=labels)
    
    # add the legend
    # g.add_legend()
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    g.savefig("../figures/lineplot_res_judgment_all.pdf")
    plt.clf()
    plt.close()
    
    
    # ### observe for the ERC tasks
    
    # perf_df = df[(df['fewshot'] != 0) & (df['task']=='ERC') & (df['turns']==5)]
    # TF_df   = df[(df['mode'] == 'TF')]
    # ID_df   = perf_df[perf_df['mode'] == 'ID']
    
    # # display the TF results for 0 shot
    # sns.set(style="darkgrid")
    # sns.set(font_scale=1.5)
    
    # all_ID_df = ID_df[(ID_df['fewshot']==-1)]
    # g     = sns.catplot(data=all_ID_df, x="model_name", y="test_f1", hue="info", kind="bar", col='tgt_dataset',\
    #     hue_order= ['utterance', 'intention', 'assumption', 'implicit_info' ,'all'], height=5, aspect=1.5, \
    #     legend='full')
    # g.savefig("../figures/catplot_ID_all_ERC_info.pdf")
    # plt.clf()
    # plt.close()
    
    # g     = sns.catplot(data=TF_df[TF_df['fewshot']==-1], x="model_name", y="test_f1", hue="info", kind="bar", col='tgt_dataset',\
    #     hue_order= ['utterance', 'intention', 'assumption', 'implicit_info' ,'all'], height=5, aspect=1.5, \
    #     legend='full')
    # g.savefig("../figures/catplot_TF_ERC_info.pdf")
    # plt.clf()
    # plt.close()
    
    # # display the ID results across different shots
    # sns.set(style="darkgrid")
    # sns.set(font_scale=1.5)
    
    # fticks  = [5, 10, 20, 50, 100, 500]
    # labels  = ['5', '10', '20', '50', '100', 'all']
    
    # ID_df   = ID_df.replace({'fewshot': {-1: 500}})
    # perf_df = perf_df.replace({'fewshot': {-1: 500}})
        
    # g = sns.relplot(data=perf_df, x="fewshot", y="test_f1", col='tgt_dataset', hue="info", style="mode", row = 'model_name',\
    #     hue_order=['utterance', 'intention', 'assumption', 'implicit_info' ,'all'], style_order=['ID','TF'], kind="line", \
    #     height=5, aspect=1.5,  dashes= True, legend='full', errorbar= None, facet_kws=dict(sharex=False, sharey=True)).set(xscale="log", xticks=fticks, xticklabels=labels)
    
    # # add the legend
    # # g.add_legend()
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    # g.savefig("../figures/lineplot_ERC_info_all.pdf")
    # plt.clf()
    # plt.close()
    
    
    # g = sns.relplot(data=perf_df[(perf_df['info'] != 'implicit_info') & (perf_df['info']!='assumption')], x="fewshot", y="test_f1", col='tgt_dataset', hue="info", style="mode", row = 'model_name',\
    #     hue_order=['utterance', 'intention','all'], style_order=['ID','TF'], kind="line", \
    #     height=5, aspect=1.5,  dashes= True, legend='full', errorbar= None, facet_kws=dict(sharex=False, sharey=True)).set(xscale="log", xticks=fticks, xticklabels=labels)
    
    # # add the legend
    # # g.add_legend()
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    # g.savefig("../figures/lineplot_ERC_info.pdf")
    # plt.clf()
    # plt.close()
    
        
def gen_csv_table():
    
    import wandb
    api = wandb.Api()
    
    runs = api.runs(path="flow-graphs-cmu/TLCONV")

    exp_dict = ddict(list)
    summary_list, config_list, name_list = [], [], []
    for run in tqdm(runs):
        try:
            # .summary contains output keys/values for
            # metrics such as accuracy.
            #  We call ._json_dict to omit large files
            exp_dict['test_f1'].append(run.summary._json_dict['test_f1'])
            # summary_list.append(run.summary._json_dict)

            exp_dict['model_name'].append(run.config['model_name'])
            exp_dict['tgt_dataset'].append(run.config['tgt_dataset'])
            exp_dict['info'].append(run.config['info'])
            exp_dict['mode'].append(run.config['mode'])
            exp_dict['fewshot'].append(run.config['fewshot'])
            exp_dict['seed'].append(run.config['seed'])
            exp_dict['turns'].append(run.config['turns'])
            exp_dict['task'].append(run.config['task'])
            # .config contains the hyperparameters.
            #  We remove special values that start with _.
            # config_list.append(
            #     {k: v for k,v in run.config.items()
            #     if not k.startswith('_')})

            # .name is the human-readable name of the run.
            # name_list.append(run.name)
        except Exception as e:
            print(run.summary._json_dict, run.config )
            continue
    
    df = pd.DataFrame(exp_dict)
    print(len(df)) 
    
    df = df.replace({'fewshot': {-1: 'all'}})
    df = df.replace({'model_name':{'bert-base-uncased': 'bert'}})
    
    curr_df = df[(df['fewshot'] != 0) & (df['task']=='res') & (df['turns']==5)]
    
    for tgt_dataset in ['CB', 'P4G']:
        exp_dict = ddict(list)
        for model_name in ['bert', 'gpt2', 't5-base']:
            for mode in ['ID', 'TF']:
                for info in ['utterance', 'intention', 'assumption', 'implicit_info' ,'all']:
                    
                    exp_dict['model_name'].append(model_name)
                    exp_dict['mode'].append(mode)
                    exp_dict['info'].append(info)
                    
                    
                
                    for fewshot in [5, 10, 20, 50, 100, 'all']:
                        
                        temp_df = curr_df[(curr_df['model_name']==model_name) &\
                            (curr_df['tgt_dataset']==tgt_dataset) & \
                            (curr_df['info']==info) & (curr_df['mode']==mode) & (curr_df['fewshot']==fewshot)]
                        
                        mean_f1 = round(100*temp_df['test_f1'].mean(),1)
                        std_f1  = round(100*temp_df['test_f1'].std(),1)
                        
                        exp_dict[fewshot].append(f'{mean_f1}\u00B1{std_f1}')
                        
                        # exp_dict['fewshot'].append(fewshot)
                        # exp_dict['mean_f1'].append(mean_f1)
                        # exp_dict['std_f1'].append(std_f1)

        exp_df = pd.DataFrame(exp_dict)
        exp_df.to_csv(f'../results/{tgt_dataset}_info.csv', index=False)                        
        
    curr_df = df[(df['fewshot'] != 0) & (df['task']=='ERC') & (df['turns']==5)]
    
    for tgt_dataset in ['friends', 'iemocap']:
        exp_dict = ddict(list)
        for model_name in ['bert', 'gpt2', 't5-base']:
            for mode in ['ID', 'TF']:
                for info in ['utterance', 'intention', 'assumption', 'implicit_info' ,'all']:
                    
                    exp_dict['model_name'].append(model_name)
                    exp_dict['mode'].append(mode)
                    exp_dict['info'].append(info)
                
                    for fewshot in [5, 10, 20, 50, 100, 'all']:
                        
                        temp_df = curr_df[(curr_df['model_name']==model_name) &\
                            (curr_df['tgt_dataset']==tgt_dataset) & \
                            (curr_df['info']==info) & (curr_df['mode']==mode) & (curr_df['fewshot']==fewshot)]
                        
                        mean_f1 = round(100*temp_df['test_f1'].mean(),1)
                        std_f1  = round(100*temp_df['test_f1'].std(),1)
                        
                        exp_dict[fewshot].append(f'{mean_f1}\u00B1{std_f1}')
                        
                        # exp_dict['fewshot'].append(fewshot)
                        # exp_dict['mean_f1'].append(mean_f1)
                        # exp_dict['std_f1'].append(std_f1)

        exp_df = pd.DataFrame(exp_dict)
        exp_df.to_csv(f'../results/{tgt_dataset}_info.csv', index=False)                        
    

### analyse the chatgpt results

def analyse_chatgpt_results():
    
    results_dir     = f'../chatgpt_results/'
    chatgpt_dict    = ddict(lambda: ddict(list))
    results_dict    = ddict(list)
    
    for fname in ['res-P4G','res-CB', 'ERC-friends','ERC-iemocap']:
        for info in ['utterance', 'intention', 'assumption', 'implicit_info' ,'all']:
            data        = json.load(open(f'{results_dir}/{fname}-chatgpt-{info}-0shot-0.json'))
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
        
            f1 = round(100* f1_score(chatgpt_dict[fname]['gold'], chatgpt_dict[fname]['pred'], average='macro'),1)
            print(f'{fname}-{info}: {f1}')


def compute_bootstrapped_df(df1, df2, df1_name, df2_name, num_iter = 1000, avg='macro'):
    
    assert len(df1) == len(df2)
    
    idx_list = [idx for idx in range(len(df1))]
    
    cnt = 0
    
    gold_lbls = df1['label'].tolist()
    df1_lbls  = df1[df1_name].tolist()
    df2_lbls  = df2[df2_name].tolist()
    
    df1_f1 = round(100* f1_score(gold_lbls, df1_lbls, average= avg),2)
    df2_f1 = round(100* f1_score(gold_lbls, df2_lbls, average= avg),2)
    
    sys_diff = df2_f1 - df1_f1
    
    if sys_diff <= 0:
        temp_df, temp_name   = df2.copy(), df2_name
        df2, df2_name        = df1.copy(), df1_name
        df1, df1_name        = temp_df.copy(), temp_name
        gold_lbls            = df1['label'].tolist()
        df1_lbls             = df1[df1_name].tolist()
        df2_lbls             = df2[df2_name].tolist()
        df1_f1               = round(100* f1_score(gold_lbls, df1_lbls, average= avg),2)
        df2_f1               = round(100* f1_score(gold_lbls, df2_lbls, average= avg),2)
        sys_diff             = df2_f1 - df1_f1
        
    assert sys_diff >= 0
    
    for i in range(0, num_iter):
        random.seed(i)
        np.random.seed(i)
        curr_idx_list = random.choices(idx_list, k=len(idx_list))
        
        df1_pred = df1.iloc[curr_idx_list][df1_name].tolist()
        df2_pred = df2.iloc[curr_idx_list][df2_name].tolist()
        gold_lbl = df1.iloc[curr_idx_list]['label'].tolist()
        
        df1_f1 = round(100* f1_score(gold_lbl, df1_pred, average= avg),2)
        df2_f1 = round(100* f1_score(gold_lbl, df2_pred, average= avg),2)
        
        curr_diff = df2_f1 - df1_f1
        
        if curr_diff > 2*sys_diff:
            cnt += 1
    
    sig_val = ''
    p_val = cnt/num_iter

    if df1_name == 'utt_pred':
        if p_val < 0.001 :
            sig_val = '***'
        elif p_val < 0.01: 
            sig_val = '**'
        elif p_val < 0.05:
            sig_val = '*'
            
    elif df2_name == 'utt_pred':
        if p_val < 0.001 :
            sig_val = '@@@'
        elif p_val < 0.01: 
            sig_val = '@@'
        elif p_val < 0.05:
            sig_val = '@'

        p_val = -p_val
            
    return sig_val, p_val


def check_exist():
    task = 'res'    
    for dataset in ['P4G', 'CB']:
        for fewshot in [5.0, 10.0, 20.0, 50.0, 100.0, -1.0]:
            for model_name in ['bert-base-uncased', 'gpt2', 't5-base']:
                info_df_list    = []
                for info in ['intention', 'assumption', 'implicit_info', 'all', 'utterance']:
                    for seed in [0, 1, 2]:
                        try:
                            info_df = pd.read_csv(f'../csv_files/{task}-ID-{dataset}-{dataset}-{model_name}-{info}-{fewshot}-5-{seed}-test.csv')
                            print('Done', end='\r')
                        except Exception as e:
                            print(f'{task}-ID-{dataset}-{dataset}-{model_name}-{info}-{fewshot}-5-{seed}-test.csv')
                        
        
    for src_dataset in ['P4G', 'CB']:
        for tgt_dataset in ['P4G', 'CB']:
            if src_dataset == tgt_dataset: continue
            
            for fewshot in [5.0, 10.0, 20.0, 50.0, 100.0, -1.0]:
                for model_name in ['bert-base-uncased', 'gpt2', 't5-base']:
                    info_df_list    = []
                    for info in ['intention', 'assumption', 'implicit_info', 'all', 'utterance']:
                        for seed in [0, 1, 2]:
                            try:
                                info_df = pd.read_csv(f'../csv_files/{task}-TF-{src_dataset}-{tgt_dataset}-{model_name}-{info}-{fewshot}-5-{seed}-test.csv')
                                print('Done', end='\r')
                            except Exception as e:
                                print(f'{task}-TF-{src_dataset}-{tgt_dataset}-{model_name}-{info}-{fewshot}-5-{seed}-test.csv')
                            




def do_sigtest():
    ### compare utt vs others in in-domain setting

    print("Inside the sigtest function")
    
    ### do for ERC strategies
    task = 'ERC'
    sig_dict = ddict(list)
    
    for dataset in ['friends', 'iemocap']:
        for fewshot in [5.0, 10.0, 20.0, 50.0, 100.0, -1.0]:
            for model_name in ['bert-base-uncased', 'gpt2', 't5-base']:
                utt_df_list     = []
                
                for seed in [0, 1, 2]:
                    
                    info_df = pd.read_csv(f'../csv_files/{task}-ID-{dataset}-{dataset}-{model_name}-utterance-{fewshot}-5-{seed}-test.csv')
                    utt_df_list.append(info_df)
                    
                utt_df = pd.concat(utt_df_list)
                utt_df = utt_df.rename(columns={'pred': 'utt_pred'})
                
                sig_dict['task'].append(task)
                sig_dict['mode'].append('ID')
                sig_dict['src_dataset'].append(dataset)
                sig_dict['tgt_dataset'].append(dataset)
                sig_dict['fewshot'].append(fewshot)
                sig_dict['model_name'].append(model_name)
            
                for info in ['intention', 'assumption', 'implicit_info', 'all']:
                    info_df_list    = []
                    for seed in [0, 1, 2]:   
                        info_df = pd.read_csv(f'../csv_files/{task}-ID-{dataset}-{dataset}-{model_name}-{info}-{fewshot}-5-{seed}-test.csv')
                        info_df_list.append(info_df)
                
                    info_df = pd.concat(info_df_list)
                    info_df = info_df.rename(columns={'pred': f'{info}_pred'})
                    
                    sig_val, p_val = compute_bootstrapped_df(utt_df, info_df, 'utt_pred', f'{info}_pred')
                                        
                    sig_dict[info].append(f'{p_val}')
                    
                    print(f'{task}-{dataset}-{fewshot}-{model_name}-{info}: {p_val} {sig_val}')
                    
    ### do for cross domain 
    for src_dataset in ['friends'   , 'iemocap']:
        for tgt_dataset in ['friends'   , 'iemocap']:
            
            if src_dataset == tgt_dataset:
                continue
            
            for fewshot in [5.0, 10.0, 20.0, 50.0, 100.0, -1.0]:
                for model_name in ['bert-base-uncased', 'gpt2', 't5-base']:
                
                    utt_df_list     = []
                    
                    for seed in [0, 1, 2]:
                        info_df = pd.read_csv(f'../csv_files/{task}-TF-{src_dataset}-{tgt_dataset}-{model_name}-utterance-{fewshot}-5-{seed}-test.csv')
                        utt_df_list.append(info_df)
                        
                    utt_df = pd.concat(utt_df_list)
                    utt_df = utt_df.rename(columns={'pred': 'utt_pred'})
                    
                    sig_dict['task'].append(task)
                    sig_dict['mode'].append('TF')
                    sig_dict['src_dataset'].append(src_dataset)
                    sig_dict['tgt_dataset'].append(tgt_dataset)
                    sig_dict['fewshot'].append(fewshot)
                    sig_dict['model_name'].append(model_name)
                
                    for info in ['intention', 'assumption', 'implicit_info', 'all']:
                        info_df_list    = []
                        
                        for seed in [0, 1, 2]:
                            info_df = pd.read_csv(f'../csv_files/{task}-TF-{src_dataset}-{tgt_dataset}-{model_name}-{info}-{fewshot}-5-{seed}-test.csv')
                            info_df_list.append(info_df)
                        
                        info_df = pd.concat(info_df_list)
                        info_df = info_df.rename(columns={'pred': f'{info}_pred'})
                        
                        sig_val, p_val = compute_bootstrapped_df(utt_df, info_df, 'utt_pred', f'{info}_pred')
                        
                        sig_dict[info].append(f'{p_val}')
                            
                        print(f'{task}-{src_dataset}-{tgt_dataset}-{fewshot}-{model_name}-{info}: {p_val} {sig_val}')
                    
    
    ### do for res strategies
    task = 'res'    
    for dataset in ['P4G', 'CB']:
        for fewshot in [5.0, 10.0, 20.0, 50.0, 100.0, -1.0]:
            for model_name in ['bert-base-uncased', 'gpt2', 't5-base']:
                utt_df_list     = []
                
                for seed in [0, 1, 2]:
                    
                    info_df = pd.read_csv(f'../csv_files/{task}-ID-{dataset}-{dataset}-{model_name}-utterance-{fewshot}-5-{seed}-test.csv')
                    utt_df_list.append(info_df)
                    
                utt_df = pd.concat(utt_df_list)
                utt_df = utt_df.rename(columns={'pred': 'utt_pred'})
                
                sig_dict['task'].append(task)
                sig_dict['mode'].append('ID')
                sig_dict['src_dataset'].append(dataset)
                sig_dict['tgt_dataset'].append(dataset)
                sig_dict['fewshot'].append(fewshot)
                sig_dict['model_name'].append(model_name)
            
                for info in ['intention', 'assumption', 'implicit_info', 'all']:
                    info_df_list    = []
                    for seed in [0, 1, 2]:   
                        info_df = pd.read_csv(f'../csv_files/{task}-ID-{dataset}-{dataset}-{model_name}-{info}-{fewshot}-5-{seed}-test.csv')
                        info_df_list.append(info_df)
                
                    info_df = pd.concat(info_df_list)
                    info_df = info_df.rename(columns={'pred': f'{info}_pred'})
                    
                    sig_val, p_val = compute_bootstrapped_df(utt_df, info_df, 'utt_pred', f'{info}_pred')
                    
                    sig_dict[info].append(f'{p_val}')
                    
                    print(f'{task}-{dataset}-{fewshot}-{model_name}-{info}: {p_val} {sig_val}')
                    
    ### do for cross domain 
    
    for src_dataset in ['P4G', 'CB']:
        for tgt_dataset in ['P4G', 'CB']:
            
            if src_dataset == tgt_dataset:
                continue
            
            for fewshot in [5.0, 10.0, 20.0, 50.0, 100.0, -1.0]:
                for model_name in ['bert-base-uncased', 'gpt2', 't5-base']:
                
                    utt_df_list     = []
                    
                    for seed in [0, 1, 2]:
                        info_df = pd.read_csv(f'../csv_files/{task}-TF-{src_dataset}-{tgt_dataset}-{model_name}-utterance-{fewshot}-5-{seed}-test.csv')
                        utt_df_list.append(info_df)
                        
                    utt_df = pd.concat(utt_df_list)
                    utt_df = utt_df.rename(columns={'pred': 'utt_pred'})
                    
                    sig_dict['task'].append(task)
                    sig_dict['mode'].append('TF')
                    sig_dict['src_dataset'].append(src_dataset)
                    sig_dict['tgt_dataset'].append(tgt_dataset)
                    sig_dict['fewshot'].append(fewshot)
                    sig_dict['model_name'].append(model_name)
                    
                    for info in ['intention', 'assumption', 'implicit_info', 'all']:
                        info_df_list    = []
                        
                        for seed in [0, 1, 2]:
                            info_df = pd.read_csv(f'../csv_files/{task}-TF-{src_dataset}-{tgt_dataset}-{model_name}-{info}-{fewshot}-5-{seed}-test.csv')
                            info_df_list.append(info_df)
                        
                        info_df = pd.concat(info_df_list)
                        info_df = info_df.rename(columns={'pred': f'{info}_pred'})
                        
                        sig_val, p_val = compute_bootstrapped_df(utt_df, info_df, 'utt_pred', f'{info}_pred')
                        
                        
                        sig_dict[info].append(f'{p_val}')
                        
                            
                        print(f'{task}-{src_dataset}-{tgt_dataset}-{fewshot}-{model_name}-{info}: {p_val} {sig_val}')
                    
    sig_df = pd.DataFrame(sig_dict)
    sig_df.to_csv(f'../results/sig_mf1_bootstrap_df.csv', index=False)




def do_sigtest_zshot():
    ### compare utt vs others in in-domain setting

    
    ### do for ERC strategies
    task = 'ERC'
    sig_dict = ddict(list)
    
    
    ### do for cross domain 
    for src_dataset in ['friends'   , 'iemocap']:
        for tgt_dataset in ['friends'   , 'iemocap']:
            
            if src_dataset == tgt_dataset:
                continue

            fewshot = 0.0
            
            for model_name in ['bert-base-uncased', 'gpt2', 't5-base']:
            
                utt_df_list     = []
                
                for seed in [0, 1, 2]:
                    info_df = pd.read_csv(f'../csv_files/{task}-TF-{src_dataset}-{tgt_dataset}-{model_name}-utterance-{fewshot}-5-{seed}-test.csv')
                    utt_df_list.append(info_df)
                    
                utt_df = pd.concat(utt_df_list)
                utt_df = utt_df.rename(columns={'pred': 'utt_pred'})
                
                sig_dict['task'].append(task)
                sig_dict['mode'].append('TF')
                sig_dict['src_dataset'].append(src_dataset)
                sig_dict['tgt_dataset'].append(tgt_dataset)
                sig_dict['fewshot'].append(fewshot)
                sig_dict['model_name'].append(model_name)
            
                for info in ['intention', 'assumption', 'implicit_info', 'all']:
                    info_df_list    = []
                    
                    for seed in [0, 1, 2]:
                        info_df = pd.read_csv(f'../csv_files/{task}-TF-{src_dataset}-{tgt_dataset}-{model_name}-{info}-{fewshot}-5-{seed}-test.csv')
                        info_df_list.append(info_df)
                    
                    info_df = pd.concat(info_df_list)
                    info_df = info_df.rename(columns={'pred': f'{info}_pred'})
                    
                    sig_val, p_val = compute_bootstrapped_df(utt_df, info_df, 'utt_pred', f'{info}_pred')
                    
                    sig_dict[info].append(f'{p_val}{sig_val}')
                        
                    print(f'{task}-{src_dataset}-{tgt_dataset}-{fewshot}-{model_name}-{info}: {p_val} {sig_val}')
                

    ### do for res strategies
    task = 'res'    
    
    for src_dataset in ['P4G', 'CB']:
        for tgt_dataset in ['P4G', 'CB']:
            
            if src_dataset == tgt_dataset:
                continue
            
            fewshot = 0.0
            for model_name in ['bert-base-uncased', 'gpt2', 't5-base']:
            
                utt_df_list     = []
                
                for seed in [0, 1, 2]:
                    info_df = pd.read_csv(f'../csv_files/{task}-TF-{src_dataset}-{tgt_dataset}-{model_name}-utterance-{fewshot}-5-{seed}-test.csv')
                    utt_df_list.append(info_df)
                    
                utt_df = pd.concat(utt_df_list)
                utt_df = utt_df.rename(columns={'pred': 'utt_pred'})
                
                sig_dict['task'].append(task)
                sig_dict['mode'].append('TF')
                sig_dict['src_dataset'].append(src_dataset)
                sig_dict['tgt_dataset'].append(tgt_dataset)
                sig_dict['fewshot'].append(fewshot)
                sig_dict['model_name'].append(model_name)
                
                for info in ['intention', 'assumption', 'implicit_info', 'all']:
                    info_df_list    = []
                    
                    for seed in [0, 1, 2]:
                        info_df = pd.read_csv(f'../csv_files/{task}-TF-{src_dataset}-{tgt_dataset}-{model_name}-{info}-{fewshot}-5-{seed}-test.csv')
                        info_df_list.append(info_df)
                    
                    info_df = pd.concat(info_df_list)
                    info_df = info_df.rename(columns={'pred': f'{info}_pred'})
                    
                    sig_val, p_val = compute_bootstrapped_df(utt_df, info_df, 'utt_pred', f'{info}_pred')
                    
                    
                    sig_dict[info].append(f'{p_val}{sig_val}')
                    
                        
                    print(f'{task}-{src_dataset}-{tgt_dataset}-{fewshot}-{model_name}-{info}: {p_val} {sig_val}')
                
    sig_df = pd.DataFrame(sig_dict)
    sig_df.to_csv(f'../results/sig_bootstrap_0shot_df.csv', index=False)


                        

def gen_full_factorial_df():
    
    import wandb
    api = wandb.Api()
    
    runs = api.runs(path="flow-graphs-cmu/TLCONV")

    exp_dict = ddict(list)
    summary_list, config_list, name_list = [], [], []
    for run in tqdm(runs):
        try:
            # .summary contains output keys/values for
            # metrics such as accuracy.
            #  We call ._json_dict to omit large files
            exp_dict['test_f1'].append(run.summary._json_dict['test_f1'])
            # summary_list.append(run.summary._json_dict)

            exp_dict['model_name'].append(run.config['model_name'])
            exp_dict['tgt_dataset'].append(run.config['tgt_dataset'])
            exp_dict['info'].append(run.config['info'])
            exp_dict['mode'].append(run.config['mode'])
            exp_dict['fewshot'].append(run.config['fewshot'])
            exp_dict['seed'].append(run.config['seed'])
            exp_dict['turns'].append(run.config['turns'])
            exp_dict['task'].append(run.config['task'])
            # .config contains the hyperparameters.
            #  We remove special values that start with _.
            # config_list.append(
            #     {k: v for k,v in run.config.items()
            #     if not k.startswith('_')})

            # .name is the human-readable name of the run.
            # name_list.append(run.name)
        except Exception as e:
            print(run.summary._json_dict, run.config )
            continue
    
    df = pd.DataFrame(exp_dict)
    print(len(df))  
    
    df     = df.replace({'fewshot': {-1: 500}})
    res_df = df[(df['task']=='res') & (df['turns']==5)]
    ERC_df = df[(df['task']=='ERC') & (df['turns']==5)]
    
    res_df.to_csv(f'../results/res_full_factorial_df.csv', index=False)
    ERC_df.to_csv(f'../results/ERC_full_factorial_df.csv', index=False)
    
    pass
    
# gen_full_factorial_df()

# analyse_chatgpt_results()

# analyse_exp_results()

# analyse_judgement_results()


def download_wandb_csv():
    import wandb
    api = wandb.Api()
    
    runs = api.runs(path="flow-graphs-cmu/TLCONV")

    summary_list, config_list, name_list = [], [], []
    for run in tqdm(runs):
        # .summary contains output keys/values for
        # metrics such as accuracy.
        #  We call ._json_dict to omit large files
        summary_list.append(run.summary._json_dict)

        # .config contains the hyperparameters.
        #  We remove special values that start with _.
        config_list.append(
            {k: v for k,v in run.config.items()
            if not k.startswith('_')})

        # .name is the human-readable name of the run.
        name_list.append(run.name)

    runs_df = pd.DataFrame({
        "summary": summary_list,
        "config": config_list,
        "name": name_list
        })
    
    import pdb; pdb.set_trace()
    print(len(runs_df))

    


#### compute metrics on the generated LLAMA results #####



def get_LLM_metrics():
    
    task    = 'ERC'
    
    if task == 'ERC':
        data_dir = '/data/shire/projects/TLDIAL-llama/13b'    
    else:
        data_dir = '/home/rdutt/babel_projects/TLDIAL/ICL_results/generations/'
    
    
    for file in os.listdir(data_dir):
        if not file.startswith(task): continue
        
        data = json.load(open(f'{data_dir}/{file}'))
        metrics = ddict(lambda: ddict(int))
        
        for elem in data:
            
            gold_lbl       = elem[f'{task}_label']
            lbl_for_prompt = elem['label_for_prompt']
            pred_ans       = 1 if 'yes' in elem['answer_generated'].lower().strip() else 0
            gold_ans       = elem['gold_answer']
            
            if pred_ans ==0 and gold_ans ==0:
                continue
            else:
                if pred_ans ==1 and gold_ans ==1:
                    metrics[gold_lbl]['TP'] += 1
                elif pred_ans ==1 and gold_ans ==0:
                    metrics[gold_lbl]['FP'] += 1
                elif pred_ans ==0 and gold_ans ==1:
                    metrics[gold_lbl]['FN'] += 1
            
        
        mF1 = []
        for lbl in metrics:
            try:
                prec = metrics[lbl]['TP']/(metrics[lbl]['TP'] + metrics[lbl]['FP'])
            except Exception as e:
                prec = 0.0
            
            try:
                rec  = metrics[lbl]['TP']/(metrics[lbl]['TP'] + metrics[lbl]['FN'])
            except Exception as e:
                rec = 0.0
            
            try: 
                f1   = 2*prec*rec/(prec+rec)
            except Exception as e:
                f1 = 0.0
                
            mF1.append(f1)
            
        print(file, round(100*np.mean(mF1),2))
    





def generate_cm_plots(dataset='CB', mode='ID', model_name='t5-base', fewshot=-1.0, info='all'):
    
    if dataset in ['P4G', 'CB']:
        task = 'res'
    else:
        task = 'ERC'
    
    src_dataset_map = {'P4G': 'CB', 'CB': 'P4G', 'friends': 'iemocap', 'iemocap': 'friends'}
    
    if mode == 'ID':
        src_dataset = dataset
        tgt_dataset = dataset
    else:
        src_dataset = src_dataset_map[dataset]
        tgt_dataset = dataset
        
    
    pred_arr, gold_arr = [], []
    lbl2idx     = {}
    idx2lbl     = {}
    f1_arr      = []
    df_list     = []

    if task == 'res':
        label_mapping = {'Counter Argumentation': 'CA',
                'Hesitance': 'HES',
                'Information Inquiry': 'INF',
                'Not a resistance strategy': 'NAS',
                'Personal Choice': 'PC',
                'Self Assertion': 'SA',
                'Self Pity': 'SP',
                'Source Derogation': 'SD',
                'UNK': 'NAS'
        }
    else:
        label_mapping = {'anger': 'anger', 
                        'disgust': 'disgust', 
                        'fear': 'fear', 
                        'joy': 'joy', 
                        'neutral': 'neutral', 
                        'sadness': 'sadness', 
                        'surprise': 'surprise',
                        'other': 'other', 
                        'UNK': 'other'
        }

    for seed in [0, 1, 2]:
        filename = f'../csv_files/{task}-{mode}-{src_dataset}-{tgt_dataset}-{model_name}-{info}-{fewshot}-5-{seed}-test.csv'
        curr_df = pd.read_csv(filename)
        df_list.append(curr_df)
        preds   = list(curr_df['pred'])
        golds   = list(curr_df['label'])
        pred_arr.extend(preds)
        gold_arr.extend(golds)
        
    label_names = list(set(pred_arr + gold_arr))
    label_names = [label_mapping[lbl] for lbl in label_names]

    for idx, label in enumerate(label_names):
        lbl2idx[label] = idx
        idx2lbl[idx]   = label

    gold_arr    = [lbl2idx[label_mapping[lbl]] for lbl in gold_arr]
    pred_arr    = [lbl2idx[label_mapping[lbl]] for lbl in pred_arr]

    cm = confusion_matrix(gold_arr, pred_arr,  normalize='true')

    cm_dict     = ddict(list)

    for i in range(len(cm)):
        for j in range(len(cm)):
            cm_dict['True Lbl'].append(idx2lbl[i])
            cm_dict['Pred Lbl'].append(idx2lbl[j])
            cm_dict['Score'].append(round(100*cm[i][j],2))
            
    cm_df = pd.DataFrame(cm_dict)          
    cm_df = cm_df.pivot(index='True Lbl', columns='Pred Lbl', values='Score')      

    sns.set(font_scale=0.9)
    # cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # ax.set_xticklabels(ax.get_xticks(), rotation = 30)

    ax = sns.heatmap(cm_df, annot=True, linewidths = 0.01, fmt=".1f", cmap='crest')
    # ax.set_yticklabels(ax.get_yticks(), rotation = 45)
    
    plt.yticks(rotation=0)
    plt.xticks(rotation=0)
    plt.savefig(f'../cm_plots/cm-{task}-{mode}-{src_dataset}-{tgt_dataset}-{model_name}-{info}-{fewshot}-test.pdf')
    plt.close()
    plt.clf()
    

#### generate cm_plots to look at the cases #####


def gen_ID_cms():
    
    generate_cm_plots(dataset='CB', mode='ID', model_name='t5-base', fewshot=-1.0, info='all')
    
    generate_cm_plots(dataset='CB', mode='ID', model_name='t5-base', fewshot=-1.0, info='utterance')
    
    generate_cm_plots(dataset='P4G', mode='ID', model_name='t5-base', fewshot=-1.0, info='all')
    
    generate_cm_plots(dataset='P4G', mode='ID', model_name='t5-base', fewshot=-1.0, info='utterance')
    
    generate_cm_plots(dataset='friends', mode='ID', model_name='bert-base-uncased', fewshot=-1.0, info='all')
    
    generate_cm_plots(dataset='friends', mode='ID', model_name='bert-base-uncased', fewshot=-1.0, info='utterance')
    
    generate_cm_plots(dataset='iemocap', mode='ID', model_name='t5-base', fewshot=-1.0, info='intention')
    
    generate_cm_plots(dataset='iemocap', mode='ID', model_name='t5-base', fewshot=-1.0, info='utterance')
    
    
#########


def gen_TF_cms():
    
    generate_cm_plots(dataset='CB', mode='TF', model_name='bert-base-uncased', fewshot=20.0, info='all')
    
    generate_cm_plots(dataset='CB', mode='TF', model_name='bert-base-uncased', fewshot=20.0, info='utterance')
    
    generate_cm_plots(dataset='P4G', mode='TF', model_name='bert-base-uncased', fewshot=20.0, info='intention')
    
    generate_cm_plots(dataset='P4G', mode='TF', model_name='bert-base-uncased', fewshot=20.0, info='utterance')
    
    generate_cm_plots(dataset='friends', mode='TF', model_name='t5-base', fewshot=20.0, info='all')
    
    generate_cm_plots(dataset='friends', mode='TF', model_name='t5-base', fewshot=20.0, info='utterance')
    
    generate_cm_plots(dataset='iemocap', mode='TF', model_name='bert-base-uncased', fewshot=20.0, info='all')
    
    generate_cm_plots(dataset='iemocap', mode='TF', model_name='bert-base-uncased', fewshot=20.0, info='utterance')


def generate_diff_plots(dataset='CB', mode='ID', model_name='t5-base', fewshot=-1.0, info='all'):
    sns.set_style('darkgrid')
    sns.set(font_scale=1.2)

    
    # fewshot         = -1.0
    # dataset         = 'P4G'
    # task            = 'res'
    # model_name      = 't5-base'
    # info            = 'all'
    
    if dataset in ['P4G', 'CB']:
        task = 'res'
    else:
        task = 'ERC'
    
    src_dataset_map = {'P4G': 'CB', 'CB': 'P4G', 'friends': 'iemocap', 'iemocap': 'friends'}
    
    if mode == 'ID':
        src_dataset = dataset
        tgt_dataset = dataset
    else:
        src_dataset = src_dataset_map[dataset]
        tgt_dataset = dataset
        
    

    idx_wise_info_scores = ddict(int)
    idx_wise_utt_scores  = ddict(int)

    for seed in [0, 1, 2]:
        info_df = pd.read_csv(f'/data/shire/projects/TLDIAL/csv_files/{task}-{mode}-{src_dataset}-{tgt_dataset}-{model_name}-{info}-{fewshot}-5-{seed}-test.csv')
        info_df = info_df.rename(columns={'Unnamed: 0': 'idx'})
        
        for idx, row in info_df.iterrows():
            if row['label'] == row['pred']:
                idx_wise_info_scores[idx]+= 1    
        
        utt_df = pd.read_csv(f'/data/shire/projects/TLDIAL/csv_files/{task}-{mode}-{src_dataset}-{tgt_dataset}-{model_name}-utterance-{fewshot}-5-{seed}-test.csv')
        for idx, row in utt_df.iterrows():
            if row['label'] == row['pred']:
                idx_wise_utt_scores[idx]+= 1    
        

    diff_mapper = {3: 'always better', 
            2: 'mostly better', 
            1: 'sometimes better',
            0: 'equally good',
            -1: 'sometimes worse',
            -2: 'mostly worse',
            -3: 'always worse'}


    curr_dataset = []

    tst_dataset = json.load(open(f'/data/shire/projects/TLDIAL/data/{task}-{dataset}-test-explained-updated.json'))
            
    for idx, elem in enumerate(tst_dataset):
        
        diff_score = idx_wise_info_scores[idx] - idx_wise_utt_scores[idx]
        diff_val   = diff_mapper[diff_score]
        elem['diff'] = diff_val
        curr_dataset.append(elem)

        
    data = pd.DataFrame(curr_dataset)
    total_counts = data[data['diff'] != 'equally good'].groupby(f'{task}_label').size().reset_index(name='total')

    grouped = data[data['diff'] != 'equally good'].groupby([f'{task}_label', 'diff']).size().reset_index(name='count')

    total_counts = total_counts.sort_values(by=['total']).reset_index(drop=True)

    grouped = grouped.merge(total_counts, on=f'{task}_label')
    # display(grouped1)

    row_order = total_counts[f'{task}_label'].values.tolist()

    grouped['percentage'] = grouped['count'] / grouped['total'] * 100

    pivot = grouped.pivot(index=f'{task}_label', columns='diff', values='percentage').fillna(0)


    categories_order = ['always worse', 'mostly worse', 'sometimes worse', 'sometimes better', 'mostly better', 'always better']
    color_mapping = ['#e06666', '#ea9999', '#f4cccc', '#a4c2f4', '#6d9eeb', '#3c78d8']
    pivot = pivot.reindex(columns=categories_order)
    pivot = pivot.reindex(index=row_order)


    ax =pivot.plot(kind='bar', stacked=True, color=color_mapping, figsize=(12, 8))

    bar_positions = range(len(pivot))

    for pos, idx in zip(bar_positions, pivot.index):
        total = total_counts[total_counts[f'{task}_label'] == idx]['total'].values[0]
        ax.text(pos, 50, str(total), ha='center', va='top') 
        
    plt.axhline(y=50, color='gray', linestyle='--')
    ax.text(-0.65, 50, '50', ha='center', va='center', color='black')

    ax.set_xlabel(f'{task} Label', fontsize=12, labelpad=20)
    ax.set_ylabel('Percentage', fontsize=12)
    # ax.set_title('Diff ' + dataset + ' Dataset', fontsize=14)
    plt.legend(title='Categories', loc='best', bbox_to_anchor=(1, 1), labels=categories_order[::-1], handles=[plt.Rectangle((0,0),1,1, color=color_mapping[i]) for i in range(len(categories_order)-1, -1, -1)])
    plt.xticks(rotation=0)
    plt.ylim(0, 100)

    labels = ax.get_xticklabels()
    wrapped_labels = [textwrap.fill(label.get_text(), width=11) for label in labels]
    ax.set_xticklabels(wrapped_labels)

    plt.tight_layout()
    plt.savefig(f'../cm_plots/dist-{task}-{mode}-{src_dataset}-{tgt_dataset}-{model_name}-{info}-{fewshot}.pdf')
    
    
def gen_diff_dists():
    
    generate_diff_plots(dataset='CB', mode='ID', model_name='t5-base', fewshot=-1.0, info='all')
    
    generate_diff_plots(dataset='P4G', mode='ID', model_name='t5-base', fewshot=-1.0, info='all')
    
    generate_diff_plots(dataset='friends', mode='ID', model_name='bert-base-uncased', fewshot=-1.0, info='all')
    
    generate_diff_plots(dataset='iemocap', mode='ID', model_name='t5-base', fewshot=-1.0, info='intention')
    
    
    generate_diff_plots(dataset='CB', mode='TF', model_name='bert-base-uncased', fewshot=20.0, info='all')
    
    generate_diff_plots(dataset='P4G', mode='TF', model_name='bert-base-uncased', fewshot=20.0, info='intention')
    
    generate_diff_plots(dataset='friends', mode='TF', model_name='t5-base', fewshot=20.0, info='all')
    
    generate_diff_plots(dataset='iemocap', mode='TF', model_name='bert-base-uncased', fewshot=20.0, info='all')
    



def gen_wf1_results():
    ### compare utt vs others in in-domain setting

    
    ### do for ERC strategies
    task = 'ERC'
    perf_dict = ddict(list)
    
    for src_dataset in ['friends', 'iemocap']:
        for tgt_dataset in ['friends', 'iemocap']:
            
            if src_dataset == tgt_dataset:
                mode = 'ID'
            else:
                mode = 'TF'
            
            for model_name in ['bert-base-uncased', 'gpt2', 't5-base']:
                for info in ['utterance','intention', 'assumption', 'implicit_info', 'all']:                
                    perf_dict['dataset'].append(src_dataset)
                    perf_dict['mode'].append(mode)
                    perf_dict['model'].append(model_name)
                    perf_dict['rationale'].append(info)
                    
                    for fewshot in [5.0, 10.0, 20.0, 50.0, 100.0, -1.0]:
                    
                        wF1_list = []
                        
                        for seed in [0, 1, 2]:
                            
                            info_df = pd.read_csv(f'../csv_files/{task}-{mode}-{src_dataset}-{tgt_dataset}-{model_name}-{info}-{fewshot}-5-{seed}-test.csv')
                            preds = list(info_df['pred'])
                            golds = list(info_df['label'])
                            wF1_list.append(f1_score(golds, preds, average='weighted'))
                            
                        perf = f"{round(100*np.mean(wF1_list), 2)}\pm{round(100*np.std(wF1_list),2)}"
                        
                        fshot = int(fewshot)
                        if fshot < 0: fshot ='All'
                        
                        perf_dict[fshot].append(perf)
                        print(f'{src_dataset}\t {mode}\t {info}\t {fewshot}\t {model_name}\t {perf}')
                    
    perf_df = pd.DataFrame(perf_dict)
    perf_df.to_csv(f'../results/wf1_ERC.csv', index=False)



# gen_wf1_results()


do_sigtest()


# gen_ID_cms()
# gen_TF_cms()
# gen_diff_dists()

# do_sigtest()