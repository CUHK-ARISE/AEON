import os
import csv
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer

batch_size = 16
k = 192

with open('data/info.csv') as f:
    c = csv.DictReader(f)
    info = []
    for row in c:
        info.append(row)
with open('data/ori.txt') as f1:
    with open('data/adv.txt') as f2:
        ori = f1.read().splitlines()
        adv = f2.read().splitlines()
        for i, (o, a) in enumerate(zip(ori, adv)):
            o = o.replace('[[[[Premise]]]]: ', '##ForPremise##')
            o = o.replace('>>>>[[[[Hypothesis]]]]: ', '##ForHypothesis##')
            a = a.replace('[[[[Premise]]]]: ', '##ForPremise##')
            a = a.replace('>>>>[[[[Hypothesis]]]]: ', '##ForHypothesis##')
            o = o.replace('[[[[Question1]]]]: ', '##ForQuestion1##')
            o = o.replace('>>>>[[[[Question2]]]]: ', '##ForQuestion2##')
            a = a.replace('[[[[Question1]]]]: ', '##ForQuestion1##')
            a = a.replace('>>>>[[[[Question2]]]]: ', '##ForQuestion2##')
            o = o.replace('[[', '').replace(']]', '')
            a = a.replace('[[', '').replace(']]', '')
            o = o.replace('##ForPremise##', '[[[[Premise]]]]: ')
            o = o.replace('##ForHypothesis##', '>>>>[[[[Hypothesis]]]]: ')
            a = a.replace('##ForPremise##', '[[[[Premise]]]]: ')
            a = a.replace('##ForHypothesis##', '>>>>[[[[Hypothesis]]]]: ')
            o = o.replace('##ForQuestion1##', '[[[[Question1]]]]: ')
            o = o.replace('##ForQuestion2##', '>>>>[[[[Question2]]]]: ')
            a = a.replace('##ForQuestion1##', '[[[[Question1]]]]: ')
            a = a.replace('##ForQuestion2##', '>>>>[[[[Question2]]]]: ')
            info[i]['ori'] = o
            info[i]['adv'] = a

mr = []
yelp = []
snli = []
mnli = []
qqp = []
imdb = []
agnews = []
for i in range(len(info)):
    if info[i]['dataset'] == 'rtmr':
        mr.append(i)
    if info[i]['dataset'] == 'yelp':
        yelp.append(i)
    if info[i]['dataset'] == 'snli':
        snli.append(i)
    if info[i]['dataset'] == 'mnli':
        mnli.append(i)
    if info[i]['dataset'] == 'qqp':
        qqp.append(i)
    if info[i]['dataset'] == 'imdb':
        imdb.append(i)
    if info[i]['dataset'] == 'agnews':
        agnews.append(i)
dataset = [mr, yelp, snli, mnli, qqp, imdb, agnews]
name = {'rotten-tomatoes': 'rtmr', 'yelp-polarity': 'yelp', 'snli': 'snli', 'MNLI': 'mnli',
        'QQP': 'qqp', 'imdb': 'imdb', 'ag-news': 'agnews'}

for d, n in zip(dataset, name.keys()):
    tokenizer = AutoTokenizer.from_pretrained('textattack/bert-base-uncased-' + n)
    model = AutoModelForSequenceClassification.from_pretrained('textattack/bert-base-uncased-' + n)
    
    if os.path.exists('baselines/NC-based-statistics/%s-avg.npy' % name[n]):
        average = np.load('baselines/NC-based-statistics/%s-avg.npy' % name[n])
        maximum = np.load('baselines/NC-based-statistics/%s-max.npy' % name[n])
        minimum = np.load('baselines/NC-based-statistics/%s-min.npy' % name[n])
    else:
        with open('data/textattack/datasets/%s_seeds.txt' % name[n]) as f:
            base_data = f.read().splitlines()
        for i in range(len(base_data)):
            base_data[i] = base_data[i].split('\t')
        if n == 'snli' or n == 'MNLI':
            base_data = ['[[[[Premise]]]]: %s>>>>[[[[Hypothesis]]]]: %s' % (i[0], i[1]) for i in base_data]
        elif n == 'QQP':
            base_data = ['[[[[Question1]]]]: %s>>>>[[[[Question2]]]]: %s' % (i[0], i[1]) for i in base_data]
        else:
            base_data = [' '.join(i[:-1]) for i in base_data]
        print(base_data[42])
        average = np.zeros((13, 768))
        maximum = np.ones((13, 768)) * (-1e8)
        minimum = np.ones((13, 768)) * (1e8)
        for i in range(0, len(base_data), batch_size):
            print('Dataset: %s, process: %d/%d' % (name[n], i, len(base_data)))
            batch_data = base_data[i:i + batch_size]
            batch_data = tokenizer(batch_data, padding=True, truncation=True, return_tensors='pt')
            outputs = model(**batch_data, return_dict=True, output_hidden_states=True, output_attentions=True)
            for layer_i in range(len(outputs.hidden_states)):
                for output in outputs.hidden_states[layer_i]:
                    average[layer_i] += output[0].detach().numpy()
                    maximum[layer_i] = (maximum[layer_i] > output[0].detach().numpy()).astype(int) * maximum[layer_i] + \
                                       (maximum[layer_i] <= output[0].detach().numpy()).astype(int) * output[0].detach().numpy()
                    minimum[layer_i] = (minimum[layer_i] < output[0].detach().numpy()).astype(int) * minimum[layer_i] + \
                                       (minimum[layer_i] >= output[0].detach().numpy()).astype(int) * output[0].detach().numpy()
        average = average / len(base_data)
        np.save('baselines/NC-based-statistics/%s-avg.npy' % name[n], average)
        np.save('baselines/NC-based-statistics/%s-max.npy' % name[n], maximum)
        np.save('baselines/NC-based-statistics/%s-min.npy' % name[n], minimum)
    
    data_ori = [info[i]['ori'] for i in d]
    data_adv = [info[i]['adv'] for i in d]
    print(data_ori[42])
    print(data_adv[42])
    NC = []
    NBC = []
    TKNC = []
    BKNC = []
    for i in range(0, len(data_adv), batch_size):
        print('Dataset: %s, process: %d/%d' % (name[n], i, len(data_adv)))
        batch_data = data_adv[i:i + batch_size]
        batch_data = tokenizer(batch_data, padding=True, truncation=True, return_tensors='pt')
        outputs_adv = model(**batch_data, return_dict=True, output_hidden_states=True, output_attentions=True)
        batch_data = data_ori[i:i + batch_size]
        batch_data = tokenizer(batch_data, padding=True, truncation=True, return_tensors='pt')
        outputs_ori = model(**batch_data, return_dict=True, output_hidden_states=True, output_attentions=True)
        nc = [0] * len(outputs_adv.hidden_states[0])
        nbc = [0] * len(outputs_adv.hidden_states[0])
        tknc = [0] * len(outputs_adv.hidden_states[0])
        bknc = [0] * len(outputs_adv.hidden_states[0])
        for layer_i in range(len(outputs_adv.hidden_states)):
            for output in range(len(outputs_adv.hidden_states[layer_i])):
                nc[output] += np.sum(outputs_adv.hidden_states[layer_i][output][0].detach().numpy() >= average[layer_i])
                nbc[output] += np.sum(outputs_adv.hidden_states[layer_i][output][0].detach().numpy() >= maximum[layer_i])
                nbc[output] += np.sum(outputs_adv.hidden_states[layer_i][output][0].detach().numpy() <= minimum[layer_i])
                adv_top_k = np.argsort(outputs_adv.hidden_states[layer_i][output][0].detach().numpy())[-k:].tolist()
                ori_top_k = np.argsort(outputs_ori.hidden_states[layer_i][output][0].detach().numpy())[-k:].tolist()
                tknc[output] += len(list(set(adv_top_k + ori_top_k))) - k
                adv_bottom_k = np.argsort(outputs_adv.hidden_states[layer_i][output][0].detach().numpy())[:k].tolist()
                ori_bottom_k = np.argsort(outputs_ori.hidden_states[layer_i][output][0].detach().numpy())[:k].tolist()
                bknc[output] += len(list(set(adv_bottom_k + ori_bottom_k))) - k
        NC += nc
        NBC += nbc
        TKNC += tknc
        BKNC += bknc
    NC = np.array(NC) / (13 * 768)
    NBC = np.array(NBC) / (13 * 768)
    TKNC = np.array(TKNC) / (13 * k)
    BKNC = np.array(BKNC) / (13 * k)
    for i in range(len(d)):
        info[d[i]]['nc'] = 1 - NC[i]
        info[d[i]]['nbc'] = 1 - NBC[i]
        info[d[i]]['tknc'] = 1 - TKNC[i]
        info[d[i]]['bknc'] = 1 - BKNC[i]

with open('baselines/results/NC.txt', 'w') as f:
    f.write('\n'.join([str(i['nc']) for i in info]))
with open('baselines/results/NBC.txt', 'w') as f:
    f.write('\n'.join([str(i['nbc']) for i in info]))
with open('baselines/results/TKNC.txt', 'w') as f:
    f.write('\n'.join([str(i['tknc']) for i in info]))
with open('baselines/results/BKNC.txt', 'w') as f:
    f.write('\n'.join([str(i['bknc']) for i in info]))

