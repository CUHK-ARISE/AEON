import csv
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
# from scipy.stats import pearsonr


all_datasets = ['rtmr', 'yelp', 'snli', 'mnli', 'qqp']
all_experiments = ['consistency', 'fluency']
all_metrics = {'consistency': ['NC', 'NBC', 'TKNC', 'BKNC', 'BLEU', 'Meteor',
                               'InferSent', 'SBERT', 'SimCSE', 'BERTScore', 'SemEval'],
               'fluency': ['NC', 'NBC', 'TKNC', 'BKNC', 'SynEval']}

def AP_AUC(csv_data, x_name, y_name, threshold=3):
    x = []
    y = []
    for i in csv_data:
        if float(i[y_name]) >= threshold:
            y.append(1)
        else:
            y.append(0)
        x.append(float(i[x_name]))
    x = np.array(x)
    y = np.array(y)
    
    fpr, tpr, threshold = roc_curve(y, x)
    auc_score = auc(fpr, tpr)
    
    ap_score = average_precision_score(y, x)

    return ap_score, auc_score


def pearson_r(x, y):
    # Compute correlation matrix
    corr_mat = np.corrcoef(x, y)
 
    # Return entry [0, 1]
    return corr_mat[0, 1]


def PCC(csv_data, x_name, y_name):
    x = [i / 2 for i in range(2, 10)]
    y = [0] * len(x)
    y_count = [0] * len(x)

    for i in csv_data:
        idx = int((float(i[y_name]) - 1) // 0.5)
        if idx == 8:
            idx = 7
        y[idx] += float(i[x_name])
        y_count[idx] += 1

    y = [y[i] for i in range(len(y_count)) if y_count[i] != 0]
    y_count = [y_count[i] for i in range(len(y_count)) if y_count[i] != 0]
    x = [x[i] for i in range(len(y_count)) if y_count[i] != 0]
    y = [i / j for i, j in zip(y, y_count)]

    pcc_score = pearson_r(x, y)
    return pcc_score


print('')
thresholds = {'rtmr': 2.0, 'yelp': 3.0, 'snli': 2.67, 'mnli': 3.25, 'qqp': 3.0}
for experiment_name in all_experiments:
    best_metric = []
    for metric_name in all_metrics[experiment_name]:
        latex = ['\\bf %s' % metric_name]
        avp_ap, avg_auc, avg_pcc = 0, 0, 0
        for dataset_name in all_datasets:
            threshold = thresholds[dataset_name]

            with open('result/annotation_%s.csv' % dataset_name) as f:
                r = csv.DictReader(f)
                csv_data = []
                for row in r:
                    csv_data.append(row)

            pcc_score = PCC(csv_data, metric_name, experiment_name)
            ap_score, auc_score = AP_AUC(csv_data, metric_name, experiment_name, threshold)

            latex += ['%.2f' % ap_score, '%.2f' % auc_score, '%.2f' % pcc_score]
            avp_ap += ap_score
            avg_auc += auc_score
            avg_pcc += pcc_score
        avp_ap, avg_auc, avg_pcc = avp_ap / len(all_datasets), avg_auc / len(all_datasets), avg_pcc / len(all_datasets)
        print(' & '.join(latex))
        best_metric.append([metric_name, avp_ap, avg_auc, avg_pcc])
    best_metric.sort(key=lambda x: x[1], reverse=True)
    print('\n'.join(['%s: %04f, %04f, %04f' % (m[0], m[1], m[2], m[3]) for m in best_metric]))
    print('')

