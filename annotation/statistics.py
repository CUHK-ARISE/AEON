import csv
import matplotlib.pyplot as plt
import numpy as np


data = []
for dataset_name in ['rtmr', 'yelp', 'mnli', 'snli', 'qqp']:
    with open('result/annotation_%s.csv' % dataset_name) as f:
        r = csv.DictReader(f)
        for row in r:
            if float(row['consistency']) <= 3.0:
                row['inconsistent'] = True
            else: row['inconsistent'] = False
            if float(row['fluency']) <= 3.0:
                row['unnatural'] = True
            else: row['unnatural'] = False
            if int(row['ground_truth']) != int(float(row['label'])):
                row['fa'] = True
            else: row['fa'] = False
            data.append(row)

a1, a2, a3, a4, a5, a6, a7, a8 = 0, 0, 0, 0, 0, 0, 0, 0
for i in data:
    if i['inconsistent'] == False and i['unnatural'] == False and i['fa'] == False:
        a1 += 1
    elif i['inconsistent'] == True and i['unnatural'] == False and i['fa'] == False:
        a2 += 1
    elif i['inconsistent'] == True and i['unnatural'] == False and i['fa'] == True:
        a3 += 1
    elif i['inconsistent'] == True and i['unnatural'] == True and i['fa'] == True:
        a4 += 1
    elif i['inconsistent'] == True and i['unnatural'] == True and i['fa'] == False:
        a5 += 1
    elif i['inconsistent'] == False and i['unnatural'] == True and i['fa'] == False:
        a6 += 1
    elif i['inconsistent'] == False and i['unnatural'] == True and i['fa'] == True:
        a7 += 1
    elif i['inconsistent'] == False and i['unnatural'] == False and i['fa'] == True:
        a8 += 1
print(a1 / 500, a2 / 500, a3 / 500, a4 / 500, a5 / 500, a6 / 500, a7 / 500, a8 / 500)

for dataset_name in ['rtmr', 'yelp', 'mnli', 'snli', 'qqp']:
    with open('result/annotation_%s.csv' % dataset_name) as f:
        r = csv.DictReader(f)
        data = []
        for row in r:
            data.append(row)

    print('Dataset: %s' % dataset_name)
    print('Avg Consistency: %.4f' % np.average(np.array([float(i['consistency']) for i in data])))
    print('Avg Naturalness: %.4f' % np.average(np.array([float(i['fluency']) for i in data])))
    print('Avg Difficulty: %.4f' % np.average(np.array([float(i['difficulty']) for i in data])))
    print('False Alarm Rate: %.2f' % np.average(np.array([int(int(i['ground_truth']) != int(float(i['label']))) for i in data])))

    data.sort(key=lambda x: float(x['SemEval']), reverse=True)
    c = []
    f = []
    x = []
    for idx in range(len(data)):
        copy_data = data[:idx + 1]
        c.append(np.average(np.array([float(i['consistency']) for i in copy_data])))
        f.append(np.average(np.array([int(int(i['ground_truth']) != int(float(i['label']))) for i in copy_data])))
        x.append(idx)

    data.sort(key=lambda x: float(x['SynEval']), reverse=True)
    n = []
    d = []
    for idx in range(len(data)):
        copy_data = data[:idx + 1]
        n.append(np.average(np.array([float(i['fluency']) for i in copy_data])))
        d.append(np.average(np.array([float(i['difficulty']) for i in copy_data])))

#     plt.plot(x, c, color='r')
#     plt.plot(x, n, color='g')
#     plt.plot(x, d, color='b')
#     plt.plot(x, f, color='c')
#     plt.savefig('result.png')

    print(c[25])
    print(n[30])
    print(d[30])
    print(f[25])

    data.sort(key=lambda x: float(x['SemEval']), reverse=True)
    print('Threshold SemEval: %.4f' % float(data[25]['SemEval']))
    data.sort(key=lambda x: float(x['SynEval']), reverse=True)
    print('Threshold SynEval: %.4f' % float(data[30]['SynEval']))