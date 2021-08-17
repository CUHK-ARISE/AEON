import csv
from nltk.tokenize import word_tokenize


root = 'data/textattack/%s-bert-%s.csv'
#'alzantot', 'bae', 'checklist', 'pso', 'textfooler'
used_methods = ['pso']
#'rtmr', 'imdb', 'mnli', 'qqp', 'yelp', 'agnews', 'snli'
used_datasets = ['rtmr', 'imdb', 'mnli', 'qqp', 'yelp', 'agnews', 'snli']

class Sample():
    def __init__(i, row, dataset, method):
        self.id = i
        self.dataset = dataset

all_success = []
sample_len = {'rtmr': [0] * 400, 'imdb': [0] * 400, 'mnli': [0] * 400, 'qqp': [0] * 400,
              'yelp': [0] * 400, 'agnews': [0] * 400, 'snli': [0] * 400}
sample_conf = {'rtmr': [0] * 400, 'imdb': [0] * 400, 'mnli': [0] * 400, 'qqp': [0] * 400,
               'yelp': [0] * 400, 'agnews': [0] * 400, 'snli': [0] * 400}
skipped_sample = {'rtmr': [], 'imdb': [], 'mnli': [], 'qqp': [], 'yelp': [], 'agnews': [], 'snli': []}
for method in used_methods:
    for dataset in used_datasets:
        with open(root % (method, dataset)) as f_csv:
            r_csv = csv.DictReader(f_csv)
            for i, row in enumerate(r_csv):
                if row['result_type'] == 'Successful':
                    all_success.append({'id': i, 'method': method, 'dataset': dataset})
                if method == 'pso':
                    sample_len[dataset][i] = len(word_tokenize(row['original_text'].replace('[[', '').replace(']]', '')))
                    sample_conf[dataset][i] = 1 - float(row['original_score'])
                if row['result_type'] == 'Skipped' and method == 'bae':
                    skipped_sample[dataset].append(i)

success_method = {'rtmr': [], 'imdb': [], 'mnli': [], 'qqp': [], 'yelp': [], 'agnews': [], 'snli': []}
for k in success_method.keys():
    for i in range(400):
        success_method[k].append([])

for i in all_success:
    success_method[i['dataset']][i['id']].append(i['method'])

count_success = {}
for k in success_method.keys():
    print(k)
    count_success[k] = [len(i) for i in success_method[k]]
    for i in skipped_sample[k]:
        count_success[k][i] = -1
    print('Skipped: %d' % count_success[k].count(-1))
    for i in range(6):
        print('%d: %d' % (i, count_success[k].count(i)))

for k in success_method.keys():
    print(k)
    for i in range(-1, 2):
        count = 0
        avg_len = 0
        for j in range(len(count_success[k])):
            if count_success[k][j] == i:
                count += 1
                avg_len += sample_len[k][j]
        print('%d: %.2f' % (i, avg_len / count))
    print('All: %.2f' % (sum(sample_len[k]) / len(sample_len[k])))

for k in success_method.keys():
    print(k)
    for i in range(-1, 2):
        count = 0
        avg_conf = 0
        for j in range(len(count_success[k])):
            if count_success[k][j] == i:
                count += 1
                avg_conf += sample_conf[k][j]
        print('%d: %.2f' % (i, avg_conf / count))
    print('All: %.2f' % (sum(sample_conf[k]) / len(sample_conf[k])))