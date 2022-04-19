from scipy import stats
import csv


datasets = ['rtmr', 'yelp', 'qqp', 'snli', 'mnli']


with open('../data/ori.txt') as f:
    ori = f.read().splitlines()
with open('../data/adv.txt') as f:
    adv = f.read().splitlines()


index_to_line_num = {} # method_dataset_num: line number (int)
baseline = {} # method_dataset_num: {baselines (string)}
with open('../data/info.csv') as csv_f:
    csv_r = csv.DictReader(csv_f)
    count = 0
    for row in csv_r:
        index = '%s_%s_%s' % (row['attack_method'], row['dataset'], row['number'])
        index_to_line_num[index] = count
        baseline[index] = {key: row[key] for key in row.keys()}
        count += 1

for dataset in datasets:

    kappa_label = []
    kappa_difficulty = []
    kappa_consistency = []
    kappa_fluency = []

    annotation_path = 'raw_annotation/%s_100.csv' % dataset

    with open(annotation_path) as csv_f:
        csv_r = csv.DictReader(csv_f)
        annotation_data = [i for i in csv_r]

    annotation = {} # method_dataset_num_question: [annotation] (int)
    for k in annotation_data[0].keys():
        if k.find(dataset) > -1:
            for i in annotation_data[2:]:
                if i[k] != '':
                    if k not in annotation.keys():
                        annotation[k] = [int(i[k])]
                    else: annotation[k].append(int(i[k]))

    save_file = {} # method_dataset_num: {all info}
    for k in annotation.keys():
        method, dataset, num, question = k.split('_')[:4]

        if question == 'label':
            # Adjust definition of label
            if dataset in ['rtmr', 'yelp', 'qqp']: annotation[k] = [2 - i for i in annotation[k]]
            elif dataset in ['snli', 'mnli']: annotation[k] = [i - 1 for i in annotation[k]]

            # Quality control
            label = [0, 0, 0]
            for i in annotation[k]:
                label[i] += 1
            if dataset in ['rtmr', 'yelp', 'qqp'] and label[1] == label[0]:
                print('dataset %s attack_method %s number %s\n' % (dataset, method, num))
            elif dataset in ['snli', 'mnli'] and label[0] == label[1] and label[1] == label[2]:
                print('dataset %s attack_method %s number %s\n' % (dataset, method, num))

            annotation[k] = [stats.mode(annotation[k])[0][0]]
            kappa_label.append(label)
        kappa = [0, 0, 0, 0, 0]
        for i in annotation[k]:
            kappa[i - 1] += 1
        if question == 'difficulty':
            kappa_difficulty.append(kappa)
        if question == 'consistency':
            kappa_consistency.append(kappa)
        if question == 'fluency':
            kappa_fluency.append(kappa)

        index = '%s_%s_%s' % (method, dataset, num)
        
        if index not in save_file.keys():
            save_file[index] = {key: baseline[index][key] for key in baseline[index].keys()}
            save_file[index]['adv_text'] = adv[index_to_line_num[index]]
            save_file[index]['ori_text'] = ori[index_to_line_num[index]]

        save_file[index][question] = float(sum(annotation[k]) / len(annotation[k]))

    # print('\n'.join([','.join([str(j) for j in i]) for i in kappa_label]))
    
    with open('result/annotation_%s.csv' % dataset, 'w') as f_csv:
        csv_w = csv.DictWriter(f_csv, fieldnames=save_file[list(save_file.keys())[0]].keys())
        csv_w.writeheader()
        for k in save_file.keys():
            csv_w.writerow(save_file[k])
