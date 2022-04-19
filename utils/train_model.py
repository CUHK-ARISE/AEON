import csv
import torch
import random
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from datasets import load_metric

dataset_name = 'rotten-tomatoes'
train_path = 'rtmr_train.txt'
test_path = 'rtmr_test.txt'
adv_files = ['../data/pso-bert-rtmr-train.csv', '../data/bae-bert-rtmr-train.csv', '../data/textfooler-bert-rtmr-train.csv', '../data/alzantot-bert-rtmr-train.csv', '../data/checklist-bert-rtmr-train.csv']
adv_scores = 'scores.txt'
sample_point_num = 20


def load_data(data_path):
    with open(data_path) as f:
        dataset = f.read()
    for i in range(161):
        if (i < 32 and i != 10 and i != 9) or i > 127:
            dataset = dataset.replace(chr(i), ' ')
    dataset = dataset.splitlines()
    dataset_text = [i.split('\t')[0] for i in dataset]
    dataset_label = [int(i.split('\t')[1]) for i in dataset]
    
    return dataset_text, dataset_label


def load_adv_data(adv_files):
    adv_text = []
    adv_label = []
    delete_tokens = ['[[', ']]', 'Premise: ', '>>>>Hypothesis:', 'Question1: ', '>>>>Question2:']
    for file in adv_files:
        with open(file) as f_csv:
            r_csv = csv.DictReader(f_csv)
            for i, row in enumerate(r_csv):
                if row['result_type'] == 'Successful':
                    text = row['perturbed_text']
                    for tok in delete_tokens:
                        text = text.replace(tok, '')
                    adv_text.append(text)
                    adv_label.append(int(float(row['ground_truth_output'])))
    return adv_text, adv_label


def select_adv_data(adv_text, adv_label, use_data=None, adv_scores=''):
    if use_data == None: use_data = len(adv_text)
    
    if adv_scores != '':
        with open(adv_scores) as f:
            scores = f.read().splitlines()
        adv = [[adv_text[i], adv_label[i]] + scores[i].split(',') for i in range(len(adv_text))]

        adv.sort(key=lambda x: x[2], reverse=True)
    else:
        adv = [[adv_text[i], adv_label[i]] for i in range(len(adv_text))]
        random.shuffle(adv)
    
    return [i[0] for i in adv[:use_data]], [i[1] for i in adv[:use_data]]


class AdvDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('Using %s' % 'GPU' if torch.cuda.is_available() else 'CPU')
model_name = 'bert-base-uncased'

tokenizer = AutoTokenizer.from_pretrained(model_name)

train_texts, train_labels = load_data(train_path)
adv_text, adv_label = load_adv_data(adv_files)

print('Dataset size: train: %d, adv: %d' % (len(train_texts), len(adv_text)))

test_texts, test_labels = load_data(test_path)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)
test_dataset = AdvDataset(test_encodings, test_labels)

test_loader = DataLoader(test_dataset, batch_size=64)

for i in range(sample_point_num):
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)
    
    use_data = int(len(adv_text) * (i + 1) / sample_point_num)
    print('Use %d data' % use_data)
    selected_adv_text, selected_adv_label = select_adv_data(adv_text, adv_label, use_data, adv_scores)
    train_texts = train_texts + selected_adv_text
    train_labels = train_labels + selected_adv_label
    
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    train_dataset = AdvDataset(train_encodings, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    optim = AdamW(model.parameters(), lr=5e-5)

    for epoch in range(10):
        model.train()
        print('Epoch %d' % epoch)
        for batch in train_loader:
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            loss.backward()
            optim.step()

        metric= load_metric("accuracy")
        model.eval()
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])

        print(metric.compute())
    model.save_pretrained('./experiments/all_rank/'+str(use_data))
