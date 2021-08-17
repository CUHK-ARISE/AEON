import csv
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW


dataset_name = 'rotten-tomatoes'
adv_file = ['data/textattack/bae-bert-rtmr-train.csv', 'data/textattack/pso-bert-rtmr-train.csv']
data_path = '/research/dept7/jthuang/data/NLP/rtmr/train.txt'


with open(data_path) as f:
    dataset = f.read().replace('\x85', ' ').splitlines()
    dataset_text = [i.split('\t')[0] for i in dataset]
    dataset_label = [int(i.split('\t')[1]) for i in dataset]

adv_text = []
adv_label = []
delete_tokens = ['[[', ']]', 'Premise: ', '>>>>Hypothesis:', 'Question1: ', '>>>>Question2:']
for file in adv_file:
    with open(file) as f_csv:
        r_csv = csv.DictReader(f_csv)
        for i, row in enumerate(r_csv):
            if row['result_type'] == 'Successful':
                text = row['perturbed_text']
                for tok in delete_tokens:
                    text = text.replace(tok, '')
                adv_text.append(text)
                adv_label.append(int(float(row['ground_truth_output'])))

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


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


tokenizer = BertTokenizer.from_pretrained(model_name)

train_texts = dataset_text + adv_text
train_labels = dataset_label + adv_label
print('Size: train: %d, adv: %d' % (len(dataset_text), len(adv_text)))

#test_texts = dataset['test']['text']
#test_labels = dataset['test']['label']

train_encodings = tokenizer(train_texts, truncation=True, padding=True)
#test_encodings = tokenizer(test_texts, truncation=True, padding=True)

train_dataset = AdvDataset(train_encodings, train_labels)
#test_dataset = AdvDataset(test_encodings, test_labels)

model_name = 'textattack/bert-base-uncased-%s' % dataset_name
model = BertForSequenceClassification.from_pretrained(model_name)
model.to(device)
model.train()

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

optim = AdamW(model.parameters(), lr=5e-5)

for epoch in range(3):
    for batch in train_loader:
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        loss.backward()
        optim.step()

model.eval()
