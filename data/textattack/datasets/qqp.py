from collections import OrderedDict

with open('data/textattack/datasets/qqp_seeds.txt') as f:
    dataset = f.read().splitlines()

for i in range(len(dataset)):
    q1, q2, l = dataset[i].split('\t')
    dataset[i] = (OrderedDict([('question1', q1), ('question2', q2)]), int(l))
    assert len(dataset[i]) == 2
