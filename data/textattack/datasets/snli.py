from collections import OrderedDict

with open('data/textattack/datasets/snli_seeds.txt') as f:
    dataset = f.read().splitlines()

label_map = {'0': 1, '1': 2, '2': 0}


for i in range(len(dataset)):
    p, h, l = dataset[i].split('\t')
    dataset[i] = (OrderedDict([('premise', p), ('hypothesis', h)]), label_map[l])
    assert len(dataset[i]) == 2
