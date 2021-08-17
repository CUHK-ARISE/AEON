with open('data/textattack/datasets/agnews_seeds.txt') as f:
    dataset = f.read().splitlines()

for i in range(len(dataset)):
    dataset[i] = dataset[i].split('\t')
    dataset[i] = tuple((' '.join(dataset[i][:-1]), int(dataset[i][-1])))
    assert len(dataset[i]) == 2
