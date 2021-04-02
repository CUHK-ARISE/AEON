with open('data/textattack/datasets/rtmr_seeds.txt') as f:
    dataset = f.read().splitlines()

for i in range(len(dataset)):
    dataset[i] = dataset[i].split('\t')
    dataset[i][1] = int(dataset[i][1])
    dataset[i] = tuple(dataset[i])
    assert len(dataset[i]) == 2
