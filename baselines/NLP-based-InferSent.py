import torch
from models import InferSent
import numpy as np

def cos_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def distance(a, b):
    return np.sqrt(sum((a - b) ** 2))

V = 2
K = 100000
MODEL_PATH = 'encoder/infersent%s.pkl' % V
W2V_PATH = 'fastText/crawl-300d-2M.vec'
ADV_PATH = '../../data/adv.txt'
ORI_PATH = '../../data/ori.txt'

with open(ADV_PATH) as f:
    adv = f.read()
    adv = adv.replace('[[[[Premise]]]]: ', '').replace('>>>>[[[[Hypothesis]]]]:', '')
    adv = adv.replace('[[', '').replace(']]', '')
    adv = adv.splitlines()

with open(ORI_PATH) as f:
    ori = f.read()
    ori = ori.replace('[[[[Premise]]]]: ', '').replace('>>>>[[[[Hypothesis]]]]:', '')
    ori = ori.replace('[[', '').replace(']]', '')
    ori = ori.splitlines()


params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': V}
infersent = InferSent(params_model)
infersent.load_state_dict(torch.load(MODEL_PATH))
infersent.set_w2v_path(W2V_PATH)
infersent.build_vocab_k_words(K)

adv_emb = infersent.encode(adv, tokenize=True)
ori_emb = infersent.encode(ori, tokenize=True)

result = [cos_sim(i, j) for i, j in zip(adv_emb, ori_emb)]
with open ('../results/InferSent.txt', 'w') as f:
    f.write('\n'.join([str(i) for i in result]))

result = [distance(i, j) for i, j in zip(adv_emb, ori_emb)]
with open ('../results/InferSent_distance.txt', 'w') as f:
    f.write('\n'.join([str(i) for i in result]))