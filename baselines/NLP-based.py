from simcse import SimCSE
from bert_score import BERTScorer
from sentence_transformers import SentenceTransformer
import numpy as np
from nltk import word_tokenize, translate

def cos_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def distance(a, b):
    return np.sqrt(sum((a - b) ** 2))

ADV_PATH = '../data/adv.txt'
ORI_PATH = '../data/ori.txt'

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

result = [translate.meteor_score.single_meteor_score(i, j) for i, j in zip(adv, ori)]
with open ('results/Meteor.txt', 'w') as f:
    f.write('\n'.join([str(i) for i in result]))

result = [translate.bleu_score.sentence_bleu([word_tokenize(i)], word_tokenize(j)) for i, j in zip(adv, ori)]
with open ('results/BLEU.txt', 'w') as f:
    f.write('\n'.join([str(i) for i in result]))

model = SimCSE("princeton-nlp/sup-simcse-bert-base-uncased")

adv_emb = model.encode(adv).cpu().detach().numpy()
ori_emb = model.encode(ori).cpu().detach().numpy()

result = [cos_sim(i, j) for i, j in zip(adv_emb, ori_emb)]
with open ('results/SimCSE.txt', 'w') as f:
    f.write('\n'.join([str(i) for i in result]))

result = [distance(i, j) for i, j in zip(adv_emb, ori_emb)]
with open ('results/SimCSE_distance.txt', 'w') as f:
    f.write('\n'.join([str(i) for i in result]))

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

adv_emb = model.encode(adv)
ori_emb = model.encode(ori)

result = [cos_sim(i, j) for i, j in zip(adv_emb, ori_emb)]
with open ('results/SBERT.txt', 'w') as f:
    f.write('\n'.join([str(i) for i in result]))

result = [distance(i, j) for i, j in zip(adv_emb, ori_emb)]
with open ('results/SBERT_distance.txt', 'w') as f:
    f.write('\n'.join([str(i) for i in result]))

model = BERTScorer(model_type='bert-base-uncased', idf=False)

SCORE_TYPE2IDX = {"precision": 0, "recall": 1, "f1": 2}
result = model.score(adv, ori)
result = result[SCORE_TYPE2IDX['f1']].numpy()

with open ('results/BERTScore.txt', 'w') as f:
    f.write('\n'.join([str(i) for i in result]))
