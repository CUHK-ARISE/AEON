import time
import numpy as np

import torch
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity as cos_sim


def get_time():
    return time.strftime('%Y-%m-%d %H:%M:%S ', time.localtime())


def load_data(ori_path, adv_path):
    print(get_time() + '[INFO] Loading original texts from: %s' % ori_path)
    print(get_time() + '[INFO] Loading adversarial texts from: %s' % adv_path)
    clean_list = ['[[', ']]', 'Premise: ', '>>>>Hypothesis: ', 'Question1: ', '>>>>Question2: ']
    with open(ori_path) as f:
        raw = f.read()
        for i in clean_list: raw = raw.replace(i, '')
        ori_texts = [i for i in raw.splitlines()]
    with open(adv_path) as f:
        raw = f.read()
        for i in clean_list: raw = raw.replace(i, '')
        adv_texts = [i for i in raw.splitlines()]
    return ori_texts, adv_texts


def zero_padding(token_list):
    length = max([len(i) for i in token_list])
    return [i + [0] * (length - len(i)) for i in token_list]


def edit_distance(ori_tokens, adv_tokens):
    ori_len = len(ori_tokens)
    adv_len = len(adv_tokens)
    matrix = [[i + j for j in range(adv_len + 1)] for i in range(ori_len + 1)]
    operation = [['' for j in range(adv_len + 1)] for i in range(ori_len + 1)]
    for i in range(1, ori_len + 1):
        operation[i][0] = operation[i - 1][0] + ',d %d %d;a %d %d' % (i - 1, ori_tokens[i - 1], i - 1, ori_tokens[i - 1])
    for j in range(1, adv_len + 1):
        operation[0][j] = operation[0][j - 1] + ',a %d %d;d %d %d' % (j - 1, adv_tokens[j - 1], j - 1, adv_tokens[j - 1])
    
    for i in range(1, ori_len + 1):
        for j in range(1, adv_len + 1):
            cost = 0 if ori_tokens[i - 1] == adv_tokens[j - 1] else 1
            matrix[i][j] = min(
                matrix[i][j - 1] + 1,
                matrix[i - 1][j] + 1,
                matrix[i - 1][j - 1] + cost
            )
            if matrix[i][j] == matrix[i - 1][j - 1] + cost:
                operation[i][j] = operation[i - 1][j - 1]
                if cost == 1:
                    operation[i][j] += ',c %d %d %d;c %d %d %d' % \
                                       (i - 1, ori_tokens[i - 1], adv_tokens[j - 1],
                                        j - 1, adv_tokens[j - 1], ori_tokens[i - 1])
            elif matrix[i][j] == matrix[i][j - 1] + 1:
                operation[i][j] = operation[i][j - 1] + ',a %d %d;d %d %d' % (i, adv_tokens[j - 1], j - 1, adv_tokens[j - 1])
            elif matrix[i][j] == matrix[i - 1][j] + 1:
                operation[i][j] = operation[i - 1][j] + ',d %d %d;a %d %d' % (i - 1, ori_tokens[i - 1], j, ori_tokens[i - 1])
    
    return matrix[-1][-1], operation[-1][-1]


class Scorer(object):
    def __init__(self, batch_size, masked_lm, embed_lm, verbose):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(get_time() + '[INFO] Using %s' % self.device)
        
        self.batch_size = batch_size
        print(get_time() + '[INFO] Using batch size: %d' % batch_size)
        
        self.tokenizer_masked_lm = AutoTokenizer.from_pretrained(masked_lm)
        self.model_masked_lm = AutoModelForMaskedLM.from_pretrained(masked_lm)
        self.model_masked_lm.eval()
        self.model_masked_lm = self.model_masked_lm.to(self.device)
        print(get_time() + '[INFO] Masked LM model: %s' % masked_lm)
        
        self.tokenizer_embed_lm = AutoTokenizer.from_pretrained(embed_lm)
        self.model_embed_lm = AutoModel.from_pretrained(embed_lm)
        self.model_embed_lm.eval()
        self.model_embed_lm = self.model_embed_lm.to(self.device)
        self.pooler = 'cls_before_pooler' if 'unsup' in embed_lm else 'cls'
        print(get_time() + '[INFO] Embedding model: %s' % embed_lm)
        
        self.verbose = verbose
        if self.verbose: print(get_time() + '[INFO] Printing more information (verbose mode)')
        
        self.lambda1 = 0.1
        self.lambda2 = 0.2
        self.phi = 0.6
    
    def text_preprocese(self, text):
        # Different models have different mask tokens (e.g., for BERT it's [MASK], for RoBERTa it's <mask>)
        mask_token = self.tokenizer_masked_lm.mask_token_id
        
        # Split sentences
        sentences = sent_tokenize(text)
        # Tokenize words
        tokenize_sentences = [self.tokenizer_masked_lm.encode(i) for i in sentences]
        
        masked_sentences = []
        for sent in tokenize_sentences:
            masked_sentences.append([])
            for masked_index in range(1, len(sent) - 1):
                masked_sentences[-1].append(sent[:masked_index] + [mask_token] + sent[masked_index + 1:])
        
        return masked_sentences, sentences

    def text_naturalness(self, masked_sentences, sentences):
        # Prepare masked inputs and their labels (real words)
        masked_input = []
        label_input = []
        for masked_sent, sent in zip(masked_sentences, sentences):
            label = self.tokenizer_masked_lm.encode(sent)[1:-1]
            for i in range(len(masked_sent)):
                masked_input.append(masked_sent[i])
                label_input.append(label[i])
        
        masked_input = torch.tensor(zero_padding(masked_input)).to(self.device)
        
        with torch.no_grad():
            outputs = []
            for i in range(0, len(label_input), self.batch_size):
                output = self.model_masked_lm(masked_input[i:min(i + self.batch_size, len(label_input))])
                outputs.append(output.logits.cpu())
            outputs = torch.cat(outputs, axis=0)
            predictions = torch.softmax(outputs, -1)
        
        # Find the prediction score of the real words
        probs = []
        for i in range(len(predictions)):
            idx = torch.argmax((masked_input[i] == self.tokenizer_masked_lm.mask_token_id).int())
            probs.append(predictions[i][idx][label_input[i]].item())
        
        # Average and min of probabilities
        avg_naturalness = np.average(probs)
        min_naturalness = np.min(probs)
        naturalness = self.phi * min_naturalness + (1 - self.phi) * avg_naturalness
        
        return naturalness

    def pair_texts_similarity(self, ori_sentences, adv_sentences):
        cls_token = self.tokenizer_embed_lm.cls_token_id
        sep_token = self.tokenizer_embed_lm.sep_token_id
        
        ori_sentences = self.tokenizer_embed_lm(ori_sentences)['input_ids']
        adv_sentences = self.tokenizer_embed_lm(adv_sentences)['input_ids']
        
        ori_exclude_ids = []
        adv_exclude_ids = []
        for ori in range(len(ori_sentences)):
            for adv in range(len(adv_sentences)):
                if ori not in ori_exclude_ids and adv not in adv_exclude_ids:
                    distance, operations = edit_distance(ori_sentences[ori], adv_sentences[adv])
                    if distance == 0:
                        ori_exclude_ids.append(ori)
                        adv_exclude_ids.append(adv)
                        break
        ori_input = []
        for i in range(len(ori_sentences)):
            if i not in ori_exclude_ids:
                ori_input += ori_sentences[i][1:-1]
        ori_input = [cls_token] + ori_input + [sep_token]
        adv_input = []
        for i in range(len(adv_sentences)):
            if i not in adv_exclude_ids:
                adv_input += adv_sentences[i][1:-1]
        adv_input = [cls_token] + adv_input + [sep_token]
        
        distance, operations = edit_distance(ori_input, adv_input)
        if distance == 0:
            return 1.0
        operations = operations[1:].split(',')
        operations_o = [int(o.split(';')[0].split()[1]) for o in operations]
        operations_a = [int(o.split(';')[1].split()[1]) for o in operations]
        partial_ids_o = [[max(operations_o[0] - 2, 0), operations_o[0] + 2]]
        partial_ids_a = [[max(operations_a[0] - 2, 0), operations_a[0] + 2]]
        for o, a in zip(operations_o[1:], operations_a[1:]):
            if o - 2 < partial_ids_o[-1][1]:
                partial_ids_o[-1][1] = o + 2
            else:
                partial_ids_o.append([o - 2, o + 2])
            if a - 2 < partial_ids_a[-1][1]:
                partial_ids_a[-1][1] = a + 2
            else:
                partial_ids_a.append([a - 2, a + 2])
        
        partial_ori = []
        partial_adv = []
        for o, a in zip(partial_ids_o, partial_ids_a):
            partial_o = ori_input[o[0]:o[1]]
            if partial_o[0] != cls_token: partial_o = [cls_token] + partial_o
            if partial_o[-1] != sep_token: partial_o = partial_o + [sep_token]
            partial_a = adv_input[a[0]:a[1]]
            if partial_a[0] != cls_token: partial_a = [cls_token] + partial_a
            if partial_a[-1] != sep_token: partial_a = partial_a + [sep_token]
            partial_ori.append(partial_o)
            partial_adv.append(partial_a)
        
        if self.verbose:
            for i in range(len(partial_ori)):
                print(get_time() + '[INFO] Modification number: %d' % i)
                print(self.tokenizer_embed_lm.convert_ids_to_tokens(partial_ori[i]))
                print(self.tokenizer_embed_lm.convert_ids_to_tokens(partial_adv[i]))
        
        ori_inputs = [ori_input] + partial_ori
        adv_inputs = [adv_input] + partial_adv
        
        with torch.no_grad():
            ori_sentence_emb = []
            for i in range(len(ori_inputs)):
                output = self.model_embed_lm(torch.tensor(ori_inputs[i]).unsqueeze(0).to(self.device))
                ori_sentence_emb.append(output.pooler_output if self.pooler == 'cls' else output.last_hidden_state[:, 0])
            ori_sentence_emb = torch.cat(ori_sentence_emb, axis=0).cpu()
            
            adv_sentence_emb = []
            for i in range(len(adv_inputs)):
                output = self.model_embed_lm(torch.tensor(adv_inputs[i]).unsqueeze(0).to(self.device))
                adv_sentence_emb.append(output.pooler_output if self.pooler == 'cls' else output.last_hidden_state[:, 0])
            adv_sentence_emb = torch.cat(adv_sentence_emb, axis=0).cpu()
        
        similarity = np.array([cos_sim(o.reshape(1, -1), a.reshape(1, -1))[0][0] \
                               for o, a in zip(ori_sentence_emb.numpy(), adv_sentence_emb.numpy())])
        
        if len(similarity) > 1:
            if self.verbose:
                print(get_time() + '[INFO] Original similarity score: %f' % similarity[0])
                print(get_time() + '[INFO] Average partial similarity score: %f' % np.average(similarity[1:]))
                print(get_time() + '[INFO] Minimum partial similarity score: %f' % similarity[1:].min())
                print(similarity[1:])
                all_sim, avg_sim, min_sim = similarity[0], np.average(similarity[1:]), similarity[1:].min()
            else:
                all_sim, avg_sim, min_sim = similarity[0], similarity[0], similarity[0]
        
        similarity = self.lambda1 * min_sim + self.lambda2 * avg_sim + (1 - self.lambda1 - self.lambda2) * all_sim
        
        return similarity
    
    def compute(self, ori_texts, adv_texts, save_file=''):
        print(get_time() + '[INFO] Calculating scores\n\n')
        
        assert type(ori_texts) == type(adv_texts)
        assert type(ori_texts) == str or type(ori_texts) == list
        assert type(adv_texts) == str or type(adv_texts) == list
        
        is_single = False
        if type(ori_texts) == str and type(adv_texts) == str:
            is_single = True
            ori_texts = [ori_texts]
            adv_texts = [adv_texts]
        
        ret = []
        for i, (ori, adv) in enumerate(zip(ori_texts, adv_texts)):
            masked_ori_sentences, ori_sentences = self.text_preprocese(ori)
            o_naturalness = self.text_naturalness(masked_ori_sentences, ori_sentences)
            
            masked_adv_sentences, adv_sentences = self.text_preprocese(adv)
            a_naturalness = self.text_naturalness(masked_adv_sentences, adv_sentences)
            
            similarity = self.pair_texts_similarity(ori_sentences, adv_sentences)
            
            print(get_time() + '[INFO] Finish text number %d:' % i)
            if self.verbose:
                print('Original text:\n%s' % ori)
                print('Adversarial text:\n%s' % adv)
                print('Original Syntactic Score: %.6f' % (o_naturalness))
                print('Adversarial Syntactic Score: %.6f' % (a_naturalness))
                print('Semantic Score: %.6f\n' % similarity)
            
            ret.append([similarity, a_naturalness, o_naturalness])
        
        if save_file != '':
            print(get_time() + '[INFO] Scores are saved in: %s' % save_file)
            with open(save_file, 'w') as f:
                f.write('\n'.join([','.join([str(j) for j in i]) for i in ret]))
        
        if is_single: ret = ret[0]
        return ret
