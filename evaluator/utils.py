import time
import numpy as np

import torch
from transformers import BertTokenizer, BertModel, BertForPreTraining
from nltk.tokenize import sent_tokenize


def get_time():
    return time.strftime('%Y-%m-%d %H:%M:%S ', time.localtime())


def load_data(ori_path, adv_path):
    print(get_time() + '[INFO] Loading original texts from: %s' % ori_path)
    print(get_time() + '[INFO] Loading adversarial texts from: %s' % adv_path)
    clean_list = ['[[', ']]', '\x85', '<br />', 'Premise: ', '>>>>Hypothesis: ']
    with open(ori_path) as f:
        raw = f.read()
        for i in clean_list: raw = raw.replace(i, '')
        ori_texts = [i for i in raw.splitlines()]
    with open(adv_path) as f:
        raw = f.read()
        for i in clean_list: raw = raw.replace(i, '')
        adv_texts = [i for i in raw.splitlines()]
    return ori_texts, adv_texts


class Scoring(object):
    def __init__(self, gpu_id, batch_size, bert_path, save_file, verbose):
        print(get_time() + '[INFO] Initializing model: %s' % bert_path)
        print(get_time() + '[INFO] Using GPU Id: %d' % gpu_id)
        print(get_time() + '[INFO] Using batch size: %d' % batch_size)
        if verbose: print(get_time() + '[INFO] Printing more information (verbose mode)')
        self.verbose = verbose
        self.gpu_id = gpu_id
        self.batch_size = batch_size
        self.model_vocab = BertForPreTraining.from_pretrained(bert_path)
        self.model_encode = BertModel.from_pretrained(bert_path)
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        self.model_vocab.eval()
        self.model_vocab.cuda(self.gpu_id)
        self.model_encode.eval()
        self.model_encode.cuda(self.gpu_id)
        self.utils_path = 'evaluator/vocabulary_table/'
        with open(self.utils_path + 'inc') as inc, open(self.utils_path + 'dec') as dec, open(self.utils_path + 'inv') as inv, open(self.utils_path + 'positive') as pos, open(self.utils_path + 'negative') as neg:
            self.inc_word_list = inc.read().splitlines()
            self.dec_word_list = dec.read().splitlines()
            self.inv_word_list = inv.read().splitlines()
            self.pos_word_list = pos.read().splitlines()
            self.neg_word_list = neg.read().splitlines()

    def sentence_preprocese(self, text):
        sent_list = sent_tokenize(text)
        tokenize_sent_list = [self.tokenizer.tokenize(i) for i in sent_list]
        total_tokens = sum([len(i) for i in tokenize_sent_list])
        tokenized_para_list = [[101] + self.tokenizer.convert_tokens_to_ids(i) + [102] for i in tokenize_sent_list]
        
        masked_para_list = []
        for para in tokenized_para_list:
            masked_para_list.append([])
            for masked_index in range(1, len(para) - 1):
                new_tokenized_text = np.array(para, dtype=int)
                new_tokenized_text[masked_index] = self.tokenizer.convert_tokens_to_ids(['[MASK]'])[0]
                masked_para_list[-1].append(new_tokenized_text)
        
        return masked_para_list, tokenized_para_list, total_tokens

    def single_sentence_grammar(self, indexed_tokens_list, real_indexs_list, total_tokens):
        # Pseudo Log-Likelihood
        pll = 0
        pll_upper = 0
        
        for indexed_tokens, real_indexs in zip(indexed_tokens_list, real_indexs_list):
            tokens_tensor = torch.tensor(indexed_tokens)
            tokens_tensor = tokens_tensor.cuda(self.gpu_id)

            with torch.no_grad():
                outputs = []
                for i in range(0, len(tokens_tensor), self.batch_size):
                    outputs.append(self.model_vocab(tokens_tensor[i:min(i + self.batch_size, len(tokens_tensor))])[0])
                outputs = torch.cat(outputs, axis=0)
                predictions = torch.softmax(outputs, -1)

            temp = 0
            for i in range(len(indexed_tokens)):
                predicted_index = torch.argmax(predictions[i][i + 1]).item()
                predicted_token = self.tokenizer.convert_ids_to_tokens([predicted_index])
                predicted_prob = predictions[i][i + 1][predicted_index].item()

                real_pos_prob = predictions[i][i + 1][real_indexs[i + 1]].item()
                real_token = self.tokenizer.convert_ids_to_tokens([real_indexs[i + 1]])

                temp += np.log2(real_pos_prob)
                pll_upper += np.log2(predicted_prob)
            pll += temp
        
        # Pseudo PerPLexity
        pppl = np.exp2(-pll / total_tokens)
        pppl_lower = np.exp2(-pll_upper / total_tokens)

        return pppl

    def polarity(self, ori, adv):
        value_matrix = [[1, -1, 1, 0, -1, 0],
                        [-1, 1, 0, 1, -1, 0],
                        [1, 0, 1, -1, -1, 0],
                        [0, 1, -1, 1, -1, 0],
                        [-1, -1, -1, -1, 1, -1],
                        [0, 0, 0, 0, -1, 1],]
        
        if ori in self.inc_word_list: ori = 0
        elif ori in self.dec_word_list: ori = 1
        elif ori in self.pos_word_list: ori = 2
        elif ori in self.neg_word_list: ori = 3
        elif ori in self.inv_word_list: ori = 4
        else: ori = 5
        
        if adv in self.inc_word_list: adv = 0
        elif adv in self.dec_word_list: adv = 1
        elif adv in self.pos_word_list: adv = 2
        elif adv in self.neg_word_list: adv = 3
        elif adv in self.inv_word_list: adv = 4
        else: adv = 5
        
        return value_matrix[ori][adv]
    
    def pair_sentence_semantic(self, ori_text, adv_text):
        # Padding
        ori_sent_lens = [len(i) for i in ori_text]
        adv_sent_lens = [len(i) for i in adv_text]
        max_sent_len = max(ori_sent_lens + adv_sent_lens)
        sent_list = [i + [0] * (max_sent_len - len(i)) for i in (ori_text + adv_text)]
        
        # Forwarding
        tokens_tensor = torch.tensor(sent_list)
        tokens_tensor = tokens_tensor.cuda(self.gpu_id)
        with torch.no_grad():
            outputs_token = []
            outputs_sent = []
            for i in range(0, len(tokens_tensor), self.batch_size):
                output = self.model_encode(tokens_tensor[i:min(i + self.batch_size, len(tokens_tensor))])
                outputs_token.append(output[0])
                outputs_sent.append(output[1])
            outputs_token = torch.cat(outputs_token, axis=0)
            outputs_sent = torch.cat(outputs_sent, axis=0)
        
        # Embedding of tokens in each sentence
        ori_token_emb = outputs_token[:len(ori_text)].cpu().detach().numpy()
        adv_token_emb = outputs_token[-len(adv_text):].cpu().detach().numpy()
        
        # Pooling embedding of sentences
        ori_sent_emb = outputs_sent[:len(ori_text)].cpu().detach().numpy()
        adv_sent_emb = outputs_sent[-len(adv_text):].cpu().detach().numpy()
        
        # Normalize to unit
        ori_sent_emb = np.array([i / np.linalg.norm(i) for i in ori_sent_emb])
        adv_sent_emb = np.array([i / np.linalg.norm(i) for i in adv_sent_emb])
        
        # Sentence cosine similarity normalized to [0, 1]
        sent_cos_sim = (np.dot(ori_sent_emb, adv_sent_emb.T) + 1) / 2
        
        # Greedily maximize cosine similarity
        ori_sent_max = np.max(sent_cos_sim, axis=1)
        adv_sent_max = np.max(sent_cos_sim, axis=0)
        
        # Greedy matching with maximum
        ori_sent_match = np.argmax(sent_cos_sim, axis=1)
        adv_sent_match = np.argmax(sent_cos_sim, axis=0)
        
        # Sentence similarity score without weighting
        sent_sim_score = (np.prod(ori_sent_max) * np.prod(adv_sent_max)) ** 0.5
        
        # Sentence similarity score weights using token similarity score
        match_list = [(i, j) for i, j in enumerate(ori_sent_match)] + [(j, i) for i, j in enumerate(adv_sent_match)]
        match_list = list(set(match_list))
        token_score_mat = np.zeros_like(sent_cos_sim)
        for i, j in match_list:
            # Normalize to unit; Delete [CLS] and [SEP]
            ori_emb_unit = np.array([k / np.linalg.norm(k) for k in ori_token_emb[i][1:ori_sent_lens[i] - 1]])
            adv_emb_unit = np.array([k / np.linalg.norm(k) for k in adv_token_emb[j][1:adv_sent_lens[j] - 1]])
            
            # Token cosine similarity normalized to [0, 1]
            token_cos_sim = (np.dot(ori_emb_unit, adv_emb_unit.T) + 1) / 2
            
            # Greedily maximize cosine similarity
            ori_token_max = np.max(token_cos_sim, axis=1)
            adv_token_max = np.max(token_cos_sim, axis=0)
            
            # Greedy matching with maximum
            ori_token_match = np.argmax(token_cos_sim, axis=1)
            adv_token_match = np.argmax(token_cos_sim, axis=0)
            
            # Token similarity score without weighting
            token_sim_score = (np.prod(ori_token_max) * np.prod(adv_token_max)) ** 0.5
            
            # Token similarity score weights using polarity
            ori_token_weights = [self.polarity(self.tokenizer.convert_ids_to_tokens(ori_text[i][k + 1]),
                                               self.tokenizer.convert_ids_to_tokens(adv_text[j][l + 1])) \
                                 for k, l in enumerate(ori_token_match)]
            adv_token_weights = [self.polarity(self.tokenizer.convert_ids_to_tokens(ori_text[i][l + 1]),
                                               self.tokenizer.convert_ids_to_tokens(adv_text[j][k + 1])) \
                                 for k, l in enumerate(adv_token_match)]
            
            # Weighted token similarity score
            ori_token_max = (ori_token_max * np.array(ori_token_weights) + 1) / 2
            adv_token_max = (adv_token_max * np.array(adv_token_weights) + 1) / 2
            Ot = np.prod(ori_token_max)
            At = np.prod(adv_token_max)
            token_score_mat[i][j] = 2 * (Ot * At) / (Ot + At)
        ori_sent_weights = np.array([token_score_mat[i][j] for i, j in enumerate(ori_sent_match)])
        adv_sent_weights = np.array([token_score_mat[j][i] for i, j in enumerate(adv_sent_match)])
        
        # Weighted sentence similarity score
        ori_sent_max = ori_sent_max * ori_sent_weights
        adv_sent_max = adv_sent_max * adv_sent_weights
        Os = np.average(ori_sent_max)
        As = np.average(adv_sent_max)
        weighted_sent_sim_score = 2 * (Os * As) / (Os + As)
        
        return weighted_sent_sim_score
    
    def from_files(self, ori_texts, adv_texts, save_file):
        print(get_time() + '[INFO] Calculating scores\n\n')
        
        save_text = []
        for i, (ori, adv) in enumerate(zip(ori_texts, adv_texts)):
            sentence_pair_scores = []
            
            masked_ori_list, ori_sent_list, ori_total = self.sentence_preprocese(ori)
            masked_adv_list, adv_sent_list, adv_total = self.sentence_preprocese(adv)
            
            o_pppl = self.single_sentence_grammar(masked_ori_list, ori_sent_list, ori_total)
            a_pppl = self.single_sentence_grammar(masked_adv_list, adv_sent_list, adv_total)
            
            weighted_sent_sim = self.pair_sentence_semantic(ori_sent_list, adv_sent_list)
            
            sentence_pair_scores += [(o_pppl / a_pppl), weighted_sent_sim]
            
            print(get_time() + '[INFO] Finish text number %d:' % i)
            if self.verbose:
                print('Original text:\n%s' % ori)
                print('Adversarial text:\n%s' % adv)
                print('Syntax Correctness: %.6f' % (o_pppl / a_pppl))
                print('Semantic Similarity: %.6f\n' % weighted_sent_sim)
            
            save_text.append(', '.join(['%.6f' % i for i in sentence_pair_scores]))
        
        if save_file != '':
            print(get_time() + '[INFO] Scores are saved in: %s' % save_file)
            with open(save_file, 'w') as f:
                f.write('\n'.join(save_text))

