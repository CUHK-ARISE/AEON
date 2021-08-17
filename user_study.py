import csv
import random
from nltk.tokenize import sent_tokenize


root = 'data/textattack/%s-bert-%s.csv'
used_methods = ['alzantot', 'bae', 'checklist', 'pso'] #'alzantot', 'bae', 'checklist', 'pso', 'textfooler'
used_datasets = ['snli'] #'rtmr', 'imdb', 'mnli', 'qqp', 'yelp', 'agnews', 'snli'
completion_code = '40886C35'
num_of_annotate_pair = 100
questions_per_annotator = 30

questionnaire_head = \
'''
[[AdvancedFormat]]

[[Block:Enter ID]]

[[Question:TE:SingleLine]]
[[ID:ID]]
Please enter your Prolific ID:
'''

body_template = \
'''
[[Block:Natural Language Understanding (English) Block %d]]
%s
'''

questionnaire_tail = \
'''
[[Block:Suggestion and Reward Code]]

[[Question:TE:SingleLine]]
[[ID:Suggestion]]
Say something about this survey. Are the questions clear? Do we provide enough choices? Is the workload too big? (Please proceed to the next page to get the completion code)

[[PageBreak]]

[[Question:DB]]
[[ID:Code]]
Thank you for your participation. You can use this code for reward on Prolific: %s
''' % completion_code

label_template = \
'''
[[Question:MC:SingleAnswer]]
[[ID:%s_%s_%d_label]]
Please see the sentence below:
<br /><br />
%s
<br /><br />
Please classify the sentence into one of these categories:

[[AdvancedChoices]]
[[Choice]]
%s

[[Question:Matrix:SingleAnswer]]
[[ID:%s_%s_%d_difficulty]]
Please score how difficult for you to classify the sentence:

[[Choices]]
Difficulty
[[AdvancedAnswers]]
[[Answer]]
Very hard
[[Answer]]
Somewhat hard
[[Answer]]
Neither easy nor hard
[[Answer]]
Somewhat easy
[[Answer]]
Very easy
'''

fluency_template = \
'''
[[Question:Matrix:SingleAnswer]]
[[ID:%s_%s_%d_fluency]]
Please see the sentence below:
<br /><br />
%s
<br /><br />
Please score the fluency (whether it reads smoothly) of the sentence:

[[Choices]]
Fluency
[[AdvancedAnswers]]
[[Answer]]
Very bad
[[Answer]]
Somewhat bad
[[Answer]]
Neither good nor bad
[[Answer]]
Somewhat good
[[Answer]]
Very good
'''

consistency_template = \
'''
[[Question:Matrix:SingleAnswer]]
[[ID:%s_%s_%d_consistency]]
Please see the two sentences below:
<br /><br />
%s
<br /><br />
%s
<br /><br />
Please score how much you think the two sentences have same meaning.

[[Choices]]
The two sentences have same meaning.
[[AdvancedAnswers]]
[[Answer]]
Strongly disagree
[[Answer]]
Somewhat disagree
[[Answer]]
Neither agree nor disagree
[[Answer]]
Somewhat agree
[[Answer]]
Strongly agree
'''


all_data = []
for method in used_methods:
    for dataset in used_datasets:
        with open(root % (method, dataset)) as f_csv:
            r_csv = csv.DictReader(f_csv)
            for i, row in enumerate(r_csv):
                if row['result_type'] == 'Successful':
                    all_data.append([method, dataset, i, row['ground_truth_output'],
                                     row['original_text'], row['perturbed_text']])

print(len(all_data))
random.seed(999)
random.shuffle(all_data)

#with open('data/ori.txt', 'w') as f: f.write('\n'.join([i[4] for i in all_data]))
#with open('data/adv.txt', 'w') as f: f.write('\n'.join([i[5] for i in all_data]))
#with open('data/info.csv', 'w') as f: f.write('attack_method,dataset,number,ground_truth\n' + '\n'.join(['%s,%s,%d,%d' % (i[0], i[1], i[2], int(float(i[3]))) for i in all_data]))

clean_all_data = []
question_pool = []
for data in all_data[:num_of_annotate_pair]:
    # Label and Difficulty
    ori = data[4].replace('[[', '').replace(']]', '')
    adv = data[5].replace('[[', '').replace(']]', '')
    # Add different choices
    if data[1] in ['rtmr', 'imdb', 'yelp']:
        choice = 'Positive Opinion\n[[Choice]]\nNegative Opinion'
    elif data[1] == 'agnews':
        choice = 'World News\n[[Choice]]\nSports News\n[[Choice]]\Business News\n[[Choice]]\nScience or Technology News'
    elif data[1] in ['mnli', 'snli']:
        choice = 'Premise is in conflict with Hypothesis\n[[Choice]]\nPremise entails Hypothesis\n[[Choice]]\nPremise is unrelated to Hypothesis'
    elif data[1] in ['qqp']:
        choice = 'Question 1 and 2 are same questions\n[[Choice]]\nQuestion 1 and 2 are different questions'
    question_pool.append(label_template % (data[0], data[1], data[2], adv, choice, data[0], data[1], data[2]))
    
    # Fluency
    ori = ori.replace('Premise: ', '').replace('>>>>Hypothesis:', '')
    ori = ori.replace('Question1: ', '').replace('>>>>Question2:', '')
    adv = adv.replace('Premise: ', '').replace('>>>>Hypothesis:', '')
    adv = adv.replace('Question1: ', '').replace('>>>>Question2:', '')
    question_pool.append(fluency_template % (data[0], data[1], data[2], adv))
    clean_all_data.append((ori, adv))
    
    # Consistency
    # For convenience, delete unperturbed sentences here
    if data[1] in ['imdb', 'yelp']:
        ori = sent_tokenize(data[4])
        ori = [sent for sent in ori if sent.find('[[') >= 0]
        ori = ' '.join(ori)
        adv = sent_tokenize(data[5])
        adv = [sent for sent in adv if sent.find('[[') >= 0]
        adv = ' '.join(adv)
    ori = ori.replace('[[', '').replace(']]', '')
    adv = adv.replace('[[', '').replace(']]', '')
    ori = ori.replace('Premise: ', '').replace('>>>>Hypothesis:', '')
    adv = adv.replace('Premise: ', '').replace('>>>>Hypothesis:', '')
    ori = ori.replace('Question1: ', '').replace('>>>>Question2:', '')
    adv = adv.replace('Question1: ', '').replace('>>>>Question2:', '')
    question_pool.append(consistency_template % (data[0], data[1], data[2], ori, adv))

#with open('data/ori_clean.txt', 'w') as f: f.write('\n'.join([i[0] for i in clean_all_data]))
#with open('data/adv_clean.txt', 'w') as f: f.write('\n'.join([i[1] for i in clean_all_data]))
random.shuffle(question_pool)

questionnaire_body = []
for i in range(0, len(question_pool), questions_per_annotator):
    block_num = i // questions_per_annotator
    questions = '\n[[PageBreak]]\n'.join(question_pool[i:i + questions_per_annotator])
    questionnaire_body.append(body_template % (block_num, questions))
questionnaire_body = '\n'.join(questionnaire_body)

with open('data/user_study.txt', 'w') as f: f.write(questionnaire_head + questionnaire_body + questionnaire_tail)

