import argparse
from evaluator.utils import *


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--masked-lm', type=str, default='bert-base-uncased', help='PLM for mask prediction.')
    parser.add_argument('--embed-lm', type=str, default='princeton-nlp/sup-simcse-bert-base-uncased', help='PLM for emebdding.')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size.')
    parser.add_argument('--ori-data', type=str, default='data/test_ori.txt', help='Original texts, one per line.')
    parser.add_argument('--adv-data', type=str, default='data/test_adv.txt', help='Adversarial texts, one per line.')
    parser.add_argument('--save-file', type=str, default='', help='Save scores to file.')
    parser.add_argument('--verbose', action='store_true', default=True, help='Print more information.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_config()
    
    scorer = Scorer(args.batch_size, args.masked_lm, args.embed_lm, args.verbose)
    
    ori_texts, adv_texts = load_data(args.ori_data, args.adv_data)
    
    scorer.compute(ori_texts, adv_texts, args.save_file)

