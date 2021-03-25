import argparse
from evaluator.utils import *


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bert-path', type=str, default='bert-base-uncased', help='Transformers-type pre-trained model.')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU Id.')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size.')
    parser.add_argument('--ori-data', type=str, default='data/ori.txt', help='Original texts, one per line.')
    parser.add_argument('--adv-data', type=str, default='data/adv.txt', help='Adversarial texts, one per line.')
    parser.add_argument('--save-file', type=str, default='', help='Save scores to file.')
    parser.add_argument('--verbose', action='store_true', default=False, help='Print more information.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_config()
    
    score = Scoring(args.gpu_id, args.batch_size, args.bert_path, args.save_file, args.verbose)
    
    ori_texts, adv_texts = load_data(args.ori_data, args.adv_data)
    
    score.from_files(ori_texts, adv_texts, args.save_file)

