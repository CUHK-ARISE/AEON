# AEON

This README introduces the functionality of each file in this repo, which includes all the cod, experiment results, and raw human annotation files in the paper: AEON: A Method for Automatic Evaluation of NLP Test Cases.

---

`scorer.py` and `utils/aeon.py` contain the code of the implementation of SemEval and SynEval of AEON.

To generate test cases and perform testing on selected models, please refer to files under `script/` which use seed data in `data/textattack/datasets/`. We pre-process the seed data using `utils/clean.py`. All generated test cases and their infomation can be found in `annotation/ori.txt`, `annotation/adv.txt` and `annotation/info.txt`.

We generate our questionnaire for human evaluation using the code in `utils/user_study.py`. `annotation/raw_annotation/` contains the raw data from user study. Quality-control record can be found in `annotation/raw_annotation/check/classification_disagree.txt` and `annotation/raw_annotation/check/high_consistency_wrong_label.txt`.

To reproduce the experiments in RQ1 (statistical analysis of human annotation), please refer to the code in `annotation/parse_annotation.py` and `annotation/statistics.py`.

To reproduce the experiments in RQ2, please refer to the code in `annotation/AP-AUC-PCC.py`, where the implementation of all the baselines methods can be found under `baselines/`.

To reproduce the experiments in RQ3 (test case selection), please refer to the code in `annotation/statistics.py`.

To reproduce the experiments in RQ4 (model re-training), please refer to the code in `utils/train_model.py`.