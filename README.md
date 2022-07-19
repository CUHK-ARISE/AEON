# AEON

This repo contains the code for [this paper](https://arxiv.org/abs/2205.06439) "AEON: A Method for Automatic Evaluation of NLP Test Cases".

This repo also includes the raw results as well as the questionnaires of the human evaluation mentioned in the paper.

## Install
Installing all the packages using `pip` is suggested:
```
$ pip install -r requirements.txt 
```

## Get started
To use AEON:
```
$ python scorer.py --ori-data PATH_TO_ORI --adv-data PATH_TO_ADV
```
The files `PATH_TO_ORI` and `PATH_TO_ADV` should be lines of texts and be paired. For example, `data/ori.txt` and `data/adv.txt`.

Check these files to see the options.

## Reproduce our Experiments
* Test case generation: please refer to files under `script/` which use seed data in `data/textattack/datasets/`. Seed data need pre-processing (cleaning) using `utils/clean.py`.
* Generate questionnaires for human evaluation: use `utils/user_study.py`.
* Perform robust re-training: use `utils/train_model.py`.
* Raw human evaluation results: see files under `annotation/raw_annotation`. The statistics can be computed using `annotation/statistics.py`.
* Baselines: both NLP-based and NC-based metrics are implemented in `baselines/` (may need extra dependencies to run).
* Raw experiment results are recorded in `annotation/result/`. The AP, AUC, and PCC are calculated using `annotation/AP-AUC-PCC.py`.

## References
For more details, please refer to [this paper](https://arxiv.org/abs/2205.06439). Please remember to cite us if you find our work helpful in your work!
```
@inproceedings{jentseissta2021aeon,
  author    = {Jen{-}tse Huang and
               Jianping Zhang and
               Wenxuan Wang and
               Pinjia He and
               Yuxin Su and
               Michael R. Lyu},
  title     = {{AEON:} {A} Method for Automatic Evaluation of {NLP} Test Cases},
  booktitle = {{ISSTA} '22: 31st {ACM} {SIGSOFT} International Symposium on Software
               Testing and Analysis},
  publisher = {{ACM}},
  year      = {2022},
}
```
