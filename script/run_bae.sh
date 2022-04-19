cd ..

textattack attack --recipe bae \
                  --model bert-base-uncased-imdb \
                  --dataset-from-file data/textattack/datasets/imdb.py \
                  --shuffle False \
                  --num-examples 400 \
                  --log-to-csv data/textattack/bae-bert-imdb.csv \
                  --query-budget 1000 \
                  --model-batch-size 32 \

textattack attack --recipe bae \
                  --model bert-base-uncased-mr \
                  --dataset-from-file data/textattack/datasets/rtmr.py \
                  --shuffle False \
                  --num-examples 400 \
                  --log-to-csv data/textattack/bae-bert-rtmr.csv \
                  --query-budget 1000 \
                  --model-batch-size 32 \

textattack attack --recipe bae \
                  --model bert-base-uncased-ag-news \
                  --dataset-from-file data/textattack/datasets/agnews.py \
                  --shuffle False \
                  --num-examples 400 \
                  --log-to-csv data/textattack/bae-bert-agnews.csv \
                  --query-budget 1000 \
                  --model-batch-size 32 \

textattack attack --recipe bae \
                  --model bert-base-uncased-mnli \
                  --dataset-from-file data/textattack/datasets/mnli.py \
                  --shuffle False \
                  --num-examples 400 \
                  --log-to-csv data/textattack/bae-bert-mnli.csv \
                  --query-budget 1000 \
                  --model-batch-size 32 \

textattack attack --recipe bae \
                  --model bert-base-uncased-snli \
                  --dataset-from-file data/textattack/datasets/snli.py \
                  --shuffle False \
                  --num-examples 400 \
                  --log-to-csv data/textattack/bae-bert-snli.csv \
                  --query-budget 1000 \
                  --model-batch-size 32 \

textattack attack --recipe bae \
                  --model bert-base-uncased-yelp \
                  --dataset-from-file data/textattack/datasets/yelp.py \
                  --shuffle False \
                  --num-examples 400 \
                  --log-to-csv data/textattack/bae-bert-yelp.csv \
                  --query-budget 1000 \
                  --model-batch-size 32 \

textattack attack --recipe bae \
                  --model bert-base-uncased-qqp \
                  --dataset-from-file data/textattack/datasets/qqp.py \
                  --shuffle False \
                  --num-examples 400 \
                  --log-to-csv data/textattack/bae-bert-qqp.csv \
                  --query-budget 1000 \
                  --model-batch-size 32 \