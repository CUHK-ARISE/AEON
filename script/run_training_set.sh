textattack attack --recipe pso \
                  --model bert-base-uncased-yelp \
                  --dataset-from-file data/textattack/datasets/yelp_train.py \
                  --shuffle False \
                  --num-examples 5000 \
                  --num-examples-offset 0 \
                  --log-to-csv /research/dept7/jthuang/projects/AutoAdvaluator/data/textattack/pso-bert-yelp-train-1.csv \
                  --query-budget 1000 \
                  --model-batch-size 32 \


textattack attack --recipe pso \
                  --model bert-base-uncased-yelp \
                  --dataset-from-file data/textattack/datasets/yelp_train.py \
                  --shuffle False \
                  --num-examples 5000 \
                  --num-examples-offset 5000 \
                  --log-to-csv /research/dept7/jthuang/projects/AutoAdvaluator/data/textattack/pso-bert-yelp-train-2.csv \
                  --query-budget 1000 \
                  --model-batch-size 32 \


textattack attack --recipe pso \
                  --model bert-base-uncased-yelp \
                  --dataset-from-file data/textattack/datasets/yelp_train.py \
                  --shuffle False \
                  --num-examples 5000 \
                  --num-examples-offset 10000 \
                  --log-to-csv /research/dept7/jthuang/projects/AutoAdvaluator/data/textattack/pso-bert-yelp-train-3.csv \
                  --query-budget 1000 \
                  --model-batch-size 32 \


textattack attack --recipe pso \
                  --model bert-base-uncased-yelp \
                  --dataset-from-file data/textattack/datasets/yelp_train.py \
                  --shuffle False \
                  --num-examples 5000 \
                  --num-examples-offset 15000 \
                  --log-to-csv /research/dept7/jthuang/projects/AutoAdvaluator/data/textattack/pso-bert-yelp-train-4.csv \
                  --query-budget 1000 \
                  --model-batch-size 32 \


textattack attack --recipe bae \
                  --model bert-base-uncased-yelp \
                  --dataset-from-file data/textattack/datasets/yelp_train.py \
                  --shuffle False \
                  --num-examples 5000 \
                  --num-examples-offset 0 \
                  --log-to-csv /research/dept7/jthuang/projects/AutoAdvaluator/data/textattack/bae-bert-yelp-train-1.csv \
                  --query-budget 1000 \
                  --model-batch-size 32 \


textattack attack --recipe bae \
                  --model bert-base-uncased-yelp \
                  --dataset-from-file data/textattack/datasets/yelp_train.py \
                  --shuffle False \
                  --num-examples 5000 \
                  --num-examples-offset 5000 \
                  --log-to-csv /research/dept7/jthuang/projects/AutoAdvaluator/data/textattack/bae-bert-yelp-train-2.csv \
                  --query-budget 1000 \
                  --model-batch-size 32 \


textattack attack --recipe bae \
                  --model bert-base-uncased-yelp \
                  --dataset-from-file data/textattack/datasets/yelp_train.py \
                  --shuffle False \
                  --num-examples 5000 \
                  --num-examples-offset 10000 \
                  --log-to-csv /research/dept7/jthuang/projects/AutoAdvaluator/data/textattack/bae-bert-yelp-train-3.csv \
                  --query-budget 1000 \
                  --model-batch-size 32 \


textattack attack --recipe bae \
                  --model bert-base-uncased-yelp \
                  --dataset-from-file data/textattack/datasets/yelp_train.py \
                  --shuffle False \
                  --num-examples 5000 \
                  --num-examples-offset 15000 \
                  --log-to-csv /research/dept7/jthuang/projects/AutoAdvaluator/data/textattack/bae-bert-yelp-train-4.csv \
                  --query-budget 1000 \
                  --model-batch-size 32 \


