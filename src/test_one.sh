#!/usr/bin/env bash
OPTIONAL_EXP_NAME=${1:-}
shift 1
echo "$OPTIONAL_EXP_NAME"
echo "$@"
CUDA_VISIBLE_DEVICES=0 python run_summarization.py --mode=decode --data_path=/home/logan/data/multidoc_summarization/TAC_Data/full_article_tf_examples/test/* --vocab_path=/home/logan/data/multidoc_summarization/cnn-dailymail/finished_files/vocab --log_root=/home/logan/data/multidoc_summarization/logs --exp_name=tac_2011"$OPTIONAL_EXP_NAME" --single_pass --coverage --max_enc_steps=100000 --max_dec_steps=120  --logan_coverage --logan_importance --logan_beta "$@" &  O=$!
