#!/usr/bin/env bash
intexit() {
    # Kill all subprocesses (all processes in the current process group)
    kill -HUP -$$
}

hupexit() {
    # HUP'd (probably by intexit)
    echo
    echo "Interrupted"
    exit
}

trap hupexit HUP
trap intexit INT

OPTIONAL_EXP_NAME=${1:-}
shift 1
echo "$OPTIONAL_EXP_NAME"
echo "$@"
DATA_PATH="/home/logan/data/multidoc_summarization/TAC_Data/full_article_tf_examples/test/*"
VOCAB_PATH="/home/logan/data/multidoc_summarization/cnn-dailymail/finished_files/vocab"
LOG_ROOT="/home/logan/data/multidoc_summarization/logs"
EXP_NAME="tac_2011"
MAX_ENC_STEPS=100000
MIN_DEC_STEPS=100
MAX_DEC_STEPS=120

CUDA_VISIBLE_DEVICES=1 python run_summarization.py --mode=decode --data_path="$DATA_PATH" --vocab_path="$VOCAB_PATH" --log_root="$LOG_ROOT" --exp_name="$EXP_NAME"_beta"$OPTIONAL_EXP_NAME" --single_pass --coverage --max_enc_steps="$MAX_ENC_STEPS" --min_dec_steps="$MIN_DEC_STEPS" --max_dec_steps="$MAX_DEC_STEPS" --logan_coverage --logan_importance --logan_beta "$@" &

wait
