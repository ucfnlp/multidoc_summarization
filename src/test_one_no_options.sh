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

DATASET_NAME=${1:-}
OPTIONAL_EXP_NAME=${2:-}
shift 2
echo "$DATASET_NAME"
echo "$OPTIONAL_EXP_NAME"
echo "$@"
VOCAB_PATH="/home/logan/data/multidoc_summarization/cnn-dailymail/finished_files/vocab"
LOG_ROOT="/home/logan/data/multidoc_summarization/logs"
MAX_ENC_STEPS=100000
MIN_DEC_STEPS=100
MAX_DEC_STEPS=120

python run_summarization.py --mode=decode --dataset_name="$DATASET_NAME" --vocab_path="$VOCAB_PATH" --actual_log_root="$LOG_ROOT" --exp_name="$DATASET_NAME""$OPTIONAL_EXP_NAME" --single_pass --coverage --max_enc_steps="$MAX_ENC_STEPS" --min_dec_steps="$MIN_DEC_STEPS" --max_dec_steps="$MAX_DEC_STEPS" "$@" &

wait
