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

DATA_SPLITS="train val test"
OPTIONAL_EXP_NAME=${1:-}
shift 1
echo "$OPTIONAL_EXP_NAME"
echo "$@"

for split in $DATA_SPLITS; do
    python run_summarization.py --mode=calc_features --dataset_name=cnn_dm --dataset_split="$split"* --vocab_path=/home/logan/data/multidoc_summarization/cnn-dailymail/finished_files/vocab --log_root=/home/logan/data/multidoc_summarization/logs --exp_name=tac_2011_beta --single_pass --coverage --max_enc_steps=100000 --min_dec_steps=100 --max_dec_steps=120 --save_path=/home/logan/data/multidoc_summarization/cnn-dailymail/importance_data"$OPTIONAL_EXP_NAME" --batch_size=100 "$@" & pids+=($!)
done

for pid in "${pids[@]}"; do
   wait "$pid"
done

wait
