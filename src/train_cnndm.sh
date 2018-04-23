#!/usr/bin/env bash

ret=100
while [ $ret -eq 100 ]
do
	echo "STARTING TRAINING"
	CUDA_VISIBLE_DEVICES=0 python run_summarization.py --mode=train --data_path=/home/logan/data/multidoc_summarization/cnn-dailymail/finished_files/chunked/train_* --vocab_path=/home/logan/data/multidoc_summarization/cnn-dailymail/finished_files/vocab --log_root=/home/logan/data/multidoc_summarization/logs --exp_name=scratch --use_pretrained=False --max_enc_steps=400 --max_dec_steps=120 --num_iterations=243000 --coverage
	ret=$?
done
echo "SUCCESS"
