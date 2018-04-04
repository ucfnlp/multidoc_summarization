CUDA_VISIBLE_DEVICES=0 python run_summarization.py --mode=decode --data_path=/home/logan/data/multidoc_summarization/cnn-dailymail/finished_files/chunked/test_* --vocab_path=/home/logan/data/multidoc_summarization/cnn-dailymail/finished_files/vocab --log_root=/home/logan/data/multidoc_summarization/logs --exp_name=cnn_dm --single_pass --coverage --max_dec_steps=120 &  O=$!
CUDA_VISIBLE_DEVICES=0 python run_summarization.py --mode=decode --data_path=/home/logan/data/multidoc_summarization/cnn-dailymail/finished_files/chunked/test_* --vocab_path=/home/logan/data/multidoc_summarization/cnn-dailymail/finished_files/vocab --log_root=/home/logan/data/multidoc_summarization/logs --exp_name=cnn_dm_coverage --single_pass --coverage --max_enc_steps=100000 --max_dec_steps=120 --logan_coverage --logan_beta &  SC=$!
CUDA_VISIBLE_DEVICES=1 python run_summarization.py --mode=decode --data_path=/home/logan/data/multidoc_summarization/cnn-dailymail/finished_files/chunked/test_* --vocab_path=/home/logan/data/multidoc_summarization/cnn-dailymail/finished_files/vocab --log_root=/home/logan/data/multidoc_summarization/logs --exp_name=cnn_dm_importance --single_pass --coverage --max_enc_steps=100000 --max_dec_steps=120 --logan_importance --logan_beta &  SI=$!
CUDA_VISIBLE_DEVICES=1 python run_summarization.py --mode=decode --data_path=/home/logan/data/multidoc_summarization/cnn-dailymail/finished_files/chunked/test_* --vocab_path=/home/logan/data/multidoc_summarization/cnn-dailymail/finished_files/vocab --log_root=/home/logan/data/multidoc_summarization/logs --exp_name=cnn_dm_beta --single_pass --coverage --max_enc_steps=100000 --max_dec_steps=120 --logan_coverage --logan_importance --logan_beta &  SCSI=$!
wait $O
wait $SC
wait $SI
wait $SCSI
cat /home/logan/data/multidoc_summarization/logs/tac_2011/decode_test_100000maxenc_4beam_35mindec_100maxdec_ckpt-238410/sheets_results.txt
cat /home/logan/data/multidoc_summarization/logs/tac_2011_coverage/decode_test_100000maxenc_4beam_35mindec_100maxdec_ckpt-238410/sheets_results.txt
cat /home/logan/data/multidoc_summarization/logs/tac_2011_importance/decode_test_100000maxenc_4beam_35mindec_100maxdec_ckpt-238410/sheets_results.txt
cat /home/logan/data/multidoc_summarization/logs/tac_2011_beta/decode_test_100000maxenc_4beam_35mindec_100maxdec_ckpt-238410/sheets_results.txt
