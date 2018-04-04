CUDA_VISIBLE_DEVICES=0 python run_summarization.py --mode=decode --data_path=/home/logan/data/multidoc_summarization/DUC/full_article_tf_examples/test/* --vocab_path=/home/logan/data/multidoc_summarization/cnn-dailymail/finished_files/vocab --log_root=/home/logan/data/multidoc_summarization/logs --exp_name=duc_2004 --single_pass --coverage --max_enc_steps=100000 --max_dec_steps=120 &  O=$!
CUDA_VISIBLE_DEVICES=0 python run_summarization.py --mode=decode --data_path=/home/logan/data/multidoc_summarization/DUC/full_article_tf_examples/test/* --vocab_path=/home/logan/data/multidoc_summarization/cnn-dailymail/finished_files/vocab --log_root=/home/logan/data/multidoc_summarization/logs --exp_name=duc_2004_coverage --single_pass --coverage --max_enc_steps=100000 --max_dec_steps=120 --logan_coverage --logan_beta &  SC=$!
CUDA_VISIBLE_DEVICES=1 python run_summarization.py --mode=decode --data_path=/home/logan/data/multidoc_summarization/DUC/full_article_tf_examples/test/* --vocab_path=/home/logan/data/multidoc_summarization/cnn-dailymail/finished_files/vocab --log_root=/home/logan/data/multidoc_summarization/logs --exp_name=duc_2004_importance --single_pass --coverage --max_enc_steps=100000 --max_dec_steps=120 --logan_importance --logan_beta &  SI=$!
CUDA_VISIBLE_DEVICES=1 python run_summarization.py --mode=decode --data_path=/home/logan/data/multidoc_summarization/DUC/full_article_tf_examples/test/* --vocab_path=/home/logan/data/multidoc_summarization/cnn-dailymail/finished_files/vocab --log_root=/home/logan/data/multidoc_summarization/logs --exp_name=duc_2004_beta --single_pass --coverage --max_enc_steps=100000 --max_dec_steps=120 --logan_coverage --logan_importance --logan_beta &  SCSI=$!
wait $O
wait $SC
wait $SI
wait $SCSI
cat /home/logan/data/multidoc_summarization/logs/duc_2004/decode_test_100000maxenc_4beam_35mindec_100maxdec_ckpt-238410/sheets_results.txt
cat /home/logan/data/multidoc_summarization/logs/duc_2004_coverage/decode_test_100000maxenc_4beam_35mindec_100maxdec_ckpt-238410/sheets_results.txt
cat /home/logan/data/multidoc_summarization/logs/duc_2004_importance/decode_test_100000maxenc_4beam_35mindec_100maxdec_ckpt-238410/sheets_results.txt
cat /home/logan/data/multidoc_summarization/logs/duc_2004_beta/decode_test_100000maxenc_4beam_35mindec_100maxdec_ckpt-238410/sheets_results.txt
