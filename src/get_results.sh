#!/bin/bash
LOG_DIR=$1
OPTIONAL_EXP_NAME=${2:-}
cat /home/logan/data/multidoc_summarization/logs/"$LOG_DIR"/decode_test_100000maxenc_4beam_35mindec_120maxdec_ckpt-238410/sheets_results.txt
cat /home/logan/data/multidoc_summarization/logs/"$LOG_DIR"_coverage"$OPTIONAL_EXP_NAME"/decode_test_100000maxenc_4beam_35mindec_120maxdec_ckpt-238410/sheets_results.txt
cat /home/logan/data/multidoc_summarization/logs/"$LOG_DIR"_importance"$OPTIONAL_EXP_NAME"/decode_test_100000maxenc_4beam_35mindec_120maxdec_ckpt-238410/sheets_results.txt
cat /home/logan/data/multidoc_summarization/logs/"$LOG_DIR"_beta"$OPTIONAL_EXP_NAME"/decode_test_100000maxenc_4beam_35mindec_120maxdec_ckpt-238410/sheets_results.txt
