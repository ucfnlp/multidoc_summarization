#!/bin/bash
EXP_NAME=$1
OPTIONAL_EXP_NAME=${2:-}
LOG_ROOT="/home/logan/data/multidoc_summarization/logs"
MAX_ENC_STEPS=100000
MIN_DEC_STEPS=70
MAX_DEC_STEPS=100
cat "$LOG_ROOT"/"$EXP_NAME"/decode_test_"$MAX_ENC_STEPS"maxenc_4beam_"$MIN_DEC_STEPS"mindec_"$MAX_DEC_STEPS"maxdec_ckpt-238410/sheets_results.txt
cat "$LOG_ROOT"/"$EXP_NAME"_coverage"$OPTIONAL_EXP_NAME"/decode_test_"$MAX_ENC_STEPS"maxenc_4beam_"$MIN_DEC_STEPS"mindec_"$MAX_DEC_STEPS"maxdec_ckpt-238410/sheets_results.txt
cat "$LOG_ROOT"/"$EXP_NAME"_importance"$OPTIONAL_EXP_NAME"/decode_test_"$MAX_ENC_STEPS"maxenc_4beam_"$MIN_DEC_STEPS"mindec_"$MAX_DEC_STEPS"maxdec_ckpt-238410/sheets_results.txt
cat "$LOG_ROOT"/"$EXP_NAME"_beta"$OPTIONAL_EXP_NAME"/decode_test_"$MAX_ENC_STEPS"maxenc_4beam_"$MIN_DEC_STEPS"mindec_"$MAX_DEC_STEPS"maxdec_ckpt-238410/sheets_results.txt
