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


DATASET_NAME=tac_2011
K_VALUES="2 5 10 25 50"
LAMBDAS="0.9 0.7 0.5 0.3"
OPTIONAL_EXP_NAME=""
FEATURE="all"
RUN_ONCE="0"

while [ $# -gt 0 ]; do
  case "$1" in
    --DATASET_NAME=*)
      DATASET_NAME="${1#*=}"
      ;;
    --K_VALUES=*)
      K_VALUES="${1#*=}"
      ;;
    --LAMBDAS=*)
      LAMBDAS="${1#*=}"
      ;;
    --OPTIONAL_EXP_NAME=*)
      OPTIONAL_EXP_NAME="${1#*=}"
      ;;
    --FEATURE=*)
      FEATURE="${1#*=}"
      ;;
    --RUN_ONCE=*)
      RUN_ONCE="${1#*=}"
      ;;
    *)
        break
  esac
  shift
done

if [[ "$RUN_ONCE" = "1" ]]; then
    K_VALUES="10"
    LAMBDAS="0.5"
fi


# shift 4
# echo "$OPTIONAL_EXP_NAME"
# echo "$@"

if [[ "$FEATURE" = "regular" || "$FEATURE" = "all"  ]]; then
    ./run_mute_tests.sh --DATASET_NAME="$DATASET_NAME" --K_VALUES="$K_VALUES" --LAMBDAS="$LAMBDAS" --OPTIONAL_EXP_NAME=_svr_regular"$OPTIONAL_EXP_NAME" --importance_model_name=importance_svr_regular"$OPTIONAL_EXP_NAME" "$@" & pids+=($!)
fi
if [[ "$FEATURE" = "average" || "$FEATURE" = "all"  ]]; then
    ./run_mute_tests.sh --DATASET_NAME="$DATASET_NAME" --K_VALUES="$K_VALUES" --LAMBDAS="$LAMBDAS" --OPTIONAL_EXP_NAME=_svr_avg_words"$OPTIONAL_EXP_NAME" --importance_model_name=importance_svr_average_over_word_states"$OPTIONAL_EXP_NAME" --sent_vec_feature_method=average "$@" & pids+=($!)
fi
if [[ "$FEATURE" = "normalize" || "$FEATURE" = "all"  ]]; then
    ./run_mute_tests.sh --DATASET_NAME="$DATASET_NAME" --K_VALUES="$K_VALUES" --LAMBDAS="$LAMBDAS" --OPTIONAL_EXP_NAME=_svr_normalized"$OPTIONAL_EXP_NAME" --importance_model_name=importance_svr_normalized"$OPTIONAL_EXP_NAME" --normalize_features "$@" & pids+=($!)
fi

for pid in "${pids[@]}"; do
   wait "$pid"
done


wait
