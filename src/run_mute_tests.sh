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
K_VALUES="7"
LAMBDAS="0.6"
OPTIONAL_EXP_NAME=""
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

echo "$OPTIONAL_EXP_NAME"
echo "$@"

cuda="1"
for lambda in $LAMBDAS; do
    for k in $K_VALUES; do
	    CUDA_VISIBLE_DEVICES="$cuda" sh test_one_no_options.sh "$DATASET_NAME" _reservoir_lambda_"$lambda"_mute_"$k""$OPTIONAL_EXP_NAME" --logan_importance --logan_beta --logan_reservoir --mute_k="$k" "$@" --lambda_val="$lambda" & pids+=($!)
        sleep 5
        if [[ "$cuda" = "0" ]]; then
            cuda="1"
        else
            cuda="0"
        fi

    done
done

for pid in "${pids[@]}"; do
   wait "$pid"
done

echo "$OPTIONAL_EXP_NAME"
for lambda in $LAMBDAS; do
    for k in $K_VALUES; do
	    sh get_results_one.sh "$DATASET_NAME" _reservoir_lambda_"$lambda"_mute_"$k""$OPTIONAL_EXP_NAME" "$lambda"_"$k";
    done
done

wait
