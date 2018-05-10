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

# DATASET_NAME=${1:-}
# K_VALUES=${2:-"2 3 5 7 10"}
# LAMBDAS=${3:-"0.25 0.5 0.75 1"}
# OPTIONAL_EXP_NAME=${4:-}
# shift 4
# echo "$OPTIONAL_EXP_NAME"
# echo "$@"

DATASET_NAME=tac_2011
K_VALUES="2 5 10 25 50"
LAMBDAS="0.9 0.7 0.5 0.3"
OPTIONAL_EXP_NAME=""

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
    *)
        break
  esac
  shift
done

echo "$OPTIONAL_EXP_NAME"
echo "$@"

for lambda in $LAMBDAS; do
    for k in $K_VALUES; do
	    sh test_one_no_options.sh "$DATASET_NAME" _reservoir_lambda_"$lambda"_mute_"$k""$OPTIONAL_EXP_NAME" --logan_importance --logan_beta --logan_reservoir --mute_k="$k" "$@" --lambda_val="$lambda" & pids+=($!)
        sleep 5
    done
done

for pid in "${pids[@]}"; do
   wait "$pid"
done

for lambda in $LAMBDAS; do
    for k in $K_VALUES; do
	    echo "$lambda"_"$k"
    done
done
echo "$OPTIONAL_EXP_NAME"
for lambda in $LAMBDAS; do
    for k in $K_VALUES; do
	    sh get_results_one.sh "$DATASET_NAME" _reservoir_lambda_"$lambda"_mute_"$k""$OPTIONAL_EXP_NAME";
    done
done

wait
