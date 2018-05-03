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

exp_name="tac_2011"

K_VALUES=${1:-"3 4 5 6 8 10 13 15"}
SIMILARITY_FN=${2:-"rouge_l"}
OPTIONAL_EXP_NAME=${3:-}
shift 3
echo "$OPTIONAL_EXP_NAME"
echo "$@"


for k in $K_VALUES; do
	sh test_one_no_options.sh _reservoir_mute_"$k"_"$SIMILARITY_FN""$OPTIONAL_EXP_NAME" --logan_importance --logan_beta --logan_reservoir --similarity_fn="$SIMILARITY_FN" --mute_k="$k" "$@" & pids+=($!)
done

for pid in "${pids[@]}"; do
   wait "$pid"
done

for k in $K_VALUES; do
	echo $k
done
for k in $K_VALUES; do
	sh get_results_one.sh "$exp_name" _reservoir_mute_"$k"_"$SIMILARITY_FN""$OPTIONAL_EXP_NAME";
done

wait
