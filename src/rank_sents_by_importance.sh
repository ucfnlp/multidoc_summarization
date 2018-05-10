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

dataset_names="tac_2011 tac_2008 duc_2004"
importance_fns="lex_rank svr"

OPTIONAL_EXP_NAME=${1:-}
shift 1
echo "$OPTIONAL_EXP_NAME"
echo "$@"


for dataset_name in $dataset_names; do
	for importance_fn in $importance_fns; do
		sh test_one_no_options.sh "$dataset_name" _reservoir_mute_5_rouge_l_"$importance_fn"_ranked"$OPTIONAL_EXP_NAME" --logan_importance --logan_beta --logan_reservoir --similarity_fn=rouge_l --mute_k=5 --importance_fn="$importance_fn" "$@" & pids+=($!)
	done
done

for pid in "${pids[@]}"; do
   wait "$pid"
done

for dataset_name in $dataset_names; do
	for importance_fn in $importance_fns; do
		echo "$dataset_name"___"$importance_fn"
	done
done
for dataset_name in $dataset_names; do
	for importance_fn in $importance_fns; do
		sh get_results_one.sh "$dataset_name" _reservoir_mute_5_rouge_l_"$importance_fn"_ranked"$OPTIONAL_EXP_NAME";
	done
done

wait




