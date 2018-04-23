#!/usr/bin/env bash

BETAS=${1:-"0.001 0.002 0.003 0.005 0.007"}
#BETAS="0.001 0.002"

for BETA in $BETAS; do
	sh test_tac.sh _beta_tau_"$BETA" --logan_beta_tau="$BETA" &
	pids[${i}]=$!;
done

for pid in ${pids[*]}; do
	wait $pid;
done

for BETA in $BETAS; do
	echo $BETA
	sh get_results.sh tac_2011 _beta_tau_"$BETA";
done
