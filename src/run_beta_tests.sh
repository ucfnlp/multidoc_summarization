#!/usr/bin/env bash

#BETAS="0.001 0.002 0.003 0.005 0.007 0.01 0.015"
BETAS="0.001 0.002"
PROCESSES=()

for BETA in $BETAS; do
	sh test_tac.sh _beta_tau_"$BETA" --logan_beta_tau="$BETA" &  $PROCESSES+=$! &
done

for PROCESS in $PROCESSES; do
	wait $PROCESS;
done

for BETA in $BETAS; do
	sh get_results.sh tac_2011 _beta_tau_"$BETA";
done
