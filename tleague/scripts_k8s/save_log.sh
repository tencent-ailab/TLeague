#!/usr/bin/env bash
sess=tr0803b
n_lrngrp=5
n_hosts=2

# copy training_log
for((i=0; i<n_lrngrp; i++))
do
    for((j=0; j<n_hosts; j++))
    do
        echo "processing lrngrp${i}-host${j}";
        kubectl cp ${sess}-lrngrp${i}-host${j}:/root/work/training_log ./training_log_lrngrp${i}_host${j}
    done;
done;

# copy league-mgr log
kubectl cp ${sess}-league-mgr:/root/work/elo.log ./elo.log
kubectl cp ${sess}-league-mgr:/root/work/example.pgn ./example.pgn
kubectl cp ${sess}-league-mgr:/root/work/league_log ./league_log
