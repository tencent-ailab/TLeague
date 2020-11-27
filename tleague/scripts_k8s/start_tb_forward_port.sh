#!/usr/bin/env bash
# batch doing kubectl port-forward, each one corresponding to a tensorboard and using a tmux window.
sess=tr1030-burnin
n_lrngrp=10
tb_port=9003
tmux_sess=tb

tmux new-session -t $tmux_sess -d
for((i=0; i<n_lrngrp; i++))
do
    tmux new-window -t $tmux_sess -n $i -d
    tmux send-keys -t $tmux_sess:$i "kubectl port-forward svc/${sess}-lrngrp${i}-host0 $((tb_port+i)):${tb_port}" Enter
done;
