#!/bin/bash
pgn_fn=$1  # tmp.pgn
model_key1=$2  # 0006:0009
model_key2=$3  # init_model:0002
tail_lines=$4  # 1000

echo "$model_key1 won/lost $model_key2"

cat $pgn_fn \
  | tail -$tail_lines \
  | grep "White \"$model_key1\"" -A 3 \
  | grep "Black \"$model_key2\"" -A 2 \
  | grep "\"1-0\"" \
  | wc -l

cat $pgn_fn \
  | tail -$tail_lines \
  | grep "White \"$model_key1\"" -A 3 \
  | grep "Black \"$model_key2\"" -A 2 \
  | grep "\"0-1\"" \
  | wc -l

cat $pgn_fn \
  | tail -$tail_lines \
  | grep "White \"$model_key2\"" -A 3 \
  | grep "Black \"$model_key1\"" -A 2 \
  | grep "\"1-0\"" \
  | wc -l

cat $pgn_fn \
  | tail -$tail_lines \
  | grep "White \"$model_key2\"" -A 3 \
  | grep "Black \"$model_key1\"" -A 2 \
  | grep "\"0-1\"" \
  | wc -l