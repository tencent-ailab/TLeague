#!/bin/bash
# view win-rates for recent_n_matches
recent_n_matches=$1

tail -$((recent_n_matches*10)) tmp_example.pgn > tmp.pgn

bayeselo<details.bayeselo

