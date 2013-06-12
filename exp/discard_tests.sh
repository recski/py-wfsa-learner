#!/bin/sh
mkdir out/discard 2> /dev/null
for c in E P M I EI PI MI MEI; do
#for c in I; do
    fn=$c
    cat ../data/sze_toks.count | grep -v [$c] > ../data/discard/$fn
    mkdir out/discard/$fn 2> /dev/null
    echo $fn
    echo "nice python exp_discard.py discard/$fn $fn > discard/$fn/$fn.out 2> discard/$fn/$fn.log "
    nice python exp_sze.py out/discard/$fn ../data/discard/$fn > out/discard/$fn/$fn.out 2> out/discard/$fn/$fn.log 
done
