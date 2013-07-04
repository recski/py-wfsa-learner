#!/bin/sh
for type in toks types; do
    file=../data/sze_$type.count
    mkdir out/$type 2> /dev/null
    echo $type
    nice python exp_sze.py out/$type $file $type > out/$type/$type.out 2> out/$type/$type.log
done
