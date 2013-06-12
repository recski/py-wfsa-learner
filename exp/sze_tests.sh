#!/bin/sh
for corp in lemma_count.tab sze_toks.count sze_toks.noM.count sze_types.count sze_types.noM.count; do
    file=../data/$corp
    for type in sze_tok sze_type l; do
        f=${corp}_${type}
        mkdir out/$f 2> /dev/null
        echo $f
        nice python exp_sze.py out/$f $file $type > out/$f/$f.out 2> out/$f/$f.log &
    done
done
