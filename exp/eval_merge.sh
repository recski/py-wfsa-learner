#!/bin/sh
for c1 in S U E P M I; do
    for c2 in S U E P M I; do
        if [ $c1 == $c2 ] ; then continue; fi
        fn=${c1}_${c2}
        ratio=`cat ../data/sze_$1.count | python ../get_char_ratio.py $c1 $c2`
        for fsa in out/merged/$1/$fn/*.wfsa; do
            python ../split_and_compare.py $fsa $c2 $c1 $ratio ../data/sze_$1.count
        done
    done
done
