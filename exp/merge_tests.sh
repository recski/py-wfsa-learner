#!/bin/sh
mkdir out/merged 2> /dev/null
outdir=out/merged/$1
mkdir $outdir 2> /dev/null
for c1 in S U E P M I; do
    for c2 in S U E P M I; do
        if [ $c1 == $c2 ] ; then continue; fi
        fn=${c1}_${c2}
        cat ../data/sze_$1.count | tr $c1 $c2 | python ../data/add_patt_freqs.py > ../data/merged/$fn
        mkdir $outdir/$fn 2> /dev/null 
        echo $fn
        nice python exp_sze.py $outdir/$fn ../data/merged/$fn sze_$1 > $outdir/$fn/$fn.out 2> $outdir/$fn/$fn.log 
    done
done
