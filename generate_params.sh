#!/bin/sh
input=data/lemma_count
for bits in 4 5 6 7 8 10 12 16
do
    for cutoff in -11 -13 -15 -17 -20 -24 -28 -32
    do
        for dist in kullback l1err squarerr
        do
            echo $input $dist $bits $cutoff
        done
    done
done
