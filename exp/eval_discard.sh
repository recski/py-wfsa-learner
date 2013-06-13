for fn in E P M I EI PI MI MEI; do
    for b in 11 12 14 16; do
        for s in e u; do
            echo -n $fn $b -20 $s
            echo -n ' '
            python ../encoder.py discard/$fn/learnt_$b-20-m-c-k-$s.wfsa ../data/sze_types.count 0.933201 $s $b -20
        done
    done
done
