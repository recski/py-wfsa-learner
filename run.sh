#!/bin/sh
inp=`echo $1 | cut -f1 -d" "`
dist=`echo $1 | cut -f2 -d" "`
bits=`echo $1 | cut -f3 -d" "`
min_=`echo $1 | cut -f4 -d" "`
python code.py -t loglinear_cutoff --min $min_ --max 0 -o run/code/code_loglin_cutoff_m${min_}_0_$bits -b $bits
bn=`basename $inp`
res=`cat $inp | nice python learner.py -a in/3states_hogy.wfsa.log -d $dist -f 0.8 -o run/wfsa/$bn.$dist.$bits.$min_.wfsa -s "#" -q run/code/code_loglin_cutoff_m${min_}_0_$bits -i 200 2>run/log/$bn.$dist.$bits.$min_`
echo $1 $res
