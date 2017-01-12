#!/usr/bin/bash
trap "exit" INT

for i in {1..100}
do
    echo Iteration $i
    set -x
    #./cudnn_stacked.py -n lstm_baseline
    #./cudnn_stacked.py -o True -n output_mixing
    ./cudnn_stacked.py -i True -n stacked
    #./cudnn_stacked.py -i True -o True -n input_output
    #./cudnn_stacked.py -m True -n memory
    #./cudnn_stacked.py -m True -i True -o True -n everything
    ./picture.py
    set +x
done
