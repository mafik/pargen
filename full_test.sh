#!/usr/bin/bash
trap "exit" INT

for i in {1..100}
do
    echo Iteration $i
    set -x
    ./stacked.py -n baseline
    ./stacked.py -i True -n input
    ./stacked.py -o True -n output
    ./stacked.py -i True -o True -n input_output
    ./stacked.py -m True -n memory
    ./stacked.py -m True -i True -o True -n everything
    #./picture.py average_loss
    set +x
done
