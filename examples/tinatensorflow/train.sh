#!/bin/bash
function die(){
    echo $@
    exit -1
}

## This script performs a simple training.
## It is meant to be run from SLURM jobs within a singularity container!

if [[ "x${SLURM_JOB_ID}" == "x" ]]; then
    die "This job should be run through the job script and should not be invoked directly unless you know what you are doing."
fi

## A notebook (as notebooks are common today)

echo "Download source if needed"
test -f cnn.ipynb || wget https://raw.githubusercontent.com/tensorflow/docs/master/site/en/tutorials/images/cnn.ipynb

# this could and should be merged into the container, here we do it on every job.
pip3 install matplotlib

echo "Run the notebook"

jupyter nbconvert --to notebook --execute cnn.ipynb
if [ -f cnn.nbconvert.ipynb ]; then
    echo "Results found"
    mv cnn.nbconvert.ipynb "slurm-result-${SLURM_JOB_ID}.ipynb"
else
    echo "No results"
fi



