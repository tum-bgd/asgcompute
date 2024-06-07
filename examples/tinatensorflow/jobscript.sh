#!/bin/bash

# SBATCH --job-name=dctemplate

CONTAINER="/glob/g01-cache/container/nvtf2310.simg"


echo "Running job on" $(hostname)
echo "SLURM info"
env |grep SLURM
echo "Container ${CONTAINER} ["$(md5sum ${CONTAINER})"]"
nvidia-smi
echo "===HEADER ENDS========================================"
#
# Now start our container
#
singularity exec -H $PWD:/home --nv "${CONTAINER}" /bin/bash -c "cd /home && ./train.sh" 

