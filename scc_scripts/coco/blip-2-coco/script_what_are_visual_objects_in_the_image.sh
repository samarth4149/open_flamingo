#!/bin/bash -l

#$ -N what_are_visual_objects_in_the_image

#$ -m bea

#$ -M sunxm@bu.edu

# Set SCC project
#$ -P ivc-ml

# Request my job to run on Buy-in Compute group hardware my_project has access to
#$ -l buyin

# Request 4 CPUs
#$ -pe omp 3

# Request 2 GPU
#$ -l gpus=1

# Specify the minimum GPU compute capability
#$ -l gpu_c=8.6

#$ -l h_rt=48:00:00

#$ -l gpu_memory=48G

# Keep track of information related to the current job
echo "=========================================================="
echo "Start date : $(date)"
echo "Job name : $JOB_NAME"
echo "Job ID : $JOB_ID  $SGE_TASK_ID"
echo "=========================================================="

module load miniconda
module load cuda/11.6
module load gcc

cd /projectnb/ivc-ml/sunxm/code/open_flamingo

echo 'activating virtual environment'
conda activate openflamingo

LM_PATH="/projectnb/ivc-ml/sunxm/ckpt/blip2-flan-t5-xl-coco"
PROCESSOR_PATH="/projectnb/ivc-ml/sunxm/ckpt/blip2-flan-t5-xl-coco"
DEVICE=0

python caption_evaluate.py  --batch_size 32 --eval_coco --coco_dataroot /projectnb/ivc-ml/sunxm/datasets/mscoco_2014/  --processor_path $PROCESSOR_PATH --device $DEVICE --coco_prompts "what are visual objects in the image" --model "blip" --lm_path $LM_PATH
