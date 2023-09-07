#!/bin/bash -l

#$ -N Could_you_describe_the_contents_of_this_image_for_me

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

CFG_PATH='minigpt4/eval_configs/minigpt4_eval.yaml'
DEVICE=0

python caption_evaluate.py  --batch_size 16 --eval_coco --coco_dataroot /projectnb/ivc-ml/sunxm/datasets/mscoco_2014/  -device $DEVICE --coco_prompts "Could you describe the contents of this image for me?" --model "minigpt4" --cfg-path $CFG_PATH --dataset_name OpenImagesV6Rare
