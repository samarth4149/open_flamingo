#!/bin/bash -l

#$ -N resisc45_clip

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
conda activate llava

python GPTScore_eval.py  --model_path /projectnb/ivc-ml/sunxm/ckpt/llava-v1.5-13b-lora/ --model_base /projectnb/ivc-ml/sunxm/ckpt/vicuna-13b-v1.5/  --model_name llava-v1.5-13b-lora  --model llava_v1_5  --coco_prompts " satellite imagery of "  --coco_dataroot    ../../datasets/  --batch_size 8 --dataset_name resisc45_clip