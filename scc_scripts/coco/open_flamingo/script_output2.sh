#!/bin/bash -l

#$ -N output2

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

LM_PATH="/projectnb/ivc-ml/sunxm/ckpt/llama/llama-7b-hf"
LM_TOKENIZER_PATH="/projectnb/ivc-ml/sunxm/ckpt/llama/llama-7b-hf"
VISION_ENCODER_NAME="ViT-L-14"
VISION_ENCODER_PRETRAINED="openai"
CKPT_PATH="/projectnb/ivc-ml/sunxm/ckpt/open-flamingo/checkpoint.pt"
DEVICE=0


python caption_evaluate.py  --batch_size 32 --eval_coco --coco_dataroot /projectnb/ivc-ml/sunxm/datasets/mscoco_2014/  --lm_path $LM_PATH --lm_tokenizer_path $LM_TOKENIZER_PATH --vision_encoder_path $VISION_ENCODER_NAME --vision_encoder_pretrained $VISION_ENCODER_PRETRAINED  --checkpoint_path $CKPT_PATH --cross_attn_every_n_layers 4 --device $DEVICE --coco_prompts "Output: "
