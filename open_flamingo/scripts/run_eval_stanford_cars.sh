echo 'activating virtual environment'
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate openflamingo
which python
module load  cuda/11.6
LM_PATH="/projectnb/ivc-ml/sunxm/llama/llama-7b-hf"
LM_TOKENIZER_PATH="/projectnb/ivc-ml/sunxm/llama/llama-7b-hf"
VISION_ENCODER_NAME="ViT-L-14"
VISION_ENCODER_PRETRAINED="openai"
CKPT_PATH="/projectnb/ivc-ml/sunxm/open-flamingo/checkpoint.pt"
DEVICE=0

#COCO_IMG_PATH="<path to coco>/train2017/"
#COCO_ANNO_PATH="<path to coco>/annotations/captions_train2017.json"
#VQAV2_IMG_PATH="<path to vqav2>/train2014"
#VQAV2_ANNO_PATH="<path to vqav2>/v2_mscoco_train2014_annotations.json"
#VQAV2_QUESTION_PATH="<path to vqav2>/v2_OpenEnded_mscoco_train2014_questions.json"

STANFORD_CARS_PATH='/projectnb/ivc-ml/sunxm/datasets/stanford_cars_20211007'

RANDOM_ID=$$
RESULTS_FILE="results_${RANDOM_ID}.json"

cd /projectnb/ivc-ml/sunxm/code/open_flamingo

python open_flamingo/eval/evaluate.py \
    --lm_path $LM_PATH \
    --lm_tokenizer_path $LM_TOKENIZER_PATH \
    --vision_encoder_path $VISION_ENCODER_NAME \
    --vision_encoder_pretrained $VISION_ENCODER_PRETRAINED \
    --checkpoint_path $CKPT_PATH \
    --cross_attn_every_n_layers 4 \
    --device $DEVICE \
    --results_file $RESULTS_FILE \
    --eval_image_cls \
    --image_cls_root $STANFORD_CARS_PATH \
    --image_cls_dataset_name 'stanford-cars' \
    --num_samples 8041 \
    --shots 1 \
    --num_trials 1 \
    --batch_size 32


echo "evaluation complete! results written to ${RESULTS_FILE}"
