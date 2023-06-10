echo 'activating virtual environment'
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate openflamingo
which python

LM_PATH="/projectnb/ivc-ml/sunxm/llama/llama-7b-hf"
LM_TOKENIZER_PATH="/projectnb/ivc-ml/sunxm/llama/llama-7b-hf"
VISION_ENCODER_NAME="ViT-L-14"
VISION_ENCODER_PRETRAINED="openai"
CKPT_PATH="/projectnb/ivc-ml/sunxm/open-flamingo/checkpoint.pt"
DEVICE=$CUDA_VISIBLE_DEVICES

#COCO_IMG_PATH="<path to coco>/train2017/"
#COCO_ANNO_PATH="<path to coco>/annotations/captions_train2017.json"
#VQAV2_IMG_PATH="<path to vqav2>/train2014"
#VQAV2_ANNO_PATH="<path to vqav2>/v2_mscoco_train2014_annotations.json"
#VQAV2_QUESTION_PATH="<path to vqav2>/v2_OpenEnded_mscoco_train2014_questions.json"

$IMAGENET_PATH='/projectnb/ivc-ml/sunxm/datasets/imagenet1k'

RANDOM_ID=$$
RESULTS_FILE="results_${RANDOM_ID}.json"

python open_flamingo/eval/evaluate.py \
    --lm_path $LM_PATH \
    --lm_tokenizer_path $LM_TOKENIZER_PATH \
    --vision_encoder_path $VISION_ENCODER_NAME \
    --vision_encoder_pretrained $VISION_ENCODER_PRETRAINED \
    --checkpoint_path $CKPT_PATH \
    --cross_attn_every_n_layers 4 \
    --device $DEVICE \
    --results_file $RESULTS_FILE \
    --eval_imagenet \
    --imagenet_root $IMAGENET_PATH \
    --num_samples 5000 \
    --shots 1 \
    --num_trials 1 \
    --batch_size 1


echo "evaluation complete! results written to ${RESULTS_FILE}"
