import os
import sys
sys.path.append(os.path.abspath('.'))
# from datasets.sugar_crepe import ALL_SPLITS

from importlib.machinery import SourceFileLoader
datasets = SourceFileLoader(
   "datasets",
   os.path.abspath('datasets/__init__.py'),
).load_module()
from datasets.sugar_crepe import ALL_SPLITS

if __name__ == '__main__':
    # cfg_path = 'minigpt4/eval_configs/minigpt4_eval_16bit_13b.yaml'
    cfg_path = 'minigpt4_v2/eval_configs/minigptv2_eval.yaml'
    
#     cmd = ['python', '-m ipdb', 'GPTScore_eval.py']
    cmd = ['python', 'GPTScore_eval.py']
    # cmd = ['python', 'clip_evaluate.py']
    cmd += ['-device', '0']
    cmd += ['--coco_dataroot', '/projectnb/ivc-ml/samarth/projects/synthetic/final/clip_benchmark_data/sugar_crepe',
            '--dataset_name', ",".join(["sugarcrepe/" + split for split in ALL_SPLITS]),]
    
    cmd += ['--save_scores']
    
    # cmd += ['--batch_size', '256',
    #         '--model', 'ViT-L-14',
    #         '--pretrained', 'openai',
    #         '--output_dir', 'expts/sugarcrepe_clip_vit-l-14_openai',
    #         ]
    
    # cmd += ['--batch_size', '8',
    #         '--model', 'minigpt_v',
    #         '--cfg-path', cfg_path,
    #         '--output_dir', 'expts/sugarcrepe_minigpt_v2_ptrain_language_only',
    #         '--ptrain', 'language_only',
    #         ]
    
    # cmd += ['--batch_size', '32',
    #         '--model', 'blip',
    #         '--processor_path', '/projectnb/ivc-ml/sunxm/ckpt/blip2-flan-t5-xl-coco',
    #         '--lm_path', '/projectnb/ivc-ml/sunxm/ckpt/blip2-flan-t5-xl-coco',
    #         '--output_dir', 'expts/sugarcrepe_blip2-flan-t5-xl-coco',]
    
    cmd += ['--batch_size', '32',
            '--model', 'old_blip',
            '--processor_path', 'Salesforce/blip-image-captioning-large',
            '--lm_path', 'Salesforce/blip-image-captioning-large',
            '--output_dir', 'expts/sugarcrepe_old_blip_large',]
#     cmd += ['--batch_size', '32',
#             '--model', 'minigpt4',
#             '--cfg-path', cfg_path,
#             '--output_dir', 'expts/sugarcrepe_minigpt4_16bit_13b',]
    

#     cmd += [
#             '--batch_size', '8',
#             '--model', 'llava_v1_5',
#             '--model_path', '/projectnb/ivc-ml/sunxm/ckpt/llava-v1.5-13b/',
#             '--model_name', 'llava-v1.5-13b',
#             '--output_dir', 'expts/sugarcrepe_llava-v1.5-13b',
#             ]

    os.system(' '.join(cmd))