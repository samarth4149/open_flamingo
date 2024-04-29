import os
import sys
sys.path.append(os.path.abspath('.'))
# from datasets.sugar_crepe import ALL_SPLITS

from importlib.machinery import SourceFileLoader
datasets = SourceFileLoader(
   "datasets",
   os.path.abspath('datasets/__init__.py'),
).load_module()

if __name__ == '__main__':
    
    # cmd = ['python -m ipdb', 'GPTScore_winoground.py']
    cmd = ['python', 'GPTScore_winoground.py']
    cmd += ['-device', '0']
    # cmd += ['--coco_dataroot', '/projectnb/ivc-ml/samarth/projects/synthetic/final/clip_benchmark_data/sugar_crepe',
    #         '--dataset_name', "sugarcrepe/add_obj",]
            # '--dataset_name', ",".join(["sugarcrepe/" + split for split in ALL_SPLITS]),]
    

    cmd += [
            '--batch_size', '16',
            '--model', 'llava_v1_5',
            # '--model_path', 'liuhaotian/llava-v1.5-13b',
            '--model_path', '/projectnb/ivc-ml/sunxm/ckpt/llava-v1.5-13b/',
            '--model_name', 'llava-v1.5-13b',
            '--output_dir', 'expts/winoground_llava-v1.5-13b',
            ]

    os.system(' '.join(cmd))
