import os
import sys
sys.path.append(os.path.abspath('.'))


from importlib.machinery import SourceFileLoader
datasets = SourceFileLoader(
    "datasets",
    os.path.abspath('datasets/__init__.py'),
).load_module()
from datasets.sugar_crepe import SugarCrepe, ALL_SPLITS

import json
import torch
import numpy as np
import argparse
import base64
from pathlib import Path

parser = argparse.ArgumentParser()



def create_html_page(imgs, pos_captions, neg_captions, outs, preds, pred_titles):
    """
    Creates an HTML page to visualize.
    Returns:
    An HTML string.
    """
    html = ["<html><body><table border=1 frame=hsides rules=rows>"]
    
    np.set_printoptions(precision=4)

    # add a row for every 3 triplets
    for i in range(len(imgs)):
        html.append("<tr>")
        image_path = imgs[i]
        pos_caption = pos_captions[i]
        neg_caption = neg_captions[i]
        html.append("<td>")
        
        image_ext = Path(image_path).suffix
        with open(image_path, "rb") as image_path:
            encoded_img = base64.b64encode(image_path.read()).decode()
        html.append(f'<img src="data:image/{image_ext};base64,{encoded_img}" width="300" height="300"><br>')
        html.append(f"Positive Caption: {pos_caption}<br>")
        html.append(f"Negative Caption: {neg_caption}<br>")
        for j, (pred_title, curr_outs, curr_preds) in enumerate(zip(pred_titles, outs, preds)):
            if curr_preds[i] == 0:
                # print output in green
                html.append(f'<font color="green"> {pred_title}: {curr_outs[i]} </font><br>')
            else:
                # print output in red
                html.append(f'<font color="red"> {pred_title}: {curr_outs[i]} </font><br>')
            # html.append(f"{pred_title}: {curr_outs[i]}<br>")
        
        html.append("</td>")
        html.append("</tr>")

    html.append("</table></body></html>")

    return "\n".join(html)


def main():
    save_dir = 'expts/sugarcrepe_result_viz2'
    os.makedirs(save_dir, exist_ok=True)
    for split in ALL_SPLITS:
        dataset = SugarCrepe(
            root='/projectnb/ivc-ml/samarth/projects/synthetic/final/clip_benchmark_data/sugar_crepe', data_split=split, transform=None)
        RNG = np.random.RandomState(44)
        # idxs = RNG.choice(len(dataset), 100, replace=False)
        
        clip_outs = torch.load('expts/sugarcrepe_clip_vit-l-14_openai/preds_sugarcrepe_' + split + '.pt')
        minigpt_v_outs = torch.load('expts/sugarcrepe_minigpt_v2/preds_sugarcrepe_' + split + '.pt')
        minigpt_v_ptrain_outs = torch.load('expts/sugarcrepe_minigpt_v2_ptrain/preds_sugarcrepe_' + split + '.pt')
        minigpt_v_ptrain_language_only_outs = torch.load('expts/sugarcrepe_minigpt_v2_ptrain_language_only/preds_sugarcrepe_' + split + '.pt')
        
        clip_outs = clip_outs.numpy()
        minigpt_v_outs = minigpt_v_outs.numpy()
        minigpt_v_ptrain_outs = minigpt_v_ptrain_outs.numpy()
        minigpt_v_ptrain_language_only_outs = minigpt_v_ptrain_language_only_outs.numpy()
        
        clip_preds = np.argmax(clip_outs, axis=1)
        minigpt_v_preds = np.argmax(minigpt_v_outs, axis=1)
        minigpt_v_ptrain_preds = np.argmax(minigpt_v_ptrain_outs, axis=1)
        minigpt_v_ptrain_language_only_preds = np.argmax(minigpt_v_ptrain_outs, axis=1)
        
        l_corr_vl_incorr = np.where((minigpt_v_ptrain_language_only_preds == 0) & (minigpt_v_preds == 1))[0]
        l_incorr_vl_corr = np.where((minigpt_v_ptrain_language_only_preds == 1) & (minigpt_v_preds == 0))[0]
        
        print('Split:', split)
        print('Language only correct VL incorrect', len(l_corr_vl_incorr))
        print('VL correct, Language only incorrect', len(l_incorr_vl_corr))
        
        l_corr_vl_incorr = RNG.choice(l_corr_vl_incorr, min(20, len(l_corr_vl_incorr)), replace=False)
        l_incorr_vl_corr = RNG.choice(l_incorr_vl_corr, min(20, len(l_incorr_vl_corr)), replace=False)
        # import ipdb; ipdb.set_trace()
        
        idxs = np.concatenate([l_corr_vl_incorr, l_incorr_vl_corr])
        
        imgs = []
        pos_captions = []
        neg_captions = []
        for i in idxs:
            img, (pos_caption, neg_caption) = dataset.get_item_path(i)
            imgs.append(img)
            pos_captions.append(pos_caption)
            neg_captions.append(neg_caption)
        
        html = create_html_page(
            imgs, pos_captions, neg_captions,
            [minigpt_v_outs[idxs], minigpt_v_ptrain_language_only_outs[idxs], clip_outs[idxs], minigpt_v_ptrain_outs[idxs]], 
            [minigpt_v_preds[idxs], minigpt_v_ptrain_language_only_preds[idxs], clip_preds[idxs], minigpt_v_ptrain_preds[idxs]], 
            ['minigpt_v', 'llama2 (language only)', 'clip', 'minigpt_v_ptrain'])
        with open(os.path.join(save_dir, f'{split}.html'), 'w') as f:
            f.write(html)
    
    
if __name__ == '__main__':
    main()