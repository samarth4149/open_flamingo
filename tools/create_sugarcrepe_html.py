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



def create_html_page(imgs, pos_captions, neg_captions, preds, pred_titles):
    """
    Creates an HTML page to visualize.
    Returns:
    An HTML string.
    """
    html = ["<html><body><table>"]

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
        for j, (pred_title, curr_preds) in enumerate(zip(pred_titles, preds)):
            html.append(f"{pred_title}: {curr_preds[i]}<br>")
        
        html.append("</td>")
        html.append("</tr>")

    html.append("</table></body></html>")

    return "\n".join(html)


def main():
    os.makedirs('expts/sugarcrepe_result_viz', exist_ok=True)
    for split in ALL_SPLITS:
        dataset = SugarCrepe(
            root='/projectnb/ivc-ml/samarth/projects/synthetic/final/clip_benchmark_data/sugar_crepe', data_split=split, transform=None)
        RNG = np.random.RandomState(44)
        idxs = RNG.choice(len(dataset), 100, replace=False)
        
        clip_preds = torch.load('expts/sugarcrepe_clip_vit-l-14_openai/preds_sugarcrepe_' + split + '.pt')
        minigpt_v_preds = torch.load('expts/sugarcrepe_minigpt_v2/preds_sugarcrepe_' + split + '.pt')
        
        clip_preds = clip_preds.numpy()[idxs]
        minigpt_v_preds = minigpt_v_preds.numpy()[idxs]
        
        imgs = []
        pos_captions = []
        neg_captions = []
        for i in idxs:
            img, (pos_caption, neg_caption) = dataset.get_item_path(i)
            imgs.append(img)
            pos_captions.append(pos_caption)
            neg_captions.append(neg_caption)
        
        html = create_html_page(imgs, pos_captions, neg_captions, [clip_preds, minigpt_v_preds], ['clip', 'minigpt_v'])
        with open(f'expts/sugarcrepe_result_viz/{split}.html', 'w') as f:
            f.write(html)
    
    
if __name__ == '__main__':
    main()