import os
from torch.utils.data import Dataset
from PIL import Image
import json
from pathlib import Path
from torchvision import transforms


ALL_SPLITS = [
    'add_obj', 'add_att', 'replace_obj', 'replace_att', 'replace_rel', 'swap_obj', 'swap_att']

class SugarCrepe(Dataset):

    def __init__(self, root, data_split='add_obj', transform=None, **kwargs):
        self.root = str(Path(root) / 'val2017')
        self.ann = json.load(open(Path(root) / f'{data_split}.json', 'r'))
        self.transform = transform
        self.idx_strings = list(self.ann.keys()) # NOTE : indices may be non-contiguous

    def get_item_path(self, idx):
        idx_str = self.idx_strings[idx]
        caption = self.ann[idx_str]['caption']
        negative_caption = self.ann[idx_str]['negative_caption']
        return os.path.join(self.root, self.ann[idx_str]['filename']), [caption, negative_caption]
    
    def __getitem__(self, idx):
        idx_str = self.idx_strings[idx]
        data = self.ann[idx_str]
        img = Image.open(os.path.join(self.root, data['filename']))
        if self.transform is not None:
            img = self.transform(img)
        caption = data['caption']
        negative_caption = data['negative_caption']
        return img, [caption, negative_caption]

    def __len__(self):
        return len(self.ann)