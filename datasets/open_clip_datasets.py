import torch
from clip_benchmark.datasets.builder import build_dataset, get_dataset_collate_fn, get_dataset_default_task, dataset_collection, get_dataset_collection_from_file
from torchvision.transforms import v2
import torch.utils.data
from tqdm import tqdm

dataset = "wds/vtab/svhn"
if dataset.startswith("wds/"):
    dataset_name = dataset.replace("wds/", "", 1)
else:
    dataset_name = dataset
dataset_root = "https://huggingface.co/datasets/clip-benchmark/wds_{dataset_cleaned}/tree/main"
dataset_root = dataset_root.format(dataset=dataset_name, dataset_cleaned=dataset_name.replace("/", "-"))
split = 'test'
task = get_dataset_default_task(dataset_name)

transform = v2.Compose([
    v2.RandomResizedCrop(size=(224, 224), antialias=True),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToDtype(torch.float32),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
dataset = build_dataset(
    dataset_name=dataset,
    root=dataset_root,
    transform=transform,
    split=split,
    annotation_file="",
    download=True,
    language="en",
    task=task,
    custom_template_file="en_zeroshot_classification_templates.json",
    custom_classname_file="en_classnames.json",
    wds_cache_dir=None,
)
dataloader =torch.utils.data.DataLoader (
                dataset.batched(10), batch_size=None,
                shuffle=False, num_workers=1,
            )
# collate_fn = get_dataset_collate_fn(dataset)
for images, target in tqdm(dataloader):
    print(images.shape)
    import pdb
    pdb.set_trace()

