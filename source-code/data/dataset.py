import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import re
from mutils.transforms import RandomIntensityChannel, RandomAffineChannel, Identity, ToRGB, NaiveNormChannel

def build_oct_transform(config, augment=True):
    intensity = RandomIntensityChannel()
    affine = RandomAffineChannel(
        degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5, fill=config.fill
    ) if config.affine else Identity()

    t_list = [
        transforms.Resize((config.input_size, config.input_size)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.ConvertImageDtype(torch.float32),
    ]

    if augment:
        t_list += [transforms.RandomHorizontalFlip(p=0.5),
                   intensity,
                   affine
                   ]

    t_list += [
        NaiveNormChannel(),
        ToRGB(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ]
    return transforms.Compose(t_list)

class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=0.02):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return torch.clamp(tensor + noise, 0.0, 1.0)

    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'

def build_cfp_transform(config, augment=True):
    t_list = [
        transforms.Resize((config.input_size, config.input_size))
    ]

    if augment:
        t_list += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=180),
        ]

    t_list += [
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ]
    return transforms.Compose(t_list)

def build_cfp_transform_advanced(config, augment=True):
    t_list = [transforms.Resize((config.input_size, config.input_size))]

    if augment:
        t_list += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=180)
        ]

    t_list += [
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ]

    return transforms.Compose(t_list)

class MMCAmdCSVDataset(Dataset):
    def __init__(self, df, cfp_dir, oct_dir, transform_cfp=None, transform_oct=None, split_name="Data"):
        self.df = df.reset_index(drop=True)
        self.cfp_dir = cfp_dir
        self.oct_dir = oct_dir
        self.transform_cfp = transform_cfp
        self.transform_oct = transform_oct
        self.split_name = split_name

        self.cached_data = []
        for idx in tqdm(range(len(self.df)), desc=f"Caching {self.split_name} into RAM"):
            cfp_name = self.df.loc[idx, 'cfp']
            oct_name = self.df.loc[idx, 'oct']
            label = self.df.loc[idx, 'label']

            if not cfp_name.endswith(('.jpg', '.png')): cfp_name += '.jpg'
            if not oct_name.endswith(('.jpg', '.png')): oct_name += '.jpg'

            with open(os.path.join(self.cfp_dir, cfp_name), 'rb') as f:
                img_cfp = Image.open(f).convert("RGB")
                img_cfp.load()
            with open(os.path.join(self.oct_dir, oct_name), 'rb') as f:
                img_oct = Image.open(f).convert("RGB")
                img_oct.load()

            self.cached_data.append((img_cfp, img_oct, label))

    def __len__(self):
        return len(self.cached_data)

    def __getitem__(self, idx):
        img_cfp, img_oct, label = self.cached_data[idx]

        if self.transform_cfp:
            img_cfp = self.transform_cfp(img_cfp.copy())
        if self.transform_oct:
            img_oct = self.transform_oct(img_oct.copy())

        return img_cfp, img_oct, torch.tensor(label, dtype=torch.long)

class GAMMADataset(Dataset):
    def __init__(self, df, cfp_dir, oct_dir, transform_cfp=None, transform_oct=None, split_name="GAMMA Data"):
        self.df = df.reset_index(drop=True)
        self.cfp_dir = cfp_dir
        self.oct_dir = oct_dir
        self.transform_cfp = transform_cfp
        self.transform_oct = transform_oct
        self.split_name = split_name

        self.cached_data = []

        for idx in tqdm(range(len(self.df)), desc=f"Caching {self.split_name} into RAM"):
            filename = str(self.df.loc[idx, 'data']).zfill(4)

            if not filename.endswith(('.jpg', '.png')):
                filename += '.jpg'

            if self.df.loc[idx, 'non'] == 1:
                label = 0
            elif self.df.loc[idx, 'early'] == 1:
                label = 1
            elif self.df.loc[idx, 'mid_advanced'] == 1:
                label = 2
            else:
                label = 0

            cfp_path = os.path.join(self.cfp_dir, filename)
            oct_path = os.path.join(self.oct_dir, filename)

            if not os.path.exists(cfp_path) or not os.path.exists(oct_path):
                print(f"[Warning] Skipping {filename} as images are not found in both branches.")
                continue

            with open(cfp_path, 'rb') as f:
                img_cfp = Image.open(f).convert("RGB")
                img_cfp.load()
            with open(oct_path, 'rb') as f:
                img_oct = Image.open(f).convert("RGB")
                img_oct.load()

            self.cached_data.append((img_cfp, img_oct, label))

    def __len__(self):
        return len(self.cached_data)

    def __getitem__(self, idx):
        img_cfp, img_oct, label = self.cached_data[idx]

        if self.transform_cfp:
            img_cfp = self.transform_cfp(img_cfp.copy())
        if self.transform_oct:
            img_oct = self.transform_oct(img_oct.copy())

        return img_cfp, img_oct, torch.tensor(label, dtype=torch.long)

class HarvardFairVisionDataset(Dataset):
    def __init__(self, cfp_dir, oct_dir, split="train", transform_cfp=None, transform_oct=None):
        self.cfp_dir = os.path.join(cfp_dir, split)
        self.oct_dir = os.path.join(oct_dir, split)
        self.transform_cfp = transform_cfp
        self.transform_oct = transform_oct
        self.split = split

        self.classes = sorted([d for d in os.listdir(self.cfp_dir) if os.path.isdir(os.path.join(self.cfp_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        temp_samples = []

        def get_core_id(filename):
            base = os.path.splitext(filename)[0]
            core = re.sub(r'^(slo_|oct_|dr_|data_)', '', base)
            return core

        for cls_name in self.classes:
            cls_path_slo = os.path.join(self.cfp_dir, cls_name)
            cls_path_oct = os.path.join(self.oct_dir, cls_name)

            if not os.path.exists(cls_path_slo) or not os.path.exists(cls_path_oct):
                continue

            slo_files = [f for f in os.listdir(cls_path_slo) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            oct_files = [f for f in os.listdir(cls_path_oct) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

            oct_dict = {get_core_id(f): f for f in oct_files}

            for slo_file in slo_files:
                core_id = get_core_id(slo_file)

                if core_id in oct_dict:
                    slo_path = os.path.join(cls_path_slo, slo_file)
                    oct_path = os.path.join(cls_path_oct, oct_dict[core_id])
                    temp_samples.append((slo_path, oct_path, self.class_to_idx[cls_name]))
                else:
                    print(f"[Warning] Data mismatch: SLO ({slo_file}) found but corresponding OCT is missing.")

        self.cached_data = []
        for slo_path, oct_path, label in tqdm(temp_samples, desc=f"Caching Harvard {split.upper()} into RAM"):
            with open(slo_path, 'rb') as f:
                img_cfp = Image.open(f).convert("RGB")
                img_cfp.load()
            with open(oct_path, 'rb') as f:
                img_oct = Image.open(f).convert("RGB")
                img_oct.load()
            self.cached_data.append((img_cfp, img_oct, label))

    def __len__(self):
        return len(self.cached_data)

    def __getitem__(self, idx):
        img_cfp, img_oct, label = self.cached_data[idx]

        if self.transform_cfp:
            img_cfp = self.transform_cfp(img_cfp.copy())
        if self.transform_oct:
            img_oct = self.transform_oct(img_oct.copy())

        return img_cfp, img_oct, torch.tensor(label, dtype=torch.long)

class TOPCONDataset(Dataset):
    def __init__(self, df, cfp_dir, oct_dir, transform_cfp=None, transform_oct=None, phase="Train"):
        self.df = df
        self.cfp_dir = cfp_dir
        self.oct_dir = oct_dir
        self.transform_cfp = transform_cfp
        self.transform_oct = transform_oct
        self.phase = phase

        self.label_cols = [c for c in df.columns if c not in ['core_id', 'cfp', 'oct']]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        cfp_path = os.path.join(self.cfp_dir, row['cfp'])
        oct_path = os.path.join(self.oct_dir, row['oct'])

        img_cfp = Image.open(cfp_path).convert('RGB')
        img_oct = Image.open(oct_path).convert('RGB')

        if self.transform_cfp:
            img_cfp = self.transform_cfp(img_cfp)

        if self.transform_oct:
            img_oct = self.transform_oct(img_oct)

        labels = row[self.label_cols].values.astype('float32')
        labels_tensor = torch.tensor(labels)

        return img_cfp, img_oct, labels_tensor