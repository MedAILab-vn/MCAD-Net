import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import re
# Import các custom transforms từ mutils của bạn
from mutils.transforms import RandomIntensityChannel, RandomAffineChannel, Identity, ToRGB, NaiveNormChannel


# =====================================================================
# 1. HÀM TẠO TRANSFORMS (AUGMENTATION)
# =====================================================================
def build_oct_transform(config, augment=True):
    """
    Tạo pipeline biến đổi cho ảnh OCT.
    """
    intensity = RandomIntensityChannel()
    affine = RandomAffineChannel(
        degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5, fill=config.fill
    ) if config.affine else Identity()

    t_list = [
        transforms.Resize((config.input_size, config.input_size)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.ConvertImageDtype(torch.float32),
        #AddGaussianNoise(mean=0.0, std=0.02)
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
        # Tạo ma trận nhiễu cùng kích thước, cộng vào ảnh gốc
        noise = torch.randn(tensor.size()) * self.std + self.mean
        # Dùng hàm clamp để kẹp chặt giá trị pixel không bị tràn khỏi dải [0.0, 1.0]
        return torch.clamp(tensor + noise, 0.0, 1.0)

    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'

def build_cfp_transform(config, augment=True):
    """
    Tạo pipeline biến đổi cho ảnh CFP.
    """
    #transforms.CenterCrop(300),  # Cắt vùng 300x300 ở tâm
    #transforms.Resize((config.input_size, config.input_size))
    t_list = [
            #transforms.CenterCrop(300),
              transforms.Resize((config.input_size, config.input_size))]

    if augment:
        t_list += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=180),
            #transforms.RandomRotation(degrees=180)
            #transforms.ColorJitter(contrast=(0.8, 1.2)),  # Contrast Transformation
            #transforms.RandomAffine(degrees=0, scale=(0.9, 1.1))  # Scaling (Tỷ lệ không gian)
        ]

    t_list += [
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ]
    return transforms.Compose(t_list)


def build_cfp_transform_advanced(config, augment=True):
    """
    Tạo pipeline biến đổi cho ảnh CFP.
    Đã tích hợp MSFF (Contrast, Scale, Gaussian Noise) rẽ nhánh riêng cho GAMMA.
    """
    t_list = [transforms.Resize((config.input_size, config.input_size))]

    if augment:
        t_list += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=180)
        ]

        # Rẽ nhánh thêm MSFF Augmentation riêng cho GAMMA
        if getattr(config, 'dataset_name', 'mmc-amd') == 'gamma':
            print('gamma')
            t_list += [
                transforms.ColorJitter(contrast=(0.8, 1.2)),  # Contrast Transformation
                transforms.RandomAffine(degrees=0, scale=(0.9, 1.1))  # Scaling (Tỷ lệ không gian)
            ]

    # Chuyển đổi sang Tensor trước khi áp dụng nhiễu toán học
    t_list += [transforms.ToTensor()]

    # Thêm Gaussian Noise (Bắt buộc phải nằm sau ToTensor)
    if augment and getattr(config, 'dataset_name', 'mmc-amd') == 'gamma':
        t_list += [AddGaussianNoise(mean=0.0, std=0.02)]

    # Chuẩn hóa cuối cùng
    t_list += [
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ]

    return transforms.Compose(t_list)

# =====================================================================
# 2. DATASET CHO MMC-AMD (GIỮ NGUYÊN BẢN GỐC)
# =====================================================================
class MMCAmdCSVDataset(Dataset):
    def __init__(self, df, cfp_dir, oct_dir, transform_cfp=None, transform_oct=None, split_name="Data"):
        self.df = df.reset_index(drop=True)
        self.cfp_dir = cfp_dir
        self.oct_dir = oct_dir
        self.transform_cfp = transform_cfp
        self.transform_oct = transform_oct
        self.split_name = split_name

        self.cached_data = []
        # Tải sẵn toàn bộ ảnh vào RAM để loại bỏ nút thắt cổ chai I/O của ổ cứng
        for idx in tqdm(range(len(self.df)), desc=f"Caching {self.split_name} into RAM"):
            cfp_name = self.df.loc[idx, 'cfp']
            oct_name = self.df.loc[idx, 'oct']
            label = self.df.loc[idx, 'label']

            # Xử lý đuôi file
            if not cfp_name.endswith(('.jpg', '.png')): cfp_name += '.jpg'
            if not oct_name.endswith(('.jpg', '.png')): oct_name += '.jpg'

            # Load PIL Image và ép vào RAM bằng hàm .load()
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

        # Tạo bản copy để Augmentation không làm hỏng ảnh gốc trong RAM
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
            # 1. Lấy tên file trực tiếp từ cột 'data'
            # 1. Lấy tên file trực tiếp từ cột 'data', ép về string và tự động điền thêm số 0 cho đủ 4 ký tự (ví dụ: '1' -> '0001')
            filename = str(self.df.loc[idx, 'data']).zfill(4)

            # Đảm bảo an toàn nếu vô tình có file thiếu đuôi .jpg
            if not filename.endswith(('.jpg', '.png')):
                filename += '.jpg'

            # 2. Xử lý gộp 3 cột nhãn One-hot thành 1 số nguyên (Label Encoding)
            if self.df.loc[idx, 'non'] == 1:
                label = 0
            elif self.df.loc[idx, 'early'] == 1:
                label = 1
            elif self.df.loc[idx, 'mid_advanced'] == 1:
                label = 2
            else:
                label = 0  # Giá trị mặc định phòng trường hợp dòng dữ liệu bị lỗi

            # 3. Tạo đường dẫn tuyệt đối
            cfp_path = os.path.join(self.cfp_dir, filename)
            oct_path = os.path.join(self.oct_dir, filename)

            if not os.path.exists(cfp_path) or not os.path.exists(oct_path):
                print(f"[Cảnh báo] Bỏ qua {filename} vì không tìm thấy ảnh ở cả 2 nhánh.")
                continue

            # 4. Load ảnh và ép vào RAM
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

        # Hàm nội bộ để gọt sạch râu ria, chỉ lấy Core ID
        def get_core_id(filename):
            base = os.path.splitext(filename)[0]  # Bỏ đuôi .jpg, .png
            # Xóa các tiền tố phổ biến, chỉ giữ lại phần ID
            core = re.sub(r'^(slo_|oct_|dr_|data_)', '', base)
            return core

        for cls_name in self.classes:
            cls_path_slo = os.path.join(self.cfp_dir, cls_name)
            cls_path_oct = os.path.join(self.oct_dir, cls_name)

            if not os.path.exists(cls_path_slo) or not os.path.exists(cls_path_oct):
                continue

            slo_files = [f for f in os.listdir(cls_path_slo) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            oct_files = [f for f in os.listdir(cls_path_oct) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

            # 1. Quét toàn bộ nhánh OCT, lưu vào từ điển: { "0001": "oct_0001.jpg" }
            oct_dict = {get_core_id(f): f for f in oct_files}

            # 2. Duyệt nhánh SLO, gọt lấy Core ID và tra cứu trong từ điển OCT
            for slo_file in slo_files:
                core_id = get_core_id(slo_file)

                if core_id in oct_dict:
                    slo_path = os.path.join(cls_path_slo, slo_file)
                    oct_path = os.path.join(cls_path_oct, oct_dict[core_id])
                    temp_samples.append((slo_path, oct_path, self.class_to_idx[cls_name]))
                else:
                    print(f"[Cảnh báo] Lệch pha dữ liệu: Có SLO ({slo_file}) nhưng không tìm thấy OCT tương ứng.")

        # Load vào RAM
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

        # Tự động trích xuất 11 cột nhãn (Bỏ qua 3 cột đầu: core_id, cfp, oct)
        self.label_cols = [c for c in df.columns if c not in ['core_id', 'cfp', 'oct']]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        cfp_path = os.path.join(self.cfp_dir, row['cfp'])
        oct_path = os.path.join(self.oct_dir, row['oct'])

        # 1. Đọc ảnh bằng PIL (Chuẩn tuyệt đối của Torchvision)
        img_cfp = Image.open(cfp_path).convert('RGB')

        # Mạng ResNet thường yêu cầu ảnh 3 kênh màu, nên ta convert OCT sang RGB luôn
        img_oct = Image.open(oct_path).convert('RGB')

        # 2. Áp dụng Transform (Cú pháp chuẩn của Torchvision)
        if self.transform_cfp:
            img_cfp = self.transform_cfp(img_cfp)

        if self.transform_oct:
            img_oct = self.transform_oct(img_oct)

        # 3. Trích xuất Vector Nhãn đa nhãn (Multi-label)
        labels = row[self.label_cols].values.astype('float32')
        labels_tensor = torch.tensor(labels)

        return img_cfp, img_oct, labels_tensor