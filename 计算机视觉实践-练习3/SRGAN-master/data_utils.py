from os import listdir
from os.path import join

from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def train_hr_transform(crop_size):
    return Compose([
        RandomCrop(crop_size),
        ToTensor(),
    ])


def train_lr_transform(crop_size, upscale_factor):
    return Compose([
        ToPILImage(),
        Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC),
        ToTensor()
    ])


def display_transform():
    return Compose([
        ToPILImage(),
        Resize(400),
        CenterCrop(400),
        ToTensor()
    ])


class TrainDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, crop_size, upscale_factor):
        super(TrainDatasetFromFolder, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.hr_transform = train_hr_transform(crop_size)
        self.lr_transform = train_lr_transform(crop_size, upscale_factor)

    def __getitem__(self, index):
        hr_image = self.hr_transform(Image.open(self.image_filenames[index]))
        lr_image = self.lr_transform(hr_image)
        return lr_image, hr_image

    def __len__(self):
        return len(self.image_filenames)


class ValDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(ValDatasetFromFolder, self).__init__()
        self.upscale_factor = upscale_factor
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]

    def __getitem__(self, index):
        hr_image = Image.open(self.image_filenames[index])
        w, h = hr_image.size
        crop_size = calculate_valid_crop_size(min(w, h), self.upscale_factor)
        lr_scale = Resize(crop_size // self.upscale_factor, interpolation=Image.BICUBIC)
        hr_scale = Resize(crop_size, interpolation=Image.BICUBIC)
        hr_image = CenterCrop(crop_size)(hr_image)
        lr_image = lr_scale(hr_image)
        hr_restore_img = hr_scale(lr_image)
        return ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

    def __len__(self):
        return len(self.image_filenames)


#  加载测试集中的图像数据 同训练集
class TestDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(TestDatasetFromFolder, self).__init__()
        self.lr_path = dataset_dir + '/SRF_' + str(upscale_factor) + '/data/'  # 构建低分辨率图像文件路径
        self.hr_path = dataset_dir + '/SRF_' + str(upscale_factor) + '/target/'  # 构建高分辨率图像文件路径
        self.upscale_factor = upscale_factor
        # 获取两个路径下的图像文件名，并保存在lr_filenames和hr_filenames列表中。
        self.lr_filenames = [join(self.lr_path, x) for x in listdir(self.lr_path) if is_image_file(x)]
        self.hr_filenames = [join(self.hr_path, x) for x in listdir(self.hr_path) if is_image_file(x)]

    def __getitem__(self, index):
        # 获取给定索引的图像数据
        image_name = self.lr_filenames[index].split('/')[-1]
        lr_image = Image.open(self.lr_filenames[index])  # 打开低分辨率图像文件
        w, h = lr_image.size  # 获取低分辨率图像的宽度和高度
        hr_image = Image.open(self.hr_filenames[index])  # 打开高分辨率图像文件
        hr_scale_1 = Resize((h // self.upscale_factor, w // self.upscale_factor), interpolation=Image.BICUBIC)
        lr_image = hr_scale_1(lr_image)
        hr_scale = Resize((self.upscale_factor * h, self.upscale_factor * w), interpolation=Image.BICUBIC)  # 缩放高分辨率图像
        hr_restore_img = hr_scale(lr_image)  # 缩放得到还原后的高分辨率图像
        # 将图像文件名、低分辨率图像、还原后的高分辨率图像和原始高分辨率图像转换为张量并返回
        return image_name, ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

    def __len__(self):
        return len(self.lr_filenames)


# class TestDatasetFromFolder(Dataset):
#     def __init__(self, dataset_dir, upscale_factor):
#         super(TestDatasetFromFolder, self).__init__()
#         self.lr_path = dataset_dir + '/SRF_' + str(upscale_factor) + '/data/'
#         self.hr_path = dataset_dir + '/SRF_' + str(upscale_factor) + '/target/'
#         self.upscale_factor = upscale_factor
#         self.lr_filenames = [join(self.lr_path, x) for x in listdir(self.lr_path) if is_image_file(x)]
#         self.hr_filenames = [join(self.hr_path, x) for x in listdir(self.hr_path) if is_image_file(x)]
#
#     def __getitem__(self, index):
#         image_name = self.lr_filenames[index].split('/')[-1]
#         lr_image = Image.open(self.lr_filenames[index])
#         w, h = lr_image.size
#         hr_image = Image.open(self.hr_filenames[index])
#         hr_scale = Resize((self.upscale_factor * h, self.upscale_factor * w), interpolation=Image.BICUBIC)
#         hr_restore_img = hr_scale(lr_image)
#         return image_name, ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)
#
#     def __len__(self):
#         return len(self.lr_filenames)
