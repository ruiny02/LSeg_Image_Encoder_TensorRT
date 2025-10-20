# data/base.py
import random
import numpy as np
from PIL import Image, ImageOps
import torch
import torch.utils.data as data

__all__ = ['BaseDataset', 'test_batchify_fn']

class BaseDataset(data.Dataset):
    def __init__(self, root, split, mode=None, transform=None,
                 target_transform=None, base_size=520, crop_size=480):
        self.root = root
        self.split = split
        self.mode = mode or split
        self.transform = transform
        self.target_transform = target_transform
        self.base_size = base_size
        self.crop_size = crop_size
        if self.mode == 'train':
            print(f'BaseDataset: base_size {base_size}, crop_size {crop_size}')

    def __getitem__(self, index):
        raise NotImplementedError

    @property
    def num_class(self):
        return self.NUM_CLASS

    @property
    def pred_offset(self):
        raise NotImplementedError

    def make_pred(self, x):
        return x + self.pred_offset

    def _val_sync_transform(self, img, mask):
        outsize = self.crop_size
        w, h = img.size
        if w > h:
            oh = outsize; ow = int(w * oh / h)
        else:
            ow = outsize; oh = int(h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        x1 = (ow - outsize) // 2
        y1 = (oh - outsize) // 2
        img = img.crop((x1, y1, x1+outsize, y1+outsize))
        mask = mask.crop((x1, y1, x1+outsize, y1+outsize))
        return img, self._mask_transform(mask)

    def _sync_transform(self, img, mask):
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        # random scale
        long_size = random.randint(int(self.base_size*0.5), int(self.base_size*2.0))
        w, h = img.size
        if h > w:
            oh = long_size; ow = int(w * long_size / h + 0.5)
            short = ow
        else:
            ow = long_size; oh = int(h * long_size / w + 0.5)
            short = oh
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad if needed
        padh = max(self.crop_size - oh, 0)
        padw = max(self.crop_size - ow, 0)
        if padh or padw:
            img = ImageOps.expand(img, border=(0,0,padw,padh), fill=0)
            mask = ImageOps.expand(mask, border=(0,0,padw,padh), fill=0)
        # random crop
        w, h = img.size
        x1 = random.randint(0, w-self.crop_size)
        y1 = random.randint(0, h-self.crop_size)
        img = img.crop((x1, y1, x1+self.crop_size, y1+self.crop_size))
        mask = mask.crop((x1, y1, x1+self.crop_size, y1+self.crop_size))
        return img, self._mask_transform(mask)

    def _mask_transform(self, mask):
        return torch.from_numpy(np.array(mask)).long()


def test_batchify_fn(data):
    error_msg = "batch must contain tensors, tuples or lists; found {}"
    if isinstance(data[0], (str, torch.Tensor)):
        return list(data)
    elif isinstance(data[0], (tuple, list)):
        transposed = list(zip(*data))
        return [test_batchify_fn(x) for x in transposed]
    else:
        raise TypeError(error_msg.format(type(data[0])))
