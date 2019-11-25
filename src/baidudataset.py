#!/usr/bin/python
# encoding: utf-8

import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import sampler
import torchvision.transforms as transforms
# import lmdb
import six
import sys
from PIL import Image
import numpy as np
import cv2
import os
import math
import matplotlib.pyplot as plt
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def random_crop(img, coor, r_int=[-10, -8, -6, -2, 0, 2, 4, 6, 8], train=True):
    '''对文字区域随机往内或外扩充
    img:cv2格式
    coor:list,8维
    '''
    # 找到最大的矩形区域
    h,w,_ = img.shape
    coor = [int(x) for x in coor]
    minx = min(coor[::2])
    maxx = max(coor[::2])
    miny = min(coor[1::2])
    maxy = max(coor[1::2])

    newx1, newy1 = minx, miny       # 左上角的点
    newx3, newy3 = maxx, maxy       # 右下角的点

    randint = random.choice(r_int)  # 随机扩大或缩小

    if len(coor) == 8:              # 8个点的区域标注,4周同时扩
        newx1 += randint;  newx1 = max(newx1, 0)
        newy1 += randint;  newy1 = max(newy1, 0)    
        newx3 -= randint;  newx3 = min(newx3, w)
        newy3 -= randint;  newy3 = min(newy3, h)

        newimg = img[newy1:newy3, newx1:newx3, :]           # crop出图像
        # 判断是否需要旋转
        if abs(newy3 - newy1) / abs(newx3-newx1) > 1.5:      # y方向的距离大于x方向的距离
            newimg = np.rot90(newimg)                       # 旋转90度
        # 批量训练的时候需要等宽，等比例缩放再填充,放到dataloader中进行，这样每一批次按最长的填充就好了
        if train:
            pass
            
    elif len(coor) == 4:            # 只扩大宽度方向
        newx1 += randint;  newx1 = max(newx1, 0) 
        newx3 -= randint;  newx3 = min(newx3, w)

        newimg = img[newy1:newy3, newx1:newx3, :]           # crop出图像

    return Image.fromarray(newimg)   



class BAIDUset(Dataset):
    '''
        baidu oc识别的数据集，图片都已经crop好了，有水平和竖直两种图片
    '''
    def __init__(self, opt, csv_root, transform=None, target_transform=None):
        self.opt = opt
        self.root = opt.root
        with open(csv_root) as f:
            self.labels = f.readlines()
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        per_label = self.labels[idx].rstrip().split('\t')
#         imgpath = os.path.join(self.root, per_label[0])         # 图片位置
#         text = per_label[1].replace(' ','')                                     # 图片的文字label
        imgpath = os.path.join(self.root, per_label[0].rstrip())                 # 图片位置
        text = per_label[1].strip()

        try:
            if self.opt.rgb:
                img = Image.open(imgpath).convert('RGB')  # for color image
            else:
                img = Image.open(imgpath).convert('L')

        except IOError:
            print(imgpath)
            print(f'Corrupted image for {idx}')
            # make dummy image and dummy label for corrupted image.
            if self.opt.rgb:
                img = Image.new('RGB', (self.opt.imgW, self.opt.imgH))
            else:
                img = Image.new('L', (self.opt.imgW, self.opt.imgH))

        '''
            for vertice text, rotate 90
        '''
        w, h = img.size
        if h > 1.5 * w and len(text) > 2:
            img = img.transpose(Image.ROTATE_90)

        return (img, text)


class BaiduCollate(object):
    '''每个batch按最宽的图像进行填充
    '''

    def __init__(self, imgH=32, imgW=128, keep_ratio=True):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio

    def __call__(self, batch):
        images, labels = zip(*batch)

        imgH = self.imgH
        resized_images = []
        if self.keep_ratio:                     # 等比例缩放
            for image in images:
                w, h = image.size
                scale = 1.0 * h / self.imgH
                image = image.resize((int(w/scale),int(h/scale)), Image.BILINEAR)
#                 image = image.resize((self.imgW, int(h/scale)), Image.BILINEAR)
                resized_images.append(image)
        else:
            for image in images:
                image = image.resize((self.imgW, self.imgH), Image.BILINEAR)
                resized_images.append(image)

        # 按最大的w进行填充
        maxw = max([x.size[0] for x in resized_images])
#         images = [baidu_pad(x, maxw, imgH) for x in resized_images]       # 填充完成
        images = resized_images

        transform = resizeNormalize((maxw, imgH))
        images = [transform(image) for image in images]
        images = torch.cat([t.unsqueeze(0) for t in images], 0)

        return images, labels


def baidu_pad(img, maxw, h=32):
    img = np.array(img)
    if img.ndim == 2:
        h, w = img.shape
        randint = random.randint(-10, -1)
        newimg = np.ones((h, maxw)) * img[randint, randint]
        newimg[:,:w] = img            # 以最大宽度填充
    else:
        h,w,c = img.shape
        randint = random.randint(-10, -1)
        newimg = np.ones((h, maxw, c), dtype='uint8') * img[randint, randint, :]
        newimg[:,:w, :] = img            # 以最大宽度填充

    return Image.fromarray(newimg)

class ImgDataset(Dataset):
    '''
        采用直接读取图片的方式读入，可以对文字区域随机加上抖动，针对4个点的标注
    '''
    def __init__(self, root=None, csv_root=None,transform=None, target_transform=None, training=True):
        self.root = root
        with open(csv_root) as f:
            self.labels = f.readlines()
        self.transform = transform
        self.target_transform = target_transform
        self.train = training
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        per_label = self.labels[idx].rstrip().split(',')
        imgpath = os.path.join(self.root, per_label[0])         # 图片位置
        img = cv2.imread(imgpath)
        coor = per_label[1:9]                                   # 标记的4个点
        if self.train:
            img = random_crop(img, coor)            # 随机裁剪和90度旋转
        else:
            img = random_crop(img, coor, r_int=[0, 0])          # 不随机裁剪

        if self.transform is not None:
            img = self.transform(img)

        label = str(per_label[9])
        label = label.decode('utf-8')           #

        if self.target_transform is not None:                   # target_transform是没有的
            label = self.target_transform(label)

        return (img, label)

class FourCoorDataset(ImgDataset):
    def __init__(self, root=None, csv_root=None,transform=None, target_transform=None, training=True):
        super(FourCoorDataset, self).__init__(root=root, csv_root=csv_root,transform=transform, target_transform=target_transform, training=training)

    def __getitem__(self, idx):
        per_label = self.labels[idx].rstrip().split('\t')
        imgpath = os.path.join(self.root, per_label[0])         # 图片位置
        img = cv2.imread(imgpath)
        img = img[:, :, ::-1]
        coor = per_label[2:6]                                   # 标记的4个点
        if self.train:
            img = random_crop(img, coor)            # 随机裁剪和90度旋转
        else:
            img = random_crop(img, coor, r_int=[0, 0])          # 不随机裁剪
            # img = img.convert('L')

        if self.transform is not None:
            img = self.transform(img)

        label = str(per_label[1].lstrip())
        # label = label.decode('utf-8')           #

        if self.target_transform is not None:                   # target_transform是没有的
            label = self.target_transform(label)

        return (img, label)


class lmdbDataset(Dataset):
    '''采用lmdb工具进行数据读取
    '''
    def __init__(self, root=None, transform=None, target_transform=None):
        self.root = root
        self.env = lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)

        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'))
            self.nSamples = nSamples

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index += 1
        with self.env.begin(write=False) as txn:
            img_key = 'image-%09d' % index
            imgbuf = txn.get(img_key)

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                # img = Image.open(buf).convert('L')
                if 'test' in self.root:
                    img = Image.open(buf).convert('L')
                else:
                    img = Image.open(buf)       # brightness 调整的时候需要是彩色图像
            except IOError:
                print('Corrupted image for %d' % index)
                return self[index + 1]

            if self.transform is not None:
                img = self.transform(img)

            label_key = 'label-%09d' % index
            label = str(txn.get(label_key))

            if self.target_transform is not None:
                label = self.target_transform(label)

        return (img, label)


class resizeNormalize(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


class randomSequentialSampler(sampler.Sampler):

    def __init__(self, data_source, batch_size):
        self.num_samples = len(data_source)
        self.batch_size = batch_size

    def __iter__(self):
        n_batch = len(self) // self.batch_size
        tail = len(self) % self.batch_size
        index = torch.LongTensor(len(self)).fill_(0)
        for i in range(n_batch):
            random_start = random.randint(0, len(self) - self.batch_size)
            batch_index = random_start + torch.arange(0, self.batch_size)
            index[i * self.batch_size:(i + 1) * self.batch_size] = batch_index
        # deal with tail
        if tail:
            random_start = random.randint(0, len(self) - self.batch_size)
            tail_index = random_start + torch.arange(0, tail)
            index[(i + 1) * self.batch_size:] = tail_index

        return iter(index)

    def __len__(self):
        return self.num_samples


class alignCollate(object):

    def __init__(self, imgH=32, imgW=100, keep_ratio=False, min_ratio=1):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio
        self.min_ratio = min_ratio

    def __call__(self, batch):
        images, labels = zip(*batch)

        imgH = self.imgH
        imgW = self.imgW
        if self.keep_ratio:
            ratios = []
            for image in images:
                w, h = image.size
                ratios.append(w / float(h))
            ratios.sort()
            max_ratio = ratios[-1]                  # 找到最大的长宽比
            imgW = int(np.floor(max_ratio * imgH))
            imgW = max(imgH * self.min_ratio, imgW)  # assure imgH >= imgW

        transform = resizeNormalize((imgW, imgH))
        images = [transform(image) for image in images]
        images = torch.cat([t.unsqueeze(0) for t in images], 0)

        return images, labels

def pad_pil(img, maxw, h=32):
    img = np.array(img)
    if img.ndim == 2:
        h, w = img.shape
        randint = random.randint(-10, -1)
        newimg = np.ones((h, maxw)) * img[randint, randint]
        newimg[:,:w] = img            # 以最大宽度填充
    else:
        h,w,c = img.shape
        randint = random.randint(-10, -1)
        newimg = np.ones((h, maxw, c), dtype='uint8') * img[randint, randint, :]
        newimg[:,:w, :] = img            # 以最大宽度填充

    return Image.fromarray(newimg)


class OwnalignCollate(object):
    '''每个batch按最宽的图像进行填充
    '''

    def __init__(self, imgH=32, keep_ratio=True, min_ratio=1):
        self.imgH = imgH
        self.keep_ratio = keep_ratio

    def __call__(self, batch):
        images, labels = zip(*batch)

        imgH = self.imgH
        resized_images = []
        if self.keep_ratio:                     # 等比例缩放
            for image in images:
                w, h = image.size
                scale = 1.0 * h / self.imgH
                image = image.resize((int(w/scale),int(h/scale)), Image.ANTIALIAS)
                resized_images.append(image)

        # 按最大的w进行填充
        maxw = max([x.size[0] for x in resized_images])
        images = [pad_pil(x, maxw, imgH) for x in resized_images]       # 填充完成

        transform = resizeNormalize((maxw, imgH))
        images = [transform(image) for image in images]
        images = torch.cat([t.unsqueeze(0) for t in images], 0)

        return images, labels


class TransformerCollate(OwnalignCollate):
    '''
    converter:将字符串转换成整数组
    '''
    def __init__(self, imgH=32, keep_ratio=True, converter=None):
        super(TransformerCollate, self).__init__(imgH=32, keep_ratio=True)
        self.converter = converter

    def __call__(self, batch):

        images, labels = zip(*batch)            #

        # 按图片的最大长度进行填充
        imgH = self.imgH
        resized_images = []
        if self.keep_ratio:                     # 等比例缩放
            for image in images:
                w, h = image.size
                scale = 1.0 * h / self.imgH
                image = image.resize((int(w/scale),int(h/scale)), Image.ANTIALIAS)
                resized_images.append(image)
        
        # 求src_seq的mask，将填充部分mask设为0
        lengthw = [x.size[0] for x in resized_images]           # 每个图片的宽度
        maxw = max([x.size[0] for x in resized_images])         # 最大宽度
        srcw = [math.floor(x / 4.0 + 1)  for x in lengthw]      # 原图有文字区域的src的长度
        max_seq = math.floor(maxw / 4.0 + 1)                    # 求图片经过encoder之后的序列长度
        src_seq = np.array([
            [Constants.UNK] * int(inst) + [Constants.PAD] * int(max_seq - inst) for inst in srcw
        ])
        src_seq = torch.LongTensor(src_seq)                     # 填充
        # 按最大的w进行填充
        
        images = [pad_pil(x, maxw, imgH) for x in resized_images]       # 填充完成

        transform = resizeNormalize((maxw, imgH))
        images = [transform(image) for image in images]
        images = torch.cat([t.unsqueeze(0) for t in images], 0)

        # 把label编码成整数，进行batch填充
        tlabel = [self.converter.encode(x) for x in labels]
        tgt_seq, tgt_pos = paired(tlabel)

        return images, src_seq, tgt_seq, tgt_pos


def paired(insts):
    ''' Pad the instance to the max seq length in batch '''

    max_len = max(len(inst) for inst in insts)

    batch_seq = np.array([
        inst + [Constants.PAD] * (max_len - len(inst))
        for inst in insts])

    batch_pos = np.array([
        [pos_i+1 if w_i != Constants.PAD else 0
         for pos_i, w_i in enumerate(inst)] for inst in batch_seq])

    batch_seq = torch.LongTensor(batch_seq)
    batch_pos = torch.LongTensor(batch_pos)

    return batch_seq, batch_pos


class TransformerConverter(object):
    """Convert between str and label.

    NOTE:
        Insert `blank` to the alphabet for CTC.

    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, alphabet, ignore_case=True):
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = 'PUBE' + alphabet  # for `-1` index

        self.dict = {}
        for i, char in enumerate(self.alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i

    def encode(self, text):
        """Support batch or single str.

        Args:
            text (str or list of str): texts to convert.

        Returns:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        """
        if isinstance(text, unicode):
            text = [
                self.dict[char.lower() if self._ignore_case else char]
                for char in text
            ]
            text.insert(0, self.dict['B'])              # 添加开始标识符
            text.append(self.dict['E'])                 # 添加结束标识符
        return text

    def decode(self, preds, raw=True):
        '''Support batch and single str decode

        :param preds: numpy.array
        :return: text: predict string
                 texts: predict string list
        '''
        assert isinstance(preds, np.ndarray), 'preds must be np.ndarray type'
        if preds.ndim == 1:
            if raw:                                     # 不剔除
                return ''.join([self.alphabet[i] for i in preds])
            else:
                charlist = [x for x in preds if x > Constants.EOS]      # 只有大于4以上的编码才有效
                return ''.join([self.alphabet[i] for i in charlist])
        else:
            # batch mode
            assert preds.ndim > 1, 'The batch mode is wrong'

            texts = [self.decode(t, raw) for t in preds]

            return texts
