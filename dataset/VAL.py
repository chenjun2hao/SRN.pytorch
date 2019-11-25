from __future__ import absolute_import
import os 

path = '/home/yangna/deepblue/OCR/data/Baidu/train_images/'
tpath = './dataset/BAIDU/images/'
with open('./dataset/BAIDU/small_train.txt') as f:
    datas = f.readlines()

for data in datas:
    name = data.rstrip().split('\t')[0]
    src = os.path.join(path, name)
    target = os.path.join(tpath, name)
    os.system('cp {} {}'.format(src, target))