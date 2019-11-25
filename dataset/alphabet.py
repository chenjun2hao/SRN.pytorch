import os
import glob

# datas = glob.glob(r'/home/yangna/deepblue/OCR/EAST2/ICDAR_2015/ch4_training_localization_transcription_gt/*.txt')
path = './dataset/BAIDU/train.list'
alphabet = []

with open(path) as f:
    perdata = f.readlines()

for pd in perdata:
    pd = pd.rstrip().split('\t')
    alphabet += pd[-1]

temp = ''.join(set(alphabet))

with open('./dataset/BAIDU/baidu_alphabet.txt', 'w') as outf:
    outf.write(temp)