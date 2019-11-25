# 2D Attentional Irregular Scene Text Recognizer

Unofficial PyTorch implementation of the [paper](https://arxiv.org/pdf/1906.05708.pdf), which transforms the irregular text with 2D layout to character sequence directly via 2D attentional scheme. They utilize a relation attention module to capture the dependencies of feature maps
and a parallel attention module to decode all characters in
parallel.

At present, the accuracy of the paper cannot be achieved. And i borrowed code from [deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark)

**Feature**
1. Output image string once not like the seqtoseq model


## Test
Now can not reach the accuracy in the original paper,so i don't upload the pretrained model. it will upload later.

## Train
I prepared a small dataset for train.The image and labels are in `./dataset/BAIDU`.
```bash
python train.py --root ./dataset/BAIDU/images/ --train_csv ./dataset/BAIDU/small_train.txt --val_csv ./dataset/BAIDU/small_train.txt
```
