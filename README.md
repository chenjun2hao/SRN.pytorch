# Towards Accurate Scene Text Recognition with Semantic Reasoning Networks

Unofficial PyTorch implementation of the [paper](https://arxiv.org/abs/2003.12294), which integrates not only global semantic reasoning module but also parallel visual attention module and visual-semantic fusion decoder.the semanti reasoning network(SRN) can be trained end-to-end.

At present, the accuracy of the paper cannot be achieved. And i borrowed code from [deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark)

**model**
<img src='./demo_image/SRN.png'> 

**result**
| IIIT5k_3000 | SVT   | IC03_860 | IC03_867 | IC13_857 | IC13_1015 | IC15_1811 | IC15_2077 | SVTP  | CUTE80 |  
| ----------- | ------| ---------| ---------| ---------| --------- | ----------| --------- | ----  | ------ |  
| 84.600      | 83.617| 92.907   | 92.849   | 90.315   | 88.177    | 71.010    | 68.064    | 71.008 | 68.641  |

**total_accuracy: 80.597**

---

**Feature**
- predict the character at once time
- DistributedDataParallel training




---
## Requirements
Pytorch >= 1.1.0


## Test
1. download the evaluation data from [deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark)

2. download the pretrained model from [Baidu](https://pan.baidu.com/s/1E5xeajIl_fvtrGWyrE9CeA), Password: d2qn 

3. test on the evaluation data
```bash
python test.py --eval_data path-to-data --saved_model path-to-model
```

---

## Train
1. download the training data from [deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark)

2. training from scratch
```bash
python train.py --train_data path-to-train-data --valid-data path-to-valid-data
```

## Reference
1. [bert_ocr.pytorch](https://github.com/chenjun2hao/Bert_OCR.pytorch)
2. [deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark)
3. [2D Attentional Irregular Scene Text Recognizer](https://arxiv.org/pdf/1906.05708.pdf)
4. [Towards Accurate Scene Text Recognition with Semantic Reasoning Networks](https://arxiv.org/abs/2003.12294)

## difference with the origin paper
- use resnet for 1D feature not resnetFpn 2D feature
- use add not gated unit for visual-semanti fusion decoder

## other
It is difficult to achieve the accuracy of the paper, hope more people to try and share