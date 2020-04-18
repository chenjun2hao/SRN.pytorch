# Towards Accurate Scene Text Recognition with Semantic Reasoning Networks

Unofficial PyTorch implementation of the [paper](https://arxiv.org/abs/2003.12294), which integrates not only global semantic reasoning module but also parallel visual attention module and visual-semantic fusion decoder.the semanti reasoning network(SRN) can be trained end-to-end.

At present, the accuracy of the paper cannot be achieved. And i borrowed code from [deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark)

**model**
<img src='./demo_image/SRN.png'> 



**Feature**
- predict the character at once time
- DistributedDataParallel training



---
## Requirements
Pytorch >= 1.1.0


## Test
coming soon ...

---

## Train
coming soon ...

## Reference
1. [bert_ocr.pytorch](https://github.com/chenjun2hao/Bert_OCR.pytorch)
2. [deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark)
3. [2D Attentional Irregular Scene Text Recognizer](https://arxiv.org/pdf/1906.05708.pdf)
4. [Towards Accurate Scene Text Recognition with Semantic Reasoning Networks](https://arxiv.org/abs/2003.12294)