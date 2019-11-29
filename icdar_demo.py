import string
import argparse
import json
import os

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data

from utils import CTCLabelConverter, AttnLabelConverter, TransformerConverter
from dataset import RawDataset, AlignCollate
from model import Model
import matplotlib.pyplot as plt
from PIL import Image


def demo(opt):
    """ model configuration """
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    elif 'Bert' in opt.Prediction:
        converter = TransformerConverter(opt.character, opt.batch_max_length)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)
    opt.alphabet_size = len(opt.character) + 2  # +2 for [UNK]+[EOS]

    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)
    print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
          opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
          opt.SequenceModeling, opt.Prediction)

    model = torch.nn.DataParallel(model)
    if torch.cuda.is_available():
        model = model.cuda()

    # load model
    print('loading pretrained model from %s' % opt.saved_model)
    model.load_state_dict(torch.load(opt.saved_model))

    # prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
    AlignCollate_demo = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    demo_data = RawDataset(root=opt.image_folder, opt=opt)  # use RawDataset
    demo_loader = torch.utils.data.DataLoader(
        demo_data, batch_size=opt.batch_size,
        shuffle=False,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_demo, pin_memory=True)

    # mkdir result
    experiment_name = os.path.join('./result', opt.image_folder.split('/')[-2])
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)
    result = {}

    # predict
    model.eval()
    for idx, (image_tensors, image_path_list) in enumerate(demo_loader):
        batch_size = image_tensors.size(0)
        with torch.no_grad():
            image = image_tensors.cuda()
            # For max length prediction
            length_for_pred = torch.cuda.IntTensor([opt.batch_max_length] * batch_size)
            text_for_pred = torch.cuda.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0)

        if 'CTC' in opt.Prediction:
            preds = model(image, text_for_pred).log_softmax(2)

            # Select max probabilty (greedy decoding) then decode index to character
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            _, preds_index = preds.permute(1, 0, 2).max(2)
            preds_index = preds_index.transpose(1, 0).contiguous().view(-1)
            preds_str = converter.decode(preds_index.data, preds_size.data)

        elif 'Bert' in opt.Prediction:
            with torch.no_grad():
                pad_mask = None
                preds = model(image, pad_mask)

                # select max probabilty (greedy decoding) then decode index to character
                _, preds_index = preds[1].max(2)
                length_for_pred = torch.cuda.IntTensor([preds_index.size(-1)] * batch_size)
                preds_str = converter.decode(preds_index, length_for_pred)

        else:
            preds = model(image, text_for_pred, is_train=False)

            # select max probabilty (greedy decoding) then decode index to character
            _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index, length_for_pred)

        print(f'{idx}/{len(demo_data) / opt.batch_size}')

        for img_name, pred in zip(image_path_list, preds_str):
            if 'Attn' in opt.Prediction:
                pred = pred[:pred.find('[s]')]  # prune after "end of sentence" token ([s])

            # for show

            # write in json
            name = f'{img_name}'.split('/')[-1].replace('gt', 'res').split('.')[0]
            value = [{"transcription": f'{pred}'}]
            result[name] = value

    with open(f'{experiment_name}/result.json', 'w') as f:
        json.dump(result, f)
        print("writed finish...")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--image_folder', default='demo_image/', help='path to image_folder which contains text images')
    parser.add_argument('--image_folder', default='/home/deepblue/deepbluetwo/chenjun/data/test_part1_task2_images/', help='path to image_folder which contains text images')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
    parser.add_argument('--saved_model', default='./saved_models/TPS-AsterRes-Bert-Bert_pred-Seed666/best_accuracy.pth', help="path to saved_model to evaluation")
    # parser.add_argument('--saved_model', default='./model/TPS-ResNet-BiLSTM-Attn.pth', help="path to saved_model to evaluation")
    """ Data processing """
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    """ Model Architecture """
    parser.add_argument('--Transformation', type=str, default='TPS', help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, default='AsterRes', help='FeatureExtraction stage. VGG|RCNN|ResNet|AsterRes')
    parser.add_argument('--SequenceModeling', type=str, default='Bert', help='SequenceModeling stage. None|BiLSTM|Bert')
    parser.add_argument('--Prediction', type=str, default='Bert_pred', help='Prediction stage. CTC|Attn|Bert_pred')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=1024,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')
    parser.add_argument('--position_dim', type=int, default=210, help='the length sequence out from cnn encoder')

    opt = parser.parse_args()

    """ vocab / character number configuration """
    if opt.sensitive:
        opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()

    demo(opt)
