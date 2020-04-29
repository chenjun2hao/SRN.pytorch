import os
import sys
import time
import random
import string
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.utils.data
import numpy as np

from utils import CTCLabelConverter, AttnLabelConverter, Averager, TransformerConverter, SRNConverter
from dataset import hierarchical_dataset, AlignCollate, Batch_Balanced_Dataset
from model import Model
from test import validation
from src.baidudataset import BAIDUset, BaiduCollate
from modules.optimizer.ranger import Ranger
# from modules.SRN_modules import cal_performance
from modules.SRN_modules import cal_performance2 as cal_performance
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train(opt):
    """ dataset preparation """
    if opt.select_data == 'baidu':
        train_set = BAIDUset(opt, opt.train_csv)
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=opt.batch_size,
            shuffle=True, num_workers=int(opt.workers),
            collate_fn=BaiduCollate(opt.imgH, opt.imgW, keep_ratio=False)
        )
        val_set = BAIDUset(opt, opt.val_csv)
        valid_loader = torch.utils.data.DataLoader(
            val_set, batch_size=opt.batch_size,
            shuffle=True,  
            num_workers=int(opt.workers),
            collate_fn=BaiduCollate(opt.imgH, opt.imgW, keep_ratio=False), pin_memory=True)

    else:
        opt.select_data = opt.select_data.split('-')
        opt.batch_ratio = opt.batch_ratio.split('-')
        train_dataset = Batch_Balanced_Dataset(opt)

        AlignCollate_valid = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
        valid_dataset = hierarchical_dataset(root=opt.valid_data, opt=opt)
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=opt.batch_size,
            shuffle=True,  # 'True' to check training progress with validation function.
            num_workers=int(opt.workers),
            collate_fn=AlignCollate_valid, pin_memory=True)
    print('-' * 80)

    """ model configuration """
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    elif 'Bert' in opt.Prediction:
        converter = TransformerConverter(opt.character, opt.max_seq)
    elif 'SRN' in opt.Prediction:
        converter = SRNConverter(opt.character, opt.SRN_PAD)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)
    print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
          opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
          opt.SequenceModeling, opt.Prediction)

    # weight initialization
    for name, param in model.named_parameters():
        if 'localization_fc2' in name:
            print(f'Skip {name} as it is already initialized')
            continue
        try:
            if 'bias' in name:
                init.constant_(param, 0.0)
            elif 'weight' in name:
                init.kaiming_normal_(param)
        except Exception as e:  # for batchnorm.
            if 'weight' in name:
                param.data.fill_(1)
            continue

    # data parallel for multi-GPU
    model = torch.nn.DataParallel(model).cuda()
    model.train()
    if opt.continue_model != '':
        print(f'loading pretrained model from {opt.continue_model}')
        model.load_state_dict(torch.load(opt.continue_model))
    print("Model:")
    print(model)

    """ setup loss """
    if 'CTC' in opt.Prediction:
        criterion = torch.nn.CTCLoss(zero_infinity=True).cuda()
    elif 'Bert' in opt.Prediction:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0).cuda()
    elif 'SRN' in opt.Prediction:
        criterion = cal_performance
    else:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=0).cuda()  # ignore [GO] token = ignore index 0
    # loss averager
    loss_avg = Averager()

    # filter that only require gradient decent
    filtered_parameters = []
    params_num = []
    for p in filter(lambda p: p.requires_grad, model.parameters()):
        filtered_parameters.append(p)
        params_num.append(np.prod(p.size()))
    print('Trainable params num : ', sum(params_num))
    # [print(name, p.numel()) for name, p in filter(lambda p: p[1].requires_grad, model.named_parameters())]

    # setup optimizer
    if opt.adam:
        optimizer = optim.Adam(filtered_parameters, lr=opt.lr, betas=(opt.beta1, 0.999))
    elif opt.ranger:
        optimizer = Ranger(filtered_parameters, lr=opt.lr)
    else:
        optimizer = optim.Adadelta(filtered_parameters, lr=opt.lr, rho=opt.rho, eps=opt.eps)
    print("Optimizer:")
    print(optimizer)
    
    lrScheduler = lr_scheduler.MultiStepLR(optimizer, [20, 50, 100], gamma=0.5)                         # 减小学习速率

    """ final options """
    # print(opt)
    with open(f'./saved_models/{opt.experiment_name}/opt.txt', 'a') as opt_file:
        opt_log = '------------ Options -------------\n'
        args = vars(opt)
        for k, v in args.items():
            opt_log += f'{str(k)}: {str(v)}\n'
        opt_log += '---------------------------------------\n'
        print(opt_log)
        opt_file.write(opt_log)

    """ start training """
    start_iter = 0
    if opt.continue_model != '':
        start_iter = int(opt.continue_model.split('_')[-1].split('.')[0])
        print(f'continue to train, start_iter: {start_iter}')

    start_time = time.time()
    best_accuracy = -1
    best_norm_ED = 1e+6
    i = start_iter
    if opt.select_data == 'baidu':
        train_iter = iter(train_loader)
        step_per_epoch = len(train_set) / opt.batch_size
        print('一代有多少step:', step_per_epoch)
    else:
        step_per_epoch = train_dataset.nums_samples / opt.batch_size
        print('一代有多少step:', step_per_epoch)

    
    while(True):
        # try:
        # train part
        for p in model.parameters():
            p.requires_grad = True

        if opt.select_data == 'baidu':
            try:
                image_tensors, labels = train_iter.next()
            except:
                train_iter = iter(train_loader)
                image_tensors, labels = train_iter.next()
        else:
            image_tensors, labels = train_dataset.get_batch()

        image = image_tensors.cuda()
        if 'SRN' in opt.Prediction:
            text, length = converter.encode(labels)
        else:
            text, length = converter.encode(labels)
        batch_size = image.size(0)

        if 'CTC' in opt.Prediction:
            preds = model(image, text).log_softmax(2)
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            preds = preds.permute(1, 0, 2)

            # (ctc_a) For PyTorch 1.2.0 and 1.3.0. To avoid ctc_loss issue, disabled cudnn for the computation of the ctc_loss
            # https://github.com/jpuigcerver/PyLaia/issues/16
            torch.backends.cudnn.enabled = False
            cost = criterion(preds, text.to(device), preds_size.to(device), length.to(device))
            torch.backends.cudnn.enabled = True

        elif 'Bert' in opt.Prediction:
            pad_mask = None
            # print(image.shape)
            preds = model(image, pad_mask)
            cost = criterion(preds[0].view(-1, preds[0].shape[-1]), text.contiguous().view(-1)) + \
                   criterion(preds[1].view(-1, preds[1].shape[-1]), text.contiguous().view(-1))

        elif 'SRN' in opt.Prediction:
            preds = model(image, None)
            cost, n_correct = criterion(preds, text, opt.SRN_PAD)

        else:
            preds = model(image, text[:, :-1]) # align with Attention.forward
            target = text[:, 1:]  # without [GO] Symbol
            cost = criterion(preds.view(-1, preds.shape[-1]), target.contiguous().view(-1))

        model.zero_grad()
        cost.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)  # gradient clipping with 5 (Default)
        optimizer.step()

        loss_avg.add(cost)

        if i % opt.disInterval == 0:
            elapsed_time = time.time() - start_time
            print(f'[{i}/{opt.num_iter}] Loss: {loss_avg.val():0.5f} elapsed_time: {elapsed_time:0.5f}')
            start_time = time.time()

        # validation part
        if i % opt.valInterval == 0 and i > start_iter:
            elapsed_time = time.time() - start_time
            print(f'[{i}/{opt.num_iter}] Loss: {loss_avg.val():0.5f} elapsed_time: {elapsed_time:0.5f}')
            # for log
            with open(f'./saved_models/{opt.experiment_name}/log_train.txt', 'a') as log:
                log.write(f'[{i}/{opt.num_iter}] Loss: {loss_avg.val():0.5f} elapsed_time: {elapsed_time:0.5f}\n')
                loss_avg.reset()

                model.eval()
                # valid_loss, current_accuracy, current_norm_ED, preds, labels, infer_time, length_of_data = validation(
                # #     model, criterion, valid_loader, converter, opt)
                valid_loss, current_accuracy, current_norm_ED, preds, labels, infer_time, length_of_data = validation(
                    model, criterion, valid_loader, converter, opt)
                model.train()

                for pred, gt in zip(preds[:5], labels[:5]):
                    if 'Attn' in opt.Prediction:
                        pred = pred[:pred.find('[s]')]
                        gt = gt[:gt.find('[s]')]
                    print(f'pred: {pred:20s}, gt: {gt:20s},   {str(pred == gt)}')
                    log.write(f'pred: {pred:20s}, gt: {gt:20s},   {str(pred == gt)}\n')

                valid_log = f'[{i}/{opt.num_iter}] valid loss: {valid_loss:0.5f}'
                valid_log += f' accuracy: {current_accuracy:0.3f}, norm_ED: {current_norm_ED:0.2f}'
                print(valid_log)
                log.write(valid_log + '\n')

                # keep best accuracy model
                if current_accuracy > best_accuracy:
                    best_accuracy = current_accuracy
                    torch.save(model.state_dict(), f'./saved_models/{opt.experiment_name}/best_accuracy.pth')
                if current_norm_ED < best_norm_ED:
                    best_norm_ED = current_norm_ED
                    torch.save(model.state_dict(), f'./saved_models/{opt.experiment_name}/best_norm_ED.pth')
                best_model_log = f'best_accuracy: {best_accuracy:0.3f}, best_norm_ED: {best_norm_ED:0.2f}'
                print(best_model_log)
                log.write(best_model_log + '\n')

        # save model per 1e+5 iter.
        if (i + 1) % opt.saveInterval == 0:
            torch.save(
                model.state_dict(), f'./saved_models/{opt.experiment_name}/iter_{i+1}.pth')

        if i == opt.num_iter:
            print('end the training')
            sys.exit()
            
        if i > 0 and i % step_per_epoch == 0:                # 调整学习速率
            lrScheduler.step()
            
        i += 1
        # except:
        #     import sys, traceback
        #     traceback.print_exc(file=sys.stdout)
        #     continue


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', help='Where to store logs and models')
    parser.add_argument('--train_data', default='/home/deepblue/deepbluetwo/chenjun/1_OCR/data/data_lmdb_release/training', help='path to training dataset')
    parser.add_argument('--valid_data', default='/home/deepblue/deepbluetwo/chenjun/1_OCR/data/data_lmdb_release/validation', help='path to validation dataset')
    parser.add_argument('--manualSeed', type=int, default=666, help='for random seed setting')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
    parser.add_argument('--num_iter', type=int, default=300000, help='number of iterations to train for')
    parser.add_argument('--valInterval', type=int, default=500, help='Interval between each validation')
    parser.add_argument('--saveInterval', type=int, default=10000, help='Interval between each save')
    parser.add_argument('--disInterval', type=int, default=5, help='Interval betweet each show')
    # parser.add_argument('--continue_model', default = '', help="path to model to continue training")
    parser.add_argument('--continue_model', default='', help="path to model to continue training")
    parser.add_argument('--adam', default=True, help='Whether to use adam (default is Adadelta)')
    parser.add_argument('--ranger', default=False, help='use RAdam + Lookahead for optimizer')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=1.0 for Adadelta')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
    parser.add_argument('--rho', type=float, default=0.95, help='decay rate rho for Adadelta. default=0.95')
    parser.add_argument('--eps', type=float, default=1e-8, help='eps for Adadelta. default=1e-8')
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping value. default=5')

    """ all baidu images """
    # parser.add_argument('--root', type=str, default='/root/shenlan/deepblue/1_OCR/data/train_images', help='the path of images')
    # parser.add_argument('--train_csv', type=str, default='/root/shenlan/deepblue/1_OCR/text_reco/dataset/BAIDU/add_train_30w.txt', help='the train samples')
    # parser.add_argument('--val_csv', type=str, default='/root/shenlan/deepblue/1_OCR/text_reco/dataset/BAIDU/add_val.txt', help='the val samples')
    # parser.add_argument('--baidu_alphabet', type=str, default='/root/shenlan/deepblue/1_OCR/text_reco/dataset/BAIDU/baidu_alphabet_30w.txt')
    
    '''a small baidu image'''
    parser.add_argument('--root', type=str, default='./dataset/BAIDU/images/', help='the path of images')
    parser.add_argument('--train_csv', type=str, default='./dataset/BAIDU/small_train.txt', help='the train samples')
    parser.add_argument('--val_csv', type=str, default='./dataset/BAIDU/small_train.txt', help='the val samples')
    parser.add_argument('--baidu_alphabet', type=str, default='./dataset/BAIDU/baidu_alphabet.txt')

    
    parser.add_argument('--max_seq', type=int, default=26, help='the maxium of the sequence length')
    parser.add_argument('--position_dim', type=int, default=65, help='the length sequence out from cnn encoder,resnet:65,resnetfpn:256')
    parser.add_argument('--alphabet_size', type=int, default=None, help='the categry of the string')

    '''SRN setting'''
    parser.add_argument('--SRN_PAD', type=int, default=37, help='refer to EOS')
    parser.add_argument('--batch_max_character', type=int, default=25, help='the max character of one image')
    parser.add_argument('--n_position', type=int, default=256, help='the sequence length of cnn out feature')
    
    parser.add_argument('--select_data', type=str, default='ICDAR2019-ICDAR2019',
                        help='select training data MJ-ST | MJ-ST-ICDAR2019 | baidu')
    parser.add_argument('--batch_ratio', type=str, default='1.0-1.0',
                        help='assign ratio for each selected data in the batch')
    parser.add_argument('--total_data_usage_ratio', type=str, default='1.0',
                        help='total data usage ratio, this ratio is multiplied to total number of data.')
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    parser.add_argument('--imgH', type=int, default=64, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=256, help='the width of the input image')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz$', help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether tlabelo keep ratio then pad for image resize')
    parser.add_argument('--data_filtering_off', action='store_true', help='for data_filtering_off mode')
    """ Model Architecture """
    parser.add_argument('--Transformation', type=str, default='None', help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, default='ResNet', help='FeatureExtraction stage. VGG|RCNN|ResNet|AsterRes|ResnetFpn')
    parser.add_argument('--SequenceModeling', type=str, default='SRN', help='SequenceModeling stage. None|BiLSTM|Bert|SRN')
    parser.add_argument('--Prediction', type=str, default='SRN', help='Prediction stage. CTC|Attn|Bert_pred|SRN')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

    opt = parser.parse_args()

    if not opt.experiment_name:
        opt.experiment_name = f'{opt.Transformation}-{opt.FeatureExtraction}-{opt.SequenceModeling}-{opt.Prediction}'
        opt.experiment_name += f'-Seed{opt.manualSeed}'
        # print(opt.experiment_name)

    os.makedirs(f'./saved_models/{opt.experiment_name}', exist_ok=True)

    """ vocab / character number configuration """
    if opt.sensitive:
        # opt.character += 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

    if opt.select_data == 'baidu':
        with open(opt.baidu_alphabet) as f:
            opt.character = f.readlines()[0]
#         opt.character = opt.baidu_alphabet
    opt.alphabet_size = len(opt.character) + 2              # +2 for [UNK]+[EOS]

    '''SRN setting'''
    opt.SRN_PAD = len(opt.character) - 1


    """ Seed and GPU setting """
    # print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed(opt.manualSeed)

    cudnn.benchmark = True
    cudnn.deterministic = True
#     opt.num_gpu = torch.cuda.device_count()
    opt.num_gpu = 1
    # print('device count', opt.num_gpu)
    if opt.num_gpu > 1:
        print('------ Use multi-GPU setting ------')
        print('if you stuck too long time with multi-GPU setting, try to set --workers 0')
        # check multi-GPU issue https://github.com/clovaai/deep-text-recognition-benchmark/issues/1
        opt.workers = opt.workers * opt.num_gpu

        """ previous version
        print('To equlize batch stats to 1-GPU setting, the batch_size is multiplied with num_gpu and multiplied batch_size is ', opt.batch_size)
        opt.batch_size = opt.batch_size * opt.num_gpu
        print('To equalize the number of epochs to 1-GPU setting, num_iter is divided with num_gpu by default.')
        If you dont care about it, just commnet out these line.)
        opt.num_iter = int(opt.num_iter / opt.num_gpu)
        """

    train(opt)
