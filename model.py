"""
Copyright (c) 2019-present NAVER Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch.nn as nn

from modules.transformation import TPS_SpatialTransformerNetwork
from modules.feature_extraction import VGG_FeatureExtractor, RCNN_FeatureExtractor, ResNet_FeatureExtractor
from modules.sequence_modeling import BidirectionalLSTM
from modules.prediction import Attention
from modules.resnet_aster import ResNet_ASTER, ResNet_ASTER2

from modules.bert import Bert_Ocr
from modules.bert import Config

from modules.SRN_modules import Transforme_Encoder, SRN_Decoder
from modules.resnet_fpn import ResNet_FPN


class Model(nn.Module):

    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt
        self.stages = {'Trans': opt.Transformation, 'Feat': opt.FeatureExtraction,
                       'Seq': opt.SequenceModeling, 'Pred': opt.Prediction}

        """ Transformation """
        if opt.Transformation == 'TPS':
            self.Transformation = TPS_SpatialTransformerNetwork(
                F=opt.num_fiducial, I_size=(opt.imgH, opt.imgW), I_r_size=(opt.imgH, opt.imgW), I_channel_num=opt.input_channel)
        else:
            print('No Transformation module specified')

        """ FeatureExtraction """
        if opt.FeatureExtraction == 'VGG':
            self.FeatureExtraction = VGG_FeatureExtractor(opt.input_channel, opt.output_channel)
        elif opt.FeatureExtraction == 'RCNN':
            self.FeatureExtraction = RCNN_FeatureExtractor(opt.input_channel, opt.output_channel)
        elif opt.FeatureExtraction == 'ResNet':
            self.FeatureExtraction = ResNet_FeatureExtractor(opt.input_channel, opt.output_channel)
            self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1
        elif opt.FeatureExtraction == 'AsterRes':
            self.FeatureExtraction = ResNet_ASTER2(opt.input_channel, opt.output_channel)
        elif opt.FeatureExtraction == 'ResnetFpn':
            self.FeatureExtraction = ResNet_FPN()
        else:
            raise Exception('No FeatureExtraction module specified')
        self.FeatureExtraction_output = opt.output_channel  # int(imgH/16-1) * 512
        

        """ Sequence modeling"""
        if opt.SequenceModeling == 'BiLSTM':
            self.SequenceModeling = nn.Sequential(
                BidirectionalLSTM(self.FeatureExtraction_output, opt.hidden_size, opt.hidden_size),
                BidirectionalLSTM(opt.hidden_size, opt.hidden_size, opt.hidden_size))
            self.SequenceModeling_output = opt.hidden_size
        elif opt.SequenceModeling == 'Bert':
            cfg = Config()
            cfg.dim = opt.output_channel; cfg.dim_c = opt.output_channel              # 降维减少计算量
            cfg.p_dim = opt.position_dim                        # 一张图片cnn编码之后的特征序列长度
            cfg.max_vocab_size = opt.batch_max_length + 1                # 一张图片中最多的文字个数, +1 for EOS
            cfg.len_alphabet = opt.alphabet_size                # 文字的类别个数
            self.SequenceModeling = Bert_Ocr(cfg)
        elif opt.SequenceModeling == 'SRN':
            self.SequenceModeling = Transforme_Encoder(n_layers=2, n_position=opt.position_dim)
            self.SequenceModeling_output = 512
        else:
            print('No SequenceModeling module specified')
            self.SequenceModeling_output = self.FeatureExtraction_output

        """ Prediction """
        if opt.Prediction == 'CTC':
            self.Prediction = nn.Linear(self.SequenceModeling_output, opt.num_class)
        elif opt.Prediction == 'Attn':
            self.Prediction = Attention(self.SequenceModeling_output, opt.hidden_size, opt.num_class)
        elif opt.Prediction == 'Bert_pred':
            pass
        elif opt.Prediction == 'SRN':
            self.Prediction = SRN_Decoder(n_position=opt.position_dim, n_class=opt.alphabet_size)
        else:
            raise Exception('Prediction is neither CTC or Attn')

    def forward(self, input, text, is_train=True):
        """ Transformation stage """
        if not self.stages['Trans'] == "None":
            input = self.Transformation(input)


        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(input)
        if self.stages['Feat'] == 'AsterRes' or self.stages['Feat'] == 'ResnetFpn':
            b, c, h, w = visual_feature.shape
            visual_feature = visual_feature.view(b, c, -1)
            visual_feature = visual_feature.permute(0, 2, 1)    # batch, seq, feature
        else:
            visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
            visual_feature = visual_feature.squeeze(3)


        """ Sequence modeling stage """
        if self.stages['Seq'] == 'BiLSTM':
            contextual_feature = self.SequenceModeling(visual_feature)
        elif self.stages['Seq'] == 'Bert':
            pad_mask = text
            contextual_feature = self.SequenceModeling(visual_feature, pad_mask)
        elif self.stages['Seq'] == 'SRN':
            contextual_feature = self.SequenceModeling(visual_feature, src_mask=None)[0]
        else:
            contextual_feature = visual_feature  # for convenience. this is NOT contextually modeled by BiLSTM


        """ Prediction stage """
        if self.stages['Pred'] == 'CTC':
            prediction = self.Prediction(contextual_feature.contiguous())
        elif self.stages['Pred'] == 'Bert_pred':
            prediction = contextual_feature
        elif self.stages['Pred'] == 'SRN':
            prediction = self.Prediction(contextual_feature)
        else:
            prediction = self.Prediction(contextual_feature.contiguous(), text, is_train, batch_max_length=self.opt.batch_max_length)
        
        return prediction
