# -*- coding: UTF-8 -*-
#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle.fluid as fluid
from paddle.fluid import layers
from paddlepalm.head.base_head import Head
import numpy as np
import os
import json


class COS(Head):
    """
    classification
    """
    # def __init__(self, config, phase, backbone_config=None):
    def __init__(self, num_classes, input_dim, dropout_prob=0.0, \
                 param_initializer_range=0.02, phase='train'):

        self._is_training = phase == 'train'
        self._hidden_size = input_dim

        self.num_classes = num_classes
    
        self._dropout_prob = dropout_prob if phase == 'train' else 0.0
        self._param_initializer = fluid.initializer.TruncatedNormal(
            scale=param_initializer_range)
        self._cos_sim = []

    @property
    def inputs_attrs(self):
        reader = {}
        bb = {"sentence_embedding": [[-1, self._hidden_size], 'float32'],
        "sentence_embedding_tb": [[-1, self._hidden_size], 'float32']}
        return {'reader': reader, 'backbone': bb}

    @property
    def outputs_attrs(self):
        if self._is_training:
            return {'loss': [[1], 'float32']}
        else:
            return {'cos_sim': [[-1, self.num_classes], 'float32']}
            

    def build(self, inputs, scope_name=''):
        sent_emb = inputs['backbone']['sentence_embedding']
        sent_emb_tb = inputs['backbone']['sentence_embedding_tb']
        # if self._is_training:
        #     label_ids = inputs['reader']['label_ids']
        #     cls_feats = fluid.layers.dropout(
        #         x=sent_emb,
        #         dropout_prob=self._dropout_prob,
        #         dropout_implementation="upscale_in_train")

        # logits = fluid.layers.fc(
        #     input=sent_emb,
        #     size=self.num_classes,
        #     param_attr=fluid.ParamAttr(
        #         name=scope_name+"cls_out_w",
        #         initializer=self._param_initializer),
        #     bias_attr=fluid.ParamAttr(
        #         name=scope_name+"cls_out_b", initializer=fluid.initializer.Constant(0.)))
        cos_sim = fluid.layers.cos_sim(sent_emb, sent_emb_tb)
        # if self._is_training:
        #     loss = fluid.layers.cross_entropy(
        #         input=cos_sim, label=label_ids)
        #     loss = layers.mean(loss)
        #     return {"loss": loss}
        # else:
        return {"cos_sim":cos_sim}

    def batch_postprocess(self, rt_outputs):
        if not self._is_training:
            # logits = rt_outputs['logits']
            cos_sim = rt_outputs['cos_sim']
            # self._preds.extend(logits.tolist())
            self._cos_sim.extend(cos_sim.tolist())


    def epoch_postprocess(self, post_inputs, output_dir=None):
        # there is no post_inputs needed and not declared in epoch_inputs_attrs, hence no elements exist in post_inputs
        if not self._is_training:
            if output_dir is None:
                raise ValueError('argument output_dir not found in config. Please add it into config dict/file.')
            with open(os.path.join(output_dir, 'predictions.json'), 'w') as writer:
                for i in range(len(self._cos_sim)):
                    label = 0 if self._cos_sim[i][0] < 0 else 1
                    result = {'index': i, 'label': label, 'cos_sim': self._cos_sim[i]}
                    result = json.dumps(result)
                    writer.write(result+'\n')
            print('Predictions saved at '+os.path.join(output_dir, 'predictions.json'))

                
