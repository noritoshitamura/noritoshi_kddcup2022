# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
"""
Modified: Change to test_x of Toy and make dummy dataset for prediction.
          load mean.pdl,scale.pdl and graph.pdl for prediction.
         This module based on predict.py of PGL baseline model.
Author: Noritoshi Tamura (tamura@feg.co.jp)
Date:    2022/07/03

"""
import os
import paddle
import paddle.nn.functional as F
import tqdm
import yaml
import numpy as np
from easydict import EasyDict as edict

import pgl
from pgl.utils.logger import log
from paddle.io import DataLoader

from wpf_dataset import PGL4WPFDataset, TestPGL4WPFDataset, Test_dummyDataset
from wpf_model import WPFModel

from utils import load_model
import time
import random

@paddle.no_grad()
def forecast2(envs):#, train_data):  #, valid_data, test_data):
    #data_mean = paddle.to_tensor(train_data.data_mean, dtype="float32")
    #data_scale = paddle.to_tensor(train_data.data_scale, dtype="float32")

    ### Add Noritoshi Tamura
    fix_seed = 3456
    random.seed(fix_seed)
    paddle.seed(fix_seed)
    np.random.seed(fix_seed)
    ###
    
    sc_file_location = envs["checkpoints"]
    sc_file1 = os.path.join(sc_file_location,"mean.pdl")
    sc_file2 = os.path.join(sc_file_location,"scale.pdl")
    
    #paddle.save(data_mean, sc_file1)
    #paddle.save(data_scale, sc_file2)
    
    data_mean = paddle.load(sc_file1)
    data_scale = paddle.load(sc_file2)     

    sc_file3 = os.path.join(sc_file_location,"graph.pdl")

    #graph = train_data.graph

    #paddle.save(graph,sc_file3)
    graph = paddle.load(sc_file3)
    graph = graph.tensor()

    model = WPFModel(config=edict(envs))

    global_step = load_model(envs["checkpoints"], model)
    model.eval()

    #test_x = sorted(glob.glob(os.path.join("predict_data", "test_x", "*")))
    #test_y = sorted(glob.glob(os.path.join("predict_data", "test_y", "*")))

    test_x_ds = TestPGL4WPFDataset(filename=envs["path_to_test_x"])
    test_x = paddle.to_tensor(
        test_x_ds.get_data()[:, :, -envs["input_len2"]:, :], dtype="float32")

    test_y_ds = Test_dummyDataset(test_x[-1,-1,-1,1].item(),test_x[-1,-1,-1,0].item())

    test_y = paddle.to_tensor(
        test_y_ds.get_data()[:, :, :envs["output_len"], :], dtype="float32")    
    pred_y = model(test_x, test_y, data_mean, data_scale, graph)
    pred_y = F.relu(pred_y * data_scale[:, :, :, -1] + data_mean[:, :, :,-1])
    pred_y = np.transpose(pred_y,[1,2,0])
    paddle.device.cuda.empty_cache()
    del model,test_x_ds,test_x,test_y

    return np.array(pred_y)



