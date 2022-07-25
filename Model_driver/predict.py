# -*-Encoding: utf-8 -*-
################################################################################
#
# Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
Description: A demo of the forecasting method
Authors: Lu,Xinjiang (luxinjiang@baidu.com)
Date:    2022/04/18
"""
import os
import time
import numpy as np
import paddle
from model import BaselineGruModel
from common import Experiment
from wind_turbine_data import WindTurbineData
from test_data import TestData
from preprocessing2 import preprocessing2
from predict_pgl import forecast2
import pickle 


def forecast_one(experiment, test_turbines, train_data):
    # type: (Experiment, TestData, WindTurbineData) -> np.ndarray
    """
    Desc:
        Forecasting the power of one turbine
    Args:
        experiment:
        test_turbines:
        train_data:
    Returns:
        Prediction for one turbine
    """
    args = experiment.get_args()
    tid = args["turbine_id"]
    model = BaselineGruModel(args)
    model_dir = '{}_t{}_i{}_o{}_ls{}_train{}_val{}'.format(
        args["filename"], args["task"], args["input_len"], args["output_len"], args["lstm_layer"],
        args["train_size"], args["val_size"]
    )
    path_to_model = os.path.join(args["checkpoints"], model_dir, "model_{}".format(str(tid)))
    model.set_state_dict(paddle.load(path_to_model))

    test_x_, _ = test_turbines.get_turbine(tid)
    test_x_2 = preprocessing2(test_x_)
    test_x_2b = test_x_2[["Ndir","Itmp","Area","Wp","Prtv","Tdiff","onoff","Pab1","Pab2","Pab3","Patv"]]
    test_x_2b.replace(to_replace=np.nan, value=0, inplace=True)
    scaler = train_data.get_scaler(tid)
    #print(test_x_2b.shape)
    test_x = scaler.transform(test_x_2b.values)

    last_observ = test_x[-args["input_len"]:]
    seq_x = paddle.to_tensor(last_observ)
    sample_x = paddle.reshape(seq_x, [-1, seq_x.shape[-2], seq_x.shape[-1]])
    prediction = experiment.inference_one_sample(model, sample_x)
    #print(prediction.shape)
    prediction = scaler.inverse_transform2(prediction, args["in_var"])
    prediction = prediction[0]
    return prediction.numpy()


def forecast(settings):
    # type: (dict) -> np.ndarray
    """
    Desc:
        Forecasting the wind power in a naive distributed manner
    Args:
        settings:
    Returns:
        The predictions
    """
    start_time = time.time()
    predictions = []
    settings["turbine_id"] = 0
    exp = Experiment(settings)
    #train_data = Experiment.train_data
    train_data = exp.load_train_data(True)
#    collection = []
#    for i in range(134):
 #       collection.append(train_data.get_scaler2(i))
        
#    with open('wind.pkl', 'wb') as f1: 
#        pickle.dump(collection, f1)
    sc_file_location = settings["checkpoints"]
    sc_file = os.path.join(sc_file_location,"wind.pkl")
    with open(sc_file, 'rb') as f1: 
        collection = pickle.load(f1)  
         
    train_data.set_scaler( collection )          
    
    if settings["is_debug"]:
        end_train_data_get_time = time.time()
        print("Load train data in {} secs".format(end_train_data_get_time - start_time))
        start_time = end_train_data_get_time
    test_x = Experiment.get_test_x(settings)
    if settings["is_debug"]:
        end_test_x_get_time = time.time()
        print("Get test x in {} secs".format(end_test_x_get_time - start_time))
        start_time = end_test_x_get_time
    for i in range(settings["capacity"]):
        settings["turbine_id"] = i
        # print('\n>>>>>>> Testing Turbine {:3d} >>>>>>>>>>>>>>>>>>>>>>>>>>\n'.format(i))
        prediction = forecast_one(exp, test_x, train_data)
        paddle.device.cuda.empty_cache()
        predictions.append(prediction)
        if settings["is_debug"] and (i + 1) % 10 == 0:
            end_time = time.time()
            print("\nElapsed time for predicting 10 turbines is {} secs".format(end_time - start_time))
            start_time = end_time
    predictions = np.array(predictions)        
    prediction2 = forecast2(settings)
    comb = np.vstack((predictions.reshape(1,134,288),prediction2.reshape(1,134,288)))
    comb2 = np.mean(comb ,axis=0)
    comb2 = comb2.reshape(134,288,1)
    return np.array(comb2)
