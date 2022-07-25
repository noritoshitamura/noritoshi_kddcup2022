# -*-Encoding: utf-8 -*-
################################################################################
#
# Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
Description: Prepare the experimental settings
"""
import paddle


def prep_env():
    # type: () -> dict
    """
    Desc:
        Prepare the experimental settings
    Returns:
        The initialized arguments
    """
    settings = {
        "path_to_test_x": "./data/sdwpf_baidukddcup2022_test_toy/test_x",
        "path_to_test_y": "./data/sdwpf_baidukddcup2022_test_toy/test_y",
        "data_path": "./data",
        "filename": "wtbdata_245days.csv",
        "task": "MS",
        "target": "Patv",
        "checkpoints": "./output",
        "input_len": 432,
        "input_len2": 144,
        "output_len": 288,
        "start_col": 3,
        "in_var": 11,
        "out_var": 1,
        "day_len": 144,
        "train_size": 214,
        "val_size": 31,
        "total_size": 245,
        "lstm_layer": 2,
        "dropout": 0.049,
        "num_workers": 5,
        "train_epochs": 20,
        "batch_size": 48,
        "patience": 3,
        "lr": 1e-4,
        "lr_adjust": "type1",
        "gpu": 0,
        "capacity": 134,
        "turbine_id": 0,
        "pred_file": "predict.py",
        "framework": "paddlepaddle",
        "is_debug": True,
        
        "var_len" : 11,
        "log_per_steps": 100,
        #"lr": 0.00005,
        "epoch": 10,
           
        "model":
            {
            "hidden_dims": 512 ,
            "nhead": 8,
            "dropout": 0.4488,
            "encoder_layers": 2,
            "decoder_layers": 1
        },
        "loss":
           {"name": "FilterMSELoss"}
    }
    
    ###
    # Prepare the GPUs
    if paddle.device.is_compiled_with_cuda():
        settings["use_gpu"] = True
        paddle.device.set_device('gpu:{}'.format(settings["gpu"]))
    else:
        settings["use_gpu"] = False
        paddle.device.set_device('cpu')

    print("The experimental settings are: \n{}".format(str(settings)))
    return settings
