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
#        "path_to_test_x": "./predict_data/test_x",
#        "path_to_test_y": "./predict_data/test_y",
#        "data_path": "./data",
#        "filename": "sdwpf_baidukddcup2022_full.csv",
#        "task": "MS",
        "target": "Patv",
        "input_len2": 144,
        "output_len": 288,
        "start_col": 0,
        
        "var_len": 11,
        
        "in_var": 11,
        "out_var": 1,
        "day_len": 144,
        "train_days": 214,
        "val_days": 16,
        "test_days": 15, 
#        "total_days": 245,

#        "num_workers": 4,
#        "epoch": 10,
#       "batch_size": 32,
        "checkpoints": "./output",
#        "output_path": "./output/baseline",
#       "log_per_steps": 100,
#       "lr": 0.00005,
           
#       "patient": 2,

        "gpu": 0,
        "capacity": 134,
        "turbine_id": 0,
        "pred_file": "predict.py",
        "framework": "paddlepaddle",
        "is_debug": True,
        "model":
            {
        "hidden_dims": 512 ,
        "nhead": 8,
        "dropout": 0.4488,
        "encoder_layers": 2,
        "decoder_layers": 1
        },
        "loss":
           {"name": "HuberLoss"}
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
