# -*-Encoding: utf-8 -*-
################################################################################
#
# Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
Description: Wind turbine dataset utilities
Authors: Lu,Xinjiang (luxinjiang@baidu.com)
Date:    2022/03/10

Modified: Add preprocessing and change normalization.
Author: Noritoshi Tamura (tamura@feg.co.jp)
Date:    2022/07/03

"""
import os
import numpy as np
import pandas as pd
import paddle
from paddle.io import Dataset
from preprocessing import preprocessing


# Turn off the SettingWithCopyWarning
pd.set_option('mode.chained_assignment', None)


class Scaler(object):
    """
    Desc: Normalization utilities
          Modifyed by Noritoshi Tamura
          Change indivisual mean and std of features from solo scalers of whole mean and std. 
    
    """
    def __init__(self):
        self.mean = 0.
        self.std = 1.

    def fit(self, data):
        # type: (paddle.tensor) -> None
        """
        Desc:
            Fit the data
        Args:
            data:
        Returns:
            None
        """
        print(data.shape)
        self.mean = np.mean(data,axis=0) # Modified by Noritoshi Tamura
        self.std = np.std(data,axis=0)   # Modified by Noritoshi Tamura
        print(self.mean)

    def transform(self, data):
        # type: (paddle.tensor) -> paddle.tensor
        """
        Desc:
            Transform the data
        Args:
            data:
        Returns:
            The transformed data
        """
        mean = paddle.to_tensor(self.mean).type_as(data).to(data.device) if paddle.is_tensor(data) else self.mean
        std = paddle.to_tensor(self.std).type_as(data).to(data.device) if paddle.is_tensor(data) else self.std
        return (data - mean) / ( std + 1e-15)

    def inverse_transform(self, data):
        # type: (paddle.tensor) -> paddle.tensor
        """
        Desc:
            Restore to the original data
            Not use.
        Args:
            data: the transformed data
        Returns:
            The original data
        """
        mean = paddle.to_tensor(self.mean) if paddle.is_tensor(data) else self.mean
        std = paddle.to_tensor(self.std) if paddle.is_tensor(data) else self.std
        return (data * std) + mean

    def inverse_transform2(self, data,nv):
        # type: (paddle.tensor) -> paddle.tensor
        """
        Desc:
            Restore to the original data for "Patv" handling. 
            Added by Noritoshi Tamura
        Args:
            data: the transformed data
            nv: in_var for pointing position for "Patv" ( minus one means last feature ) 
        Returns:
            The original data
        """
        mean = paddle.to_tensor(self.mean[nv-1]) if paddle.is_tensor(data) else self.mean
        std = paddle.to_tensor(self.std[nv-1]) if paddle.is_tensor(data) else self.std
        return (data * std) + mean

class WindTurbineData(Dataset):
    """
    Desc: Wind turbine power generation data
          Here, e.g.    15 days for training,
                        3 days for validation
                       
    Modified by Noritoshi Tamura
    
          1. Add preprocessing and reading raw data and preprocessing run  at first time 
             then keep them to 134'th turbine model building.
          2. Change Scaling factors for normarization as Tensor.      
          
    """
    first_initialized = False
    scaler_collection = None
    df_raw = None

    def __init__(self, data_path,
                 filename='sdwpf_baidukddcup2022_full.csv',
                 flag='train',
                 size=None,
                 turbine_id=0,
                 task='MS',
                 target='Patv',
                 scale=True,
                 start_col=3,       # the start column index of the data one aims to utilize
                 day_len=24 * 6,
                 train_days=15,     # 15 days
                 val_days=3,        # 3 days
                 total_days=30,     # 30 days
                 farm_capacity=134,
                 is_test=False
                 ):
        super().__init__()
        self.unit_size = day_len
        if size is None:
            self.input_len = self.unit_size
            self.output_len = self.unit_size
        else:
            self.input_len = size[0]
            self.output_len = size[1]
        # initialization
        assert flag in ['train', 'val']
        type_map = {'train': 0, 'val': 1}
        self.set_type = type_map[flag]
        self.task = task
        self.target = target
        self.scale = scale
        self.start_col = start_col
        self.data_path = data_path
        self.filename = filename
        self.tid = turbine_id
        self.farm_capacity = farm_capacity
        self.is_test = is_test
        # If needed, we employ the predefined total_size (e.g. one month)
        self.total_size = self.unit_size * total_days
        #
        self.train_size = train_days * self.unit_size
        self.val_size = val_days * self.unit_size
        #
        if  not is_test:
            if not WindTurbineData.first_initialized: # work only once at first time.
                self.__read_data__()
                WindTurbineData.first_initialized = True
 
        if WindTurbineData.scaler_collection is None: # Work only once at first time.
            WindTurbineData.scaler_collection = []

            for i in range(self.farm_capacity):
                WindTurbineData.scaler_collection.append(None)
        if not is_test:
            self.data_x, self.data_y = self.__get_data__(self.tid)
            
    def __read_data__(self):
        df_raw_ = pd.read_csv(os.path.join(self.data_path, self.filename))
        WindTurbineData.df_raw = preprocessing(df_raw_)
        WindTurbineData.df_raw.replace(to_replace=np.nan, value=0, inplace=True)

    def __get_turbine__(self, turbine_id):
        border1s = [turbine_id * self.total_size,
                    turbine_id * self.total_size + self.train_size - self.input_len
                    ]
        border2s = [turbine_id * self.total_size + self.train_size,
                    turbine_id * self.total_size + self.train_size + self.val_size
                    ]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.task == 'MS':
            cols = WindTurbineData.df_raw.columns[self.start_col:]
            df_data_ = WindTurbineData.df_raw[cols]
            df_data = df_data_[["Ndir","Itmp","Area","Wp","Prtv","Tdiff","onoff","Pab1","Pab2","Pab3","Patv"]] #Feature selection
        elif self.task == 'S':
            df_data = WindTurbineData.df_raw[[turbine_id, self.target]]
        else:
            raise Exception("Unsupported task type ({})! ".format(self.task))
        train_data = df_data[border1s[0]:border2s[0]]
        if WindTurbineData.scaler_collection[turbine_id] is None:
            scaler = Scaler()
            scaler.fit(train_data.values)
            WindTurbineData.scaler_collection[turbine_id] = scaler
        self.scaler = WindTurbineData.scaler_collection[turbine_id]
        res_data = df_data[border1:border2]
        if self.scale:
            res_data = self.scaler.transform(res_data.values)
        else:
            res_data = res_data.values
        return res_data

    def __get_data__(self, turbine_id):
        data_x = self.__get_turbine__(turbine_id)
        data_y = data_x
        return data_x, data_y

    def get_scaler(self, turbine_id): 
        return WindTurbineData.scaler_collection[turbine_id]

    def get_scaler2(self, turbine_id):
        if WindTurbineData.scaler_collection[turbine_id] is None:
            self.__get_turbine__(turbine_id)
        return WindTurbineData.scaler_collection[turbine_id]

    def __getitem__(self, index):
        #
        # Rolling window with the size of input_len + output_len
        s_begin = index
        s_end = s_begin + self.input_len
        r_begin = s_end
        r_end = r_begin + self.output_len
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        return seq_x, seq_y

    def __len__(self):
        # In our case, the rolling window is adopted, the number of samples is calculated as follows
        if self.set_type < 2:
            return len(self.data_x) - self.input_len - self.output_len + 1
        # Otherwise,
        return int((len(self.data_x) - self.input_len) / self.output_len)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
        
    def set_scaler(self,sc): # for loading scaling factors from pickle.
        WindTurbineData.scaler_collection = sc
        
