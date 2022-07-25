# -*-Encoding: utf-8 -*-
################################################################################
#
# Copyright (c) 2022 Noritosh Tamura(tamura@feg.co.jp). All Rights Reserved
#
################################################################################
"""
Description: Preprocessing function for PGL Model.
Author: Noritoshi Tamura (tamura@feg.co.jp)
Date:    2022/07/03

"""
import numpy as np
import pandas as pd
import math


def preprocessing3(dat):
    dat2 = dat.copy()
    x_list = dat.columns.values

    for x in x_list[3:]:
        dat2[f"n_{x}"] = dat2[[f"{x}"]]
    
    def wind_dir(xx):
        xx0 = np.nan if ( math.isnan(xx[0])) else xx[0] % 360
        xx1 = np.nan if ( math.isnan(xx[1]))  else xx[1] % 360
        ret = np.nan if  math.isnan( xx0 + xx1) else xx0 + xx1
        return ret % 360
   
    dat2["wind_dir"] = dat2[["Wdir","Ndir"]].apply(wind_dir,axis=1)

    dat2.loc[dat2["n_Etmp"]>=80,"n_Etmp"] = 80
    dat2.loc[dat2["n_Itmp"]>=80,"n_Itmp"] = 80
    dat2.loc[dat2["n_Etmp"]< -10,"n_Etmp"] = -10
    dat2.loc[dat2["n_Itmp"]< -10,"n_Itmp"] = -10
    dat2.loc[dat2["Wspd"] >= 26.29,"Wspd"] = 26.29

    f = lambda x: math.cos(math.radians(x[0]))
    dat2["Area"] = dat2[["Wdir"]].apply(f,axis=1)
    dat2["Wp"] = dat2["Area"] * dat2["Wspd"]**3.0   
    dat2["Tdiff"] = dat2["n_Itmp"]-dat2["n_Etmp"]
    
            
    def unknown_judge(x):
        if (x[0]<= 0 and x[1]>2.5) or (x[2]>89 or x[3]>89 or x[4]>89 or x[5] < -180 or x[5] > 180 or x[6] < -720 or x[6] > 720):
            return 0
        else:
            return 1                       

    dat2["onoff"] = dat2[["Patv","Wspd","Pab1","Pab2","Pab3","Wdir","Ndir"]].apply(unknown_judge,axis=1)
               
    return   dat2[["weekday","time",\
             "Wspd",\
             "Ndir",\
             "Pab1","Pab2","Pab3","Wdir",\
             "wind_dir","Etmp","Tdiff","Prtv","Patv"]]
