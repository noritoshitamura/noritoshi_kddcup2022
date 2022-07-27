Manual on Building Models and using      (2022.07.25)                                                                              

  Author: Noritosh Tamura (tamura@feg.co.jp)
          Financial Engineering Group, Inc.
                                                                                              

 Codes in "Model_driver" contains for final prediction.If you provide test_x.zip and test_y.zip in ./ ,using evaluation.py
in the holder ,you can evaluate final prediction locally.  
To use code for building Model and useing them need to following instruction.

I. Data preparation
  All provied data has deleted in shared holders and you provide them first. 
 
 1. Put wtbdata_245days.csv to model holders both
   - Destination   
     ./LSTM_MODEL/data
     ./PGL_MODEL/data 
 2. Put files in text_x and test_y of sdwpf_baidukddcup2022_test_toy 
   - Destination
     ./LSTM_MODEL/data/sdwpf_baidukddcup2022_test_toy 
     ./PGL_MODEL/data/predict_data 
 3. Put test_x.zip and test_y.zip in ./ ,  to run evaluation.py in Model_driver holder. 

II. Models build
   Model building codes have to run individually.
   
  1. LSTM model
     Do commands like,
     > cd LSTM_MODEL
     > python train.py
     > python evaluation.py ( To make wind.pkl and check results)
      
  2. PGL model(=STGT Model in technical report)  
     Do commands like,
     > cd PGL_MODEL
     > python main.py
     > python predict2.py ( To make graph.pdl,mean.pdl and scale.pdl  and check results)
     
III. Gather Models in ./Model_driver/output holder
     copy files in ./PGL_MODEL/output, directory in ./LSTM_MODEL/checkpoints and wind.pkl together 
     to /Model_driver/output holder, same as following location.
     
     
               | --- ./output
               | --- graph.pdl
               | --- mean.pdl
               | --- scale.pdl
               | --- wind.pkl
               | --- ./wtbdata_245days.csv_tMS_i432_o288_ls2_train214_val31
                     | --- model_0
                      .
                      .
                      .
                     | --- model_133
               | --- ./model_1
                     | --- ckpt.pdparams
                     | --- step
                     
     Now, you can run evaluation.py for final prediction( the average of two predictions of Models .
      Do commands like,
      > cd Model_driver 
      > python eavluation.py
       
Directory structure 

  * () mean file deleted from shared holder. 
  
./noritoshi_kddcup2022
 | --- (test_x.zip)
 | --- (test_y.zip)
 | --- readme.txt                                                           This file
 | --- ./LSTM_MODEL
        | --- __init__.py 
        | --- common.py
        | --- evaluation.py
        | --- metrics.py
        | --- model.py
        | --- predict.py
        | --- prepare.py
        | --- preprocessing.py                                              Preprocessing code for train
        | --- preprocessing2.py                                             Preprocessing code for evaluation
        | --- test_data.py
        | --- train.py
        | --- wind.pkl                                                      Saved factors for normalize features 
        | --- wind_turbine_data.py         
        | --- ./checkpoints
              | --- ./wtbdata_245days.csv_tMS_i432_o288_ls2_train214_val31
                     | --- model_0
                      .
                      .
                      .
                     | --- model_133
        | --- ./data        
              | ---  (wtbdata_245days.csv)   
              | --- ./sdwpf_baidukddcup2022_test_toy   
                     | --- ./test_x
                            | --- (0001in.csv)
                     | --- ./test_y
                            | --- (0001out.csv)

 | --- ./PGL_MODEL
        | --- __init__.py
        | --- common.py
        | --- config.yaml
        | --- evaluation.py
        | --- loss.py
        | --- main.py
        | --- metrics.py                                                    For train
        | --- metrics2.py                                                   For local evaluation with evaluation.py
        | --- optimization.py
        | --- predict.py                                                    Prediction code
        | --- predict2.py                                                   Local evaluation 
        | --- prepare.py
        | --- preprocessing.py                                              Preprocessing code for PGL_MODEL
        | --- test_vis.png
        | --- utils.py
        | --- val_vis.png
        | --- wind_turbine_data.py
        | --- wpf_dataset.py
        | --- wpf_model.py
        | --- ./output
               | --- graph.pdl
               | --- mean.pdl
               | --- scale.pdl
               | --- ./model_0
                     | --- ckpt.pdparams
                     | --- step
               | --- ./model_1
                     | --- ckpt.pdparams
                     | --- step      
        | --- ./data
               | --- (wtbdata_245days.csv)
        | --- ./predict_data
               | --- ./test_x
                      | --- (0001in.csv)
               | --- ./test_y
                      | --- (0001out.csv)               
  | --- ./Model_driver ( = submited code )
        | --- __init__.py 
        | --- common.py
        | --- evaluation.py
        | --- metrics.py
        | --- model.py
        | --- predict.py                                                        Prediction LSTM and take mean with PGL_Model
        | --- predict_pgl.py                                                    Predction code of PGL_MODEL part
        | --- prepare.py
        | --- preprocessing.py
        | --- preprocessing2.py
        | --- preprocessing3.py                                                 Preprocessing code for PGL_MODEL
        | --- test_data.py
        | --- train.py
        | --- utils.py
        | --- wind_turbine_data.py
        | --- wpf_dataset.py
        | --- wpf_model.py
        | --- ./output
               | --- graph.pdl
               | --- mean.pdl
               | --- scale.pdl
               | --- wind.pkl
               | --- ./wtbdata_245days.csv_tMS_i432_o288_ls2_train214_val31
                     | --- model_0
                      .
                      .
                      .
                     | --- model_133
               | --- ./model_1
                     | --- ckpt.pdparams
                     | --- step

