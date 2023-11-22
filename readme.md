# config introduction

| name                             | function                                        | need to be reset recording to your environment | note                                                                                                                                    |
| -------------------------------- | ----------------------------------------------- | ---------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| **Random parameter setting**     |
| seed                             | Random seed for dataset shuffle                 |
| **distributed training setting** |
| gpus                             | How many GPU you use in distributed training    |
| world_size                       | The number of processes in distributed training |
| backend                          | The platform used for process communication     |
| init_method                      | Ip and port of your server                      | $\surd$                                        |
| syncbn                           | Switch for synchronized batch normalization     |
| **transform setting**            |
| face                             | Type of face corping                            |
| size                             | Size of corpped face                            |
| **training parameter setting**   |
| debug                            | Switch of debug mode                            |
| logint                           | Interval of logging                             |
| modelperiod                      | Interval of saving model                        |
| valint                           | Interval of validation                          |
| batch                            | The size of training batch size                 | $\surd$                                        |
| epochs                           | The maximum epoch                               | $\surd$                                        |
| net_s                            | Net architecture for student model              |
| net_t                            | Net architecture for teacher model              |
| traindb                          | Decide quality and training set segmentation    |                                                | ["ff-c23-720-140-140"] raw/c23/c40:chosen quality <br>  720:training set size<br> 140:validation set size<br> 140: testing set size<br> |
| trainIndex                       | Choose temper type                              |                                                | 0:Deepfake<br> 1:Face2Face<br> 2:Face Swap<br> 3:Neural Texture                                                                         |
| tagnote                          | Note part for tag                               |
| **dataset setting**              |
| ffpp_faces_df_path               | The path of dataset dataframe                   | $\surd$                                        |
| ffpp_faces_dir                   | The path of dataset                             | $\surd$                                        |
| workers                          | The process number for dataset loading          |
| **optimizer setting**            |
| lr                               | Training  learning rate                         | $\surd$                                        |
| patience                         | Adam parameter                                  |
| **model loading setting**        |
| models_dir                       | The path of models repository                   | $\surd$                                        |
| mode                             | Model loading type                              | $\surd$                                        | 0:choose the best trained model <br> 1: choose the last one<br> 2:choose the one of specific iteration <br>                             |
| index                            | The model of a specific iteration               |
| **log setting**                  |
| log_dir                          | The path of logs repository                     | $\surd$                                        |

---
---
# Preprocess instruction
* use **index_dataset.py** generating the dataframe of corresponding dataset
* use **extract_faces.py** transform video into frames and corresponding dataframe
---
---
# Running instruction 
* setting the **config.py** in folder **config**
* train your teacher using **train_starter.py**
* train your expanded model **using train_starter.py**
* trained model for domain adaptation using **train_starter.py** 
