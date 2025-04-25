# SIIM-ISIC-Melanoma-2020

This repository provides a PyTorch implemenation of the classifcation problem in the Kaggle SIIM-ISIC Melanoma 2020 Competition, https://www.kaggle.com/competitions/siim-isic-melanoma-classification/overview, (It can be modified for other problems though).
![download](https://github.com/user-attachments/assets/02bc5e95-f4e1-431b-9d55-6267956fbac4)

Using an ensemble of models produced by this code as well as other TensorFlow models, our team managed to be in top 10% and win a bronze medal in the competition.

The implementation uses a 5-fold cross-validation method. 

## Running
To train a model, specify configurations in config_file.py and run :

```
python train.py  --save_folder config55   --dev_num 0
```
where save_folder will be the name of the created folders to save best checkpoints and predictions.

(Some updates might be introduced later).
