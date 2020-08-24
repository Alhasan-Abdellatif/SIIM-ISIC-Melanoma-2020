# SIIM-ISIC-Melanoma-2020

This repository provides a PyTorch implemenation of the classifcation problem in the Kaggle SIIM-ISIC Melanoma 2020 Competition (It can be modified for other problems though).

Using an ensemble of models produced by this code as well as other TensorFlow models, our team managed to be in 323th place out of 3314 participants and win a bronze medal in the competition.

The implementation uses a 5-fold cross-validation method. 

## Running
To train a model, specify configurations in config_file.py and :

```
python train.py  --save_folder config55   --dev_num 0
```
where save_folder will be the name of the created folders to save best checkpoints and predictions.

(Many updates should be introduced later).
