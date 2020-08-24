import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import torchtoolbox.transform as transforms
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, roc_auc_score,precision_score,confusion_matrix,f1_score
from sklearn.model_selection import StratifiedKFold, GroupKFold,KFold
from catalyst.data.sampler import BalanceClassSampler
import pandas as pd
import numpy as np
import gc
import os
import cv2
import time
import datetime
import warnings
import random
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from utils import *
from models import *
import importlib
import config_file
import sys
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
config = {}

print('training')

parser = argparse.ArgumentParser()

parser.add_argument('--dev_num', type=int, default=0
                        ,help = 'the index of a gpu to be used')
parser.add_argument('--save_folder', type=str, default=None
                        ,help = 'folder to save things')
parser.add_argument('--ngpu', type=int, default=1)
args = parser.parse_args()


device = torch.device("cuda:"+str(args.dev_num) if torch.cuda.is_available() else "cpu")
print(device)

# Import config
#cfg = importlib.import_module(sys.argv[1])
config.update(config_file.config)

#if config.get('data_folder',None) is not None:


#if len(sys.argv) >1:
#    config['data_folder'] = sys.argv[1]
#else:
#    config['data_folder'] = 'data/jpeg'

if args.save_folder is not None:
    config['save_folder'] = 'results/'+args.save_folder
#else:
#    config['save_folder'] = ''

if not os.path.exists(config['save_folder']):
    os.makedirs(config['save_folder'])
    
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#train_df = pd.read_csv('data/train.csv')
train_df = pd.read_csv(config['train_meta_path'])
test_df = pd.read_csv(os.path.join(config['data_folder'],'test.csv'))


#load external data
if config.get('external',False):
    train_ext_df = pd.read_csv(config['external_meta_path'])
    if config.get('external_2018',False):
        train_ext_df = train_ext_df[train_ext_df['tfrecord']%2==0].reset_index(drop=True)
    train_ext_df.replace('anterior torso','torso',inplace = True)
    train_ext_df.replace('posterior torso','torso',inplace = True)
    train_ext_df.replace('lateral torso','torso',inplace = True)
#else:
#    train_ext_df = pd.DataFrame({})

# preprocessing of meta data

# One-hot encoding of anatom_site_general_challenge feature
if config.get('external',False):
    concat = pd.concat([train_df['anatom_site_general_challenge'],
     test_df['anatom_site_general_challenge'],train_ext_df['anatom_site_general_challenge']], ignore_index=True)
    dummies = pd.get_dummies(concat, dummy_na=True, dtype=np.uint8, prefix='site')
    train_df = pd.concat([train_df, dummies.iloc[:train_df.shape[0]]], axis=1)
    test_df = pd.concat([test_df, dummies.iloc[train_df.shape[0]:test_df.shape[0]+train_df.shape[0]].reset_index(drop=True)], axis=1)
    train_ext_df = pd.concat([train_ext_df, dummies.iloc[test_df.shape[0]+train_df.shape[0]:].reset_index(drop=True)], axis=1)
    train_ext_df = train_ext_df.drop(['site_nan'],axis = 1)
else:
    concat = pd.concat([train_df['anatom_site_general_challenge'], test_df['anatom_site_general_challenge']], ignore_index=True)
    dummies = pd.get_dummies(concat, dummy_na=True, dtype=np.uint8, prefix='site')
    train_df = pd.concat([train_df, dummies.iloc[:train_df.shape[0]]], axis=1)
    test_df = pd.concat([test_df, dummies.iloc[train_df.shape[0]:].reset_index(drop=True)], axis=1)

train_df = train_df.drop(['site_nan'],axis = 1)
test_df = test_df.drop(['site_nan'],axis = 1)


# Sex features
if config['sex_ohe']:
    if config.get('external',False):
        concat = pd.concat([train_df['sex'], test_df['sex'],train_ext_df['sex']], ignore_index=True)
        dummies = pd.get_dummies(concat, dummy_na=True, dtype=np.uint8, prefix='sex')
        dummies = dummies.drop(['sex_nan'],axis = 1)
        dummies = dummies.drop(['sex_unknown'],axis = 1)
        train_df = pd.concat([train_df, dummies.iloc[:train_df.shape[0]]], axis=1)
        test_df = pd.concat([test_df, dummies.iloc[train_df.shape[0]:test_df.shape[0]+train_df.shape[0]].reset_index(drop=True)], axis=1)
        train_ext_df = pd.concat([train_ext_df, dummies.iloc[test_df.shape[0]+train_df.shape[0]:].reset_index(drop=True)], axis=1)
    else:
        concat = pd.concat([train_df['sex'], test_df['sex']], ignore_index=True)
        dummies = pd.get_dummies(concat, dummy_na=True, dtype=np.uint8, prefix='sex')
        train_df = pd.concat([train_df, dummies.iloc[:train_df.shape[0]]], axis=1)
        test_df = pd.concat([test_df, dummies.iloc[train_df.shape[0]:].reset_index(drop=True)], axis=1)
        train_df = train_df.drop(['sex_nan'],axis = 1)
        test_df = test_df.drop(['sex_nan'],axis = 1)
else:
    train_df['sex_num'] = train_df['sex']
    test_df['sex_num'] = test_df['sex']
    train_df['sex_num'] = train_df['sex_num'].map({'male': 1, 'female': 0})
    test_df['sex_num'] = test_df['sex_num'].map({'male': 1, 'female': 0})
    train_df['sex_num'] = train_df['sex_num'].fillna(-1)
    test_df['sex_num'] = test_df['sex_num'].fillna(-1)
    if config.get('external',False):
        train_ext_df['sex_num'] = train_ext_df['sex']
        train_ext_df['sex_num'] = train_ext_df['sex_num'].map({'male': 1, 'female': 0})
        train_ext_df['sex_num'] = train_ext_df['sex_num'].fillna(-1)

        
# Age features
train_df['age_num'] = train_df['age_approx']
test_df['age_num'] = test_df['age_approx']

train_df['age_num'] = train_df['age_num'].fillna(-5)
test_df['age_num'] = test_df['age_num'].fillna(-5)

if config.get('external',False):
    train_ext_df['age_num'] = train_ext_df['age_approx']
    train_ext_df['age_num'] = train_ext_df['age_num'].fillna(-5)


if config['age_normalize']:
    train_df['age_num'] /= train_df['age_num'].max()
    test_df['age_num'] /= test_df['age_num'].max()
    if config.get('external',False):
        train_ext_df['age_num'] /= train_ext_df['age_num'].max()

# remove duplicates if any
if 'tfrecord' in train_df:
    dup_ind = train_df[train_df['tfrecord']==-1].index
    train_df.drop(dup_ind , inplace=True)
    train_df.reset_index(drop=True, inplace=True)
    if config.get('external',False):
        dup_ind = train_ext_df[train_ext_df['tfrecord']==-1].index
        train_ext_df.drop(dup_ind , inplace=True)
        train_ext_df.reset_index(drop=True, inplace=True)


meta_features = [col for col in train_df.columns if 'sex' in col] + [col for col in train_df.columns if 'site_' in col]+['age_num']
meta_features.remove('anatom_site_general_challenge')
meta_features.remove('sex')


if config['test']:
    # define test set
    test = MelanomaDataset(df=test_df,
                                 config = config,
                           imfolder=config['data_folder']+'/test', 
                           split='test',
                           meta_features=meta_features)

#skf = GroupKFold(n_splits=5)
#skf = StratifiedKFold(n_splits=5, random_state=999, shuffle=True)

#SEED = random.randint(1, 10000) # use if you want new results
#if config.get('seed',False):
#    SEED = random.randint(1, 10000) # use if you want new results
#else:    
#    SEED = config['seed']

SEED = config.get('seed', random.randint(1, 10000))

print(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

skf = KFold(n_splits=5,shuffle=True,random_state=SEED)


epochs = config['epochs']  # Number of epochs to run
#model_path = 'best_local_cv_model.pth'  # Path and filename to save model to
es_patience = 3  # Early Stopping patience - for how many epochs with no improvements to wait
TTA = config['TTA'] # Test Time Augmentation rounds



oof = np.zeros((len(train_df), 1))  # Out Of Fold predictions

if config['test']:
    preds = torch.zeros((len(test), 1), dtype=torch.float32, device=device)  # Predictions for test test


auc_cvs = np.zeros(skf.n_splits) 
# train on each fold loop
for fold,(idxT,idxV) in enumerate(skf.split(np.arange(15)),1):
#for fold, (train_idx, val_idx) in enumerate(skf.split(X=np.zeros(len(train_df)), y=train_df['target'], groups=train_df['patient_id'].tolist()), 1):
    train_idx = train_df[train_df.tfrecord.isin(idxT)].index.tolist()
    val_idx = train_df[train_df.tfrecord.isin(idxV)].index.tolist()
    #print(val_idx,train_idx)
    #print(len(val_idx))
    print('=' * 20, 'Fold', fold, '=' * 20)
    print('Val: # of +1 class:',sum(train_df.loc[val_idx]['target'].values==1),', # of 0 class:',sum(train_df.loc[val_idx]['target'].values==0))
    #print('Train: # of +1 class:',sum(train_df.loc[train_idx]['target'].values==1),', # of 0 class:',sum(train_df.loc[train_idx]['target'].values==0))
    start_time = time.time()

    #fold_path = os.path.join(config['save_folder'],str(fold))
    #if not os.path.exists(fold_path):
    #    os.makedirs(fold_path)
    model_path = 'best_cv'+str(fold)+'.pth'
    # define train, test and val sets
    #print(config['data_folder'])
    if config.get('external',False):
        train_all_df = pd.concat([train_df.loc[train_idx],train_ext_df],axis=0)
    else:
        train_all_df = train_df.loc[train_idx]

    if config.get('upsample',False) and config['upsample']>0:
        print('upsampling ' +str(config['upsample']) + "x")
        #is_mal = train_all_df['target']==1
        #train_mal = train_all_df[is_mal]
        #train_all_df=train_all_df.append([train_mal]*config['upsample'])
        train_train_df = train_df.loc[train_idx]
        is_mal = train_train_df['target']==1
        train_mal = train_train_df[is_mal]
        train_all_df = train_all_df.append([train_mal]*config['upsample'])

    train = MelanomaDataset(df=train_all_df.reset_index(drop=True),
                            config = config,
                            imfolder=config['data_folder']+'/train', 
                            split='train',
                            meta_features=meta_features)

    val = MelanomaDataset(df=train_df.loc[val_idx].reset_index(drop=True),
                            config = config,
                            imfolder=config['data_folder']+'/train', 
                            split='val', 
                            meta_features=meta_features)
    
    if config.get('balanced_batches',False):
        train_loader = DataLoader(dataset=train, sampler=BalanceClassSampler(train,list(train_all_df['target'])),batch_size=config['batch_size'], num_workers=2)
        print('balanced_batches')
    else:    
        train_loader = DataLoader(dataset=train, batch_size=config['batch_size'], shuffle=True, num_workers=2)

    val_loader = DataLoader(dataset=val, batch_size=16, shuffle=False, num_workers=2)

    if config['test']:
        test_loader = DataLoader(dataset=test, batch_size=16, shuffle=False, num_workers=2)

    #print(train.composed)

    # select model
    model =  getModel({'numClasses':config['numClasses'],'model_type':config['model_type']}) ()

    # modify number of classes
    if 'Dense' in config['model_type']:
        if config['input_size'][0] != 224:
            model = utils.modify_densenet_avg_pool(model)
            #print(model)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, config['numClasses'])
        #print(model)
    elif 'dpn' in config['model_type']:
        num_ftrs = model.classifier.in_channels
        model.classifier = nn.Conv2d(num_ftrs,config['numClasses'],[1,1])
        #model.add_module('real_classifier',nn.Linear(num_ftrs, config['numClasses']))
        #print(model)
    elif 'efficient' in config['model_type']:
        # Do nothing, output is prepared
        if 'ns' in config['model_type']:
            num_cnn_features = model.classifier.in_features
        else:     
        	num_cnn_features = model._fc.in_features 
        #num_ftrs = model._fc.in_features
        model._fc = nn.Linear(num_cnn_features, config['numClasses'])    
    elif 'wsl' in config['model_type'] or 'resnest' in config['model_type']:
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, config['numClasses'])          
    else:
        num_ftrs = model.last_linear.in_features
        model.last_linear = nn.Linear(num_ftrs, config['numClasses'])    

    # take care of meta
    if config['train_meta']:
        # freeze cnn first
        if config['freeze_cnn']:
            # deactivate all
            for param in model.parameters():
                param.requires_grad = False

            if 'efficient' in config['model_type']:
                # Activate fc
                for param in model._fc.parameters():
                    param.requires_grad = True
            elif 'wsl' in config['model_type']:
                # Activate fc
                for param in model.fc.parameters():
                    param.requires_grad = True
            else:
                # Activate fc
                for param in model.last_linear.parameters():
                    param.requires_grad = True                                
        else:
            # mark cnn parameters
            for param in model.parameters():
                param.is_cnn_param = True
            # unmark fc
            if 'efficient' in config['model_type']:
                for param in model._fc.parameters():
                    param.is_cnn_param = False

            elif 'wsl' in config['model_type'] or 'resnest' in config['model_type']:
                for param in model.fc.parameters():
                    param.is_cnn_param = False
        # modify model
        config['n_meta_features'] = len(meta_features)
        model = modify_meta(config,model)
        for param in model.parameters():
            if not hasattr(param, 'is_cnn_param'):
                param.is_cnn_param = False 

    #model = Net(arch=Model, n_meta_features=len(meta_features))  # New model for each fold
    if args.ngpu > 1:
        os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
        model = nn.DataParallel(model,[0,1])
    
    model = model.to(device)

    # optimizer
    if config.get('train_meta',None) is not None:
        if config['freeze_cnn']:
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config['learning_rate_meta'])
            # sanity check
            for param in filter(lambda p: p.requires_grad, model.parameters()):
                print(param.name,param.shape)
        else:
            optimizer = optim.Adam([
                                    {'params': filter(lambda p: not p.is_cnn_param, model.parameters()), 'lr': config['learning_rate_meta']},
                                    {'params': filter(lambda p: p.is_cnn_param, model.parameters()), 'lr': config['learning_rate']}
                                    ], lr=config['learning_rate'])
    else:
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    # scheduler
    if config['step_decay']:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=config['lowerLRAfter'], gamma=1/np.float32(config['LRstep']))
    elif config['plateau_decay']:    
        scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=config['patience_steps'], verbose=True, factor=config['plateau_decay_factor'])
    else:
        scheduler = None

    # loss
    if config.get('focalloss',False):
        criterion = Sigmoid_focal_loss(config['focal_gamma'],config['focal_alpha'])
    elif config['weighted_BCE']:
        criterion = WeightedBCELoss(weight=config['bce_wieghts'])
    else:
        criterion = nn.BCEWithLogitsLoss()
    # metric variabels

    best_val = None  # Best validation score within this fold
    patience = es_patience  # Current patience counter

    if config['mixed_prec']:
        scaler = GradScaler()
    # train on each epoch
    for epoch in range(epochs):

        correct = 0
        epoch_loss = 0
        model.train()

        # train on each mini-batch
        for x, y in train_loader:
            if y.size(0) ==1:
                continue
            x[0] = x[0].float().to(device)
            x[1] = x[1].float().to(device)
            y = y.float().to(device)
            optimizer.zero_grad()
            #print(x[1].shape)
            if config['mixed_prec']:
                with autocast():
                    logit = model(x)
                    #print(logit)
                    loss = criterion(logit, y.unsqueeze(1))
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
            else:
                logit = model(x)
                loss = criterion(logit, y.unsqueeze(1))
                loss.backward()
                optimizer.step()

            pred = torch.round(torch.sigmoid(logit))  # round off sigmoid to obtain predictions
            correct += (pred.cpu() == y.cpu().unsqueeze(1)).sum().item()  # tracking number of correctly predicted samples
            epoch_loss += loss.item()
            #break
            #print(epoch_loss)
            #exit()
        train_acc = correct / len(train)
        #p#rint(logit)
        #print(train_acc)
        model.eval()  # switch model to the evaluation mode
        val_preds = torch.zeros((len(val), 1), dtype=torch.float32, device=device)
        #print(len(val_idx))    
        with torch.no_grad():  # Do not calculate gradient since we are only predicting
            # Predicting on validation set
            for j, (x_val, y_val) in enumerate(val_loader):
                x_val[0] = x_val[0].float().to(device)
                x_val[1] = x_val[1].float().to(device)
                y_val = y_val.float().to(device)
                #print(x_val,y_val)
                logit_val = model(x_val)
                #print(logit_val)
                val_pred = torch.sigmoid(logit_val)
                #val_preds[j*x_val[0].shape[0]:j*x_val[0].shape[0] + x_val[0].shape[0]] = val_pred
                #print(val_pred)
                val_preds[j*val_loader.batch_size:j*val_loader.batch_size + x_val[0].shape[0]] = val_pred
                #print(val_preds)
                #exit()
                #print(val_pred.shape)
            #print(len(train_df.loc[val_idx]['target'].values),len(val_preds.cpu()))
            #print(train_df.loc[val_idx]['target'].values,val_preds.cpu())    
            val_acc = accuracy_score(train_df.loc[val_idx]['target'].values, torch.round(val_preds.cpu()))
            val_F1 = f1_score(train_df.loc[val_idx]['target'].values, torch.round(val_preds.cpu()),zero_division=0)
            val_CM = confusion_matrix(train_df.loc[val_idx]['target'].values, torch.round(val_preds.cpu()))
            val_roc = roc_auc_score(train_df.loc[val_idx]['target'].values, val_preds.cpu())

            print('Epoch {:03}: | Loss: {:.3f} | Train acc: {:.3f} | Val acc: {:.3f} | Val roc_auc: {:.4f} | Val F1: {:.3f} | Val CM: {} | Training time: {}'.format(
            epoch + 1, 
            epoch_loss, 
            train_acc, 
            val_acc, 
            val_roc,
            val_F1, 
            val_CM.flatten(),
            str(datetime.timedelta(seconds=time.time() - start_time))[:7]))

            if config['step_decay']:
                scheduler.step()
            elif config['plateau_decay']:
                scheduler.step(val_roc)

            # During the first iteration (first epoch) best validation is set to None
            if not best_val:
                best_val = val_roc  # So any validation roc_auc we have is the best one for now
                auc_cvs[fold-1] = val_roc
                torch.save(model.state_dict(), os.path.join(config['save_folder'],model_path))  # Saving the model
                #torch.save(model.state_dict(),
                #               f"Fold{fold}_Epoch{epoch+1}_ValidAcc_{val_acc:.3f}_ROC_{val_roc:.3f}.pth")
                continue
                
            if val_roc >= best_val:
                best_val = val_roc
                auc_cvs[fold-1] = val_roc
                patience = es_patience  # Resetting patience since we have new best validation accuracy
                torch.save(model.state_dict(), os.path.join(config['save_folder'],model_path))  # Saving current best model
            else:
                patience -= 1
                if patience == 0:
                    print('Early stopping. Best Val roc_auc: {:.4f}'.format(best_val))
                    break    
        #break
    #model = torch.load(os.path.join(config['save_folder'],model_path))  # Loading best model of this fold
    #print(model)
    #print(os.path.join(config['save_folder'],model_path))
    model.load_state_dict(torch.load(os.path.join(config['save_folder'],model_path)))  # Loading best model of this fold
    model.eval()  # switch model to the evaluation mode
    val_preds = torch.zeros((len(val), 1), dtype=torch.float32, device=device)

    # evaluation best model on val/test sets 
    with torch.no_grad():
        # Predicting on validation set once again to obtain data for OOF
        for j, (x_val, y_val) in enumerate(val_loader):
            x_val[0] = x_val[0].float().to(device)
            x_val[1] = x_val[1].float().to(device)
            y_val = y_val.float().to(device)
            z_val = model(x_val)
            val_pred = torch.sigmoid(z_val)
            #val_preds[j*x_val[0].shape[0]:j*x_val[0].shape[0] + x_val[0].shape[0]] = val_pred
            val_preds[j*val_loader.batch_size:j*val_loader.batch_size + x_val[0].shape[0]] = val_pred
        oof[val_idx] = val_preds.cpu().numpy()
        #exit()
        # Predicting on test set
        if config['test']:
            tta_preds = torch.zeros((len(test), 1), dtype=torch.float32, device=device)
            for _ in range(TTA):
                for i, x_test in enumerate(test_loader):
                    x_test[0] = x_test[0].float().to(device)
                    x_test[1] = x_test[1].float().to(device)
                    z_test = model(x_test)
                    z_test = torch.sigmoid(z_test)
                    #preds[i*x_test[0].shape[0]:i*x_test[0].shape[0] + x_test[0].shape[0]] += z_test
                    #preds[i*test_loader.batch_size:i*test_loader.batch_size + x_test[0].shape[0]] += z_test
                    tta_preds[i*test_loader.batch_size:i*test_loader.batch_size + x_test[0].shape[0]] += z_test
            
            fold_pred = tta_preds/TTA  
            subـfold = pd.read_csv(os.path.join(config['data_folder'],'sample_submission.csv'))
            subـfold['target'] = fold_pred.cpu().numpy().reshape(-1,)
            subـfold.to_csv(os.path.join(config['save_folder'],"fold_" +str(fold)+'_submission.csv'), index=False)
            preds += tta_preds/TTA        
            #preds /= TTA

    del train, val, train_loader, val_loader, x, y, x_val, y_val
    gc.collect()

if config['test']:
    preds /= skf.n_splits

print('OOF: {:.4f}'.format(roc_auc_score(train_df['target'], oof)))


# Saving OOF predictions so stacking would be easier
pd.Series(oof.reshape(-1,)).to_csv(os.path.join(config['save_folder'],'oof.csv'), index=False)
pd.Series(auc_cvs).to_csv(os.path.join(config['save_folder'],'auc_cvs.csv'), index=False)

#save submission
if config['test']:
    sub = pd.read_csv(os.path.join(config['data_folder'],'sample_submission.csv'))
    sub['target'] = preds.cpu().numpy().reshape(-1,)
    sub.to_csv(os.path.join(config['save_folder'],'submission.csv'), index=False)



