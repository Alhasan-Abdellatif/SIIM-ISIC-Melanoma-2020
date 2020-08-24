import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import torchtoolbox.transform as transforms
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, GroupKFold
import pandas as pd
import numpy as np
import gc
import os
import cv2
import time
import datetime
import warnings
import random
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from albumentations import (Cutout,HueSaturationValue)

class MelanomaDataset(Dataset):
    def __init__(self, df: pd.DataFrame,config, imfolder: str, split = 'train', meta_features = None):
        """
        Class initialization
        Args:
            df (pd.DataFrame): DataFrame with data description
            imfolder (str): folder with images
            split : train ,val,test
            transforms: image transformation method to be applied
            meta_features (list): list of features with meta information, such as sex and age
            
        """
        self.df = df
        self.imfolder = imfolder
        self.split = split
        self.meta_features = meta_features
        self.input_size = config['input_size']
        self.same_sized_crop = config['same_sized_crop']
        self.hair_aug = config['hair_aug']
        self.microscope_aug = config['microscope_aug']
        self.config = config
        if split == 'train' or split == 'test':
            all_transforms = []
            if self.hair_aug :
                all_transforms.append(AdvancedHairAugmentation(hairs_folder='melanoma_hair/'))
            if self.same_sized_crop:
                all_transforms.append(transforms.RandomCrop(self.input_size))
            else:
                all_transforms.append(transforms.RandomResizedCrop(self.input_size,scale=(config.get('scale_min',0.08),1.0)))

             
            all_transforms.append(transforms.RandomHorizontalFlip())
            all_transforms.append(transforms.RandomVerticalFlip())   

            #if config.get('full_rot',0) > 0:
            #    if config.get('scale',False):
            #        all_transforms.append(transforms.RandomChoice([transforms.RandomAffine(config['full_rot'], scale=config['scale'], shear=config.get('shear',0), resample=Image.NEAREST),
            #                                                    transforms.RandomAffine(config['full_rot'],scale=config['scale'],shear=config.get('shear',0), resample=Image.BICUBIC),
            #                                                    transforms.RandomAffine(config['full_rot'],scale=config['scale'],shear=config.get('shear',0), resample=Image.BILINEAR)])) 
            #    else:
            #        all_transforms.append(transforms.RandomChoice([transforms.RandomRotation(config['full_rot'], resample=Image.NEAREST),
            #                                                    transforms.RandomRotation(config['full_rot'], resample=Image.BICUBIC),
            #                                                    transforms.RandomRotation(config['full_rot'], resample=Image.BILINEAR)]))

            if config.get('full_rot',0) > 0:
                if config.get('scale',False):
                    all_transforms.append(transforms.RandomAffine(config['full_rot'], scale=config['scale'], shear=config.get('shear',0)))
            #                                                    transforms.RandomAffine(config['full_rot'],scale=config['scale'],shear=config.get('shear',0), resample=Image.BICUBIC),
            #                                                    transforms.RandomAffine(config['full_rot'],scale=config['scale'],shear=config.get('shear',0), resample=Image.BILINEAR)])) 
                else:
                    all_transforms.append(transforms.RandomRotation(config['full_rot']))
            #                                                    transforms.RandomRotation(config['full_rot'], resample=Image.BICUBIC),
            #                                                    transforms.RandomRotation(config['full_rot'], resample=Image.BILINEAR)]))


            all_transforms.append(transforms.ColorJitter(brightness=32. / 255.,saturation=0.5))   

            if self.microscope_aug:
                all_transforms.append(Microscope(p=0.6))
            if config['cutout']:
                all_transforms.append(Cutout_v0(n_holes=1,length=config['cutout_length']))                
                #all_transforms.append(transforms.Cutout(scale=(0.05, 0.007), value=(0, 0)))       

            all_transforms.append(transforms.ToTensor())
            all_transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]))   
            self.composed = transforms.Compose(all_transforms)                  

        else:
             self.composed = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                ])

    def __getitem__(self, index):
        #print(self.df.iloc[index]['image_name'],self.config['img_ext'])
        if self.df.iloc[index]['patient_id'] == -1:
            im_path = os.path.join(self.config['external_path'], self.df.iloc[index]['image_name'] + self.config['img_ext'])
        else:
            im_path = os.path.join(self.imfolder, self.df.iloc[index]['image_name'] + self.config['img_ext'])
        #print(self.imfolder)
        #print(im_path)
        x = cv2.imread(im_path)
        #x =Image.open(im_path)
        meta = np.array(self.df.iloc[index][self.meta_features].values, dtype=np.float32)

        if self.composed:
            x = self.composed(x)
            
        if self.split == 'train' or self.split == 'val':
            y = self.df.iloc[index]['target']
            return (x, meta), y
        else:
            return (x, meta)
    
    def __len__(self):
        return len(self.df)
    
class AdvancedHairAugmentation:
    """
    Impose an image of a hair to the target image

    Args:
        hairs (int): maximum number of hairs to impose
        hairs_folder (str): path to the folder with hairs images
    """

    def __init__(self, hairs: int = 5, hairs_folder: str = ""):
        self.hairs = hairs
        self.hairs_folder = hairs_folder

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to draw hairs on.

        Returns:
            PIL Image: Image with drawn hairs.
        """
        n_hairs = random.randint(0, self.hairs)
        
        if not n_hairs:
            return img
        
        height, width, _ = img.shape  # target image width and height
        hair_images = [im for im in os.listdir(self.hairs_folder) if 'png' in im]
        
        for _ in range(n_hairs):
            hair = cv2.imread(os.path.join(self.hairs_folder, random.choice(hair_images)))
            hair = cv2.flip(hair, random.choice([-1, 0, 1]))
            hair = cv2.rotate(hair, random.choice([0, 1, 2]))

            h_height, h_width, _ = hair.shape  # hair image width and height
            #print(img.shape[0] - hair.shape[0])
            #print(img.shape[1] - hair.shape[1])
            roi_ho = random.randint(0, img.shape[0] - hair.shape[0])
            roi_wo = random.randint(0, img.shape[1] - hair.shape[1])
            roi = img[roi_ho:roi_ho + h_height, roi_wo:roi_wo + h_width]

            # Creating a mask and inverse mask
            img2gray = cv2.cvtColor(hair, cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
            mask_inv = cv2.bitwise_not(mask)

            # Now black-out the area of hair in ROI
            img_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

            # Take only region of hair from hair image.
            hair_fg = cv2.bitwise_and(hair, hair, mask=mask)

            # Put hair in ROI and modify the target image
            dst = cv2.add(img_bg, hair_fg)

            img[roi_ho:roi_ho + h_height, roi_wo:roi_wo + h_width] = dst
                
        return img

    def __repr__(self):
        return f'{self.__class__.__name__}(hairs={self.hairs}, hairs_folder="{self.hairs_folder}")'


class Microscope:
    """
    Cutting out the edges around the center circle of the image
    Imitating a picture, taken through the microscope

    Args:
        p (float): probability of applying an augmentation
    """

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to apply transformation to.

        Returns:
            PIL Image: Image with transformation.
        """
        if random.random() < self.p:
            circle = cv2.circle((np.ones(img.shape) * 255).astype(np.uint8), # image placeholder
                        (img.shape[0]//2, img.shape[1]//2), # center point of circle
                        random.randint(img.shape[0]//2 - 3, img.shape[0]//2 + 15), # radius
                        (0, 0, 0), # color
                        -1)

            mask = circle - 255
            img = np.multiply(img, mask)
        
        return img

    def __repr__(self):
        return f'{self.__class__.__name__}(p={self.p})'


class Cutout_v0(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        img = np.array(img)
        #print(img.shape)
        h = img.shape[0]
        w = img.shape[1]

        mask = np.ones((h, w), np.uint8)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        #mask = torch.from_numpy(mask)
        #mask = mask.expand_as(img)
        img = img * np.expand_dims(mask,axis=2)
        #img = Image.fromarray(img)
        return img   
    
def weighted_binary_cross_entropy(sigmoid_x, targets, pos_weight, weight=None, size_average=True, reduce=True):
    """
    Args:
        sigmoid_x: predicted probability of size [N,C], N sample and C Class. Eg. Must be in range of [0,1], i.e. Output from Sigmoid.
        targets: true value, one-hot-like vector of size [N,C]
        pos_weight: Weight for postive sample
    """
    if not (targets.size() == sigmoid_x.size()):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(targets.size(), sigmoid_x.size()))

    sigmoid_x = torch.sigmoid(sigmoid_x)

    loss = -pos_weight* targets * sigmoid_x.log() - (1-targets)*(1-sigmoid_x).log()

    if weight is not None:
        loss = loss * weight

    if not reduce:
        return loss
    elif size_average:
        return loss.mean()
    else:
        return loss.sum()


class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight=1, weight=None, PosWeightIsDynamic= False, WeightIsDynamic= False, size_average=True, reduce=True):
        """
        Args:
            pos_weight = Weight for postive samples. Size [1,C]
            weight = Weight for Each class. Size [1,C]
            PosWeightIsDynamic: If True, the pos_weight is computed on each batch. If pos_weight is None, then it remains None.
            WeightIsDynamic: If True, the weight is computed on each batch. If weight is None, then it remains None.
        """
        super().__init__()

        self.register_buffer('weight', weight)
        self.register_buffer('pos_weight', pos_weight)
        self.size_average = size_average
        self.reduce = reduce
        self.PosWeightIsDynamic = PosWeightIsDynamic

    def forward(self, input, target):
        # pos_weight = Variable(self.pos_weight) if not isinstance(self.pos_weight, Variable) else self.pos_weight
        if self.PosWeightIsDynamic:
            positive_counts = target.sum(dim=0)
            nBatch = len(target)
            self.pos_weight = (nBatch - positive_counts)/(positive_counts +1e-5)

        if self.weight is not None:
            # weight = Variable(self.weight) if not isinstance(self.weight, Variable) else self.weight
            return weighted_binary_cross_entropy(input, target,
                                                 self.pos_weight,
                                                 weight=self.weight,
                                                 size_average=self.size_average,
                                                 reduce=self.reduce)
        else:
            return weighted_binary_cross_entropy(input, target,
                                                 self.pos_weight,
                                                 weight=None,
                                                 size_average=self.size_average,
                                                 reduce=self.reduce)


class Sigmoid_focal_loss(nn.Module):
    def __init__(self, gamma=2.0, alpha=-1, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
         return sigmoid_focal_loss(inputs,targets,self.alpha,self.gamma,self.reduction)

def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = -1,
    gamma: float = 2,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


class FocalLoss(nn.Module):

    def __init__(self, gamma=2.0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)                         # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))    # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        #print("before gather",logpt)
        #print("target",target)
        logpt = logpt.gather(1,target)
        #print("after gather",logpt)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            #print("alpha",self.alpha)
            #print("gathered",at)
            logpt = logpt * at

        loss = -1 * (1 - pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()  