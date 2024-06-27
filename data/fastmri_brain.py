
import pickle
import numpy as np
import yaml
import argparse
import os
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
from data.mri import sense

# FastMRI brain subjects
# 108 FLAIR, 104 T1, 113 T1POST, 115 T2

#%%        

def get_labels_key(path):
    subdirs = sorted(os.listdir(path))
    labels_key = dict(enumerate([subdir.split('_')[0] for subdir in subdirs]))
    return labels_key

def preload(path, start_sub=0, num_sub_per_type=2, acc=4.0, num_sl=10, classes=[0,1,2,3]):
    subdirs = sorted(os.listdir(path))
    train_ksp, train_csm, labels, first_sub = None, None, None, True
    for i, subdir in enumerate(subdirs):
        if i in classes:
            fnames = [filename for filename in sorted(os.listdir(path+subdir)) if filename.endswith('.pickle')]
            print(subdir, '- loading', num_sub_per_type, 'of', len(fnames), 'subjects')
            
            subpath = os.path.join(path, subdir)
            train_fnames = fnames[start_sub:start_sub+num_sub_per_type]
            
            for j, train_fname in enumerate(train_fnames):
                with open(os.path.join(subpath, train_fname), 'rb') as f:
                    ksp, csm = pickle.load(f)
                    ksp, csm = torch.tensor(ksp[:num_sl]), torch.tensor(csm[:num_sl])
                    
                    coil_imgs = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(ksp, dim=[-1,-2]),dim=[-1,-2],norm="ortho"), dim=[-1,-2])
                    org = torch.sum(coil_imgs*torch.conj(csm),1,True)
#                     print('org', org.shape)
                    
#                     mean_vals = org.std(dim=1)
#                     std_vals = org.mean(dim=1)
#                     normalizer = transforms.Normalize(mean=mean_vals, std=std_vals) 
                    
#                     org = normalizer(org)
                    
                    org = org / org.abs().max()
                    print('min/max:', org.abs().min(), '/', org.abs().max())
                    
                    ksp = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(org, dim=[-1,-2]),dim=[-1,-2],norm="ortho"), dim=[-1,-2])
                    ksp = ksp * csm
                    
                    if first_sub:
                        train_ksp = ksp
                        train_csm = csm
                        labels = torch.ones(ksp.shape[0],)*i
                        first_sub = False
                    else:
                        train_ksp = torch.cat((train_ksp, ksp))
                        train_csm = torch.cat((train_csm, csm))
                        labels = torch.cat((labels, torch.ones(ksp.shape[0],)*i))
                    print('ksp:', ksp.shape, '\tcsm:', csm.shape)
        
    # print('ksp:', train_ksp.shape, '\ncsm:', train_csm.shape, '\nlabels:', labels.shape,)
    
    if acc == 0:
        mask = torch.ones_like(train_ksp)
    elif acc != None:
        mask_filename = f'poisson_mask_2d_acc{acc:.1f}_320by320.npy'
        # mask = np.load(mask_filename)
        # mask = torch.tensor(mask)
        mask = np.load(mask_filename).astype(np.complex64)  
        mask = torch.tensor(np.tile(mask, [train_ksp.shape[0],train_ksp.shape[1],1,1]))
        # print("mask:", mask.shape)
    else:
        mask = None
    
    labels_key = dict(enumerate([subdir.split('_')[0] for subdir in subdirs]))
    print(f"Loaded dataset of {train_ksp.shape[0]} slices\n")
    
    return train_ksp, train_csm, mask, labels.long(), labels_key

def preprocess(ksp, csm, mask):    
    coil_imgs = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(ksp, dim=[-1,-2]),dim=[-1,-2],norm="ortho"), dim=[-1,-2])
    org = torch.sum(coil_imgs*torch.conj(csm),1,True)
    us_ksp = ksp * mask
    
    return org, us_ksp.type(torch.complex64), csm.type(torch.complex64), mask

def preprocess_imgs_complex(ksp, csm):
    coil_imgs = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(ksp, dim=[-1,-2]),dim=[-1,-2],norm="ortho"), dim=[-1,-2])
    org = torch.sum(coil_imgs*torch.conj(csm),1,True)
    
    # print('min/max:', org.abs().min(), '/', org.abs().max()) 
    
    return org.to(torch.float)

def preprocess_imgs_mag(ksp, csm):
    coil_imgs = torch.fft.fftshift(torch.fft.ifft2(torch.fft.ifftshift(ksp, dim=[-1,-2]),dim=[-1,-2],norm="ortho"), dim=[-1,-2])
    org = torch.sum(coil_imgs*torch.conj(csm),1,True).abs()
    
    org = org.to(torch.float)
    print('org', org.shape)

    # for i in range(org.shape[0]):
    #     # org[i] = (org[i] - org[i].mean()) / (org[i].std())
    #     org[i] = (org[i]) / (org[i].std())
    #     print('min/max:', org[i].min(), '/', org[i].max())
                    
    # for i in range(org.shape[0]):
    #     org[i] = org[i]/org[i].max()
    # print('min/max:', org.abs().min(), '/', org.abs().max()) 
    
    return org

#%%
class DataGenImagesOnly(Dataset):
    def __init__(self, start_sub=0, num_sub=2):
        # self.path = '/Shared/lss_jcb/aniket/FastMRI brain data/'
        self.path = '/home/mrjohn/LSS/lss_jcb/aniket/FastMRI brain data/'
        
        self.start_sub = start_sub
        self.num_sub = num_sub
        ksp, csm, _, self.labels, self.labels_key = preload(self.path, self.start_sub, self.num_sub, None)
        # self.org = preprocess_imgs_complex(ksp, csm)
        self.org = preprocess_imgs_mag(ksp, csm)
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, i):
        return self.org[i]
    
    
def inf_train_gen(batch_size, start_sub=0, num_sub=2):

    loader = torch.utils.data.DataLoader(
        DataGenImagesOnly(start_sub, num_sub), batch_size, drop_last=True, shuffle=True
    )
    while True:
        for img in loader:
            yield img
            

#%%
class DataGenImagesDownsampled(Dataset):
    def __init__(self, start_sub=0, num_sub=2, device=None, res=0, complex_in=False, classes=[0,1,2,3]):
        # self.path = '/Shared/lss_jcb/aniket/FastMRI brain data/'
        self.path = '/home/mrjohn/LSS/lss_jcb/aniket/FastMRI brain data/'
        self.start_sub = start_sub
        self.num_sub = num_sub
        self.device = device
        ksp, csm, _, self.labels, self.labels_key = preload(self.path, self.start_sub, self.num_sub, acc=None, classes=classes)
        if complex_in:
            self.org = preprocess_imgs_complex(ksp, csm)
            if res > 0:
                self.org = self.downsample_2d(self.org.real, res) + self.downsample_2d(self.org.imag, res)*1j
        else:
            self.org = preprocess_imgs_mag(ksp, csm).float()
            if res > 0:
                print(f"Resizing to {res}x{res}")
                self.org = self.downsample_2d(self.org, res).cpu()
            
        print('Resized org:', self.org.shape)
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, i):
        return self.org[i]
    
    def downsample_2d(self, X, sz):
        """
        Downsamples a stack of square images.
        Args: X: a stack of images (batch, channels, ny, ny), sz: the desired size of images.
        Returns: The downsampled images, a tensor of shape (batch, channel, sz, sz)
        """
        with torch.no_grad():
            kernel = torch.tensor([[.25, .5, .25], 
                                   [.5, 1, .5], 
                                   [.25, .5, .25]], device=X.device).reshape(1, 1, 3, 3)
            kernel = kernel.repeat((X.shape[1], 1, 1, 1))
            while sz < X.shape[-1] / 2:
                # Downsample by a factor 2 with smoothing
                mask = torch.ones(1, *X.shape[1:])
                mask = F.conv2d(mask, kernel, groups=X.shape[1], stride=2, padding=1)
                X = F.conv2d(X.float(), kernel, groups=X.shape[1], stride=2, padding=1)

                # Normalize the edges and corners.
                X = X = X / mask

        return F.interpolate(X, size=sz, mode='bilinear')
    
    
def inf_train_gen_downsampled(batch_size, start_sub=0, num_sub=2, device=None, res=0, complex_in=False, classes=[0,1,2,3]):

    loader = torch.utils.data.DataLoader(
        DataGenImagesDownsampled(start_sub, num_sub, device, res, complex_in, classes), batch_size, drop_last=True, shuffle=True
    )
    while True:
        for img in loader:
            yield img
            
