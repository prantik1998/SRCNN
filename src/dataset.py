import os
import h5py

import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.utils.data as data

class Image91(data.Dataset):
	def __init__(self,datadir):
		super(Image91, self).__init__()
		self.filename = datadir

	def __getitem__(self,idx):
		f = h5py.File(self.filename,'r')
		return np.expand_dims(f['lr'][idx] / 255., 0), np.expand_dims(f['hr'][idx] / 255., 0)

	def __len__(self):
		f = h5py.File(self.filename,'r')		
		return len(f['lr'])

class Set5Dataset(data.Dataset):
	def __init__(self,datadir):
		super(Set5Dataset, self).__init__()
		self.filename = datadir

	def __len__(self):
		f = h5py.File(self.filename,'r')
		return len(f['lr'])

	def __getitem__(self,idx):
		f = h5py.File(self.filename,'r')
		idx = str(idx)
		return np.expand_dims(f['lr'][idx][:,:] / 255., 0), np.expand_dims(f['hr'][idx][:,:] / 255., 0)


