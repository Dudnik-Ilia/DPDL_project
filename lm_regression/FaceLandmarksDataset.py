import numpy as np
import os 
from skimage import io, transform
import random
import torch 
from torch.utils.data import Dataset, DataLoader
import torchvision



class FaceLandmarksDataset(Dataset):

    def __init__(self, data_frame, transform=None, output_shape=(-1, 2)):
        """
        Args:
            data_frame (pandas.DataFrame): Dataframe with all information 
            necessary to build the dataset
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = data_frame
        self.transform = transform
        self.output_shape = output_shape
    
    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.landmarks_frame.iloc[idx]['image_path'])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 2:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(self.output_shape)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample



