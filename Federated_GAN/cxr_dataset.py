import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import os
from PIL import Image
from skimage import exposure
from skimage import util


class PatchDataset(Dataset):

    def __init__(self, path_to_images, df, fold='train', sample=0, transform=None):

        self.transform = transform
        self.path_to_images = path_to_images
        self.df = df
        self.fold = fold
        # the 'fold' column says something regarding the train/valid/test seperation
        self.df = self.df[self.df['fold'] == fold]
        if(sample > 0 and sample < len(self.df)):
            self.df = self.df.sample(sample, random_state=42)
        
        self.df = self.df.set_index("Image Index")
        # self.PRED_LABEL = ['No Finding', 'Cardiomegaly', 'Edema', 
        #                     'Consolidation', 'Pneumonia', 'Atelectasis',
        #                     'Pneumothorax', 'Pleural Effusion']
        self.PRED_LABEL = ['No Finding']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        filename = '{0:06d}'.format(self.df.index[idx])
        image = Image.open(
            os.path.join(self.path_to_images, filename+'.png')
            )
        image = image.convert('L')
        label = np.zeros(len(self.PRED_LABEL), dtype=int)
        for i in range(0, len(self.PRED_LABEL)):
             # can leave zero if zero, else make one
            if(self.df[self.PRED_LABEL[i].strip()].iloc[idx].astype('int') > 0):
                # df.series.str.strip: remove leading and traling characters
                label[i] = self.df[self.PRED_LABEL[i].strip()].iloc[idx].astype('int')
                # Becareful with the 'int' type here !!!
        if self.transform:
            image = self.transform(image)

        return (image, label)
