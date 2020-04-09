import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import os
from PIL import Image
from skimage import exposure
from skimage import util


class CXRDataset(Dataset):

    def __init__(
            self,
            path_to_images,
            fold,
            transform=None,
            sample=0,
            finding="any",
            starter_images=False):

        self.transform = transform
        self.path_to_images = path_to_images
        self.df = pd.read_csv("nih_labels.csv")
        # the 'fold' column says something regarding the train/valid/test seperation
        self.df = self.df[self.df['fold'] == fold]

        if(starter_images):
            starter_images = pd.read_csv("starter_images.csv")
            self.df=pd.merge(left=self.df,right=starter_images, how="inner",on="Image Index")
            
        # can limit to sample, useful for testing
        # if fold == "train" or fold =="val": sample=500
        if(sample > 0 and sample < len(self.df)):
            self.df = self.df.sample(sample)
        # df.sample: return a random sample of items from an axis of object

        if not finding == "any":  # can filter for positive findings of the kind described; useful for evaluation
            if finding in self.df.columns:
                if len(self.df[self.df[finding] == 1]) > 0:
                    self.df = self.df[self.df[finding] == 1]
                else:
                    print("No positive cases exist for "+LABEL+", returning all unfiltered cases")
            else:
                print("cannot filter on finding " + finding +
                      " as not in data - please check spelling")

        self.df = self.df.set_index("Image Index")
        # df.set_index: set the dataframe index using existing columns. 
        self.PRED_LABEL = [
            'Atelectasis',
            'Cardiomegaly',
            'Effusion',
            'Infiltration',
            'Mass',
            'Nodule',
            'Pneumonia',
            'Pneumothorax',
            'Consolidation',
            'Edema',
            'Emphysema',
            'Fibrosis',
            'Pleural_Thickening',
            'Hernia']
        
        RESULT_PATH = "results/"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        image = Image.open(
            os.path.join(
                self.path_to_images,
                self.df.index[idx]))
        image = image.convert('RGB')

        ################## convert image from RGB to our 3 channel image ###############
        '''
        # print(str(image.size))
        image_np = np.array(image)
        test_img_1 = image_np[:, :, 1]
        img = (test_img_1 - np.amin(test_img_1))
        img = img / np.amax(img) * 2.0 - 1.0
        img_eq = exposure.equalize_hist(img)
        image_np[:, :, 1] = img_eq
        image_np[:, :, 2] = util.invert(image_np[:, :, 2])
        # print(str(image_np.shape))
        image = Image.fromarray(image_np)
        # print('Convert back to pillow image: '+ str(image.size))
        ################## convert image from RGB to our 3 channel image ###############
        '''

        label = np.zeros(len(self.PRED_LABEL), dtype=int)
        for i in range(0, len(self.PRED_LABEL)):
             # can leave zero if zero, else make one
            if(self.df[self.PRED_LABEL[i].strip()].iloc[idx].astype('int') > 0):
                # df.series.str.strip: remove leading and traling characters
                label[i] = self.df[self.PRED_LABEL[i].strip()
                                   ].iloc[idx].astype('int')
                # Becareful with the 'int' type here !!!

        if self.transform:
            image = self.transform(image)

        return (image, label,self.df.index[idx])
