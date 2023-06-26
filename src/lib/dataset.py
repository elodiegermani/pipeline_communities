from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
import nibabel as nib

class ClassifDataset(Dataset):
    '''
    Create a Dataset object used to load training data and train a model using pytorch.

    Parameters:
        - data_dir, str: directory where images are stored
        - id_file, str: path to the text file containing ids of images of interest
        - label_file, str: path to the csv file containing labels of images of interest
        - label_column, str: name of the column to use as labels in label_file
        - label_list, list: list of unique labels sorted in alphabetical order

    Attributes:
        - data, list of str: list containing all images of the dataset selected
        - ids, list of int: list containing all ids of images of the selected dataset
        - labels, list of str: list containing all labels of each data
    '''
    def __init__(self, data_dir, label_file, label_column, label_list, contrast_list):
        self.df = pd.read_csv(label_file)

        if contrast_list != 'All':
            self.df = self.df[self.df['Contrast'].isin(contrast_list)]
        
        self.data = self.df['Filename'].tolist()
        self.ids = self.df['Ids'].tolist()
        self.labels = self.df[label_column].tolist()
        self.classe = label_column
        self.subject = self.df['Group'].tolist()

        self.label_list = sorted(label_list)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fname = self.data[idx]
        label = self.labels[idx]
        label_vect = [0 for i in range(len(self.label_list))]

        for i in range(len(self.label_list)):
            if label == self.label_list[i]:
                label_vect[i] = 1
                
        sample = nib.load(fname).get_fdata().copy().astype(float)
        sample = np.nan_to_num(sample)[0:48,0:56,0:48]

        sample = torch.tensor(sample).view((1), *sample.shape)
        label_vect = torch.tensor(label_vect)
        
        return sample, label_vect

    def get_original_ids(self):
        return self.ids

    def get_original_labels(self):
        return self.labels

    def get_original_subject(self):
        return self.subject