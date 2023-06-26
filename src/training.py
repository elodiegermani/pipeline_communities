from lib import model_4layers as md 
from lib import train, dataset 
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
import sys
import os
from sklearn import metrics

def main():
	# Parameters:
	### TO MODIFY
	dpath = '/srv/tempdd/egermani/pipeline_distance'
	lr=1e-4
	str_lr = "{:.0e}".format(lr)
	batch_size=64
	epochs=500
	out_dir = f'{dpath}/data/derived'
	data_dir = f'{dpath}/data/preprocessed'
	model = md.Classifier3D()

	# Datasets
	label_file = f'{dpath}/data/train_dataset.csv'
	label_column = 'Pipeline'
	label_list = np.unique(pd.read_csv(label_file)[label_column])
	contrast_list = ['right-foot', 'right-hand', 'tongue']
	c_name = 'con-right-tongue'

	train_dataset = dataset.ClassifDataset(f'{data_dir}/resampled_mni_masked_normalized_res_4', 
	                               label_file, label_column, label_list, contrast_list)
	print('Train dataset:', len(train_dataset.get_original_ids()))

	label_file = f'{dpath}/data/valid_dataset.csv'
	label_column = 'Pipeline'
	label_list = np.unique(pd.read_csv(label_file)[label_column])

	valid_dataset = dataset.ClassifDataset(f'{data_dir}/resampled_mni_masked_normalized_res_4', 
	                               label_file, label_column, label_list, contrast_list)

	print('Valid dataset:', len(valid_dataset.get_original_ids()))

	# Reproducibility constraints
	random_seed=0
	np.random.seed(random_seed)
	torch.manual_seed(random_seed)
	np.random.seed(random_seed)
	torch.random.manual_seed(random_seed)
	torch.cuda.manual_seed(random_seed)

	torch.backends.cudnn.deterministic = True 
	torch.backends.cudnn.benchmark = False

	# Create output dir to save models
	if not os.path.isdir(out_dir): 
	    os.mkdir(out_dir)

	# GPU or CPU
	if torch.cuda.is_available():
	    device = torch.device("cuda")
	    torch.cuda.manual_seed(random_seed)
	    print('Using GPU.')
	else:
	    device = "cpu"
	    print('Using CPU.')

	# Loss to use
	distance = nn.CrossEntropyLoss()
	# Model
	print('Model:', model)
	model = model.to(device)
	# Optimizer
	print(f'Optimizer: ADAM, lr={lr}')
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)  

	# DataLoader
	train_dataset = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	valid_dataset = DataLoader(valid_dataset, batch_size=batch_size)

	# START TRAINING
	print('Start training...')
	training_loss = []
	validation_loss = []

	for epoch in range(epochs):
		# TRAINING LOOP
	    current_training_loss = train.train(model, train_dataset, distance, optimizer, device)
	    training_loss.append(current_training_loss)

	    # VALIDATION LOOP
	    current_validation_loss = train.validate(model, valid_dataset, distance, device)
	    validation_loss.append(current_validation_loss)
	    
	    print('epoch [{}/{}], loss:{:.4f}, validation:{:.4f}'.format(epoch + 1, epochs, current_training_loss,
	                                                                 current_validation_loss))
	    if device != 'cpu':
	        if device.type == 'cuda':
	            torch.cuda.empty_cache()

	    if epoch%10==0:
	        torch.save(model,  f"{out_dir}/model-4l_b-{batch_size}_lr-{str_lr}_{c_name}_epochs_{epoch}.pt") 

	print('Training ended')

if __name__ == '__main__':
	main()