# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 09:58:10 2022

@author: Rana
"""

import pickle
import sys
import timeit
import math
import numpy as np
import random
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import mean_squared_error,r2_score
from scipy import stats


class KcatPrediction(nn.Module):
    def __init__(self):
        super(KcatPrediction, self).__init__()
        self.embed_fingerprint = nn.Embedding(n_fingerprint, dim)
        self.embed_word = nn.Embedding(n_word, dim)
        self.W_gnn = nn.ModuleList([nn.Linear(dim, dim)
                                    for _ in range(layer_gnn)])
        self.W_cnn = nn.ModuleList([nn.Conv2d(
                     in_channels=1, out_channels=1, kernel_size=2*window+1,
                     stride=1, padding=window) for _ in range(layer_cnn)])
        self.W_attention = nn.Linear(dim, dim)
        self.W_out = nn.ModuleList([nn.Linear(2*dim, 2*dim)
                                    for _ in range(layer_output)])
        # self.W_interaction = nn.Linear(2*dim, 2)
        self.W_interaction = nn.Linear(2*dim, 1)

    def gnn(self, xs, A, layer):
        for i in range(layer):
            hs = F.leaky_relu(self.W_gnn[i](xs))
            xs = xs + torch.matmul(A, hs)
        # return torch.unsqueeze(torch.sum(xs, 0), 0)
        return torch.unsqueeze(torch.mean(xs, 0), 0)

    def attention_cnn(self, x, xs, layer):
        """The attention mechanism is applied to the last layer of CNN."""

        xs = torch.unsqueeze(torch.unsqueeze(xs, 0), 0)
        for i in range(layer):
            xs = F.leaky_relu(self.W_cnn[i](xs))
        xs = torch.squeeze(torch.squeeze(xs, 0), 0)

        h = F.leaky_relu(self.W_attention(x))
        hs = F.leaky_relu(self.W_attention(xs))
        weights = torch.tanh(F.linear(h, hs))
        ys = torch.t(weights) * hs

        # return torch.unsqueeze(torch.sum(ys, 0), 0)
        return torch.unsqueeze(torch.mean(ys, 0), 0)

    def forward(self, inputs):

        fingerprints, adjacency, words = inputs

        """Compound vector with GNN."""
        fingerprint_vectors = self.embed_fingerprint(fingerprints)
        compound_vector = self.gnn(fingerprint_vectors, adjacency, layer_gnn)

        """Protein vector with attention-CNN."""
        word_vectors = self.embed_word(words)
        protein_vector = self.attention_cnn(compound_vector,
                                            word_vectors, layer_cnn)

        """Concatenate the above two vectors and output the interaction."""
        cat_vector = torch.cat((compound_vector, protein_vector), 1)
        for j in range(layer_output):
            cat_vector = F.leaky_relu(self.W_out[j](cat_vector))
        interaction = self.W_interaction(cat_vector)
        # print(interaction)

        return interaction

    def __call__(self, data, train=True):

        inputs, correct_interaction = data[:-1], data[-1]
        predicted_interaction = self.forward(inputs)
        # print(predicted_interaction)

        if train:
            loss = F.mse_loss(predicted_interaction, correct_interaction)
            correct_values = correct_interaction.to('cpu').data.numpy()
            predicted_values = predicted_interaction.to('cpu').data.numpy()[0]
            return loss, correct_values, predicted_values
        else:
            correct_values = correct_interaction.to('cpu').data.numpy()
            predicted_values = predicted_interaction.to('cpu').data.numpy()[0]
            # correct_values = np.concatenate(correct_values)
            # predicted_values = np.concatenate(predicted_values)
            # ys = F.softmax(predicted_interaction, 1).to('cpu').data.numpy()
            # predicted_values = list(map(lambda x: np.argmax(x), ys))
            # print(correct_values)
            # print(predicted_values)
            # predicted_scores = list(map(lambda x: x[1], ys))
            return correct_values, predicted_values


class Trainer(object):
    def __init__(self, model):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=lr, weight_decay=weight_decay)

    def train(self, dataset):
        np.random.shuffle(dataset)
        N = len(dataset)
        loss_total = 0
        trainCorrect, trainPredict = [], []
        for data in dataset:
            loss, correct_values, predicted_values = self.model(data)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_total += loss.to('cpu').data.numpy()

            correct_values = math.log10(math.pow(2,correct_values))
            predicted_values = math.log10(math.pow(2,predicted_values))
            trainCorrect.append(correct_values)
            trainPredict.append(predicted_values)
        rmse_train = np.sqrt(mean_squared_error(trainCorrect,trainPredict))
        r2_train = r2_score(trainCorrect,trainPredict)
        p_correlation_train, p_value_train = stats.pearsonr(trainCorrect, trainPredict)
        s_correlation_train, p_value_train = stats.spearmanr(trainCorrect, trainPredict)
        return loss_total, rmse_train, r2_train, p_correlation_train, s_correlation_train


class Tester(object):
    def __init__(self, model):
        self.model = model

    def test(self, dataset):
        N = len(dataset)
        SAE = 0  # sum absolute error.
        testY, testPredict = [], []
        for data in dataset :
            (correct_values, predicted_values) = self.model(data, train=False)
            correct_values = math.log10(math.pow(2,correct_values))
            predicted_values = math.log10(math.pow(2,predicted_values))
            SAE += np.abs(predicted_values-correct_values)
            # SAE += sum(np.abs(predicted_values-correct_values))
            testY.append(correct_values)
            testPredict.append(predicted_values)
        MAE = SAE / N  # mean absolute error.
        rmse = np.sqrt(mean_squared_error(testY,testPredict))
        r2 = r2_score(testY,testPredict)
        p_correlation_test, p_value_test = stats.pearsonr(testY, testPredict)
        s_correlation_test, p_value_test = stats.spearmanr(testY, testPredict)
        return MAE, rmse, r2, p_correlation_test, s_correlation_test, testY, testPredict

    def save_MAEs(self, MAEs, filename):
        with open(filename, 'a') as f:
            f.write('\t'.join(map(str, MAEs)) + '\n')
    
    def save_ys(self, ys, filename):
        with open(filename, 'a') as f:
            f.write('\t'.join(map(str, ys)) + '\n')

    def save_model(self, model, filename):
        torch.save(model.state_dict(), filename)

def load_tensor(file_name, dtype):
    return [dtype(d).to(device) for d in np.load(file_name + '.npy', allow_pickle=True)]

def load_array(file_name):
    x = np.load(file_name, allow_pickle=True)
    return x

def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        return pickle.load(f)

def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset

def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2

def split_data(data, input_data=['Compounds', 'Adjacencies', 'Proteins', 'Sequences', 'Interactions', 'SMILES'], cluster_file=None, split_type='type_0', split=(70, 10, 20), random_state=1234):
    """
    Parameters
    ----------
    data : list of zipped numpy arrays
        THIS IS THE INPUT DATA FOR THE DATA SPLITTING THAT CAN BE INTEGRATED INTO THE PIPELINE.
    input_data : LIST, optional
        The default is ['Compounds', 'Adjacencies', 'Proteins', 'Interactions', 'SMILES', 'Clusters'].
    cluster_file: Numpy array of pandas df
        The default is None. If available, insert as needed.
    split_type : STRING, optional
        The default is 'type_0'. Currently, the model can split for 'type_2' and 'type_3'. More options will be added soon.
    split : TUPLE, optional
        The default is (70, 10, 20).You can also use (80, 10, 10). 
    random_state : int, optional
        This is the random state of the random splits. The default is 1234.
    Returns
    -------
    dataset_train : list of zipped outputs
        The output of the dataset will be the following for the TRAINING set: (compounds, adjacencies, proteins, interactions)
    dataset_test : list of zipped outputs
        The output of the dataset will be the same as the dataset_train, but for the TESTING set.
    dataset_val : TYPE
        The output of the dataset will be the same as the dataset_train, but for the VALIDATION set.
    """
    
    df = pd.DataFrame(data, columns=input_data)
    compounds = input_data[0]
    adjacencies = input_data[1]
    proteins = input_data[2]
    sequences = input_data[3]
    values = input_data[4]
    smiles = input_data[5]
    
    if cluster_file is not None:
        cluster_df = (cluster_file)
        
    


    if split_type=='type_0': 
        train_len = int(len(df)*split[0]/100)
        valid_len = int(len(df)*split[1]/100)
        test_len = int(len(df)*split[2]/100)
       
        train, valid, test = [], [], []
        split_arr = []
    
        new_data = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        new_data['index'] = new_data.index
    
        count = 0
        for i  in range(len(new_data)):
            if i < train_len:
                train.append(count)
            elif i < valid_len+train_len:
                valid.append(count)
            else:
                test.append(count)
            count += 1
        for i, val in new_data.iterrows():
            if val['index'] in train:
                split_arr.append(0)
            elif val['index'] in valid:
                split_arr.append(1)
            elif val['index'] in test:
                split_arr.append(2)
    
        unique, counts = np.unique(split_arr, return_counts=True)
    
        print(len(split_arr))
        print("Train:", counts[np.where(unique==0)])
        print("Val:", counts[np.where(unique==1)])
        print("Test:", counts[np.where(unique==2)])
    
        new_data['split_label'] = split_arr
    
        train_dataset = new_data.loc[new_data['split_label']==0]
        val_dataset = new_data.loc[new_data['split_label']==1]
        test_dataset = new_data.loc[new_data['split_label']==2]
    
        compounds_train = train_dataset[compounds].to_numpy()
        adj_train = train_dataset[adjacencies].to_numpy()
        prot_train = train_dataset[proteins].to_numpy()
        seq_train = train_dataset[sequences].to_numpy()
        values_train = train_dataset[values].to_numpy()
        smiles_train = train_dataset[smiles].to_numpy()
    
        compounds_test = test_dataset[compounds].to_numpy()
        adj_test = test_dataset[adjacencies].to_numpy()
        prot_test = test_dataset[proteins].to_numpy()
        seq_test = train_dataset[sequences].to_numpy()
        values_test = test_dataset[values].to_numpy()
        smiles_test = test_dataset[smiles].to_numpy()
    
        compounds_val = val_dataset[compounds].to_numpy()
        adj_val = val_dataset[adjacencies].to_numpy()
        prot_val = val_dataset[proteins].to_numpy()
        seq_val = train_dataset[sequences].to_numpy()
        values_val = val_dataset[values].to_numpy()
        
        dataset_train = list(zip(compounds_train, adj_train, prot_train, values_train))
        dataset_test = list(zip(compounds_test, adj_test, prot_test, values_test))
        dataset_val = list(zip(compounds_val, adj_val, prot_val, values_val))
    
    elif split_type=='type_2':
        new_data = df 
        
        unique_sequences = new_data[sequences].unique()
        count_dict = {}
        train, valid, test = [], [], []
        split_arr = []
        for seq in unique_sequences:
            count_dict[seq] = new_data[new_data[sequences]==seq].shape[0]
        l = list(count_dict.items())
        random.seed(random_state)
        random.shuffle(l)
        sorted_count_dict = dict(l)
        
        train_len = int(len(new_data)*split[0]/100) # This will need to be changed
        valid_len = int(len(new_data)*split[1]/100) # This will need to be changed
        test_len = int(len(new_data)*split[2]/100) # This will need to be changed

        print('Train, val, and test lengths, respectively: ', train_len, valid_len, test_len)

        count = 0
        for seq, c in sorted_count_dict.items():
            if count < train_len:
                train.append(seq)
            elif count < valid_len+train_len:
                valid.append(seq)
            else:
                test.append(seq)
            count += c

        for i, val in new_data.iterrows():
            if val[sequences] in train:
                split_arr.append(0)
            elif val[sequences] in valid:
                split_arr.append(1)
            else:
                split_arr.append(2)

        unique, counts = np.unique(split_arr, return_counts=True)

        #print(len(split_arr))
        print("Train:", counts[np.where(unique==0)])
        print("Val:", counts[np.where(unique==1)])
        print("Test:", counts[np.where(unique==2)])
        
        new_data['split_label'] = split_arr
        
        train_dataset = new_data.loc[new_data['split_label']==0]
        val_dataset = new_data.loc[new_data['split_label']==1]
        test_dataset = new_data.loc[new_data['split_label']==2]
    
        compounds_train = train_dataset[compounds].to_numpy()
        adj_train = train_dataset[adjacencies].to_numpy()
        prot_train = train_dataset[proteins].to_numpy()
        values_train = train_dataset[values].to_numpy()
    
        compounds_test = test_dataset[compounds].to_numpy()
        adj_test = test_dataset[adjacencies].to_numpy()
        prot_test = test_dataset[proteins].to_numpy()
        values_test = test_dataset[values].to_numpy()
    
        compounds_val = val_dataset[compounds].to_numpy()
        adj_val = val_dataset[adjacencies].to_numpy()
        prot_val = val_dataset[proteins].to_numpy()
        values_val = val_dataset[values].to_numpy()
        
        dataset_train = list(zip(compounds_train, adj_train, prot_train, values_train))
        dataset_test = list(zip(compounds_test, adj_test, prot_test, values_test))
        dataset_val = list(zip(compounds_val, adj_val, prot_val, values_val))
    
    elif split_type=='type_3':
         new_data = df 
         
         unique_compounds = new_data[smiles].unique()
         count_dict = {}
         train, valid, test = [], [], []
         split_arr = []
         for comp in unique_compounds:
             count_dict[comp] = new_data[new_data[smiles]==comp].shape[0]
         l = list(count_dict.items())
         random.seed(random_state)
         random.shuffle(l)
         sorted_count_dict = dict(l)
         
         train_len = int(len(new_data)*split[0]/100) # This will need to be changed
         valid_len = int(len(new_data)*split[1]/100) # This will need to be changed
         test_len = int(len(new_data)*split[2]/100) # This will need to be changed

         print('Train, val, and test lengths, respectively: ', train_len, valid_len, test_len)

         count = 0
         for seq, c in sorted_count_dict.items():
             if count < train_len:
                 train.append(seq)
             elif count < valid_len+train_len:
                 valid.append(seq)
             else:
                 test.append(seq)
             count += c

         for i, val in new_data.iterrows():
             if val[smiles] in train:
                 split_arr.append(0)
             elif val[smiles] in valid:
                 split_arr.append(1)
             else:
                 split_arr.append(2)

         unique, counts = np.unique(split_arr, return_counts=True)

         #print(len(split_arr))
         print("Train:", counts[np.where(unique==0)])
         print("Val:", counts[np.where(unique==1)])
         print("Test:", counts[np.where(unique==2)])
         
         new_data['split_label'] = split_arr
         
         train_dataset = new_data.loc[new_data['split_label']==0]
         val_dataset = new_data.loc[new_data['split_label']==1]
         test_dataset = new_data.loc[new_data['split_label']==2]
     
         compounds_train = train_dataset[compounds].to_numpy()
         adj_train = train_dataset[adjacencies].to_numpy()
         prot_train = train_dataset[proteins].to_numpy()
         values_train = train_dataset[values].to_numpy()
     
         compounds_test = test_dataset[compounds].to_numpy()
         adj_test = test_dataset[adjacencies].to_numpy()
         prot_test = test_dataset[proteins].to_numpy()
         values_test = test_dataset[values].to_numpy()
     
         compounds_val = val_dataset[compounds].to_numpy()
         adj_val = val_dataset[adjacencies].to_numpy()
         prot_val = val_dataset[proteins].to_numpy()
         values_val = val_dataset[values].to_numpy()
         
         dataset_train = list(zip(compounds_train, adj_train, prot_train, values_train))
         dataset_test = list(zip(compounds_test, adj_test, prot_test, values_test))
         dataset_val = list(zip(compounds_val, adj_val, prot_val, values_val))
         
    elif split_type=='type_4':
        new_data = df
        print('OG DF:', new_data)
        train_len = int(len(new_data)*split[0]/100) # This will need to be changed
        valid_len = int(len(new_data)*split[1]/100) # This will need to be changed
        test_len = int(len(new_data)*split[2]/100) # This will need to be changed
    
        clustered_df = new_data.merge(cluster_df, how='left', on='Sequences')
        groups = [clustered_df for _, df in clustered_df.groupby('Cluster')]
        random.seed(random_state)
        random.shuffle(groups)
        new_data = pd.concat(groups).reset_index(drop=True)
        
        print(new_data)
        
        print('Length of df for type 4: ', len(new_data))
        
        unique_clusters = new_data['Cluster'].unique()
        count_dict = {}
        train, valid, test = [], [], []
        split_arr = []
        for clust in unique_clusters:
             count_dict[clust] = new_data[new_data['Cluster']==clust].shape[0]
        sorted_count_dict = count_dict
         
        train_len = int(len(new_data)*split[0]/100) # This will need to be changed
        valid_len = int(len(new_data)*split[1]/100) # This will need to be changed
        test_len = int(len(new_data)*split[2]/100) # This will need to be changed

        print('Train, val, and test lengths, respectively: ', train_len, valid_len, test_len)

        count = 0
        for seq, c in sorted_count_dict.items():
            if count < train_len:
                train.append(seq)
            elif count < valid_len+train_len:
                valid.append(seq)
            else:
                test.append(seq)
            count += c

        for i, val in new_data.iterrows():
            if val[smiles] in train:
                split_arr.append(0)
            elif val[smiles] in valid:
                split_arr.append(1)
            else:
                split_arr.append(2)

        unique, counts = np.unique(split_arr, return_counts=True)

        print("Train:", counts[np.where(unique==0)])
        print("Val:", counts[np.where(unique==1)])
        print("Test:", counts[np.where(unique==2)])
         
        new_data['split_label'] = split_arr
         
        train_dataset = new_data.loc[new_data['split_label']==0]
        val_dataset = new_data.loc[new_data['split_label']==1]
        test_dataset = new_data.loc[new_data['split_label']==2]
     
        compounds_train = train_dataset[compounds].to_numpy()
        adj_train = train_dataset[adjacencies].to_numpy()
        prot_train = train_dataset[proteins].to_numpy()
        values_train = train_dataset[values].to_numpy()
     
        compounds_test = test_dataset[compounds].to_numpy()
        adj_test = test_dataset[adjacencies].to_numpy()
        prot_test = test_dataset[proteins].to_numpy()
        values_test = test_dataset[values].to_numpy()
     
        compounds_val = val_dataset[compounds].to_numpy()
        adj_val = val_dataset[adjacencies].to_numpy()
        prot_val = val_dataset[proteins].to_numpy()
        values_val = val_dataset[values].to_numpy()
         
        dataset_train = list(zip(compounds_train, adj_train, prot_train, values_train))
        dataset_test = list(zip(compounds_test, adj_test, prot_test, values_test))
        dataset_val = list(zip(compounds_val, adj_val, prot_val, values_val))
        
    return dataset_train, dataset_test, dataset_val

if __name__ == "__main__":

    """Hyperparameters."""
    (DATASET, radius, ngram, dim, layer_gnn, window, layer_cnn, layer_output,
     lr, lr_decay, decay_interval, weight_decay, iteration,
     setting) = sys.argv[1:]
    (dim, layer_gnn, window, layer_cnn, layer_output, decay_interval,
     iteration) = map(int, [dim, layer_gnn, window, layer_cnn, layer_output,
                            decay_interval, iteration])
    lr, lr_decay, weight_decay = map(float, [lr, lr_decay, weight_decay])

    # print(type(radius))

    """CPU or GPU."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('The code uses GPU...')
    else:
        device = torch.device('cpu')
        print('The code uses CPU!!!')

    """Load preprocessed data."""
    dir_input = ('../../Data/input/')
    compounds = load_tensor(dir_input + 'compounds', torch.LongTensor)
    adjacencies = load_tensor(dir_input + 'adjacencies', torch.FloatTensor)
    proteins = load_tensor(dir_input + 'proteins', torch.LongTensor)
    interactions = load_tensor(dir_input + 'regression', torch.FloatTensor)
    sequences = load_array(dir_input + 'sequences.npy')
    smiles = load_array(dir_input + 'smiles.npy')
    #cluster_df = load_array(dir_input + 'clusters.npy')
    cluster_df = pd.read_csv(dir_input + 'clusters_df.csv')
    fingerprint_dict = load_pickle(dir_input + 'fingerprint_dict.pickle')
    word_dict = load_pickle(dir_input + 'sequence_dict.pickle')
    n_fingerprint = len(fingerprint_dict)
    n_word = len(word_dict)
    # print(n_fingerprint)  # 3958
    # print(n_word)  # 8542
    # 394 and 474 when radius=1 and ngram=2

    """Create a dataset and split it into train/dev/test."""
    
    dataset = list(zip(compounds, adjacencies, proteins, sequences, interactions, smiles))
    dataset_train, dataset_test, dataset_dev = split_data(dataset, split_type='type_4', cluster_file=cluster_df, random_state=1234)
    
    # dataset = list(zip(compounds, adjacencies, proteins, interactions))
    # dataset = shuffle_dataset(dataset, 1234)
    # dataset_train, dataset_ = split_dataset(dataset, 0.7)
    # dataset_dev, dataset_test = split_dataset(dataset_, 0.33)

    """Set a model."""
    torch.manual_seed(1234)
    model = KcatPrediction().to(device)
    trainer = Trainer(model)
    tester = Tester(model)

    """Output files."""
    file_MAEs = '../../Data/Results/output/MAEs--' + setting + '.txt'
    file_model = '../../Data/Results/output/' + setting
    file_ys = '../../Data/Results/output/test_ys--' + setting + '.txt'
    MAEs = ('Epoch\tTime(sec)\tRMSE_train\tR2_train\tMAE_dev\tMAE_test\tRMSE_dev\tRMSE_test\tR2_dev\tR2_test\tR_dev (Pearson)\tR_test (Pearson)\tR_dev (Spearman)\tR_test (Spearman)')
    ys = ('Real\tPredicted')
    
    with open(file_MAEs, 'w') as f:
        f.write(MAEs + '\n')

    with open(file_ys, 'w') as f:
        f.write(ys + '\n')
        
    """Start training."""
    print('Training...')
    print(MAEs)
    start = timeit.default_timer()

    for epoch in range(1, iteration+1):

        if epoch % decay_interval == 0:
            trainer.optimizer.param_groups[0]['lr'] *= lr_decay

        loss_train, rmse_train, r2_train, pearson_r_train, spearman_r_train = trainer.train(dataset_train)
        MAE_dev, RMSE_dev, R2_dev, pearson_r_dev, spearman_r_dev, yreal_dev, ypred_dev = tester.test(dataset_dev)
        MAE_test, RMSE_test, R2_test, pearson_r_test, spearman_r_test, yreal_test, ypred_test = tester.test(dataset_test)

        end = timeit.default_timer()
        time = end - start

        MAEs = [epoch, time, rmse_train, r2_train, pearson_r_train, spearman_r_train, MAE_dev,
                MAE_test, RMSE_dev, RMSE_test, R2_dev, R2_test, pearson_r_dev, pearson_r_test, spearman_r_dev, spearman_r_test]
        ys = [yreal_test, ypred_test]
        tester.save_MAEs(MAEs, file_MAEs)
        tester.save_ys(ys, file_ys)
        tester.save_model(model, file_model)

        print('\t'.join(map(str, MAEs)))
        #print('\t'.join(map(str, ys)))
