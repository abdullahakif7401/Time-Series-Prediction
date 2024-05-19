#!/usr/bin/env python
# coding: utf-8

# In[40]:


# from google.colab import drive
# drive.mount('/content/drive')


# In[41]:


# import os
# os.chdir('/content/drive/MyDrive/FIT3162')


# In[42]:


# ls


# In[43]:


# Try on Proshphet model from facebook to train and predict


# In[44]:


import json
import os
import matplotlib.pyplot as plt
import numpy as np

# In[45]:


# pip install pyts


# In[46]:


# pip install torch numpy pandas scikit-learn mlxtend pyts


# 

# In[47]:


# Preprocess the dataset

# Preprocess the data
import torch
import numpy as np;
from torch.autograd import Variable
from pyts.approximation import SymbolicAggregateApproximation
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from mlxtend.frequent_patterns import apriori, association_rules


def normalise_sd(x):
    # x.std() standard deviation: measure of the spread or variability of the
    # data. The standard deviation is computed by taking the square root of the
    # average squared deviation from the mean
    # This factor adjusts the standard deviation to account for the fact that
    # the sample mean is not the same as the population mean, and it is based
    # on the degrees of freedom of the sample, which is one less than the sample size
    # Bessel's correction, adjust the sd by mulpliying with square of (n-1)/n
    # important when sample size is small -> provide better estimate of the population sd
    # correction makes the sd slighly larger as compensating the fact
    # that sample variance typically understimate of the population variance for small sample
    return x.std() * np.sqrt((len(x) - 1) / len(x))


class Data_util(object):
    # train and valid is the ratio of training set and validation set. test = 1 - train - valid
    # ntp: next token prediciton
    # re: rolling evaluation
    def __init__(self, file_name, train, valid, cuda, ntp, re, normalise=2):
        self.cuda_is_available = cuda
        self.re = re
        self.ntp = ntp
        data = open(file_name);
        # load txt file to be file object
        # separate dat by ,
        self.rawdat = np.loadtxt(data, delimiter=',')
        data.close()
        # perform arm on data
        self._arm(5, 0.7, 0.7, 50)
        self.dat = np.zeros(self.rawdat.shape)
        self.n, self.m = self.dat.shape;
        self.normalise = 2
        self.scale = np.ones(self.m)
        # _for private
        self._normalised(normalise)
        # 0.6 * whole dataset size -> number of rows to train
        # 0.8 end index for the rows for valid
        self._split(int(train * self.n), int((train + valid) * self.n), self.n)
        # tensor from numpy array
        self.scale = torch.from_numpy(self.scale).float()
        # reshape the dimension of the original data
        # with row_count = test[1].size(0) and self.m =
        tmp = self.test[1] * self.scale.expand(self.test[1].size(0), self.m)

        if self.cuda_is_available:
            self.scale = self.scale.cuda()  # move scaling parameters into GPU from CPU
            # wrap the tensor for pytorch to track the history of operations to th
            # this scaling tensor for auto-diffentiation
        self.scale = Variable(self.scale)
        # self.rse = normal_std(tmp); calculates the root squared error (RSE) of the model output tmp
        self.rse = normalise_sd(tmp)
        # calculates the relative absolute error (RAE) of the model output tmp
        self.rae = torch.mean(torch.abs(tmp - torch.mean(tmp)))

    def _arm(self, n_bins, min_support, min_threshold, n_rules):
        data = self.rawdat
        df = pd.DataFrame(data)
        scaler = MinMaxScaler()
        norm_data = scaler.fit_transform(df)
        sax = SymbolicAggregateApproximation(n_bins=n_bins, strategy='uniform')
        sax_df = pd.DataFrame(sax.fit_transform(norm_data))
        binary_sax_df = pd.get_dummies(sax_df)
        frequent_itemsets = apriori(binary_sax_df, min_support=min_support, use_colnames=True)
        if len(frequent_itemsets) == 0:
            self.cols = list(range(df.shape[1]))
            return
        rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=min_threshold)
        rules = rules.sort_values(by=['zhangs_metric'], ascending=False).iloc[:n_rules, :]
        unique_items = set()
        for index, row in rules.iterrows():
            unique_items.update([int(item[:-2]) for item in row['antecedents']])
            unique_items.update([int(item[:-2]) for item in row['consequents']])
        self.cols = list(unique_items)
        df = df.iloc[:, self.cols]
        self.rawdat = df.values

    def _normalised(self, normalise):
        # normalised by the max value of entire matrix
        if (normalise == 0):
            self.dat = self.rawdat
        elif (normalise == 1):
            self.dat = self.rawdat / np.max(self.rawdat)
        elif (normalise == 2):
            # normalised by the max value of each row
            for i in range(self.m):
                # list of scaling values for each of the data
                self.scale[i] = np.max(np.abs(self.rawdat[:, 1]))
                # normalise the value in individual column based on the largest value in column
                # some column largest values might be negative
                self.dat[:, i] = self.rawdat[:, i] / np.max(np.abs(self.rawdat[:, i]))

    def _split(self, train, valid, test):
        # leave for the self.re: rolling_evaluation (validation)
        # self.ntp: next few tokens prediction
        # 0.6
        train_set = range(self.re + self.ntp - 1, train);  # save the front valus for re and ntp
        valid_set = range(train, valid);  # 0.6 - 0.8
        test_set = range(valid, self.n);  # remaining for text 0.2, 0.8 - 1.0
        # train dataset
        self.train = self._batchify(train_set, self.ntp);
        self.valid = self._batchify(valid_set, self.ntp);
        self.test = self._batchify(test_set, self.ntp);

    # self.train = self._batchify(train_set, self.ntp);
    def _batchify(self, idx_set, ntp):  # horizon for next prediciton
        # number of samples in one batch
        # index set for dataset (train_set if train_set passed)
        # each dataset has a n
        n = len(idx_set)  # rolling_evaluation , size of the input column
        X = torch.zeros((n, self.re, self.m))
        Y = torch.zeros((n, self.m))

        for i in range(n):  # end: start of the next token prediction
            # for each row/entry set the region for the training
            # [train] + [rolling_evaluation]
            end = idx_set[i] - self.ntp + 1  # save the last few token/value for next token prediciton
            # start: start of rolling evaluation
            start = end - self.re  # save the values infront values save for self.ntp to do rolling_evalution
            # slice the input data for training
            # by slicing each sample/entry/row into X
            # between start and end to be the training dataset

            # create a PyTorch tensor 'X' with 5 batches, each containing a slice of 20 rows from 'data'
            # X = torch.empty(5, 20, 10)  # Pre-initialize X with the desired shape
            # X will contain 5 separate slices from data

            # slice the many rows of data except for
            X[i, :, :] = torch.from_numpy(self.dat[start:end, :])
            # for batching multiple rows of data together to train from start to end
            # his line is assigning a 2D slice of the numpy array self.dat to the i-th
            # index in the first dimension of the tensor X
            # start:end indicates a range of rows, so you’re selecting multiple rows and all columns within that range.
            # This line is assigning a 1D slice (a single row) of the numpy array self.dat
            # to the i-th index in the first dimension of the tensor Y
            Y[i, :] = torch.from_numpy(self.dat[idx_set[i], :]);
            # idx_set[i] is an index for a specific row, so you’re selecting just that row and all columns in it.
            # The result is that Y[i,:] will be a 1D tensor with the same number of elements as there are columns in self.dat.

        return [X, Y]

        # train_loss = train(Data, x, y, model, criterion, optim, args.batch_size)
        # data.get_batches(X,Y, batch_size, True):

    def get_batches(self, inputs, targets, batch_size, shuffle=True):
        # get_batches used in training loop to iterate over the generator to ge the batche of data
        # for each training step
        length = len(inputs)
        if shuffle:  # permutation to shuffle the whole row/entry, so different first value in en
            # creates a tensor named index that contains a random permutation of
            # integers from 0 to length-1. Then shuffle the valus according the random valus
            index = torch.randperm(
                length)  # randperm uses a Fisher-Yates shuffle algorithm to create a random permutation of numbers
        else:  # create a long type tensor with original arrangement
            index = torch.LongTensor(range(length))
        start_idx = 0
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            data_idx = index[start_idx:end_idx]
            X = inputs[data_idx]
            Y = targets[data_idx]
            if (self.cuda_is_available):
                X = X.cuda()
                Y = Y.cuda()
                # return multiple values in generator-level fashion
                # Data.train[1] = Variable(X)
            yield Variable(X), Variable(Y)
            start_idx += batch_size


# In[48]:


# python main.py --gpu 3 --horizon 24 --data data/electricity.txt --save save/elec.pt --output_fun Linear
# args = parser.parse_args()
# Data = Data_util(args.data, 0.6, 0.2, args.cuda, args.horizon, args.window, args.normalise);

# In[49]:


def evaluate(data, X, Y, model, evaluateL2, evaluateL1, batch_size):
    model.eval()
    total_loss = 0
    total_loss_l1 = 0
    n_samples = 0
    predict = None
    test = None

    for X, Y in data.get_batches(X, Y, batch_size, False):
        output = model(X)
        # predict can be chanegd durring the for loop
        if predict is None:
            predict = output;
            test = Y;
        else:
            predict = torch.cat((predict, output))
            test = torch.cat((test, Y))

        scale = data.scale.expand(output.size(0), data.m)
        # computer L2 loss = mean squared error (MSE)
        # .data[0] contain the loss value
        # scale to bring back the value if they were normalised
        # total_loss += evaluateL2(output * scale, Y * scale).data[0]
        # .item() to get the loss value
        total_loss += evaluateL2(output * scale, Y * scale).item()
        total_loss_l1 += evaluateL1(output * scale, Y * scale).item()
        # output.size(0) = batch_size * number of columns in the dataset
        n_samples += (output.size(0) * data.m)
    # rse: Root relative squared error: predictive accuracy of a model in statistics and machine learning
    #   = mse/ population sd +-= (nomalised sample sd)
    rse = math.sqrt(total_loss / n_samples) / data.rse
    rae = (total_loss_l1 / n_samples) / data.rae

    # converts a PyTorch tensor to a NumPy array
    # safe way:
    # predict = predict.detach().cpu().numpy()
    # can lead to potential issues with the computation graph and gradient tracking.
    predict = predict.data.cpu().numpy()
    # make the numpy array to refer the memory locaiton of the tensor
    # change to numpy array -. from tensor
    # singma_p contains the sd of the prediciton for partical feature
    # across all samples -. for understanding the variability of the model's prediction
    # for each deature
    # In a machine learning context, this operation is often performed after making
    # predictions with a model to analyze the consistency of the predictions.
    # A lower standard deviation indicates that the model’s predictions for that
    # feature are more consistent, while a higher standard deviation indicates
    # greater variability.
    Ytest = test.data.cpu().numpy()
    sd_p = predict.std(axis=0)
    sd_g = Ytest.std(axis=0)
    # calculate the mean for each column (across the rows)
    # Ytest = np.array([[1, 4],
    # [2, 5],
    # [3, 6]])
    # print(mean_g)  # Output: [2. 5.]
    mean_p = predict.mean(axis=0)
    mean_g = Ytest.mean(axis=0)
    # True if the corresponding element in sigma_g is not equal to zero.
    # filter out the columns/features with sd = 0  to avoid divison by 0 problem
    # in subsequent correlation function
    # output a boolean index array with true/false
    # index: boolean array
    index = (sd_g != 0)
    # calculate the correlation for each column
    correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis=0) / (sd_p * sd_g)
    # correlation for the column that has index with true in boolean index array
    correlation = (correlation[index]).mean()
    return rse, rae, correlation


# In[50]:


import argparse

# argparse to parse the input argument which take a path
parser = argparse.ArgumentParser(description='Pytorch Time series forecasting')
# parser.add_argument('--data', type = str, required=True, help='location of the data file')
# parser.add_argument('--ntp', type=int, default=12)
# args = parser.parse_args()


import torch
import torch.nn as nn
import numpy as np


class Gate(nn.Module):
    def __init__(self, input_size, output_size):
        super(Gate, self).__init__()
        # Initialize weights and bias for the gate
        self.W = nn.Parameter(torch.randn(output_size * 2, output_size))
        self.b = nn.Parameter(torch.zeros(output_size, 1))
        # Bottleneck transformation layer [321, 128]
        self.bottleneck = nn.Linear(input_size, output_size)
        self.bn_concat = nn.Linear(input_size, output_size * 2)

    def forward(self, x_t, h_prev):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x_t = x_t.to(device)
        h_prev = h_prev.to(device)
        # print(x_t.shape)
        # Transform x_t to the correct size using the bottleneck layer
        x_t_transformed = self.bottleneck(x_t)
        # print(x_t_transformed.shape)
        # Concatenate transformed input and previous hidden state along the feature dimension
        concat = torch.cat((x_t_transformed.t(), h_prev), dim=1)
        # print(concat.shape)
        # Compute the gate's output
        # torch.matmul = dot product

        z = torch.matmul(concat, self.W) + self.b
        gate_output = torch.sigmoid(z)

        return gate_output

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))


class ForgetGate(Gate):
    def __init__(self, input_size, output_size):
        super().__init__(input_size, output_size)

    def forward(self, x_t, h_prev):
        f_t = super().forward(x_t, h_prev)
        return f_t


class InputGate(Gate):
    def __init__(self, input_size, output_size):
        super().__init__(input_size, output_size)

        # Control mu embedding
        self.w_C = nn.Parameter(torch.randn(output_size * 2, output_size))
        self.b_C = nn.Parameter(torch.zeros(output_size, 1))

    def control_forward(self, x_t, h_prev):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print(x_t.shape)
        x_t_transformed = self.bottleneck(x_t).to(device)
        # print(x_t_transformed.shape)
        h_prev = h_prev.to(device)
        concat = torch.cat((x_t_transformed, h_prev), dim=1)

        # Use matmul for matrix multiplication
        temp = torch.matmul(concat, self.w_C)
        # print(temp.shape)
        C_mu = torch.tanh(temp + self.b_C)
        return C_mu

    def forward(self, x_t, h_prev):
        i_t = super().forward(x_t, h_prev)
        C_t = self.control_forward(x_t, h_prev)
        return i_t, C_t


class OutputGate(Gate):
    def __init__(self, input_size, output_size):
        super().__init__(input_size, output_size)

    def forward(self, x_t, h_prev):
        o_t = super().forward(x_t, h_prev)
        return o_t


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # Bottleneck transformation layer
        self.bottleneck = nn.Linear(input_size + hidden_size, hidden_size)

        # Initialisise gates
        self.input_gate = InputGate(input_size, hidden_size)
        self.forget_gate = ForgetGate(input_size, hidden_size)
        self.output_gate = OutputGate(input_size, hidden_size)
        self.cell_state = nn.Parameter(torch.zeros(hidden_size, 1))

    def forward(self, x_t, h_prev, c_prev):
        # Move all tensors to the same device as the model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x_t, h_prev, c_prev = x_t.to(device), h_prev.to(device), c_prev.to(device)

        i_t, c_mu = self.input_gate.forward(x_t, h_prev)
        f_t = self.forget_gate.forward(x_t, h_prev)
        o_t = self.output_gate.forward(x_t, h_prev)

        c_t = f_t * c_prev + i_t * c_mu

        # Compute the current hidden state
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128):
        # = super(LSTM,self).__init__()

        super().__init__()
        self.hidden_size = hidden_size
        self.lstm_cell = LSTMCell(input_size, hidden_size)
        self.inver_bottleneck = nn.Linear(hidden_size, input_size)

    def forward(self, batch_size, input_sequence):
        # Initialize hidden state and cell state for each sequence in the batch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        h_t = torch.zeros(batch_size, self.hidden_size).to(device)
        c_t = torch.zeros(batch_size, self.hidden_size).to(device)

        for t in range(len(input_sequence)):
            x_t = input_sequence[:, t, :].to(device)
            # print(x_t.shape)
            h_t, c_t = self.lstm_cell(x_t, h_t, c_t)

        return h_t, self.inver_bottleneck(c_t)


# Train
import math
import time
import torch
import torch.nn as nn
import numpy as np;
import importlib


def train(data, X, Y, model, criterion, optim, batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # set model to training mode
    model.train()
    total_loss = 0
    n_samples = 0
    for X, Y in data.get_batches(X, Y, batch_size, True):
        # reset the model parameters before training
        model.zero_grad()
        output = model(batch_size, X)
        output = output[1]
        # expnad the scaling matrix into the of output
        # .size() method on the output tensor to get its size.
        # The arguments (0, data.m) indicate that you want to expand data.scale to
        # have the same size as the first dimension of output and the size
        # of data.m for the second dimension.

        # By expanding data.scale, you can ensure that it has the same size as output
        # for broadcasting purposes, which is often needed in operations like element-
        # wise multiplication or addition.
        data.scale = data.scale
        scale = data.scale.expand(output.shape[0], data.m)
        # print(scale.shape)
        # criterion: loss function measure difference between the predicted outputs and the true values.
        Y = Y
        # print(output.shape)
        loss = criterion(output * scale, Y * scale)
        loss.backward()
        # gradient clipping to prevent gradient explosion or diminishing
        # grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        # step function to call the model to update the parameters based on the computed gradients.
        grad_norm = optim.step()
        total_loss += loss.item()  ## Extract the loss value as a Python float
        # number of samples = first dimension size of the output * dataset second dimension
        # calculates the total number of elements in the current batch by multiplying the
        # batch size by the number of features or time steps. This product is then added
        # to the n_samples variable, which accumulates the total number of elements processed
        # over multiple batches
        n_samples += (output.size(0) * data.m)
    return total_loss / n_samples


import torch
import torch.nn as nn

##Optimasation
import math
import torch.optim as optim


class Optim(object):
    def _makeOptimizer(self):
        if self.method == 'sgd':
            self.optimizer = optim.SGD(self.params, lr=self.lr)
        elif self.method == 'adagrad':
            self.optimizer = optim.Adagrad(self.params, lr=self.lr)
        elif self.method == 'adadelta':
            self.optimizer = optim.Adadelta(self.params, lr=self.lr)
        elif self.method == 'adam':
            self.optimizer = optim.Adam(self.params, lr=self.lr)
        else:
            raise RuntimeError("Invalid optim method: " + self.method)

    def __init__(self, params, method, lr, max_grad_norm, lr_decay=1, start_decay_at=None):
        self.params = list(params)  # params may be a generator
        self.lr = lr
        # max_grad_norm maximum normalise gradient allowed before clipping
        self.max_grad_norm = max_grad_norm
        self.method = method
        self.lr_decay = lr_decay
        self.start_decay_at = start_decay_at
        self.start_decay = False

        self._makeOptimizer()

    def step(self):
        # computer gradients norm
        grad_norm = 0
        for param in self.params:
            # etrieves the gradient data for a parameter of the neural network
            # norm() squares the L2 norm (Euclidean norm) of the gradient tensor.
            grad_norm += math.pow(param.grad.data.norm(), 2)  # since 2
            # accumulates the sum of the squared norms of all parameters’ gradients.
        grad_norm = math.sqrt(grad_norm)
        # squre root to get the overal L2 norm

        if grad_norm > 0:
            shrinkage = self.max_grad_norm / grad_norm
        else:  # 1.: float-point number
            shrinkage = 1.

        for param in self.params:
            # if meet the threshold, apply gradient cliping
            if shrinkage < 1:
                # apply gradient clipping for each parameter's gradient
                param.grad.data.mul_(shrinkage)

        self.optimizer.step()
        return grad_norm

    # decay learning rate if not improve on val perf
    # or change start_decay_limit to true
    def upadateLearningRate(self, ppl, epoch):
        # decide which epoch to start decaying the learning rate
        if self.start_decay_at is not None and epoch >= self.start_decay_at:
            self.start_decay = True
            # stores the perplexity value from the last evaluation
            # ppl: Perplexity (PPL): It is a metric used to evaluate language models.
            # It’s defined as the exponentiated average negative log-likelihood of a sequence. The lower the perplexity, the better the model is at predicting the sequence1.
        if self.last_ppl is not None and ppl > self.last_ppl:
            self.start_decay = True

        if self.start_decay:
            self.lr = self.lr * self.lr_decay
            print("Decaying learning rate to %g" % self.lr)

        # only decay for one epoch
        self.start_decay = False

        self.last_ppl = ppl

        self._makeOptimizer()


# model = Model(args,Data)
# optim = Optim(
#     model.parameters(), args.optim, args.lr, args.clip,
# )


# optim = Optim(
#     model.parameters(), args.optim, args.lr, args.clip,
# )


def train_and_evaluate(model, data, args, train_func, evaluate_func):
    best_val = 10000000;
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # move everything to the same device - GPU
    if torch.cuda.is_available():
        model = model.cuda()
        # moving X to GPU
        data.train[0] = data.train[0].cuda()
        data.train[1].cuda()

        if not args.cuda:
            print("WARNING, have gpu, should run with --cuda")
        else:
            torch.cuda.manual_seed(args.seed)

    if args.L1Loss:
        # L1 loss = MAE (Mean Absolute Error) loss  mean of absolute difference
        # between target value and predictions

        # criterion = nn.L1Loss(size_average=False)
        # average = loss -> losses are summed
        criterion = nn.L1Loss(reduction='sum')


    else:
        criterion = nn.MSELoss(reduction='sum')

        # set up L2 loss - MSE loss during validation or testing
    evaluateL2 = nn.MSELoss(reduction='sum')
    evaluateL1 = nn.L1Loss(reduction='sum')

    # .cuda(): This method transfers the loss function to the GPU.
    if args.cuda:
        criterion = criterion.cuda()
        evaluateL1 = evaluateL1.cuda()
        evaluateL2 = evaluateL2.cuda()

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # At any point you can hit Ctrl + C to break out of training early.
    try:
        print('begin training');
        # 'Namespace' object has no attribute 'epochs' = args.epoch is not defined
        model = model.to(device)
        for epoch in range(1, args.epochs + 1):
            epoch_start_time = time.time()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            X = data.train[0].to(device)
            Y = data.train[1].to(device)

            for x, y in data.get_batches(X, Y, args.batch_size, True):
                if len(x) < args.batch_size:
                    # overlapping batch_size, sequence length, feature_size/sensor num

                    # print(input_sequence.size(0)) [128, 168, 321]

                    padding_size = args.batch_size - len(x)
                    # Create a tensor of zeros for padding

                    x_padding = torch.zeros(padding_size, x.size(1), x.size(2)).to(device)
                    y_padding = torch.zeros(padding_size, x.size(2)).to(device)
                    # print(padding.size())
                    # Concatenate the padding to the x
                    x = torch.cat((x, x_padding), dim=0)
                    # Concatenate the padding to the x
                    y = torch.cat((y, y_padding), dim=0)
                # reset the model parameters before training
                model.zero_grad()
                hidden_state, output = model(args.batch_size, x)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch}, Loss: {loss.item()}")

            if epoch % 2 == 0:
                X_val = data.valid[0].to(device)
                Y_val = data.valid[1].to(device)
                for x_val, y_val in data.get_batches(X_val, Y_val, args.batch_size, True):
                    if len(x_val) < args.batch_size:
                        val_padding_size = args.batch_size - len(x_val)
                        xval_padding = torch.zeros(val_padding_size, x_val.size(1), x_val.size(2)).to(device)
                        yval_padding = torch.zeros(val_padding_size, x_val.size(2)).to(device)
                        # print(padding.size())
                        # Concatenate the padding to the x
                        x_val = torch.cat((x_val, xval_padding), dim=0)
                        # Concatenate the padding to the x
                        y_val = torch.cat((y_val, yval_padding), dim=0)

                    hidden_state, valid_output = model(args.batch_size, x_val)
                    val_loss = criterion(valid_output, y_val)

                    if val_loss.item() < best_val:
                        with open(args.save, 'wb') as f:
                            torch.save(model, f)
                        best_val = val_loss.item()
                # torch.device('cuda:0') for the first GPU or torch.device('cpu') for the CPU1.
                # then .to(device) would go to either CPU or GPU
                # .cude(1) go to first GPU
                # model.to('cuda')

                print(f"Validation: Epoch {epoch}, Loss: {val_loss.item()}")
                if val_loss.item() < best_val:
                    with open(args.save, 'wb') as f:
                        torch.save(model, f)
                    best_val = val_loss.item()
            # if epoch % 5 == 0:
            #   #torch.device('cuda:0') for the first GPU or torch.device('cpu') for the CPU1.
            #   #then .to(device) would go to either CPU or GPU
            #   #.cude(1) go to first GPU
            #   # model.to('cuda')
            #   test_acc, test_rae, test_corr  = evaluate(Data, x_test, y_test, model, evaluateL2, evaluateL1, args.batch_size);
            #   print ("test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f}".format(test_acc, test_rae, test_corr))

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    # test_acc, test_rae, test_corr  = evaluate(Data, x_test, y_test, model, evaluateL2, evaluateL1, args.batch_size);
    # print ("test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f}".format(test_acc, test_rae, test_corr))


# train_and_evaluate(model, Data, args, train, evaluate)


class PredData_util:
    def __init__(self, file_name, ntp, re, cols, normalise=2):
        self.re = re
        self.ntp = ntp
        data = open(file_name)
        self.rawdat = pd.DataFrame(np.loadtxt(data, delimiter=',')).iloc[:, cols].values
        data.close()
        self.dat = np.zeros(self.rawdat.shape)
        self.n, self.m = self.dat.shape
        self.scale = np.ones(self.m)
        self._normalised(normalise)
        self._batchify(range(self.re + self.ntp - 1, self.n))

    def _normalised(self, normalise):
        if normalise == 0:
            self.dat = self.rawdat
        elif normalise == 1:
            self.dat = self.rawdat / np.max(self.rawdat)
        elif normalise == 2:
            for i in range(self.m):
                self.scale[i] = np.max(np.abs(self.rawdat[:, 1]))
                self.dat[:, i] = self.rawdat[:, i] / np.max(np.abs(self.rawdat[:, i]))

    def _batchify(self, idx_set):
        n = len(idx_set)
        X = torch.zeros((n, self.re, self.m))
        Y = torch.zeros((n, self.m))

        for i in range(n):
            end = idx_set[i] - self.ntp + 1
            start = end - self.re
            X[i, :, :] = torch.from_numpy(self.dat[start:end, :])
            Y[i, :] = torch.from_numpy(self.dat[idx_set[i], :])

        self.x = X
        self.y = Y


def model_pred(model, args, data, X, Y):
    try:
        for x, y in data.get_batches(X, Y, args.batch_size, True):
            with torch.no_grad():
                if len(x) < args.batch_size:
                    # overlapping batch_size, sequence length, feature_size/sensor num

                    # print(input_sequence.size(0)) [128, 168, 321]

                    padding_size = args.batch_size - len(x)
                    # Create a tensor of zeros for padding
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    x_padding = torch.zeros(padding_size, x.size(1), x.size(2)).to(device)
                    y_padding = torch.zeros(padding_size, x.size(2)).to(device)
                    # print(padding.size())
                    # Concatenate the padding to the x
                    x = torch.cat((x, x_padding), dim=0)
                    # Concatenate the padding to the x
                    y = torch.cat((y, y_padding), dim=0)
                    hidden_state, output = model(args.batch_size, x)
                    # print(output)
                    return hidden_state, output
    except Exception as ex:
        print(ex)


# !/usr/bin/env python
# coding: utf-8

# (Rest of your existing code...)

def main(file_path_traing, file_path_pred):
    # Initialize and configure the arguments
    args = argparse.Namespace()
    args.data = file_path_traing
    args.window = 24 * 7
    args.hidRNN = 100
    args.hidCNN = 100
    args.hidSkip = 5
    args.CNN_kernel = 6
    args.skip = 24
    args.gpu = 1
    args.cuda = False  # Set to True if you want to use GPU
    args.highway_window = 24
    args.dropout = 0.2
    args.output_fun = 'sigmoid'
    args.model = 'LSTNet'
    args.batch_size = 128
    args.seed = 54321
    args.L1Loss = True
    args.optim = 'adam'
    args.lr = 0.01
    args.clip = 10
    args.epochs = 2
    args.save = 'model/model.pt'
    args.horizon = 24
    args.pred_data = file_path_pred

    # Ensure the predictions folder exists
    predictions_folder = os.path.join(os.getcwd(), 'predictions_folder')
    if not os.path.exists(predictions_folder):
        os.makedirs(predictions_folder)

    Data = Data_util(args.data, 0.6, 0.2, args.cuda, args.horizon, args.window, 2)

    model = LSTM((len(Data.rawdat[0])))
    # optim = Optim(model.parameters(), args.optim, args.lr, args.clip)

    train_and_evaluate(model, Data, args, train, evaluate)

    # # Predict using the trained model
    # with open(args.save, 'rb') as f:
    #     model = torch.load(f)
    PredData = PredData_util(args.pred_data, Data.ntp, Data.re, Data.cols, Data.normalise)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = PredData.x.to(device)
    Y = PredData.y.to(device)

    hidden_state, output = model_pred(model, args, Data, X, Y)
    output = output.cpu().numpy()

    # Save the output predictions to a JSON file
    predictions_file_path = os.path.join(predictions_folder, 'predictions.json')
    with open(predictions_file_path, 'w') as f:
        json.dump(output.tolist(), f)

    return output


if __name__ == "__main__":
    import sys

    main(sys.argv[1])

# # In[58]:


# # Load the best saved model.
# #output = None
# with open(args.save, 'rb') as f:
#   model = torch.load(f)
#   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#   X = Data.test[0].to(device)
#   Y = Data.test[1].to(device)

#   hidden_state, output = model_pred(Data,X,Y)
#   output = output.cpu()
#   output = output.numpy()
#   print(output)

#   for i in range(len(output)):
#     for j in range(len(output[0])):
#       print(output[0][j])
