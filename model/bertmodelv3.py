# -*- coding: utf-8 -*-
"""Bert_v3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1dRaBph13We-66BwI_ci65yWirDCdmnIn

Import Libraries
"""

# Libraries
#originated from https://towardsdatascience.com/bert-text-classification-using-pytorch-723dfb8b6b5b
#https://towardsdatascience.com/beginners-guide-to-bert-for-multi-classification-task-92f5445c2d7c
#!pip install transformers
import matplotlib.pyplot as plt
import pandas as pd
import torch

# Preliminaries

from torchtext.data import Field, TabularDataset, BucketIterator, Iterator

# Models

import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification

# Training

import torch.optim as optim

# Evaluation

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
print("libraries imported")

#mounting google drive
#from google.colab import drive
#drive.mount('/content/gdrive')

"""Pre-Process"""

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Model parameter
MAX_SEQ_LEN = 128
PAD_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
UNK_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)
if torch.cuda.is_available():
  dev = "cuda:0"
else:
  dev = "cpu"
device1 = torch.device(dev)

# Fields

label_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)
text_field = Field(use_vocab=False, tokenize=tokenizer.encode, lower=False, include_lengths=False, batch_first=True,
                   fix_length=MAX_SEQ_LEN, pad_token=PAD_INDEX, unk_token=UNK_INDEX)
fields = [('class_id', label_field), ('tweet', text_field)]
Path="~/Hate-Speech-Detection-With-BERT/"
destination_folder=Path
Train='labeled_data_bert.csv'
Validation='bert_test.csv'
Test='anti_racist_tweets.csv'
# TabularDataset
#update the args
train, valid, test = TabularDataset.splits(path=Path, train=Train, validation=Validation,
                                           test=Train, format='CSV', fields=fields, skip_header=True)

# Iterators
#update this iterator with https://mlexplained.com/2018/02/08/a-comprehensive-tutorial-to-torchtext/
train_iter = BucketIterator(train, batch_size=16, sort_key=lambda x: len(x.tweet),
                            device=device1, train=True, sort=True, sort_within_batch=True)
valid_iter = BucketIterator(valid, batch_size=16, sort_key=lambda x: len(x.tweet),
                            device=device1, train=True, sort=True, sort_within_batch=True)
test_iter = Iterator(test, batch_size=16, device=device1, train=False, shuffle=False, sort=False)
print("working")
"""Step 3: Build Model"""
#num_labels=3
class BERT(nn.Module):

    def __init__(self):
        #self.num_labels=3
        super(BERT, self).__init__()
        #self.num_labels=3

        options_name = "bert-base-uncased"
        self.encoder = BertForSequenceClassification.from_pretrained(options_name, num_labels=3)

    def forward(self, text, label):
        loss, text_fea = self.encoder(text, labels=label)

        return loss, text_fea



"""Save and Load Functions"""

# Save and Load Functions

def save_checkpoint(save_path, model, valid_loss):

    if save_path == None:
        return

    state_dict = {'model_state_dict': model.state_dict(),
                  'valid_loss': valid_loss}

    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')

def load_checkpoint(load_path, model):

    if load_path==None:
        return

    state_dict = torch.load(load_path, map_location=device1)
    print(f'Model loaded from <== {load_path}')

    model.load_state_dict(state_dict['model_state_dict'])
    return state_dict['valid_loss']


def save_metrics(save_path, train_loss_list, valid_loss_list, global_steps_list):

    if save_path == None:
        return

    state_dict = {'train_loss_list': train_loss_list,
                  'valid_loss_list': valid_loss_list,
                  'global_steps_list': global_steps_list}

    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


def load_metrics(load_path):

    if load_path==None:
        return

    state_dict = torch.load(load_path, map_location=device1)
    print(f'Model loaded from <== {load_path}')

    return state_dict['train_loss_list'], state_dict['valid_loss_list'], state_dict['global_steps_list']

"""Train"""

# Training Function

def train(model,
          optimizer,
          criterion = nn.BCELoss(),
          train_loader = train_iter,
          valid_loader = valid_iter,
          num_epochs = 30,
          eval_every = len(train_iter) // 2,
          file_path = destination_folder,
          best_valid_loss = float("Inf")):

    # initialize running values
    running_loss = 0.0
    valid_running_loss = 0.0
    global_step = 0
    train_loss_list = []
    valid_loss_list = []
    global_steps_list = []

    # training loop
    model.train()
    for epoch in range(num_epochs):
        for (class_id, tweet), _ in train_loader:
            #print("Yes")
            labels = class_id.type(torch.LongTensor)
           #print(len(labels))
            labels = labels.to(device1)
            text = tweet.type(torch.LongTensor)
            text = text.to(device1)
            output = model(text, labels)
            loss, _ = output

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update running values
            running_loss += loss.item()
            global_step += 1

            # evaluation step
            if global_step % eval_every == 0:
                model.eval()
                with torch.no_grad():

                    # validation loop
                    for (labels, text), _ in valid_loader:
                        labels = labels.type(torch.LongTensor)
                        labels = labels.to(device1)
                        text = text.type(torch.LongTensor)
                        text = text.to(device1)
                        output = model(text, labels)
                        loss, _ = output

                        valid_running_loss += loss.item()

                # evaluation
                average_train_loss = running_loss / eval_every
                average_valid_loss = valid_running_loss / len(valid_loader)
                train_loss_list.append(average_train_loss)
                valid_loss_list.append(average_valid_loss)
                global_steps_list.append(global_step)

                # resetting running values
                running_loss = 0.0
                valid_running_loss = 0.0
                model.train()

                # print progress
                print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}'
                      .format(epoch+1, num_epochs, global_step, num_epochs*len(train_loader),
                              average_train_loss, average_valid_loss))

                # checkpoint
                if best_valid_loss > average_valid_loss:
                    best_valid_loss = average_valid_loss
                    save_checkpoint('modelgpu.pt', model, best_valid_loss)
                    save_metrics('metricsgpu.pt', train_loss_list, valid_loss_list, global_steps_list)

    save_metrics('metricsgpu.pt', train_loss_list, valid_loss_list, global_steps_list)
    print('Finished Training!')
print("good before model")
model = BERT().to(device1)
print("issue is with optimizer")
lr=2e-5
optimizer = optim.Adam(model.parameters(), lr=lr)
print("good pre-train")
train(model=model, optimizer=optimizer)
print("training good")
train_loss_list, valid_loss_list, global_steps_list = load_metrics('metricsgpu.pt')
plt.plot(global_steps_list, train_loss_list, label='Train')
plt.plot(global_steps_list, valid_loss_list, label='Valid')
plt.xlabel('Global Steps')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Evaluation Function

def evaluate(model, test_loader):
    y_pred = []
    y_true = []


    model.eval()
    with torch.no_grad():
        for (labels, text), _ in test_loader:

                labels = labels.type(torch.LongTensor)
                labels = labels.to(device1)
                text = text.type(torch.LongTensor)
                text = text.to(device1)
                output = model(text, labels)

                _, output = output
                y_pred.extend(torch.argmax(output, 1).tolist())
                y_true.extend(labels.tolist())

    print('Classification Report:')
    print(y_true, y_pred)
    print(classification_report(y_true, y_pred, labels=[1,0], digits=4))

    cm = confusion_matrix(y_true, y_pred, labels=[1,0])
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax, cmap='Blues', fmt="d")

    ax.set_title('Confusion Matrix')

    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')

    ax.xaxis.set_ticklabels(['FAKE', 'REAL'])
    ax.yaxis.set_ticklabels(['FAKE', 'REAL'])

best_model = BERT().to(device1)
print("MOdel Loaded")
load_checkpoint('modelgpu.pt', best_model)
print("starting model evaluation")
evaluate(best_model, test_iter)
print("done")
print("testing with outstanding data")
import csv

with open('anti_racist_tweets.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)
    for tweet in data:
        print(data[tweet], best_model(tweet))
