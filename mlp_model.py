from datasets import *
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForTextEncoding
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from transformers import TrainingArguments, Trainer
import pandas as pd
import re
import random
from matplotlib import pyplot as plt
## load and subset data
sample_posts_all = pd.read_csv("/home/venu/projects/p8/input/reddit/askscience_data.csv",header=0)
sample_size = 100
test_size = int(sample_size*0.8)
train_size = int(sample_size*0.2)
rand_indx = random.sample(list(range(sample_posts_all.shape[0])),k=sample_size)
sample_posts = sample_posts_all.iloc[rand_indx].reset_index(drop=True)
BASE_MODEL = "google/bert_uncased_L-4_H-256_A-4"
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
model = AutoModelForTextEncoding.from_pretrained(BASE_MODEL)
truncate_to = 512
feature_list = list()
url_pattern = re.compile(r'https?://\S+|www\.\S+')
## cleanup text and generate feature vectors
for i in range(sample_posts.shape[0]):
    body = sample_posts["body"][i]
    title = sample_posts["title"][i]
    if type(body) != str:
        body = "NA"
    if type(title) != str:
        title = "NA"
    body = url_pattern.sub("",body)
    title = url_pattern.sub("",title)
    body_list = body.split("\n")
    body_list_clean = list()
    for k in range(len(body_list)):
        if body_list[k] != "":
            body_list_clean.append(body_list[k])
    body = ' '.join(body_list_clean)
    if len(body_list_clean) == 0:
        next
    encoding = tokenizer(body, return_tensors="pt", truncation=True,padding="max_length", max_length=truncate_to)

    # forward pass
    outputs = model(**encoding)

    # get feature vector 
    body_feature = outputs.last_hidden_state[:,0,:]

    encoding = tokenizer(title, return_tensors="pt", truncation=True,padding="max_length", max_length=truncate_to)

    # forward pass
    outputs = model(**encoding)

    # get feature vector 
    title_feature = outputs.last_hidden_state[:,0,:]

    feature_list.append(torch.cat((body_feature,title_feature)))
## prepare input data
X = torch.stack(feature_list)
X_test = X[0:test_size,:,:]
X_val = X[test_size+1:,:,:]
Y_flat = sample_posts["score"].values.reshape(sample_posts.shape[0],1)
Y = torch.from_numpy(Y_flat)
Y = torch.tensor(Y, dtype=torch.float)
Y_test = Y[0:test_size,:]
Y_val = Y[test_size+1:,:]
## define model structure and loss function
model = nn.Sequential(
     nn.Linear(256, 128),
     nn.ReLU(),
     nn.Linear(128, 1)
)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
n_iters = 200
trainingEpoch_loss = []
validationEpoch_loss = []
# setup training
for epoch in range(n_iters):
        # prediction = forward pass
    y_pred = model(X_test)
    y_pred = torch.mean(y_pred,dim=1)
        #loss
    l = loss_fn(Y_test, y_pred)

        #gradient = backward pass
    l.backward(retain_graph=True)

        # update weights
    optimizer.step()

        #empty gradients
    optimizer.zero_grad()

    # accuracy
    accuracy = sum(Y_test == y_pred) / len(y_pred)
    trainingEpoch_loss.append(l.detach().numpy().item())
    # validation
    y_pred = model(X_val)
    y_pred = torch.mean(y_pred,dim=1)
    l_val= loss_fn(Y_val, y_pred)
    validationEpoch_loss.append(l_val.detach().numpy().item())
    if epoch % 1 == 0:
        print(f'epoch {epoch+1}: loss = {l:.8f}: accuracy = {accuracy.numpy().item():.3f}: val_loss = {l_val:.8f}')
## plot training and validation loss
plt.plot(trainingEpoch_loss, label='train_loss')
plt.plot(validationEpoch_loss,label='val_loss')
plt.legend()
plt.savefig("/home/venu/projects/p8/output/tt_loss.png")
# save model
torch.save(model.state_dict(), "/home/venu/projects/p8/model.pt")