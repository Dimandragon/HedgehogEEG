import sys
import torch
from torch import nn
import math
import mne
import os
from mne.datasets import eegbci
from mne import channels
import numpy as np
import torch.utils.data as data
import torch.nn.functional as F
from torch import optim
import time
import random
import braindecode

from braindecode.datasets import MOABBDataset, BaseConcatDataset
from numpy import multiply

sys.path.append("hedgehog-tokenizer/out/")
import extractor
import encoding
DIM_COUNT = 15
TGT_VOCAB_SIZE = 5
NUM_HEADS=5
NUM_LAYERS=4
FF_DIM = 512
DROPOUT = 0.5
N_CHANNELS = 22
SEQ_LEN = 500
MAX_EMB_LEN = 1300
device = "cuda" if torch.cuda.is_available() else "cpu"

print("-----------------------------------------------------------------")
print(device)
print("-----------------------------------------------------------------")


def weighted_loss(pred, lab):
    loss = 0.
    true_preds = torch.argmax(lab)
    sum = 0.
    loss += (1-pred[0][true_preds])/gistogram[true_preds]
    sum += 1/gistogram[true_preds]
    return loss/sum
    

def tokensFrom2DTensor(data, dim_count):
    tokenizer = extractor.InstFreqNormSincTokenizer()
    tokenizer.locality_coeff = 2
    tokenizer.period_muller = 1.1
    raw_tokens = [] # list of lists

    for channel in data:
        v = extractor.DoubleVector(len(channel))
        for i in range(len(channel)):
            #print(channel[i])
            v[i] = channel[i].item()
        tokenizer.computeVec(v)
        temp_raw_tokens = tokenizer.getTokens()
        raw_tokens.append(temp_raw_tokens)
    
    max_t = 0.0
    max_val = 0.0
    max_inst_freq = 0.0
    max_inst_ampl = 0.0
    max_phase = 0.0
    max_mode_num = 0.0 

    for raw_tokens_channels in raw_tokens:
        for raw_token in raw_tokens_channels:
            if abs(raw_token.t) * 2 > max_t:
                max_t = abs(raw_token.t) * 2
            if abs(raw_token.val) * 2 > max_val:
                max_val = abs(raw_token.val) * 2
            if abs(raw_token.inst_freq) * 2 > max_inst_freq:
                max_inst_freq = abs(raw_token.inst_freq) * 2
            if abs(raw_token.inst_ampl) * 2 > max_inst_ampl:
                max_inst_ampl = abs(raw_token.inst_ampl) * 2
            if abs(raw_token.phase) * 2 > max_phase:
                max_phase = abs(raw_token.phase) * 2 
            if abs(raw_token.mode_num) * 2 > max_mode_num:
                max_mode_num = abs(raw_token.mode_num) * 2     

    #print(max_t, max_val, max_inst_freq, max_inst_ampl, max_phase, max_mode_num)

    encoded_tokens = []

    for i in range(len(raw_tokens)):
        raw_tokens_channel = raw_tokens[i]
        for raw_token in raw_tokens_channel:
            tensor = torch.zeros(7,dim_count)
            tensor[0] = encoding.periodicalFunctionEncodding(raw_token.mode_num, max_mode_num,dim_count)
            tensor[1] = encoding.periodicalFunctionEncodding(raw_token.t, max_t,dim_count)
            tensor[2] = encoding.periodicalFunctionEncodding(raw_token.val, max_val,dim_count)
            tensor[3] = encoding.periodicalFunctionEncodding(raw_token.inst_freq, max_inst_freq,dim_count)
            tensor[4] = encoding.periodicalFunctionEncodding(raw_token.inst_ampl, max_inst_ampl,dim_count)
            tensor[5] = encoding.periodicalFunctionEncodding(raw_token.phase, max_phase,dim_count)                                
            tensor[6] = encoding.periodicalFunctionEncodding(i, N_CHANNELS, dim_count) 
            
            encoded_tokens.append(tensor.reshape(7*dim_count))
            
    return encoded_tokens

class Tokenizator():
    def __init__(self, dim_count):    
        self.dim_count = dim_count

    def forward(self, data):
        tokens = tokensFrom2DTensor(data, self.dim_count)
        #print(tokens[0].size())
        out = torch.zeros(MAX_EMB_LEN, self.dim_count*7) #(tokens_count, 7, 15)
        tokens_count = len(tokens)
        if tokens_count <= MAX_EMB_LEN:
            for i in range(0, tokens_count):
                out[i] = tokens[i]
        else:
            for i in range(0, MAX_EMB_LEN):
                out[i] = tokens[i] #torch.Size([1300, 7, 15]) 
        return out

class Transformer(nn.Module):
    def __init__(self, num_heads, num_layers, d_ff, tgt_vocab_size,dim_count):
        super(Transformer, self).__init__()
        self.d_model = 7*dim_count
        self.encoder_embedding = Tokenizator(dim_count)
        self.encoder_layer = nn.TransformerEncoderLayer(self.d_model, num_heads, d_ff, dropout = 0., activation= "gelu")

        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)
        self.pooling = nn.Linear(self.d_model, tgt_vocab_size)
        self.pooling_ = nn.Linear(MAX_EMB_LEN * tgt_vocab_size, tgt_vocab_size)
        self.dense = nn.Linear(tgt_vocab_size, tgt_vocab_size)
        self.optimizer = optim.AdamW(self.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
        self.softmax = nn.Softmax(dim = 1)
        self.mask = torch.triu(torch.ones(N_CHANNELS,self.d_model), diagonal=1)
        self.mask = self.mask.int().float().to(device)

        self.loss = torch.nn.MSELoss()

    def forward(self,src):
        embeddings = self.encoder_embedding.forward(src)     
        enc_output = self.encoder.forward(embeddings).to(device)
        fc_output = self.pooling(enc_output).to(device)
        emb_reshaped = torch.reshape(fc_output, (1,-1))
        emb_pooled = self.pooling_(emb_reshaped)
        dense_output = self.dense(emb_pooled)
        predictions = self.softmax(dense_output).to(device)
        #print(predictions.size())
        return predictions

    def weightedAccuracy(self, output, label):
        correct = 0
        pred = torch.argmax(output)
        true_pred = torch.argmax(label)
        sum = 0
        if pred == true_pred:
            correct += 1/gistogram[pred]
        sum += 1/gistogram[true_pred]    
        return correct/sum

    def training_step(self, data, lab):
        output = self.forward(data)
        loss_res = self.loss(output, lab)
        loss_res.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        is_true = False
        if torch.argmax(lab) == torch.argmax(output):
            is_true = True
        return [loss_res, is_true]

    def valid_step(self,batch):
        data,y,true_pred = batch
        output = self.forward(data)
        loss_res = self.loss(output, true_pred)
        is_true = False
        if torch.argmax(true_pred) == torch.argmax(output):
            is_true = True
        return [loss, is_true,embeddings]


def slice_to_batches(raw_data, batch_size, n_batches, n_chans):
  batch_list = []
  for b in range(n_batches):
    single_batch = []
    for i in range(n_chans):
      element = raw_data[i][(b*batch_size):((b+1)*batch_size)]
      element = element.unsqueeze(0).to(device)
      single_batch.append(element)
    tensored = torch.cat(single_batch,0).type(torch.FloatTensor).to(device)
    batch_list.append(tensored)
  return batch_list



training_set = []
validating_set = []
for id in range(1,10):
    raw_dataset = MOABBDataset(dataset_name="BNCI2014_001", subject_ids=[id])
    training_set += raw_dataset.datasets[0:8]
    validating_set += raw_dataset.datasets[8:12]

training_datasets = []
labels_batches = []
validating_datasets = []
pred_batches = []
gistogram = [0,0,0,0,0]

preds_for_one = []

for i in range(len(validating_set)):
    valid_raw = validating_set[i].raw
    raw_data = torch.from_numpy(valid_raw.get_data()).to(device)
    n_batches = raw_data.size(1)//SEQ_LEN
    validating_datasets += slice_to_batches(raw_data, SEQ_LEN, n_batches, N_CHANNELS)
    true_preds = torch.from_numpy(mne.events_from_annotations(valid_raw)[0]).to(device)
    pred_dict = {}
    counter = 0
    for j in range (n_batches):
        start = SEQ_LEN * j
        flag = False
        label = torch.zeros(1, 5)
        for id in range (start, start + SEQ_LEN):
            if (id >= raw_data.size(1)):
                break
            if (counter >= true_preds.size(1)):
                break
            if (id == true_preds[counter][0]):
                #label = labels[counter][2]
                label[0][true_preds[counter][2] - 1] = 1.0
                label[0][4] = 0.
                counter += 1

        preds_for_one.append(label)
    for pred in range(true_preds.size(0)):
        gistogram[true_preds[pred][2]-1] += 1
    gistogram[4] += (raw_data.size(1)/500 + 1)  - true_preds.size(0)
labels_for_one = []

for i in range(len(training_set)):
    train_raw = training_set[i].raw
    raw_data = torch.from_numpy(train_raw.get_data()).to(device)
    n_batches = raw_data.size(1)//SEQ_LEN
    training_datasets += slice_to_batches(raw_data, SEQ_LEN, n_batches, N_CHANNELS)
    labels = torch.from_numpy(mne.events_from_annotations(train_raw)[0]).to(device)
    counter = 0
    for j in range (n_batches):
        start = SEQ_LEN * j
        flag = False
        label = torch.zeros(1, 5)
        for id in range (start, start + SEQ_LEN):
            if (id >= raw_data.size(1)):
                break
            if (counter >= labels.size(1)):
                break
            if (id == labels[counter][0]):
                label[0][labels[counter][2] - 1] = 1.0
                label[0][4] = 0.0
                counter += 1

        labels_for_one.append(label)
    for l in range(labels.size(0)):
        gistogram[labels[l][2]-1] += 1
    gistogram[4] += (raw_data.size(1)/500+1) - labels.size(0)

gistogram_tensor = torch.tensor(gistogram).to(device)
norm_cf = 0.
normalized_list = []
for i in gistogram:
    norm_cf += 1/i
for i in gistogram:
    normalized_list.append((1/i)/norm_cf)
print(gistogram)
transformer = Transformer(NUM_HEADS,NUM_LAYERS,FF_DIM,TGT_VOCAB_SIZE,DIM_COUNT)#d_model, num_heads, num_layers, d_ff, seq_lenght, dropout,in_d,tgt_vocab_size
transformer = transformer.to(device)
torch.save(transformer, "model.onnx")
running_loss = 0
last_loss = 0
running_acc = 0
EPOCHS = 1
output = 0
start_time = time.time()
best_loss = 9999999999999999.9
epoch_loss = 0.
epoch_accuracy = 0.
model = torch.load("model.onnx")
batches_pack = []
for j in range(EPOCHS):
    running_acc = 0.
    running_loss = 0.
    for i in range(len(training_datasets)):
        transformer.train()
        loss, acc = transformer.training_step(training_datasets[i],labels_for_one[i])
        running_loss += loss.item()
        running_acc += acc
        epoch_loss += loss.item()
        epoch_accuracy += acc
        if loss < best_loss:
            best_loss = loss
            os.remove("model.onnx")
            torch.save(transformer, "model.onnx")
        if i % 200 == 0:
            last_loss = running_loss / 200.
            print("training step")
            print(f"batch {i+1} mean loss: {last_loss}, mean accuracy: {running_acc/200.}")
            running_loss = 0
            running_acc = 0
    with open("results.txt", mode = "w") as file:
        file.write(f"{epoch_loss/len(training_datasets)} {epoch_accuracy/len(training_datasets)} {best_loss}")
    print(f"Epoch {j} loss {epoch_loss/len(training_datasets)}  accuracy {epoch_accuracy/len(training_datasets)}")
    epoch_accuracy = 0
    epoch_loss = 0
valid_loss = 0
last_loss = 0
valid_acc = 0

for i in range(len(validating_datasets)):
    loss,acc,embeddings = transformer.valid_step([validating_datasets[i],embeddings, pred_batches[i]])
    valid_loss += loss.item()
    valid_acc += acc
    if i % 10 == 9:
        last_loss = valid_loss / 10 
        print("validating step")
        print(f"batch {i+1} mean loss: {last_loss}, mean accuracy: {valid_acc/10}")
        valid_loss = 0
        valid_acc = 0