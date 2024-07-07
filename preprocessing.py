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

sys.path.append("hedgehog-tokenizer/linux_x86_64/")
import extractor
import encoding

TGT_VOCAB_SIZE = 5
DIM=64
NUM_HEADS=4
NUM_LAYERS=4
FF_DIM = 64
DROPOUT = 0.5
N_CHANNELS = 22
SEQ_LEN = 500
device = "cuda" if torch.cuda.is_available() else "cpu"

print("-----------------------------------------------------------------")
print(device)
print("-----------------------------------------------------------------")


def weighted_loss(pred, lab):
    loss = 0.
    all_points = pred.size(0)
    true_preds = torch.argmax(lab, dim = 1)
    sum = 0.
    for i in range(all_points):
        loss += (1-pred[i][true_preds[i]])/gistogram[true_preds[i]]
        sum += 1/gistogram[true_preds[i]]
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
            tensor[6] = i
            encoded_tokens.append(tensor)
    return encoded_tokens



class PositionalEncoding(nn.Module):
    def __init__(self, seq_length, model_dim):
        super(PositionalEncoding,self).__init__()
        self.sl = seq_length
        self.md = model_dim
        self.encodings_matrices = torch.zeros(self.sl, self.md).to(device)
        position = torch.arange(0,self.sl,dtype = torch.float).unsqueeze(1).to(device)
        for p in range(self.sl):
          for i in range(self.md):
            d = 2*seq_length/math.pi
            t = p/(d**(2*i/self.md))
            if i%2 == 0:
                self.encodings_matrices[p,i] = math.cos(t)
            else:
                self.encodings_matrices[p,i] = math.sin(t)
        self.copy = torch.clone(self.encodings_matrices)
    def forward(self,x):
      output = torch.cat((x,self.copy),1).to(device)
      return output

class Tokenizator():
    def __init__(self, dim_count):    
        self.dim_count = dim_count

    def forward(self, data):
        tokens = tokensFrom2DTensor(data, self.dim_count)
        #todo
        return torch.FloatTensor(tokens)

class Transformer(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, d_ff, seq_lenght,in_d,tgt_vocab_size,dim_count):
        super(Transformer, self).__init__()
        self.in_d = in_d
        self.seq_lenght = seq_lenght
        self.encoder_embedding = Tokenizator(dim_count)
        self.positional_encoding = PositionalEncoding(seq_lenght + dim_count*6, d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model*2, num_heads, d_ff, dropout = 0., activation= "gelu")

        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)
        self.pooling = nn.Linear(d_model*2, tgt_vocab_size)
        self.dense = nn.Linear(tgt_vocab_size, tgt_vocab_size)
        self.optimizer = optim.AdamW(self.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
        self.softmax = nn.Softmax(dim = 1)
        self.mask = torch.triu(torch.ones(seq_lenght,d_model*2), diagonal=1)
        self.mask = self.mask.int().float().to(device)

    def forward(self,src):
        embeddings = self.encoder_embedding.forward(src)     
        src_embedded = self.positional_encoding(embeddings).to(device)
        enc_output = self.encoder.forward(src_embedded).to(device)
        fc_output = self.pooling(enc_output).to(device)
        dense_output = self.dense(fc_output)
        output = self.softmax(dense_output).to(device)
        return [output, src_embedded]

    def weightedAccuracy(self, output, labels):
        correct = 0
        all_points = output.size(0)
        pred = torch.argmax(output,dim = 1) #[tensor of indices]
        true_pred = torch.argmax(labels, dim = 1)
        sum = 0  #[tensor of indices]
        for i in range(all_points):
            if pred[i] == true_pred[i]:
                correct += 1/gistogram[pred[i]]
            sum += 1/gistogram[true_pred[i]]    
        return correct/sum

    def training_step(self, batch):
        data, lab = batch
        self.optimizer.zero_grad()
        output, _ = self.forward(data)
        f_output = output.view(-1, TGT_VOCAB_SIZE)
        loss = weighted_loss(f_output, lab)
        loss.backward(loss)
        self.optimizer.step()
        accuracy = self.weightedAccuracy(f_output,lab)
        return [loss, accuracy]

    def valid_step(self,batch):
        data,y,true_pred = batch
        output, embeddings = self.forward(data)
        f_output = output.view(-1,TGT_VOCAB_SIZE)
        loss = weighted_loss(f_output, true_pred)
        accuracy = self.weightedAccuracy(f_output, true_pred)
        return [loss, accuracy,embeddings]


def labels_to_matrices(labels, tgt_vocab_size, seq_len):
    local_gistogram = [0,0,0,0,0]
    class_matrices = torch.zeros(seq_len, tgt_vocab_size-1).to(device)
    last_class = torch.ones(seq_len, 1).to(device)
    labels_matrices = torch.cat([class_matrices, last_class],1).to(device)
    coordinates = list(labels.keys())
    for i in coordinates:
        labels_matrices[i][labels[i]-1] = 1.0
        local_gistogram[labels[i]-1] += 1
        labels_matrices[i][tgt_vocab_size-1] = 0.0
    local_gistogram[4] = seq_len - len(coordinates)
    labels_matrices = labels_matrices.type(torch.FloatTensor).to(device)
    return [labels_matrices, local_gistogram]

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

for i in range(len(validating_set)):
    valid_raw = validating_set[i].raw
    raw_data = torch.from_numpy(valid_raw.get_data()).to(device)
    n_batches = raw_data.size(1)//SEQ_LEN
    validating_datasets += slice_to_batches(raw_data, SEQ_LEN, n_batches, N_CHANNELS)
    true_preds = torch.from_numpy(mne.events_from_annotations(valid_raw)[0]).to(device)
    pred_dict = {}
    for l in true_preds:
        pred_dict[l[0].item()] = l[2].item()
    pred_matrices,valid_gistogram = labels_to_matrices(pred_dict, TGT_VOCAB_SIZE, n_batches * SEQ_LEN)
    for j in range(TGT_VOCAB_SIZE):
        gistogram[j] += valid_gistogram[j]
    pred_batches += torch.split(pred_matrices, SEQ_LEN)

for i in range(len(training_set)):
    train_raw = training_set[i].raw
    raw_data = torch.from_numpy(train_raw.get_data()).to(device)
    n_batches = raw_data.size(1)//SEQ_LEN
    training_datasets += slice_to_batches(raw_data, SEQ_LEN, n_batches, N_CHANNELS)
    labels = torch.from_numpy(mne.events_from_annotations(train_raw)[0]).to(device)
    labels_dict = {}
    for l in labels:
        labels_dict[l[0].item()] = l[2].item()
    labels_matrices,training_gistogram = labels_to_matrices(labels_dict, TGT_VOCAB_SIZE, n_batches * SEQ_LEN)
    for j in range(TGT_VOCAB_SIZE):
        gistogram[j] += training_gistogram[j]
    labels_batches += torch.split(labels_matrices, SEQ_LEN)

gistogram_tensor = torch.tensor(gistogram).to(device)
norm_cf = 0.
normalized_list = []
for i in gistogram:
    norm_cf += 1/i
for i in gistogram:
    normalized_list.append((1/i)/norm_cf)
print(gistogram)
transformer = Transformer(DIM,NUM_HEADS,NUM_LAYERS,FF_DIM,SEQ_LEN,N_CHANNELS,TGT_VOCAB_SIZE,15)#d_model, num_heads, num_layers, d_ff, seq_lenght, dropout,in_d,tgt_vocab_size
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
for j in range(EPOCHS):
    running_acc = 0
    for i in range(len(training_datasets)):
        transformer.train()
        loss, acc = transformer.training_step([training_datasets[i],labels_batches[i]])
        running_loss += loss.item()
        running_acc += acc
        epoch_loss += loss.item()
        epoch_accuracy += acc
        if loss < best_loss:
            best_loss = loss
            os.remove("model.onnx")
            torch.save(transformer, "model.onnx")
        if i % 1000 == 999:
            last_loss = running_loss / 1000 
            print("training step")
            print(f"batch {i+1} mean loss: {last_loss}, mean accuracy: {running_acc/1000}")
            running_loss = 0
            running_acc = 0
    with open("results.txt", mode = "w") as file:
        file.write(f"{epoch_loss/len(training_datasets)} {epoch_accuracy/len(training_datasets)} {best_loss}")
    print(f"Epoch {j} loss {epoch_loss/len(training_datasets)}  accuracy {epoch_accuracy/len(training_datasets)}")
    epoch_accuracy = 0
    epoch_loss = 0
embeddings = torch.randn(SEQ_LEN,DIM*2)
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
