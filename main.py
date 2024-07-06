#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
import nltk
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import os
nltk.download('punkt')

import torch.nn as nn
import torch.nn.functional as F

import dataset
import tokenizer


from tokenizer import SimpleTokenizer
from dataset import SpeechesClassificationDataset, LanguageModelingDataset

print('set')

seed = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import transformer




""" Hyperparameters to use for training to roughly match
the numbers mentioned in the assignment description """
batch_size = 16  # Number of independent sequences  we will process in parallel
block_size = 32  # Maximum context length for predictions
learning_rate = 1e-3  # Learning rate for the optimizer
n_embd = 64  # Embedding dimension
n_head = 2  # Number of attention heads
n_layer = 4  # Number of transformer layers
dropout=0.0


eval_interval = 100  # How often to evaluate train and test perplexity during training
max_iters = 500 # For language modeling, we can process all the batches for the entire dataset, but that takes a while, so we'll limit it to 500 iterations. For batch size of 16 and block size of  32, this is roughly, this is  500 * 16 * 32 = 256000 tokens, SOTA LMs are trained on trillions of tokens, so this is a very small dataset.
eval_iters = 200  # Number of iterations to evaluate perplexity on the test set


## classifier training hyperparameters. It is a simple 1 hidden layer feedforward network, with input
## size of 64, hidden size of 50 and output size of 3.

n_input = 64  # Input size for the classifier, should match the embedding size of the transformer
n_hidden = 100  # Hidden size for the classifier
n_output = 3  # Output size for the classifier, we have 3 classes
epochs_CLS = 15 # epochs for classifier training

from transformer import classifier
from transformer import TransformerEncoder
from transformer import TransformerDecoder
from transformer import classifier_A

def load_texts(directory):
    """
    This function loads all texts from the specified directory, ignoring any files with "test" in their name. The text is used for "training" the tokenizer. Since our tokenizer is simple, we don't need to do any training, but we still need to ignore the test data.
    """

    texts = []
    files = os.listdir(directory)
    for filename in files:
        if "test" in filename:  ## don't "read test files"
            continue
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
            texts.append(file.read())
    return texts



def collate_batch(batch):
    """ Collate a batch of data into a single tensor with padding."""
    data, labels = zip(*batch)  # Separate the data and labels
    # Pad sequences to the fixed length
    padded_sequences = pad_sequence(data, batch_first=True, padding_value=0)
    padded_sequences = padded_sequences[:, :block_size]  # Truncate if longer
    # Add padding if shorter
    padded_sequences = torch.nn.functional.pad(padded_sequences, (0, max(0, block_size - padded_sequences.shape[1])), "constant", 0)
    labels = torch.stack(labels)
    return padded_sequences, labels

def compute_classifier_accuracy(classifier, data_loader):
    """ Compute the accuracy of the classifier on the data in data_loader."""
    classifier.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for X, Y in data_loader:
            X, Y = X.to(device), Y.to(device)
            outputs = classifier(X)
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == Y).sum().item()
            total_samples += Y.size(0)
        accuracy = (100 * total_correct / total_samples)
        classifier.train()
        return accuracy


def compute_perplexity(decoderLMmodel, data_loader, eval_iters=100):
    """ Compute the perplexity of the decoderLMmodel on the data in data_loader.
    Make sure to use the cross entropy loss for the decoderLMmodel.
    """
    decoderLMmodel.eval()
    criterion = nn.CrossEntropyLoss()
    losses = []
    total_loss = 0
    with torch.no_grad():
        for X, Y in data_loader:
            X, Y = X.to(device), Y.to(device)
            outputs,_ = decoderLMmodel(X)  # Call model without passing Y
            loss = criterion(outputs, Y.view(-1))  # Compute loss
            losses.append(loss.item())
            total_loss += loss.item()
            if len(losses) >= eval_iters:
                break

    mean_loss = total_loss / len(losses)  # Compute mean loss
    perplexity = torch.exp(torch.tensor(mean_loss))  # Calculate perplexity as exp(mean loss)

    decoderLMmodel.train()
    return perplexity.item()


    losses = torch.tensor(losses)
    mean_loss = losses.mean()
    perplexity = torch.exp(mean_loss).item()  # Calculate perplexity as exp(mean loss)

    decoderLMmodel.train()
    return perplexity

def main():
    
    # part 1 - Encoder,part 2- decoder , part 3- Exploration
    
    part=input("Enter as part1 or part2 or part3:  ")

    print("Loading data and creating tokenizer ...")
    texts = load_texts('speechesdataset')
    tokenizer = SimpleTokenizer(' '.join(texts)) # create a tokenizer from the data
    print("Vocabulary size is", tokenizer.vocab_size)

    train_CLS_dataset = SpeechesClassificationDataset(tokenizer, "speechesdataset/train_CLS.tsv")
    train_CLS_loader = DataLoader(train_CLS_dataset, batch_size=batch_size,collate_fn=collate_batch,shuffle=True)

    test_cls_dataset=SpeechesClassificationDataset(tokenizer,"speechesdataset/test_CLS.tsv")
    test_CLS_loader=DataLoader(test_cls_dataset,batch_size=batch_size,collate_fn=collate_batch,shuffle=False)

    vocab_size=tokenizer.vocab_size


    inputfile = "speechesdataset/train_LM.txt"
    with open(inputfile, 'r', encoding='utf-8') as f:
        lmtrainText = f.read()
    train_LM_dataset = LanguageModelingDataset(tokenizer, lmtrainText,  block_size)
    train_LM_loader = DataLoader(train_LM_dataset, batch_size=batch_size, shuffle=True)


    inputfile_o = "speechesdataset/test_LM_obama.txt"
    with open(inputfile_o, 'r', encoding='utf-8') as f:
        lmtrainText_O = f.read()
    train_LM_dataset_O = LanguageModelingDataset(tokenizer, lmtrainText_O,  block_size)
    train_LM_loader_O =DataLoader(train_LM_dataset_O, batch_size=batch_size, shuffle=True)


    inputfile_H = "speechesdataset/test_LM_hbush.txt"
    with open(inputfile_H, 'r', encoding='utf-8') as f:
        lmtrainText_H = f.read()
    train_LM_dataset_H = LanguageModelingDataset(tokenizer, lmtrainText_H,  block_size)
    train_LM_loader_H =DataLoader(train_LM_dataset_H, batch_size=batch_size, shuffle=True)

    inputfile_w = "speechesdataset/test_LM_wbush.txt"
    with open(inputfile_w, 'r', encoding='utf-8') as f:
        lmtrainText_w = f.read()
    train_LM_dataset_w = LanguageModelingDataset(tokenizer, lmtrainText_w,  block_size)
    train_LM_loader_w =DataLoader(train_LM_dataset_w, batch_size=batch_size, shuffle=True)

    if part=='part1':

        # for the classification  task, you will train for a fixed number of epochs like this:
        model=None
        model=classifier(n_embd,n_head,block_size,n_layer,dropout,vocab_size).to(device)
        print(sum(p.numel() for p in model.parameters()),'PARAMETERS')
        optimizer_cls=torch.optim.AdamW(model.parameters(),lr=learning_rate)

        for epoch in range(epochs_CLS):
            loss=0
            for xb, yb in train_CLS_loader:
                xb, yb = xb.to(device), yb.to(device)
                model.train()
                logits = model(xb)
                loss_cls=F.cross_entropy(logits,yb)
                loss=loss+loss_cls.item()

                optimizer_cls.zero_grad(set_to_none=True)
                loss_cls.backward()
                optimizer_cls.step()
            avg=loss/len(train_CLS_loader)
            test_accuracy = compute_classifier_accuracy(model, test_CLS_loader)
            print('Epoch no',epoch,'Loss', avg,"Test Accuracy:", test_accuracy)

        # for sanity check and attention map
        encoder_model=TransformerEncoder(n_embd, n_head, block_size, n_layer, dropout,vocab_size).to(device)
        s1='They have affiliates in many countries and are trying to expand their reach.'
        import utilities
        from utilities import Utilities
        ad=Utilities(tokenizer,encoder_model)
        ad.sanity_check(s1,block_size)
        


    if part=='part2':

        lm_model=None
        de_mask = torch.tril(torch.full((batch_size, n_head, block_size, block_size), float('-inf')), diagonal=-1).to(device)
        lm_model=TransformerDecoder(n_embd, n_head, block_size, n_layer,dropout,de_mask,vocab_size).to(device)
        print(sum(p.numel() for p in lm_model.parameters()),'PARAMETERS')

        optimizer_lm = torch.optim.AdamW(lm_model.parameters(), lr=learning_rate)

        lm_model.train()
        print('FOR TRAINING SET')

        for i, (xb, yb) in enumerate(train_LM_loader):
            loss=0
            if i > max_iters:
                break
            xb, yb = xb.to(device), yb.to(device)
            yb_reshaped = yb.view(-1)


            outputs_lm,_ = lm_model(xb)
            loss_lm = F.cross_entropy(outputs_lm, yb_reshaped)
            optimizer_lm.zero_grad(set_to_none=True)
            loss_lm.backward()
            optimizer_lm.step()

            train_perplexity = compute_perplexity(lm_model, train_LM_loader, eval_iters)
            if (i>5 and i%100==0):
                print('Iteration No',i,"Train Perplexity:", train_perplexity)

        print('FOR TEST_LM_OBAMA')

        lm_model=None
        de_mask = torch.tril(torch.full((batch_size, n_head, block_size, block_size), float('-inf')), diagonal=-1).to(device)
        lm_model=TransformerDecoder(n_embd, n_head, block_size, n_layer,dropout,de_mask,vocab_size).to(device)

        optimizer_lm = torch.optim.AdamW(lm_model.parameters(), lr=learning_rate)

        lm_model.train()

        for i, (xb, yb) in enumerate(train_LM_loader):
            loss=0
            if i > max_iters:
                break
            xb, yb = xb.to(device), yb.to(device)
            yb_reshaped = yb.view(-1)


            outputs_lm,_ = lm_model(xb)
            loss_lm = F.cross_entropy(outputs_lm, yb_reshaped)
            optimizer_lm.zero_grad(set_to_none=True)
            loss_lm.backward()
            optimizer_lm.step()

            train_perplexity = compute_perplexity(lm_model, train_LM_loader_O, eval_iters)
            if (i>5 and i%100==0):
                print('Iteration No',i,"Test Perplexity-OBAMA:", train_perplexity)


        print('FOR TEST_LM_hbush')

        lm_model=None
        de_mask = torch.tril(torch.full((batch_size, n_head, block_size, block_size), float('-inf')), diagonal=-1).to(device)
        lm_model=TransformerDecoder(n_embd, n_head, block_size, n_layer,dropout,de_mask,vocab_size).to(device)

        optimizer_lm = torch.optim.AdamW(lm_model.parameters(), lr=learning_rate)

        lm_model.train()

        for i, (xb, yb) in enumerate(train_LM_loader):
            loss=0
            if i > max_iters:
                break
            xb, yb = xb.to(device), yb.to(device)
            yb_reshaped = yb.view(-1)


            outputs_lm,_ = lm_model(xb)
            loss_lm = F.cross_entropy(outputs_lm, yb_reshaped)
            optimizer_lm.zero_grad(set_to_none=True)
            loss_lm.backward()
            optimizer_lm.step()

            train_perplexity = compute_perplexity(lm_model, train_LM_loader_H, eval_iters)
            if (i>5 and i%100==0):
                print('Iteration No',i,"Test Perplexity-HBUSH:", train_perplexity)


        print('FOR TEST_LM_wbush')

        lm_model=None
        de_mask = torch.tril(torch.full((batch_size, n_head, block_size, block_size), float('-inf')), diagonal=-1).to(device)
        lm_model=TransformerDecoder(n_embd, n_head, block_size, n_layer,dropout,de_mask,vocab_size).to(device)

        optimizer_lm = torch.optim.AdamW(lm_model.parameters(), lr=learning_rate)

        lm_model.train()

        for i, (xb, yb) in enumerate(train_LM_loader):
            loss=0
            if i > max_iters:
                break
            xb, yb = xb.to(device), yb.to(device)
            yb_reshaped = yb.view(-1)


            outputs_lm,_ = lm_model(xb)
            loss_lm = F.cross_entropy(outputs_lm, yb_reshaped)
            optimizer_lm.zero_grad(set_to_none=True)
            loss_lm.backward()
            optimizer_lm.step()

            train_perplexity = compute_perplexity(lm_model, train_LM_loader_w, eval_iters)
            if (i>5 and i%100==0):
              print('Iteration No',i,"Test Perplexity-WBUSH:", train_perplexity)

        # for sanity check and attention map for a sentence

        de_mask = torch.tril(torch.full((batch_size, n_head, block_size, block_size), float('-inf')), diagonal=-1).to(device)
        decoder_model=TransformerDecoder(n_embd, n_head, block_size, n_layer,dropout,de_mask,vocab_size).to(device)
        s2='Others traveled to Baghdad in a variety of efforts to restore peace and justice." '
        import utilities
        from utilities import Utilities
        ad=Utilities(tokenizer,decoder_model)
        ad.sanity_check(s2,block_size)


    if part=='part3':
        model=None
        model=classifier_A(n_embd,n_head,block_size,n_layer,dropout,vocab_size).to(device)
        print(sum(p.numel() for p in model.parameters()),'PARAMETERS')
        optimizer_cls=torch.optim.AdamW(model.parameters(),lr=learning_rate)


        for epoch in range(epochs_CLS):
            loss=0
            for xb, yb in train_CLS_loader:
                xb, yb = xb.to(device), yb.to(device)
                model.train()
                logits = model(xb)
                loss_cls=F.cross_entropy(logits,yb)
                loss=loss+loss_cls.item()

                optimizer_cls.zero_grad(set_to_none=True)
                loss_cls.backward()
                optimizer_cls.step()
            avg=loss/len(train_CLS_loader)
            test_accuracy = compute_classifier_accuracy(model, test_CLS_loader)
            print('Epoch no',epoch,'Loss', avg,"Test Accuracy:", test_accuracy)
            

if __name__ == "__main__":
    main()

