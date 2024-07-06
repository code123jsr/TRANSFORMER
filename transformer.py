
batch_size = 16  # Number of independent sequences  we will process in parallel
block_size = 32  # Maximum context length for predictions
learning_rate = 1e-3  # Learning rate for the optimizer
n_embd = 64  # Embedding dimension
n_head = 2  # Number of attention heads 
n_layer = 4  # Number of transformer layers
dropout=0.0
import tokenizer
# device = "cuda"
from tokenizer import SimpleTokenizer

# texts = load_texts('speechesdataset')
# tokenizer = SimpleTokenizer(' '.join(texts)) # create a tokenizer from the data

# vocab_size=tokenizer.vocab_size

eval_interval = 100  # How often to evaluate train and test perplexity during training
max_iters = 500 # For language modeling, we can process all the batches for the entire dataset, but that takes a while, so we'll limit it to 500 iterations. For batch size of 16 and block size of  32, this is roughly, this is  500 * 16 * 32 = 256000 tokens, SOTA LMs are trained on trillions of tokens, so this is a very small dataset.
eval_iters = 200  # Number of iterations to evaluate perplexity on the test set


## classifier training hyperparameters. It is a simple 1 hidden layer feedforward network, with input 
## size of 64, hidden size of 50 and output size of 3.

n_input = 64  # Input size for the classifier, should match the embedding size of the transformer
n_hidden = 100  # Hidden size for the classifier
n_output = 3  # Output size for the classifier, we have 3 classes
epochs_CLS = 15 # epochs for classifier training

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




# PART 1- ENCODER IMPLEMENTATION


class MultiHeadAttention(nn.Module):
    
    
    def __init__(self, n_embd: int, n_head:int,dropout:float , mask=None):
        super(MultiHeadAttention, self).__init__()

        assert n_embd % n_head == 0
        
        self.a=nn.Linear(n_embd,3*n_embd,bias=False)
        self.ad=nn.Dropout(dropout)
        self.rd=nn.Dropout(dropout)
        
        self.w=nn.Linear(n_embd,n_embd,bias=False)
        

        self.n_embd = n_embd
        self.n_head = n_head
        self.mask=mask
        self.dropout=dropout


    def forward(self, x):
        
        B,T,C=x.size()
        
        query,key,value=self.a(x).split(self.n_embd,dim=2)


        key=key.view(B,T,self.n_head, C//self.n_head).transpose(1,2)
        query=query.view(B,T,self.n_head, C//self.n_head).transpose(1,2)
        value=value.view(B,T,self.n_head, C//self.n_head).transpose(1,2)
        dk=key.size(-1)
        
        
        atten=(query@key.transpose(-2,-1))/math.sqrt(dk)

        if self.mask is not None:
            atten.masked_fill(self.mask == 0, float('-inf'))
        
        atten= F.softmax(atten,dim=-1)
            
        if self.dropout is not None:
            atten=self.ad(atten)
        
        y=atten@value
        y=y.transpose(1,2).contiguous().view(B,T,C)
        
        y=self.rd(self.w(y))

        return y,atten
    

    
class FeedFoward(nn.Module):

    def __init__(self, n_embd,dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
    
    
class EncoderLayer(nn.Module):
    def __init__(self, n_embd, n_head,dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(n_embd, n_head,dropout)
        self.ff=FeedFoward(n_input,dropout)
        self.norm1 = nn.LayerNorm(n_embd)
        self.norm2 = nn.LayerNorm(n_embd)

    def forward(self, x):
       
        
        x_norm = self.norm1(x)
        self_attn_output,att = self.self_attn(x_norm)
        x = x + self_attn_output
        x = self.norm2(x)
        ff_output = self.ff(x)
        x = x + ff_output
        
        return x,att

class TransformerEncoder(nn.Module):
    def __init__(self,n_embd, n_head,block_size, n_layer,dropout,vocab_size):
        dropout=0.0
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, n_embd)
        self.position_encoding =nn.Embedding(block_size,n_embd)
        self.layers = nn.ModuleList([EncoderLayer(n_embd, n_head,dropout) for _ in range(n_layer)])
        self.norm = nn.LayerNorm(n_embd)
        self.n_embd=n_embd
   

    def forward(self, x):
        
        at_m=[]
        at=None

        e = self.embedding(x)*math.sqrt(self.n_embd)
        p = self.position_encoding(torch.arange(x.shape[1]).to(device))
        t=e+p
        for layer in self.layers:
            t,at = layer(t)
            at_m.append(at)
        t = self.norm(t)  
        return t,at_m


class FeedForwardClassifier(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(FeedForwardClassifier, self).__init__()
        self.fc1 = nn.Linear(n_input, n_hidden)
        
        self.fc2 = nn.Linear(n_hidden, n_output)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    
class classifier(nn.Module):
    def __init__(self, n_embd,n_head,block_size,n_layer,dropout,vocab_size):
        super().__init__()
        self.trans=TransformerEncoder(n_embd,n_head,block_size,n_layer,0.0, vocab_size)
        self.feed=FeedForwardClassifier(n_input, n_hidden, n_output)
    
    def forward(self,x):
        
        y,at_m=self.trans(x)
        y_mean=torch.mean(y,dim=1)
        logits=self.feed(y_mean)
        return logits
    
    
    
# PART 2 - DECODER IMPLEMENTATION


class MultiHeadAttention(nn.Module):
    dropout=0.0
    def __init__(self, n_embd: int, n_head:int,dropout:float , mask=None):
        super(MultiHeadAttention, self).__init__()

        assert n_embd % n_head == 0
        
        self.a=nn.Linear(n_embd,3*n_embd,bias=False)
        self.ad=nn.Dropout(dropout)
        self.rd=nn.Dropout(dropout)
        
        self.w=nn.Linear(n_embd,n_embd,bias=False)
        

        self.n_embd = n_embd
        self.n_head = n_head
        self.mask=mask
        self.dropout=dropout


    def forward(self, x):
        
        B,T,C=x.size()
        
        query,key,value=self.a(x).split(self.n_embd,dim=2)


        key=key.view(B,T,self.n_head, C//self.n_head).transpose(1,2)
        query=query.view(B,T,self.n_head, C//self.n_head).transpose(1,2)
        value=value.view(B,T,self.n_head, C//self.n_head).transpose(1,2)
        dk=key.size(-1)
        
        
        atten=(query@key.transpose(-2,-1))/math.sqrt(dk)

        if self.mask is not None:
            atten.masked_fill(self.mask == 0, float('-inf'))
        
        atten= F.softmax(atten,dim=-1)
            
        if self.dropout is not None:
            atten=self.ad(atten)
        
        y=atten@value
        y=y.transpose(1,2).contiguous().view(B,T,C)
        
        y=self.rd(self.w(y))

        return y,atten
    
    
class FeedFoward(nn.Module):

    def __init__(self, n_embd,dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class DecoderLayer(nn.Module):
    def __init__(self, n_embd, n_head,dropout,de_mask):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(n_embd, n_head,dropout,mask=de_mask)
        self.ff=FeedFoward(n_input,dropout)
        self.norm1 = nn.LayerNorm(n_embd)
        self.norm2 = nn.LayerNorm(n_embd)


    def forward(self, x):


        x_norm = self.norm1(x)
        self_attn_output,att = self.self_attn(x_norm)
        x = x + self_attn_output
        x = self.norm2(x)
        ff_output = self.ff(x)
        x = x + ff_output

        return x,att
    
    
class TransformerDecoder(nn.Module):
    def __init__(self,n_embd, n_head,block_size, n_layer,dropout,de_mask,vocab_size):
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, n_embd)
        self.position_encoding =nn.Embedding(block_size,n_embd)
        self.layers = nn.ModuleList([DecoderLayer(n_embd, n_head,dropout,de_mask) for _ in range(n_layer)])
        self.norm = nn.LayerNorm(n_embd)
        self.n_embd=n_embd
        self.fc=nn.Linear(n_embd,vocab_size)

    def forward(self, x):

        at_m=[]
        at=None

        e = self.embedding(x)*math.sqrt(self.n_embd)
        p = self.position_encoding(torch.arange(x.shape[1]).to(device))
        t_d=e+p
        for layer in self.layers:
            t_d,at = layer(t_d)
            at_m.append(at)
        t_d = self.norm(t_d)
        y=self.fc(t_d)
        y_reshaped = y.view(-1, y.size(-1))
        return y_reshaped,at_m
    
    
    
# PART 3 - EXPLORATION

# Here, tried with different architectures disentangled attention pattern, positional encoding like Alibi and sinusidal , with different constructional design of multihead attention, positional encoding, and also varied the parameters . Some of the models are given below. 

# WITH ALIBI POSITIONAL ENCODING


import torch
import torch.nn as nn

class AliBiPositionalEncoding(nn.Module):
    def __init__(self, n_embd, block_size):
        super(AliBiPositionalEncoding, self).__init__()
        self.position_embeddings = nn.Embedding(block_size, n_embd)

    def forward(self, x):
        seq_length = x.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=x.device)
        position_embeddings = self.position_embeddings(position_ids)
        return x + position_embeddings
    
class EncoderLayer(nn.Module):
    def __init__(self, n_embd, n_head,dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(n_embd, n_head,dropout)
        self.ff=FeedFoward(n_input,dropout)
        self.norm1 = nn.LayerNorm(n_embd)
        self.norm2 = nn.LayerNorm(n_embd)

    def forward(self, x):
       
        
        x_norm = self.norm1(x)

        self_attn_output,att = self.self_attn(x_norm)

        x = x + self_attn_output
        x = self.norm2(x)
        ff_output = self.ff(x)

        x = x + ff_output
        
        
        return x,att
    
class TransformerEncoder_A(nn.Module):
    def __init__(self, n_embd, n_head, block_size, n_layer, dropout,vocab_size):
        super(TransformerEncoder_A, self).__init__()
        self.embedding = nn.Embedding(vocab_size, n_embd)
        self.position_encoding = AliBiPositionalEncoding(n_embd, block_size)
        self.layers = nn.ModuleList([EncoderLayer(n_embd, n_head, dropout) for _ in range(n_layer)])
        self.norm = nn.LayerNorm(n_embd)
        self.n_embd = n_embd

    def forward(self, x):
        at_m = []
        at = None

        e = self.embedding(x) * math.sqrt(self.n_embd)
        f = self.position_encoding(e).to(device)
        t=e+f
        for layer in self.layers:
            t, at = layer(t)
            at_m.append(at)
        t = self.norm(t)
        return t, at_m
    
    
    
class FeedForwardClassifier(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(FeedForwardClassifier, self).__init__()
        self.fc1 = nn.Linear(n_input, n_hidden)
        
        self.fc2 = nn.Linear(n_hidden, n_output)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    
class classifier_A(nn.Module):
    def __init__(self,n_embd,n_head,block_size,n_layer,dropout,vocab_size):
        super().__init__()
        self.trans=TransformerEncoder_A(n_embd,n_head,block_size,n_layer,dropout,vocab_size)
        self.feed=FeedForwardClassifier(n_input, n_hidden, n_output)
    
    def forward(self,x):
        
        y,at_m=self.trans(x)
        y_mean=torch.mean(y,dim=1)
        logits=self.feed(y_mean)
        return logits
    
    
# WITH DISENTANGLED MULTIHEAD ATTENTION

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DisentangledMultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout, mask=None):
        super(DisentangledMultiHeadAttention, self).__init__()

        assert n_embd % n_head == 0

        self.n_embd = n_embd
        self.n_head = n_head
        self.mask = mask
        self.dropout = dropout

        self.query_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.key_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.value_proj = nn.Linear(n_embd, n_embd, bias=False)
        self.position_bias = nn.Parameter(torch.zeros(n_head, n_embd // n_head))

        self.output_proj = nn.Linear(n_embd, n_embd, bias=False)

        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.size()

        query = self.query_proj(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        key = self.key_proj(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        value = self.value_proj(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        dk = key.size(-1)

        content_atten = (query @ key.transpose(-2, -1)) / math.sqrt(dk)
        position_atten = self.position_bias.unsqueeze(0).unsqueeze(2) 

        atten = content_atten + position_atten

        if self.mask is not None:
            atten.masked_fill_(self.mask == 0, float('-inf'))

        atten = F.softmax(atten, dim=-1)

        atten = self.dropout_layer(atten)

        y = atten @ value
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.output_proj(y)

        return y, atten


class EncoderLayer_D(nn.Module):
    def __init__(self, n_embd, n_head,dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = DisentangledMultiHeadAttention(n_embd, n_head,dropout)
        self.ff=FeedFoward(n_input,dropout)
        self.norm1 = nn.LayerNorm(n_embd)
        self.norm2 = nn.LayerNorm(n_embd)

    def forward(self, x):
       
        
        x_norm = self.norm1(x)
        self_attn_output,att = self.self_attn(x_norm)

        x = x + self_attn_output

        x = self.norm2(x)
        ff_output = self.ff(x)
        x = x + ff_output
        
        return x,att
    
    

class TransformerEncoder_D(nn.Module):
    def __init__(self,n_embd, n_head,block_size, n_layer,dropout,vocab_size):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, n_embd)
        self.position_encoding =nn.Embedding(block_size,n_embd)
        self.layers = nn.ModuleList([EncoderLayer_D(n_embd, n_head,dropout) for _ in range(n_layer)])
        self.norm = nn.LayerNorm(n_embd)
        self.n_embd=n_embd

    def forward(self, x):
        
        at_m=[]
        at=None

        e = self.embedding(x)*math.sqrt(self.n_embd)
        p = self.position_encoding(torch.arange(x.shape[1]))
        t=e+p
        for layer in self.layers:
            t,at = layer(t)
            at_m.append(at)
        t = self.norm(t)  
        return t,at_m
    
    
class FeedForwardClassifier(nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(FeedForwardClassifier, self).__init__()
        self.fc1 = nn.Linear(n_input, n_hidden)
        
        self.fc2 = nn.Linear(n_hidden, n_output)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    
class classifier_D(nn.Module):
    def __init__(self,n_embd,n_head,block_size,n_layer,dropout,vocab_size):
        super().__init__()
        self.trans=TransformerEncoder_D(n_embd,n_head,block_size,n_layer,dropout,vocab_size)
        self.feed=FeedForwardClassifier(n_input, n_hidden, n_output)
    
    def forward(self,x):
        
        y,at_m=self.trans(x)
        y_mean=torch.mean(y,dim=1)
        logits=self.feed(y_mean)
        return logits












    
    

    
    

    
    
