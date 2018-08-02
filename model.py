import torch
import torch.nn as nn
import numpy as np 
import torch.nn.Function as F 
from torchvision import models

class EncoderCNN(nn.Module):

    def __init__(self):
        super(EncoderCNN,self).__init__()
        vgg16=models.vgg16(pretrained=True)
        for param in vgg16.parameters():
            param.requires_grad_(False)
        modules=list(vgg16.children())[0]
        self.vgg16=modules[:30]

    def forward(self,images):
        features=self.vgg16(images)
        return features


class DecoderRNN(nn.Module):

    def __init__(self,vocab_size,embed_size,hidden_size,num_layer=1):
        super(DecoderRNN,self).__init__()
        self.embedding=nn.Embedding(vocab_size,embed_size)
        self.lstm=nn.LSTM(512+embed_size,hidden_size,num_layer)
        self.linear=nn.Linear(hidden_size,vocab_size)
        self.attention_layer=nn.Linear(512+hidden_size,1)
        self.init_c=nn.Linear(512,hidden_size)
        self.init_h=nn.Linear(512,hidden_size)


    def forward(self,context,caption,hc):
        caption_vector=self.embedding(caption)
        input_vector=torch.cat((context,caption_vector),dim=-1)
        input_vector=torch.unsqueeze(input_vector,0)
        output_vector,hc=self.lstm(input_vector,hc)
        score=self.linear(output_vector)
        return score,hc



    def attention(self,features,hc):
        hidden_state=hc[0]
        hidden_state=torch.transpose(hidden_state,1,0)
        hidden_state=hidden_state.expand(-1,196,-1)
        features=features.view(-1,features.size(2)*features.size(3),features.size(1))
        attention_vector=torch.cat((features,hidden_state),dim=-1)
        scores=self.attention_layer(attention_vector)
        alpha=F.softmax(scores.squeeze(),dim=1)
        alpha=torch.unsqueeze(alpha,-1)
        context=torch.sum(features*alpha,dim=1)
        return context,alpha



    def _init_hidden(self,features):
        features=features.view(-1,features.size(2)*features.size(3),features.size(1))
        init_input=torch.mean(features,dim=1)
        init_c=self.init_c(init_input)
        init_h=self.init_h(init_input)
        init_c=torch.unsqueeze(init_c,0)
        init_h=torch.unsqueeze(init_h,0)
        return init_h,init_c

    
    def sample(self,features):
        word_index=[0]
        context=
        while start!=1 and len(word_index)
        


