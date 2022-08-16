import torch
import torch.nn as nn
from utils.utils import get_train_data, load_dynamic_data
import numpy as np



class encoder_cnn(nn.Module):
    def __init__(
        self,
        premodel_path ,
        pre_output = 30,
        output_num = 32,
        channel_size = 7
        ):
        super(encoder_cnn,self).__init__()
        self.pre = torch.load(premodel_path)
        self.cnn = nn.Conv1d(in_channels=pre_output,out_channels=output_num,kernel_size=channel_size)
        self.norm = nn.LayerNorm([output_num,1])

    def forward(self,x):
        
        x = self.pre(x)[0]
        x = self.cnn(x)
        x = self.norm(x)
        return x

class static_unnum(nn.Module):
    def __init__(
        self,
        inputsize = 123,
        outpustsize = 16
    ):
        super(static_unnum,self).__init__()
        self.fc = nn.Linear(inputsize,outpustsize)
        self.norm = nn.LayerNorm(outpustsize)
    def forward(self,x):
        x = self.fc(x)
        x = self.norm(x)
        return x


class static_num(nn.Module):
    def __init__(
        self,
        inputsize = 6 ,
        outpustsize = 16
    ):
        super(static_num,self).__init__()
        self.fc = nn.Linear(inputsize,outpustsize)
        self.norm = nn.LayerNorm(outpustsize)
    def forward(self,x):
        x = self.fc(x)
        x = self.norm(x)
        return x