import torch
import torch.nn as nn
import torch.nn.functional as F
from model.pyGAT import GAT
import torch
import torch.nn as nn
import numpy as np
from model import transformer 
import copy

class CNN1DEncoder(torch.nn.Module):
    def __init__(self):
        super(CNN1DEncoder, self).__init__()
        self.layer1 = nn.Conv1d(7, 128, padding=2, kernel_size=(3,), dtype=torch.float)
        self.layer2 = nn.Conv1d(128, 128, padding=2, kernel_size=(3,), dtype=torch.float)
        self.layer3 = nn.Conv1d(128, 128, padding=2, kernel_size=(3,), dtype=torch.float)
        self.pool = nn.AdaptiveAvgPool1d(1)
        

    def forward(self, x):
        
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        x = self.layer3(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(509,6,128)
        return x

class cnn_lstm_attention(nn.Module):
    def __init__(self):
        super(cnn_lstm_attention, self).__init__()
        self.cnn = CNN1DEncoder()
        self.rnn = nn.LSTM(
            input_size=128, 
            hidden_size=256, 
            batch_first=True,
            bidirectional=True
            )
        self.hid = nn.Linear(129,512)
        self.cel = nn.Linear(129,512)
    def forward(self, x,static):
        h = self.hid(static).view(509,2,256).permute(1, 0, 2).contiguous()
        c = self.cel(static).view(509,2,256).permute(1, 0, 2).contiguous()
        x = self.cnn(x)
        out ,(hidden,cell) = self.rnn(x,(h,c))
        
       
        return out


class cnn_lstm_attention_gat(nn.Module):
    def __init__(self):
        super(cnn_lstm_attention_gat, self).__init__()
        self.cnn_lstm = cnn_lstm_attention()
        self.gnn = GAT(
            nfeat = 512,
            nhid = 512,
            nclass = 2,
            dropout = 0.5,
            nheads = 8,
            alpha = 0.2,
        )
        self.graph_th = 1/2**0.5
        self.w_omega = nn.Parameter(torch.Tensor(
            2*256 , 2*256 ))
        self.u_omega = nn.Parameter(torch.Tensor(2*256 , 1))
        

        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)

    def makeadj(self,p):
        pdot = torch.matmul(p,p.permute(1,0))
        psqu = torch.sum(p**2,dim=1)
        psqrt = psqu**0.5
        pfr =torch.matmul(psqrt.view(psqrt.shape[0],-1),psqrt.view(-1,psqrt.shape[0])) 
        pcos = pdot / pfr
        zero_vec = torch.zeros_like(pcos)
        pdo_n = torch.where(pcos > self.graph_th , pcos, zero_vec)
        pdo_n_s = 1/torch.sum(pdo_n,axis=1)**0.5
        pdi = torch.diag(pdo_n_s)
        dad = torch.matmul(pdi , torch.matmul(pdo_n,pdi))
        return dad
    def forward(self, xss,static):
        outs = self.cnn_lstm(xss,static)
        x = outs
        u = torch.tanh(torch.matmul(x, self.w_omega))
        # u形状是(batch_size, seq_len, 2 * num_hiddens)
        att = torch.matmul(u, self.u_omega)
        # att形状是(batch_size, seq_len, 1)
        att_score = F.softmax(att, dim=1)
        # att_score形状仍为(batch_size, seq_len, 1)
        scored_x = x * att_score
        # scored_x形状是(batch_size, seq_len, 2 * num_hiddens)
        # Attention过程结束
        
        feat = torch.sum(scored_x, dim=1) #加权求和

        outs = feat

        adj = self.makeadj(outs)
        
        gout = self.gnn(outs,adj)
        return gout[0],gout[1],gout[2],adj



def get_data():
    from utils.utils import load_dynamic_data
    import torch
    seq , col = load_dynamic_data()
    import numpy as np
    cnn_inputs = []
    for i in seq:
        state = np.array(i[1:])
        slices = np.split(state, list(range(5, state.shape[1]-5, 5)), axis=1)
        cnns = [torch.from_numpy(np.expand_dims(s, axis=0).copy()).float().to('cuda:0') for s in slices]
        cnn_inputs.append(cnns)
    from utils.utils import get_train_data
    trainseq,valseq,testseq,\
    static_no_num_train,static_no_num_val,static_no_num_test,  \
    static_num_train,static_num_val,static_num_test, \
    y_train,y_val,y_test , \
    allseq,all_static_no_num,all_static_num, \
    idx_train,idx_val,idx_test,y = get_train_data()
    allstatic = []
    for i,j in  zip(all_static_no_num,all_static_num):
        t = i+j
        allstatic.append(t)


    static_tensors = []
    for i in allstatic:
        static_tensors.append(torch.from_numpy(np.array(i)).to('cuda:0').to(torch.float32).view(1,1,129)) 

    allseq2 = []
    for i in range(509):
        t = []
        t1 = allseq[i]
        for j in range(6):
            allseq2.append(t1[0][j*5:(j+1)*(5)])
    allseq2t = torch.from_numpy(np.array(allseq2)).view(-1,7,5).to(torch.float32).to('cuda:0')

    allst = np.array(allstatic)
    allst = torch.from_numpy(allst).to(torch.float32).to('cuda:0')
    return allseq2t,allst,idx_train,idx_val,idx_test,y