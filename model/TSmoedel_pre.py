import torch
import torch.nn as nn
import numpy as np
from model import transformer 
import copy
def smape_loss(y_pred, target):
    loss = 2 * (y_pred - target).abs() / (y_pred.abs() + target.abs() + 1e-8)
    return loss.mean()


def gen_trg_mask(length, device):
    mask = torch.tril(torch.ones(length, length, device=device)) == 1

    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )

    return mask


class Positional_Encoding(nn.Module):
    def __init__(self, embed_size, seqlen, dropout, device):
        super(Positional_Encoding, self).__init__()
        self.device = device
        self.pe = torch.tensor([[pos / (10000.0 ** (i // 2 * 2.0 / embed_size)) for i in range(embed_size)] for pos in range(seqlen)])
        self.pe[:, 0::2] = torch.sin(self.pe[:, 0::2])
        self.pe[:, 1::2] = torch.cos(self.pe[:, 1::2])
        self.pe = self.pe.to(device)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        out = x + nn.Parameter(self.pe, requires_grad=False).to(self.device)
        out = self.dropout(out)
        return out
    def getpe(self):
        return self.pe



class TSencoder(nn.Module):
    def __init__(
        self,
        channels,
        encoder_head_num,
        dropout,
        dimfeedforward,
        encoder_layer_num,
        seqlen,
        device
    ):
        super(TSencoder,self).__init__()

        self.position_embed = Positional_Encoding(
            embed_size = channels,
            seqlen = seqlen,
            dropout = dropout,
            device = device
        )
        self.encoder_layer = transformer.TransformerEncoderLayer(
            d_model= channels,
            nhead= encoder_head_num,
            dropout= dropout,
            dim_feedforward= dimfeedforward
        )

        self.encoder = transformer.TransformerEncoder(
            self.encoder_layer,
            num_layers=encoder_layer_num
        )


        

    def forward(self, x):
        x = self.position_embed(x).permute(1, 0, 2)
        x,att = self.encoder(x)
        return x.permute(1, 0, 2),att


class TSdecoder(nn.Module):
    def __init__(
        self,
        channels,
        decoder_head_num,
        dropout,
        dimfeedforward,
        decoder_layer_num,
        de_seqlen,
        device

    ):
        super(TSdecoder,self).__init__()
        self.deseqlen = de_seqlen
        self.device = device
        self.position_embed = Positional_Encoding(
            embed_size = channels,
            seqlen = de_seqlen,
            dropout = dropout,
            device = device
        )
        self.decoder_layer = transformer.TransformerDecoderLayer(
            d_model=channels,
            nhead=decoder_head_num,
            dropout=dropout,
            dim_feedforward=dimfeedforward,
        )
        self.decoder = transformer.TransformerDecoder(self.decoder_layer, num_layers=decoder_layer_num)

    def forward(self,x,memory):
        x = self.position_embed(x).permute(1, 0, 2)
        x_mask = gen_trg_mask(self.deseqlen,self.device)
        x,att,self_Att = self.decoder(tgt=x , memory = memory , tgt_mask = x_mask)
        return x.permute(1, 0, 2),att,self_Att






class TSF(nn.Module):
    def __init__(
        self,
        channels,
        encoder_head_num,
        decoder_head_num,
        dropout,
        en_dimfeedforward,
        de_dimfeedforward,
        encoder_layer_num,
        decoder_layer_num,
        seqlen,
        de_seqlen,
        device

        ):
        super(TSF,self).__init__()
        self.device = device
        self.encoder = TSencoder(
            channels = channels,
            encoder_head_num = encoder_head_num,
            dropout = dropout,
            dimfeedforward = en_dimfeedforward,
            encoder_layer_num = encoder_layer_num,
            seqlen = seqlen,
            device =device
        )

        self.decoder = TSdecoder(
            channels = channels,
            decoder_head_num = decoder_head_num,
            dropout = dropout,
            dimfeedforward = de_dimfeedforward,
            decoder_layer_num = decoder_layer_num,
            de_seqlen = de_seqlen,
            device = device
        )

    def forward(self,src,trg,needall):
        src = src.to(self.device)
        trg = trg.to(self.device)
        src,en_self_att = self.encoder(src)
        out,de_att,de_self_att= self.decoder(trg,src)

        if needall:
            return out , src , en_self_att,de_att , de_self_att
        return out        
    def __return__encoder__(self):
        return self.encoder


class ncoder_lin(nn.Module):
    def __init__(
        self,
        premodel_path,
        pre_output,
        output_num
        ):
        super(ncoder_lin,self).__init__()
        self.pre = torch.load(premodel_path)
        self.fc = nn.Linear(pre_output,output_num)

    def forward(self,x):
        x = self.pre(x)[0]
        x = self.fc(x)
        return x

if __name__ == '__main__':
    


    x = np.random.rand(2,64,10)
    x = torch.from_numpy(x).view(2,64,10).to('cuda:0').to(torch.float32)

    xt = np.random.rand(2,32,10)
    xt = torch.from_numpy(xt).view(2,32,10).to('cuda:0').to(torch.float32)

    tsf = TSF(
        channels= 10,
        encoder_head_num= 1,
        decoder_head_num= 1 ,
         dropout= 0.4,
         en_dimfeedforward= 4 * 10,
         de_dimfeedforward= 4 * 10,
         encoder_layer_num= 4,
         decoder_layer_num=  4,
         seqlen= 64,
         de_seqlen= 32,
         device= 'cuda:0'
    )
    tsf.cuda()
    tsf.float()

    out , src , en_self_att,de_att , de_self_att = tsf(x,xt,needall= True)


    print(out)
    encnet = tsf.__return__encoder__()

    torch.save(encnet , '/home/zhaoxiaohui/20210927/model_save/testenc.pkl')

    encnetload = torch.load('/home/zhaoxiaohui/20210927/model_save/testenc.pkl')


    outl = encnetload(x)
    print(outl)


    el = ncoder_lin(
        premodel_path= '/home/zhaoxiaohui/20210927/model_save/testenc.pkl',
        pre_output= 10,
        output_num= 2
    )

    el.cuda()
    el.float()

    oel = el(x)

    print(el)





    
    
    



