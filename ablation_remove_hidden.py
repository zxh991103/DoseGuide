from model.AblationModel import cnn_lstm_attention_gat,get_data

import datetime
from model.pyGAT import accuracy
from utils.utils import test_para
import torch
import torch.nn.functional as F
import numpy as np
savepath = "/home/zhaoxiaohui/20210927_v3/model_save/ablation_remove_hidden/model_epo_{}_loss_{}_acc_{}_new.pkl"
jspath = "/home/zhaoxiaohui/20210927_v3/model_save/train_log/ablation_remove_hidden_epo_{}_noval_fc.json"

tg = cnn_lstm_attention_gat()
LR = 0.0001
tg.float()
tg.cuda()
loss_function = F.nll_loss
optimizer = torch.optim.Adam(tg.parameters(), lr=LR)

epochs = 200000

if __name__ == '__main__':
    allseq2t,allst,idx_train,idx_val,idx_test,y = get_data()


    
    y = y.to('cuda:0')
    starttime = datetime.datetime.now()
    for epoch in  range(epochs):

        
        tg.train()
        out,_,_ ,adj= tg(allseq2t,allst)
        optimizer.zero_grad()
        loss = loss_function(out[idx_train],y[idx_train])
        loss.backward()
        optimizer.step()
        acc = accuracy(out[idx_train],y[idx_train])
        accp , lr = test_para(out,y,idx_train)
    
        if epoch % 100 ==0 or epoch < 10:
            endtime = datetime.datetime.now()

            print("epoch : {} , loss : {} , acc : {} , time : {}s".format(
                epoch,
                float(loss),
                float(acc),
                (endtime - starttime).seconds
            ))
            print(lr)

            starttime = datetime.datetime.now()


        if epoch < 10 or epoch % 100 == 0:
            torch.save(tg,savepath.format(epoch,float(loss),float(acc)))

            with open(jspath.format(epochs),'a+') as f:
                tg.eval()
                # loss = loss_function(out[idx_val],y[idx_val])
                # acc = accuracy(out[idx_val],y[idx_val])
                # accp , lr = test_para(out,y,idx_val)
                # print("val epoch: {} , loss: {} , accuracy : {} , TP : {} , TN : {} , FP : {} ,FN : {} , SENSITIVITY : {} , SPECIFICITY : {}".format(epoch,float(loss),float(acc),lr[0],lr[1],lr[2],lr[3],lr[4],lr[5]),file=f)
                loss = loss_function(out[idx_test],y[idx_test])
                acc = accuracy(out[idx_test],y[idx_test])
                accp , lr = test_para(out,y,idx_test)
                print("test epoch: {} , loss: {} , accuracy : {} , TP : {} , TN : {} , FP : {} ,FN : {} , SENSITIVITY : {} , SPECIFICITY : {}".format(epoch,float(loss),float(acc),lr[0],lr[1],lr[2],lr[3],lr[4],lr[5]),file=f)
            f.close()

