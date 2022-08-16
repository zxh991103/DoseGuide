import torch
import torch.nn as nn
from utils.utils import get_train_data , load_dynamic_data
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import datetime

from model.embedding_model import static_unnum,static_num,encoder_cnn # 16 16 32 
from model.pyGAT import tsGAT
from model.pyGAT import accuracy
from utils.utils import test_para

savepath = "/home/zhaoxiaohui/20210927/model_save/lr_{}_ep_200000_tsgat_91/model_epo_{}_loss_{}_acc_{}_new.pkl"
jspath = "/home/zhaoxiaohui/20210927/model_save/train_log/model_lr_0.0001_tsgat_epo_{}_noval_new.json"

premodel_path = '/home/zhaoxiaohui/20210927/model_save/lr_0.001_all_encoder/model_epo_520_loss_0.10194980871930451_.pkl'
tg = tsGAT(
        prepath= premodel_path
)
LR = 0.0001
tg.float()
tg.cuda()
loss_function = F.nll_loss
optimizer = torch.optim.Adam(tg.parameters(), lr=LR)

epochs = 200000


if __name__ == '__main__':
    
    

    
    
    

    trainseq,valseq,testseq,\
    static_no_num_train,static_no_num_val,static_no_num_test,  \
    static_num_train,static_num_val,static_num_test, \
    y_train,y_val,y_test , \
    allseq,all_static_no_num,all_static_num, \
    idx_train,idx_val,idx_test,y = get_train_data()


    allseq = torch.from_numpy(np.array(allseq)).to('cuda:0').to(torch.float32).view(-1,30,7)
    all_static_no_num = torch.from_numpy(np.array(all_static_no_num)).to('cuda:0').to(torch.float32).view(-1,123)
    all_static_num = torch.from_numpy(np.array(all_static_num)).to('cuda:0').to(torch.float32).view(-1,6)
    y = y.to('cuda:0')
    starttime = datetime.datetime.now()
    for epoch in  range(epochs):

        
        tg.train()
        out,_,_ ,_,_,_,_= tg(allseq,all_static_no_num,all_static_num)
        optimizer.zero_grad()
        loss = loss_function(out[idx_train],y[idx_train])
        loss.backward()
        optimizer.step()
        acc = accuracy(out[idx_train],y[idx_train])
        accp , lr = test_para(out,y,idx_train)
    
        if epoch % 200 ==0 or epoch < 10:
            endtime = datetime.datetime.now()

            print("epoch : {} , loss : {} , acc : {} , time : {}s".format(
                epoch,
                float(loss),
                float(acc),
                (endtime - starttime).seconds
            ))
            print(lr)

            starttime = datetime.datetime.now()


        # if epoch < 10 or epoch % 200 == 0:
            # torch.save(tg,savepath.format(LR,epoch,float(loss),float(acc)))

            # with open(jspath.format(epochs),'a+') as f:
            #     tg.eval()
            #     # loss = loss_function(out[idx_val],y[idx_val])
            #     # acc = accuracy(out[idx_val],y[idx_val])
            #     # accp , lr = test_para(out,y,idx_val)
            #     # print("val epoch: {} , loss: {} , accuracy : {} , TP : {} , TN : {} , FP : {} ,FN : {} , SENSITIVITY : {} , SPECIFICITY : {}".format(epoch,float(loss),float(acc),lr[0],lr[1],lr[2],lr[3],lr[4],lr[5]),file=f)
            #     loss = loss_function(out[idx_test],y[idx_test])
            #     acc = accuracy(out[idx_test],y[idx_test])
            #     accp , lr = test_para(out,y,idx_test)
            #     print("test epoch: {} , loss: {} , accuracy : {} , TP : {} , TN : {} , FP : {} ,FN : {} , SENSITIVITY : {} , SPECIFICITY : {}".format(epoch,float(loss),float(acc),lr[0],lr[1],lr[2],lr[3],lr[4],lr[5]),file=f)
            # f.close()
            



   

    

