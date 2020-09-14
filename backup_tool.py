import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F



def Evaluation(Y_hat,Y):
    #print(Y_hat)
    #print(Y)
    from sklearn.metrics import f1_score
    TP,FP,FN,TN = 0,0,0,0
    for i in range(len(Y_hat)):
        if int(Y_hat[i]) == 1 and int(Y[i]) ==1:
            TP+=1
        elif int(Y_hat[i]) == 1 and int(Y[i]) ==0:
            FP +=1
        elif int(Y_hat[i]) == 0 and int(Y[i]) ==1:
            FN +=1
        elif int(Y_hat[i]) == 0 and int(Y[i]) ==0:
            TN +=1
        else:
            print('[ERROR]')
    Accuracy = (TP+TN+1)/(TP+FP+FN+TN+1)
    Precision = (TP+1)/(TP+FP+1)
    Recall = (TP+1)/(TP+FN+1)
    F1 = f1_score(Y, Y_hat)

    print('Accuracy:',Accuracy)
    print('Precision:',Precision)
    print('Recall:',Recall)
    print('F1:',F1)

def batch_str2batch_tensor(batch_X,batch_Y,Sent_BERT_model):
    batch_Y_tensor = torch.tensor(batch_Y,dtype=torch.long).cuda()
    batch_X_tensor = list()
    for i in range(batch_X.size()[0]):
        batch_X_tensor.append([torch.tensor(batch_X[i,0,:]).cuda(),torch.tensor(batch_X[i,1,:]).cuda()])
    return batch_X_tensor,batch_Y_tensor

def Train_Eval_Process_Layer(train_X,train_Y,test_X,test_Y,Sent_BERT_model):
    # RetaGNN + Self Attention
    # train_X = [batch,batch,...] where batch = [batch_num,L,dim]_tensor
    # train_Y = [batch,batch,...] where batch = [batch_num,1]_tensor
    # test_X = [one_batch]where one_batch = [1,L,dim]_tensor
    # test_Y = list 
    import pyprind
    import pickle
    epoch_num = 15
    model = NLI_model(input_dim=256).cuda()
    #model.aux_logits = False
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    for epoch_  in range(epoch_num):
        model.train() 
        for i in pyprind.prog_bar(range(len(train_X))):
            batch_X,batch_Y = train_X[i],train_Y[i] 
            batch_X = torch.tensor(batch_X).cuda()
            batch_Y = torch.tensor(batch_Y).cuda()
            #batch_X,batch_Y = batch_str2batch_tensor(batch_X,batch_Y,Sent_BERT_model)
            batch_Y_hat,_ = model(batch_X)
            loss = criterion(batch_Y_hat, batch_Y)#.float())
            #loss = F.cross_entropy(batch_Y_hat, batch_Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #print('loss:',loss)
        model.eval()
        pred_Y_list,true_Y_list = list(),list()
        for i in range(len(test_X)):
            batch_X,batch_Y = train_X[i],train_Y[i]
            batch_X = torch.tensor(batch_X).cuda()
            batch_Y = torch.tensor(batch_Y).cuda()
            #batch_X,batch_Y = batch_str2batch_tensor(batch_X,batch_Y,Sent_BERT_model)
            pred_Y,_ = model(batch_X)
            pred_Y = pred_Y.argmax(dim=1)
            pred_Y = list(pred_Y.cpu().data.numpy())
            true_Y = list(batch_Y.cpu().data.numpy())
            pred_Y_list += pred_Y
            true_Y_list += true_Y
        Evaluation(pred_Y_list,true_Y_list)
    torch.save(model.state_dict(), 'demo2.pkl')

class NLI_model(nn.Module):
    def __init__(self,input_dim):
        super(NLI_model,self).__init__()
        hidden_dim = input_dim
        self.input_dim = input_dim
        self.nli_embedding = nn.Linear(input_dim*3, hidden_dim)
        self.fc = nn.Linear(input_dim, 3)
        self.m = nn.LogSoftmax()
    def forward(self,s1_s2):
        s1,s2 = s1_s2[:,0,:],s1_s2[:,1,:]
        s1_minus_s2_abs = torch.abs(s1-s2)
        s1_2 = torch.cat([s1,s2,s1_minus_s2_abs],1)
        embedding = self.nli_embedding(s1_2)
        pred_Y = self.fc(embedding)
        return pred_Y,embedding

class Cross_NLI_Model(nn.Module):
    def __init__(self):
        super(Cross_NLI_Model,self).__init__()
        input_dim = 256
        self.input_dim = 256
        self.nli_model = NLI_model(input_dim=256).cuda()
        self.D2D_FC = nn.Linear(input_dim,input_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.fc = nn.Linear(input_dim,1)
        self.sigmoid_layer = nn.Sigmoid()
    def forward(self,X,sent_embedding):
        sent_embedding = nn.Embedding.from_pretrained(sent_embedding)
        X = sent_embedding(X) #bz,l,d,2
        pair_embedding_list = list()
        for i in range(X.size()[0]):
            X_i = X[i,:,:,:]
            _,pair_embedding = self.nli_model(X_i) #l ,d
            pair_embedding = pair_embedding.view(1,-1,self.input_dim)
            pair_embedding_list.append(pair_embedding)
        pair_embedding_tensor = torch.cat(pair_embedding_list,0)
        #'bz_pair_embedding_torch: torch.Size([32, 1225, 256])'
        attention = self.softmax(torch.sum(self.D2D_FC(pair_embedding_tensor),-1))
        #'attention: torch.Size([32, 1225])'
        out = torch.sum(torch.matmul(attention,pair_embedding_tensor),1)
        #'out: torch.Size([32, 256])'
        pred_Y = self.sigmoid_layer(self.fc(out))
        return pred_Y

def Train_Eval_Process_Layer_for_CROSS_NLI(train_X,train_Y,test_X,test_Y):
    import pyprind
    import pickle
    epoch_num = 15
    model = Cross_NLI_Model().cuda()
    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCELoss()
    for epoch_  in range(epoch_num):
        model.train() 
        for i in pyprind.prog_bar(range(len(train_X))):
            batch_X,batch_Y = train_X[i],train_Y[i] 
            bz_L_2,sent_embedding = batch_X[0],batch_X[1]
            bz_L_2 = torch.tensor(bz_L_2).cuda()
            sent_embedding = torch.FloatTensor(sent_embedding).cuda()
            batch_Y = torch.tensor(batch_Y).cuda()
            batch_Y_hat = model(bz_L_2,sent_embedding)
            loss = criterion(batch_Y_hat, batch_Y.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #print('loss:',loss)
    return model



class One_Sent2Other_Sent(nn.Module):
    def __init__(self):
        super(One_Sent2Other_Sent,self).__init__()
        input_dim = 256
        self.input_dim = 256
        #self.nli_model = NLI_model(input_dim=256).cuda()
        self.D2D_FC = nn.Linear(input_dim,input_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.single_sent2other_sent_FCL = nn.Linear(input_dim*3,input_dim)
        self.sigmoid_layer = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.rnn = nn.LSTM(input_dim, input_dim)
        self.attention_weight = nn.Linear(self.input_dim, 1)
        self.fc = nn.Linear(self.input_dim, 1)
    
    def Attention_Layer(self,hidden_vec):
        self.attn = self.attention_weight(hidden_vec)
        out = torch.sum(hidden_vec * self.attn,1)
        return out    
    
    def forward(self,X):
        single_sent2other_info = list()
        for i in range(X.size()[0]):
            X_i = X[i,:]
            print(X_i)
            X_i = X_i.view(1,-1)
            if i != 0 and i != int(X.size()[0] -1):
                X1 = X[:i,:].view(-1,256)
                X2 = X[i+1:,:].view(-1,256)
                X_other = torch.cat([X1,X2],0)
            elif i == int(X.size()[0] -1):
                X_other = X[:i-1,:]
            elif i == 0 :
                X_other = X[i+1:,:]
            else:
                print('[ERROR]: nan index exist.')
            X_other_len = X_other.size()[0]
            if X_other_len !=0:
                X_i_long_tensor = torch.cat([X_i for j in range(X_other_len)],0)
                X_i_ohter_residual = torch.cat([X_i_long_tensor,X_other,torch.abs(X_i_long_tensor-X_other)],1)
                X_i_ohter_residual = self.relu(self.single_sent2other_sent_FCL(X_i_ohter_residual))
                X_i_ohter_residual = torch.max(X_i_ohter_residual,0)[0].view(1,-1)
                single_sent2other_info.append(X_i_ohter_residual)
        single_sent2other_info = torch.cat(single_sent2other_info,0).view(1,-1,self.input_dim)
        hidden_vec, (h_n, c_n) = self.rnn(single_sent2other_info)
        attn_layer_out = self.Attention_Layer(hidden_vec)
        out = self.fc(attn_layer_out)
        pred_Y = self.sigmoid_layer(out)
        return pred_Y


import pyprind
import torch
import torch.nn as nn
import torch.optim as optim
#train_X = [(b,l,d),(b,l,d),...] ; train_Y = [(b,),(b,),...]
#test_X = (N,l,d)  test_Y = (N,)

def Train_Eval_Process_Layer_v2(train_X,train_Y,test_X,test_Y):
    # LSTM
    epoch_num = 10
    #model = LSTM_model(input_dim=8,hidden_dim=8)
    model = One_Sent2Other_Sent()
    optimizer = optim.Adam(model.parameters())
    criterion = nn.BCELoss()
    for epoch_  in pyprind.prog_bar(range(epoch_num)):
        model.train()
        for i in range(len(train_X)):
            X = torch.tensor(train_X[i])#.cuda()
            pred_train_Y = model(X)
            Y = torch.tensor([train_Y[i]])#.cuda()
            true_train_Y = Y.squeeze(dim=-1)
            loss = criterion(pred_train_Y, true_train_Y.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #print('loss:',loss)
        model.eval()
        pred_test_Y = list()
        for i in range(len(test_X)):
            X = torch.tensor(test_X[i])#.cuda()
            pred_test_Y_i = model(X).cpu().data.numpy().reshape(1,1)
            pred_test_Y.append(pred_test_Y_i)
        test_Y_hat = np.concatenate(pred_test_Y,0)
        test_Y_hat_list = list()
        for i in range(test_Y_hat.shape[0]):
            if test_Y_hat[i,0] >= 0.5:
                test_Y_hat_list.append(1)
            else:
                test_Y_hat_list.append(0)
        Evaluation(test_Y_hat_list,test_Y)


def Evaluation(Y_hat,Y):
    from sklearn.metrics import f1_score
    TP,FP,FN,TN = 0.0001,0.0001,0.0001,0.0001
    for i in range(len(Y_hat)):
        if int(Y_hat[i]) == 1 and int(Y[i]) ==1:
            TP+=1
        elif int(Y_hat[i]) == 1 and int(Y[i]) ==0:
            FP +=1
        elif int(Y_hat[i]) == 0 and int(Y[i]) ==1:
            FN +=1
        elif int(Y_hat[i]) == 0 and int(Y[i]) ==0:
            TN +=1
        else:
            print('[ERROR]')
    Accuracy = (TP+TN)/(TP+FP+FN+TN)
    Precision = (TP)/(TP+FP)
    Recall = (TP)/(TP+FN)
    F1 = f1_score(Y, Y_hat)

    print('Accuracy:',Accuracy)
    print('Precision:',Precision)
    print('Recall:',Recall)
    print('F1:',F1)