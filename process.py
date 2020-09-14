from sentence_transformers import SentenceTransformer, models
from torch import nn
from backup_tool import Train_Eval_Process_Layer_for_CROSS_NLI,Evaluation
import random
import pyprind
import numpy as np
import torch


# entailment 0
# contradiction 1
# neutral 2

class MNLI_multi_sent:
    def __init__(self):
        word_embedding_model = models.Transformer('sentence-transformers/bert-large-nli-max-tokens', max_seq_length=256)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(), out_features=256, activation_function=nn.Tanh())
        self.model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model])
        path = 'multinli_1.0/'
        self.MNLI_train_path = path + 'multinli_1.0_train.txt'
        self.MNLI_matched_test_path = path + 'multinli_1.0_dev_matched.txt'
        self.MNLI_mismatched_test_path = path + 'multinli_1.0_dev_mismatched.txt'


    def wiki_load(self):
        import pickle
        with open('clean_POS_NEG_data.pickle', 'rb') as file:
            clean_POS_NEG_data =pickle.load(file)
        self.pos_X = clean_POS_NEG_data['pos_X']
        self.neg_X = clean_POS_NEG_data['neg_X']
        pos_title = clean_POS_NEG_data['pos_title']
        neg_title = clean_POS_NEG_data['neg_title']
        pos_X_SC = clean_POS_NEG_data['pos_X_SC']
        neg_X_SC = clean_POS_NEG_data['neg_X_SC']

    def Read_Data_X_Y(self,data_path):
        with open(data_path,'r') as f:
            data = f.readlines()
        X,Y = list(),list()
        for i in range(len(data)-1):
            data_i = data[i+1]
            label = data_i.split('\t')[0]
            s1 = data_i.split('\t')[5]
            s2 = data_i.split('\t')[6]
            Y_value = None
            if label == 'entailment':
                Y_value = 0
            elif label == 'contradiction':
                Y_value = 1
            elif label == 'neutral':
                Y_value = 2
            if Y_value is not None:      
                X.append([s1,s2])
                Y.append(Y_value)
        return X,Y

    def Seperate_Y2Data(self,X,Y,Y2data):
        for i in range(len(Y)):
            if Y[i] not in Y2data:
                Y2data[Y[i]] = list()
            Y2data[Y[i]].append(X[i])
        return Y2data

    def Sent_Num_Bound(self,interval,max_sent_num):
        sent_num_bound = list()
        for i in range(max_sent_num):
            if int(i+1) % interval ==0:
                sent_num_bound.append(i+1)
        return sent_num_bound

    def Contradiction_Fator(self,data_size,one_factor_rate=0.9,two_factor_rate=0.1):
        one_factor_num = int(data_size * one_factor_rate)
        two_factor_num = int(data_size * two_factor_rate)
        contradiction_fator_list = [1 for i in range(one_factor_num)] + [2 for i in range(two_factor_num)]
        contradiction_fator_list = random.sample(contradiction_fator_list,len(contradiction_fator_list))
        return contradiction_fator_list

    def Generate_Pair_Sentence_list(self,sent_num_bound,contradiction_fator_list,Y2data,contradiction_or_not=True):
        if contradiction_or_not:
            # randomly select contradiction_fator_num (one or two)
            contradiction_fator_num = random.sample(contradiction_fator_list,1)[0]
            # randomly select contradiction_fator
            s1_s2_contradiction_list = random.sample(Y2data[1],contradiction_fator_num)
        else:
            contradiction_fator_num = 0
            s1_s2_contradiction_list = []
        # determine non-contradiction_fator
        pair_num = sent_num_bound * (sent_num_bound-1)/2
        X0_2 = Y2data[0] + Y2data[2]
        s1_s2_non_contradiction_list = random.sample(X0_2,int(pair_num - contradiction_fator_num))
        s1_s2_list = s1_s2_contradiction_list + s1_s2_non_contradiction_list
        s1_s2_list = random.sample(s1_s2_list,len(s1_s2_list))
        return s1_s2_list

    def Batch_Data_X_Y(self,info_):
        sent_num_bound_list = info_[0]
        sent_num_bound = random.sample(sent_num_bound_list,1)[0]
        contradiction_fator_list = info_[1]
        Y2data = info_[2]
        batch_size = info_[3]
        pair_sentence_list = self.Generate_Pair_Sentence_list(sent_num_bound,contradiction_fator_list,Y2data,contradiction_or_not=True)
        pos_num = int(batch_size/2)
        neg_num = batch_size - pos_num
        idx = 0
        POS_part,NEG_part,sent_list = list(),list(),list()
        for i in range(pos_num):
            pair_sentence_list = self.Generate_Pair_Sentence_list(sent_num_bound,contradiction_fator_list,Y2data,contradiction_or_not=True)
            POS_part_i = list()
            for j in range(len(pair_sentence_list)):
                sent_1_idx = idx
                sent_list.append(pair_sentence_list[j][0])
                idx +=1
                sent_2_idx = idx
                sent_list.append(pair_sentence_list[j][1])
                POS_part_i.append([sent_1_idx,sent_2_idx])
            POS_part.append(POS_part_i)
        for i in range(neg_num):
            pair_sentence_list = self.Generate_Pair_Sentence_list(sent_num_bound,contradiction_fator_list,Y2data,contradiction_or_not=False)
            NEG_part_i = list()
            for j in range(len(pair_sentence_list)):
                sent_1_idx = idx
                sent_list.append(pair_sentence_list[j][0])
                idx +=1
                sent_2_idx = idx
                sent_list.append(pair_sentence_list[j][1])
                NEG_part_i.append([sent_1_idx,sent_2_idx])
            NEG_part.append(NEG_part_i)
        POS_NEG_X_part = POS_part + NEG_part
        POS_NEG_Y_part = [1 for i in range(len(POS_part))] + [0 for i in range(len(NEG_part))]
        shuffle_index = random.sample([i for i in range(len(POS_NEG_X_part)) ],len(POS_NEG_X_part))
        POS_NEG_X_part = [POS_NEG_X_part[shuffle_index[i]] for i in range(len(shuffle_index))]
        POS_NEG_X_part = np.array(POS_NEG_X_part)
        POS_NEG_Y_part = [POS_NEG_Y_part[shuffle_index[i]] for i in range(len(shuffle_index))]
        batch_Y = np.array(POS_NEG_Y_part).reshape(-1,1)
        sent_embedding = self.model.encode(sent_list)
        batch_X = [POS_NEG_X_part,sent_embedding]
        return [batch_X,batch_Y]
            
    def Data_X_Y_Function(self,past_info):
        sent_num_bound_list = past_info[0]
        contradiction_fator_list = past_info[1]
        Y2data = past_info[2]
        batch_size = past_info[4]
        POS_or_NEG = past_info[3]
        data_X,data_Y = list(),list()
        if POS_or_NEG == 'POS' :
            contradiction_or_not=True
            Y_value = 1
        else:
            contradiction_or_not=False
            Y_value = 0      
        batch_X,batch_Y = list(),list()
        sent_num_bound = random.sample(sent_num_bound_list,1)[0]
        idx,sent_list,batch_pair_sent_idx_list = 0,list(),list()
        for j in range(batch_size):
            pair_sentence_list = self.Generate_Pair_Sentence_list(sent_num_bound,contradiction_fator_list,Y2data,contradiction_or_not)
            pair_sent_idx_list = list()
            for k in range(len(pair_sentence_list)):
                sent1 = pair_sentence_list[k][0]
                sent2 = pair_sentence_list[k][1]
                sent_list.append(sent1)
                sent_list.append(sent2)
                pair_sent_idx_list.append([idx,idx+1])
                idx +=2
            batch_pair_sent_idx_list.append(pair_sent_idx_list)
            batch_Y.append(Y_value)
        batch_Y = np.array(batch_Y)
        batch_pair_sent_idx_array = np.array(batch_pair_sent_idx_list) #bz,L,2
        sent_embedding = self.model.encode(sent_list) #l,d
        batch_X = [batch_pair_sent_idx_array,sent_embedding]
        return [batch_X,batch_Y]

    def Train_Test_Function_v0(self,data_X,data_Y,train_rate,batch_num):
        sample_index = random.sample([i for i in range(batch_num)],batch_num)
        sample_train_index = sample_index[:int(len(sample_index)*train_rate)]
        sample_test_index = sample_index[int(len(sample_index)*train_rate):]

        train_X = [data_X[sample_train_index[i]] for i in range(len(sample_train_index))]
        train_Y = [data_Y[sample_train_index[i]] for i in range(len(sample_train_index))]
        test_X = [data_X[sample_test_index[i]] for i in range(len(sample_test_index))]
        test_Y = [data_Y[sample_test_index[i]] for i in range(len(sample_test_index))]

        shuffle_index = random.sample([i for i in range(len(train_X)) ],len(train_X))
        train_X = [train_X[shuffle_index[i]] for i in range(len(shuffle_index))]
        train_Y = [train_Y[shuffle_index[i]] for i in range(len(shuffle_index))]

        shuffle_index = random.sample([i for i in range(len(test_X)) ],len(test_X))
        test_X = [test_X[shuffle_index[i]] for i in range(len(shuffle_index))]
        test_Y = [test_Y[shuffle_index[i]] for i in range(len(shuffle_index))]
        return train_X,train_Y,test_X,test_Y

    def Restrict_Sent_Num(self,X,restricted_sent_num=200):
        X_restrict_sent_num = list()
        for i in range(len(X)):
            if len(X[i]) <= restricted_sent_num and len(X[i])>1:
                X_restrict_sent_num.append(X[i])
        return X_restrict_sent_num

    def Prediction_BY_Cross_NLI_Model(self,X,cross_nli_model):
        pred_Y_list = list()
        for i in range(len(X)):
            X_i = X[i]
            batch_pair_sent_idx_array,sent_embedding = self._wiki_sent2pair_sent_(X_i)
            batch_pair_sent_idx = torch.tensor(batch_pair_sent_idx_array).cuda()
            sent_embedding = torch.tensor(sent_embedding).cuda()
            pred_Y = cross_nli_model(batch_pair_sent_idx,sent_embedding)
            pred_Y_list.append(pred_Y)
        pred_Y_array = torch.cat(pred_Y_list,0).cpu().data.numpy()
        test_Y_hat_list = list()
        for i in range(pred_Y_array.shape[0]):
            if pred_Y_array[i,0] >= 0.5:
                test_Y_hat_list.append(1)
            else:
                test_Y_hat_list.append(0)
        return test_Y_hat_list

    def _wiki_sent2pair_sent_(self,sent_list):
        pair_sentence_list = list()
        for i in range(len(sent_list)):
            for j in range(len(sent_list)):
                if i > j:
                    pair_sentence_list.append([sent_list[i],sent_list[j]])
        sent_list = list()
        pair_sent_idx_list = list()
        idx = 0
        for k in range(len(pair_sentence_list)):
            sent1 = pair_sentence_list[k][0]
            sent2 = pair_sentence_list[k][1]
            sent_list.append(sent1)
            sent_list.append(sent2)
            pair_sent_idx_list.append([idx,idx+1])
            idx +=2
        batch_pair_sent_idx_array = np.array([pair_sent_idx_list]) #bz,L,2
        sent_embedding = self.model.encode(sent_list) #l,d
        return batch_pair_sent_idx_array,sent_embedding

    def Wiki_Test_Part(self,cross_nli_model):
        self.wiki_load()
        pos_X = self.Restrict_Sent_Num(self.pos_X)
        neg_X = self.Restrict_Sent_Num(self.neg_X)

        POS_pred_Y_list = self.Prediction_BY_Cross_NLI_Model(pos_X,cross_nli_model)
        NEG_pred_Y_list = self.Prediction_BY_Cross_NLI_Model(neg_X,cross_nli_model)
        pred_Y_list = POS_pred_Y_list + NEG_pred_Y_list
        test_Y = [1 for i in range(len(POS_pred_Y_list))]+[0 for i in range(len(NEG_pred_Y_list))]
        Evaluation(pred_Y_list,test_Y)

    def main(self):
        # load data
        X1,Y1 = self.Read_Data_X_Y(self.MNLI_train_path)
        X2,Y2 = self.Read_Data_X_Y(self.MNLI_matched_test_path)
        X3,Y3 = self.Read_Data_X_Y(self.MNLI_mismatched_test_path)
        Y2data = dict()
        Y2data = self.Seperate_Y2Data(X1,Y1,Y2data)
        Y2data = self.Seperate_Y2Data(X2,Y2,Y2data)
        Y2data = self.Seperate_Y2Data(X3,Y3,Y2data)
        # parameter setting
        train_rate = 0.8
        #data_size = len(Y2data[0]) + len(Y2data[1]) + len(Y2data[2])
        #batch_size = 32
        data_size = 12
        batch_size = 10
        batch_num = int(data_size/batch_size) + 1
        contradiction_fator_list = self.Contradiction_Fator(data_size,one_factor_rate=0.9,two_factor_rate=0.1)
        interval = 10
        #max_sent_num = 400
        max_sent_num = 20
        sent_num_bound_list = self.Sent_Num_Bound(interval,max_sent_num)
        sent_num_bound = random.sample(sent_num_bound_list,1)[0]
        # generate batch data
        info_ = [sent_num_bound_list,contradiction_fator_list,Y2data,batch_size]
        batch_X_Y_list = list()
        for i in range(batch_num):
            batch_X_Y_list.append(self.Batch_Data_X_Y(info_))
        data_X = [batch_X_Y_list[i][0] for i in range(len(batch_X_Y_list))]
        data_Y = [batch_X_Y_list[i][1] for i in range(len(batch_X_Y_list))]
        # split each batch data into train,test data.
        print('Start to Train NLI-Text Model.')
        train_X,train_Y,test_X,test_Y = self.Train_Test_Function_v0(data_X,data_Y,train_rate,batch_num)
        # Training NLI-text model
        cross_nli_model = Train_Eval_Process_Layer_for_CROSS_NLI(train_X,train_Y,test_X,test_Y)
        # Testing Wiki Dataset
        print('Start to Evaluate Wikipedia Article.')
        self.Wiki_Test_Part(cross_nli_model)

if __name__ == '__main__':
    mnli_multi_sent= MNLI_multi_sent()
    mnli_multi_sent.main()