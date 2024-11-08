import torch.nn as nn
from transformers import BertForPreTraining
import torch
from torchcrf import CRF

class BertModel(nn.Module):
    def __init__(self, classnum, criterion):
        super().__init__()
        self.bert = BertForPreTraining.from_pretrained(pretrained_model_name_or_path='./bert-base-uncased').bert
        self.classifier = nn.Linear(768, classnum)
        self.criterion = criterion

    def forward(self, batchdata, batchlabel=None):
        bertOut = self.bert(batchdata)
        bertOut0, bertOut1 = bertOut[0], bertOut[1]
        pre = self.classifier(bertOut0)
        if batchlabel is not None:
            # import pdb; pdb.set_trace()
            loss = self.criterion(pre.reshape(-1, pre.shape[-1]), batchlabel.reshape(-1))
            return loss
        else:
            return torch.argmax(pre, dim=-1)
        
class Bert_Crf_Model(nn.Module):
    def __init__(self, classnum, criterion):
        super().__init__()
        self.bert = BertForPreTraining.from_pretrained(pretrained_model_name_or_path='./bert-base-uncased').bert
        self.classifier = nn.Linear(768, classnum)
        self.dropout = nn.Dropout(0.1)
        self.crf = CRF(classnum, batch_first=True)
        self.criterion = criterion

    def forward(self, batchdata, batchlabel=None):
        bertOut = self.bert(batchdata)
        bertOut0 = bertOut[0]
        dropoutOut = self.dropout(bertOut0)
        pre = self.classifier(dropoutOut)
        if batchlabel is not None:
            loss =  -self.crf(pre, batchlabel, mask=batchdata.gt(0), reduction='mean')
            return loss
        else:
            return self.crf.decode(pre, mask=batchdata.gt(0))
        
class Bert_lstm(nn.Module):
    def __init__(self, classnum, criterion, hidden_dim=1024, dropout_prob=0.1):
        super().__init__()
        self.bert = BertForPreTraining.from_pretrained(pretrained_model_name_or_path='./bert-base-uncased').bert
        self.dropout = nn.Dropout(dropout_prob)
        self.bilstm = nn.LSTM(input_size=768, hidden_size=hidden_dim, num_layers=2, bidirectional=True, batch_first=True, dropout=0.5)
        self.classifier = nn.Linear(hidden_dim * 2, classnum)  # BiLSTM输出维度是hidden_dim的2倍
        self.criterion = criterion

    def forward(self, batchdata, batchlabel=None):
        bertOut = self.bert(batchdata)[0]  # 只取BERT的最后一层输出
        dropoutOut = self.dropout(bertOut)
        lstmOut, _ = self.bilstm(dropoutOut)
        # [256, 77, 2048]
        pre = self.classifier(lstmOut)
        if batchlabel is not None:
            loss = self.criterion(pre.reshape(-1, pre.shape[-1]), batchlabel.reshape(-1))
            return loss
        else:
            return torch.argmax(pre, dim=-1)
        

class Bert_CNN(nn.Module):
    def __init__(self, classnum, criterion, dropout_prob=0.1, num_filters=3, filter_size=3):
        super().__init__()
        self.bert = BertForPreTraining.from_pretrained(pretrained_model_name_or_path='./bert-base-uncased').bert
        self.dropout = nn.Dropout(dropout_prob)
        self.conv = nn.Conv2d(1, num_filters, kernel_size=(filter_size, 768), padding=(filter_size // 2, 0))
        self.classifier = nn.Linear(num_filters, classnum)
        self.criterion = criterion

    def forward(self, batchdata, batchlabel=None):
        bertOut = self.bert(batchdata)[0]  # 只取BERT的最后一层输出
        bertOut = bertOut.unsqueeze(1)  # 增加一个通道维度
        convOut = self.conv(bertOut)
        convOut = convOut.squeeze(3)  # 去掉最后一个维度
        dropoutOut = self.dropout(convOut)
        # torch.Size([256, 3, 77])
        pre = self.classifier(dropoutOut.transpose(1, 2))  # 转置以匹配分类器的输入
        if batchlabel is not None:
            loss = self.criterion(pre.view(-1, pre.shape[-1]), batchlabel.view(-1))
            return loss
        else:
            return torch.argmax(pre, dim=-1)
        
class Bert_CNN_LSTM(nn.Module):
    def __init__(self, classnum, criterion, hidden_dim=512, dropout_prob=0.1, num_filters=256, filter_size=3):
        super().__init__()
        self.bert = BertForPreTraining.from_pretrained(pretrained_model_name_or_path='./bert-base-uncased').bert
        self.dropout = nn.Dropout(dropout_prob)
        
        # CNN部分
        self.conv = nn.Conv2d(1, num_filters, kernel_size=(filter_size, 768), padding=(filter_size // 2, 0))
        
        # LSTM部分
        self.bilstm = nn.LSTM(input_size=768, hidden_size=hidden_dim, num_layers=2, bidirectional=True, batch_first=True, dropout=0.5)
        
        # 分类器
        self.classifier = nn.Linear(num_filters + hidden_dim * 2, classnum)  # CNN和BiLSTM输出维度之和
        self.criterion = criterion

    def forward(self, batchdata, batchlabel=None):
        bertOut = self.bert(batchdata)[0]  # 只取BERT的最后一层输出
        
        # CNN部分
        cnnOut = bertOut.unsqueeze(1)  # 增加一个通道维度
        cnnOut = self.conv(cnnOut)
        cnnOut = cnnOut.squeeze(3)  # 去掉最后一个维度
        cnnOut = self.dropout(cnnOut)
        cnnOut = cnnOut.transpose(1, 2)  # 转置以匹配LSTM的输入
        
        # LSTM部分
        lstmOut, _ = self.bilstm(bertOut)
        
        # 拼接CNN和LSTM的输出
        concatOut = torch.cat((cnnOut, lstmOut), dim=2)
        
        # 分类器
        pre = self.classifier(concatOut)
        
        if batchlabel is not None:
            loss = self.criterion(pre.view(-1, pre.shape[-1]), batchlabel.view(-1))
            return loss
        else:
            return torch.argmax(pre, dim=-1)
