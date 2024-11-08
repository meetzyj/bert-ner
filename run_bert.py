import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from seqeval.metrics import accuracy_score, f1_score
import time
from tqdm import tqdm
from matplotlib import pyplot as plt
import logging
from dataset import readFile, label2index, NerDataset
from model import BertModel

# 设置logger
logging.basicConfig(filename='training2.log', level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger()

if __name__ == '__main__':
    # 超参数
    batchsize = 256
    epoch = 250
    maxlength = 75
    lr = 0.03
    weight_decay = 0.00001
    patience = 50

    # 读取数据
    trainData, trainLabel = readFile('train.txt')
    devData, devLabel = readFile('dev.txt')
    testData, testLabel = readFile('test.txt')

    # 构建词表
    labelIndex, indexLabel = label2index(trainLabel)

    # 构建数据集,迭代器
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path='./bert-base-uncased')
    trainDataset = NerDataset(trainData, trainLabel, labelIndex, tokenizer, maxlength)
    trainDataloader = DataLoader(trainDataset, batch_size=batchsize, shuffle=False)
    devDataset = NerDataset(devData, devLabel, labelIndex, tokenizer, maxlength)
    devDataloader = DataLoader(devDataset, batch_size=batchsize, shuffle=False)
    testDataset = NerDataset(testData, testLabel, labelIndex, tokenizer, maxlength)
    testDataloader = DataLoader(testDataset, batch_size=batchsize, shuffle=False)

    # 建模
    criterion = nn.CrossEntropyLoss()
    model = BertModel(len(labelIndex), criterion).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 使用余弦退火学习率
    scheduler = CosineAnnealingLR(optimizer, T_max=epoch)

    # 绘图准备
    epochPlt = []
    trainLossPlt = []
    devAccPlt = []
    devF1Plt = []

    best_f1 = 0
    patience_counter = 0

    # 训练验证
    for e in range(epoch):
        if patience_counter >= patience:
            logger.info(f'Early stopping at epoch {e+1}')
            break

        # 训练
        time.sleep(0.1)
        epochPlt.append(e+1)
        epochloss = 0
        model.train()
        for batchdata, batchlabel, batchlen in tqdm(trainDataloader, total=len(trainDataloader), leave=False, desc="train"):
            batchdata = batchdata.to(device)
            batchlabel = batchlabel.to(device)
            loss = model.forward(batchdata, batchlabel)
            epochloss += loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        epochloss /= len(trainDataloader)
        trainLossPlt.append(float(epochloss))

        # 更新学习率
        scheduler.step()

        # 验证
        time.sleep(0.1)
        epochbatchlabel = []
        epochpre = []
        model.eval()
        for batchdata, batchlabel, batchlen in tqdm(devDataloader, total=len(devDataloader), leave=False, desc="dev"):
            batchdata = batchdata.to(device)
            batchlabel = batchlabel.to(device)
            pre = model.forward(batchdata)
            pre = pre.cpu().numpy().tolist()
            batchlabel = batchlabel.cpu().numpy().tolist()

            for b, p, l in zip(batchlabel, pre, batchlen):
                b = b[1:l+1]
                p = p[1:l+1]
                b = [indexLabel[i] for i in b]
                p = [indexLabel[i] for i in p]
                epochbatchlabel.append(b)
                epochpre.append(p)
        acc = accuracy_score(epochbatchlabel, epochpre)
        f1 = f1_score(epochbatchlabel, epochpre)
        devAccPlt.append(acc)
        devF1Plt.append(f1)

        # Early stopping
        if f1 > best_f1:
            best_f1 = f1
            patience_counter = 0
        else:
            patience_counter += 1

        # 记录日志
        logger.info(f'Epoch: {e+1}, Loss: {epochloss:.5f}, Acc: {acc:.4f}, F1: {f1:.4f}, LR: {scheduler.get_last_lr()[0]}')

        # 绘图
        plt.plot(epochPlt, trainLossPlt)
        plt.plot(epochPlt, devAccPlt)
        plt.plot(epochPlt, devF1Plt)
        plt.ylabel('Loss/Accuracy/F1')
        plt.xlabel('Epoch')
        plt.legend(['TrainLoss', 'DevAcc', 'DevF1'], loc='best')
        plt.savefig("bert.png")

    # 测试集评估
    epochbatchlabel = []
    epochpre = []
    model.eval()
    for batchdata, batchlabel, batchlen in tqdm(testDataloader, total=len(testDataloader), leave=False, desc="test"):
        batchdata = batchdata.to(device)
        batchlabel = batchlabel.to(device)
        pre = model.forward(batchdata)
        pre = pre.cpu().numpy().tolist()
        batchlabel = batchlabel.cpu().numpy().tolist()

        for b, p, l in zip(batchlabel, pre, batchlen):
            b = b[1:l+1]
            p = p[1:l+1]
            b = [indexLabel[i] for i in b]
            p = [indexLabel[i] for i in p]
            epochbatchlabel.append(b)
            epochpre.append(p)
    test_acc = accuracy_score(epochbatchlabel, epochpre)
    test_f1 = f1_score(epochbatchlabel, epochpre)
    logger.info(f'Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}')