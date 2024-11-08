import torch
from torch.utils.data import Dataset

def readFile(name):
    data = []
    label = []
    dataSentence = []
    labelSentence = []
    with open(name, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if not line.strip():
                data.append(dataSentence)
                label.append(labelSentence)
                dataSentence = []
                labelSentence = []
            else:
                content = line.strip().split()
                dataSentence.append(content[0].lower())
                labelSentence.append(content[-1])
    return data, label

def label2index(label):
    label2index = {}
    for sentence in label:
        for i in sentence:
            if i not in label2index:
                label2index[i] = len(label2index)
    return label2index, list(label2index)

class NerDataset(Dataset):
    def __init__(self, data, label, labelIndex, tokenizer, maxlength):
        self.data = data
        self.label = label
        self.labelIndex = labelIndex
        self.tokenizer = tokenizer
        self.maxlength = maxlength

    def __getitem__(self, item):
        thisdata = self.data[item]
        thislabel = self.label[item][:self.maxlength]
        thisdataIndex = self.tokenizer.encode(thisdata, add_special_tokens=True, max_length=self.maxlength + 2,
                                               padding="max_length", truncation=True, return_tensors="pt")
        thislabelIndex = [self.labelIndex['O']] + [self.labelIndex[i] for i in thislabel] + [self.labelIndex['O']] * (
                    self.maxlength + 1 - len(thislabel))
        thislabelIndex = torch.tensor(thislabelIndex)
        return thisdataIndex[-1], thislabelIndex, len(thislabel)

    def __len__(self):
        return len(self.data)