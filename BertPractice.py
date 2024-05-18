import torch.nn as nn
from transformers import AdamW
from torch.utils.data import Dataset
import pandas as pd
import torch
from transformers import BertConfig, BertForSequenceClassification
from transformers import BertTokenizer
from torch.utils.data import DataLoader

# 超参数
hidden_dropout_prob = 0.3
num_labels = 2
learning_rate = 1e-5
weight_decay = 1e-2
epochs = 1
batch_size = 8
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_path = "sentiment/"  # linux
# data_path = "sentiment\\"  # windows
vocab_file = data_path + "vocab.txt"  # 词汇表
train_data = data_path + "sentiment.train.data"  # 训练数据集
valid_data = data_path + "sentiment.valid.data"  # 验证数据集


# 定义 Dataset
class SentimentDataset(Dataset):
    def __init__(self, path_to_file):
        self.dataset = pd.read_csv(path_to_file, sep="\t", names=["text", "label"])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # 根据 idx 分别找到 text 和 label
        text = self.dataset.loc[idx, "text"]
        label = self.dataset.loc[idx, "label"]
        sample = {"text": text, "label": label}
        # 返回一个 dict
        return sample


# 加载训练集
sentiment_train_set = SentimentDataset(data_path + "sentiment.train.data")
sentiment_train_loader = DataLoader(sentiment_train_set, batch_size=batch_size, shuffle=True, num_workers=0)
# 加载验证集
sentiment_valid_set = SentimentDataset(data_path + "sentiment.valid.data")
sentiment_valid_loader = DataLoader(sentiment_valid_set, batch_size=batch_size, shuffle=False, num_workers=0)

# 定义 tokenizer，传入词汇表
tokenizer = BertTokenizer(vocab_file)

# 加载模型
config = BertConfig.from_pretrained("bert-base-uncased", num_labels=num_labels, hidden_dropout_prob=hidden_dropout_prob)
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", config=config)
model.to(device)

# 定义优化器和损失函数
# Prepare optimizer and schedule (linear warmup and decay)
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
     'weight_decay': weight_decay},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

# optimizer = AdamW(model.parameters(), lr=learning_rate)
optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
criterion = nn.CrossEntropyLoss()


# 定义训练的函数
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    for i, batch in enumerate(dataloader):
        # 标签形状为 (batch_size, 1)
        label = batch["label"].to(device)
        text = batch["text"]

        # tokenized_text 包括 input_ids， token_type_ids， attention_mask
        tokenized_text = tokenizer(text, max_length=100, add_special_tokens=True, truncation=True, padding=True,
                                   return_tensors="pt")
        tokenized_text = tokenized_text.to(device)
        # 梯度清零
        optimizer.zero_grad()

        # output: (loss), logits, (hidden_states), (attentions)
        output = model(**tokenized_text, labels=label)

        # y_pred_prob = logits : [batch_size, num_labels]
        y_pred_prob = output[1]
        y_pred_label = y_pred_prob.argmax(dim=1)

        # 计算loss
        # 这个 loss 和 output[0] 是一样的
        loss = criterion(y_pred_prob.view(-1, 2), label.view(-1))

        # 计算acc
        acc = ((y_pred_label == label.view(-1)).sum()).item()

        # 反向传播
        loss.backward()
        optimizer.step()

        # epoch 中的 loss 和 acc 累加
        # loss 每次是一个 batch 的平均 loss
        epoch_loss += loss.item()
        # acc 是一个 batch 的 acc 总和
        epoch_acc += acc
        if i % 200 == 0:
            print("current loss:", epoch_loss / (i + 1), "\t", "current acc:", epoch_acc / ((i + 1) * len(label)))

    # len(dataloader) 表示有多少个 batch，len(dataloader.dataset.dataset) 表示样本数量
    return epoch_loss / len(dataloader), epoch_acc / len(dataloader.dataset.dataset)


def evaluate(model, iterator, device):
    # model = BertForSequenceClassification.from_pretrained("bert-base-uncased", config=config)
    # sentiment_valid_loader = DataLoader(sentiment_valid_set, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    print('@@@@' * 40)
    with torch.no_grad():
        for _, batch in enumerate(iterator):
            text = batch["text"]

            tokenized_text = tokenizer(text, max_length=100, add_special_tokens=True, truncation=True, padding=True,
                                       return_tensors="pt")
            tokenized_text = tokenized_text.to(device)

            label = batch["label"].to(device)
            output = model(**tokenized_text, labels=label)
            for eachText in text:
                print('the eachText is :', eachText[:5], 'the label is :', label)

            # # 假设我们有两个样本，每个样本有三个类别的 logits
            # logtis = torch.tensor([[0.9, 0.5, 0.4], [0.8, 0.1, 0.9],[3, 38, 0.4],[0.1, 0.5, 0.4],[0.1, 0.5, 0.4],[0.3, 0.3, 0.4]])
            # pred_labels = logtis.argmax(dim=1)
            # print(pred_labels)
            # 则输出tensor([0, 2, 1, 1, 1, 2])
            print('The output is :', output[0], output[1])
            y_pred_label = output[1].argmax(dim=1)
            loss = output[0]
            labelview = label.view(-1)
            validLv = y_pred_label == labelview
            sum = validLv.sum()
            acc = sum.item()
            # epoch 中的 loss 和 acc 累加
            # loss 每次是一个 batch 的平均 loss
            epoch_loss += loss.item()
            # acc 是一个 batch 的 acc 总和
            epoch_acc += acc
            print('len(iterator) is :', len(iterator))
            print('len(iterator.dataset.dataset) is :', len(iterator.dataset.dataset))
            print('&&' * 30)

    # len(dataloader) 表示有多少个 batch，len(dataloader.dataset.dataset) 表示样本数量
    return epoch_loss / len(iterator), epoch_acc / len(iterator.dataset.dataset)


# logtis = torch.tensor([[-0.2165, -0.3768]])
# pred_labe = logtis.argmax(dim=1)
# print(pred_labe)
#
# # 假设我们有两个样本，每个样本有三个类别的 logits
# logtis = torch.tensor([[0.9, 0.5, 22], [0.8, 0.1, 0.9]])
# sum = logtis.sum()
# item = sum.item()
# print('sum is :', sum ,'item is :', item)
# pred_labels = logtis.argmax(dim=1)
# print(pred_labels)


# 开始训练和验证
# for i in range(epochs):
#     # train_loss, train_acc = train(model, sentiment_train_loader, optimizer, criterion, device)
#     # print("train loss: ", train_loss, "\t", "train acc:", train_acc)
#     valid_loss, valid_acc = evaluate(model, sentiment_valid_loader, device)
#     print("valid loss: ", valid_loss, "\t", "valid acc:", valid_acc)
#
#

model.eval()
my_dict = {}
for _, batch in enumerate(sentiment_valid_loader):

    label = batch["label"].to(device)
    text = batch["text"]
    tokenized_text = tokenizer(text, max_length=100, add_special_tokens=True, truncation=True, padding=True,
                               return_tensors="pt")
    output = model(**tokenized_text, labels=label)
    y_pred_label = output[1].argmax(dim=1)
    print(y_pred_label)

    count = 0
    for num in y_pred_label:
        eachTxt = text[count]
        if num.item() == 0:
            my_dict.update({eachTxt: '好'})
        if num.item() == 1:
            my_dict.update({eachTxt: '坏'})
        count += 1
for txt, res in my_dict.items():
    print('the txt is :', txt, '\nand the value is :', res, '\n', '*' * 300)
