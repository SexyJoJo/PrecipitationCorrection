import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

dtype = torch.FloatTensor

sentence = (
    'GitHub Actions makes it easy to automate all your software workflows '
    'from continuous integration and delivery to issue triage and more'
)

# 算法过程（分类问题：所有的词中选一个作为预测值）：
# 'Github ? ? ? ... ?' => 'Actions'
# 'Github Actions ? ? ? ... ?' => 'makes'
# ...

word2idx = {w: i for i, w in enumerate(list(set(sentence.split())))}
idx2word = {i: w for i, w in enumerate(list(set(sentence.split())))}
n_class = len(word2idx)  # 分类的个数
max_len = len(sentence.split())    # 21
n_hidden = 5


def make_data(sentence):
    input_batch = []
    target_batch = []

    words = sentence.split()
    for i in range(max_len - 1):
        input = [word2idx[n] for n in words[:(i + 1)]]  # 把输入单词列表转为单词索引列表
        input = input + [0] * (max_len - len(input))    # 其余单词补为0
        target = word2idx[words[i + 1]]
        input_batch.append(np.eye(n_class)[input])
        target_batch.append(target)

    return torch.Tensor(input_batch), torch.LongTensor(target_batch)


# input_batch维度: [max_len-1(循环的次数), max_len, n_class(每个单词的编码长度)]
input_batch, target_batch = make_data(sentence)
dataset = Data.TensorDataset(input_batch, target_batch)
loader = Data.DataLoader(dataset, 16, True)

class BiLSTM(nn.Module):
    def __init__(self):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=n_class, hidden_size=n_hidden, bidirectional=True)
        # 全连接
        self.fc = nn.Linear(n_hidden * 2, n_class)

    def forward(self, X):
        # X: [batch_size, max_len, n_class]
        batch_size = X.shape[0]
        input = X.transpose(0, 1)  # input : [max_len, batch_size, n_class]

        hidden_state = torch.randn(1*2, batch_size, n_hidden)   # 维度[num_layers(BiLSTM层数) * num_directions(双向，取2), batch_size, n_hidden]
        cell_state = torch.randn(1*2, batch_size, n_hidden)     # 维度[num_layers(BiLSTM层数) * num_directions(双向，取2), batch_size, n_hidden]

        # 全部输出
        outputs, (_, _) = self.lstm(input, (hidden_state, cell_state))  # 维度[max_len, batch_size, n_hidden * 2]
        # 最后一个细胞的输出
        outputs = outputs[-1]  # [batch_size, n_hidden * 2]
        model = self.fc(outputs)  # model : [batch_size, n_class]
        return model

model = BiLSTM()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
