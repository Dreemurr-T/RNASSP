# import pandas as pd
# import torch

# test_file = pd.read_csv('data_csv/5s_Acanthamoeba-castellanii-1.csv',header=None)

# print(test_file.loc[:,0],len(test_file.index))
import torch
import torch.nn as nn

rnn = nn.LSTM(8, 3, 2,bidirectional =True)
input = torch.randn(23, 113, 8)
# h0 = torch.randn(2, 3, 3)
# c0 = torch.randn(2, 3, 3)
output, (hn, cn) = rnn(input)

print(output.shape)