# Customized dataset to load RNA data
# Resize to 300*4(aborted)

from torch.utils.data import Dataset
import pandas as pd
import torch
import os

notation_dict = {
    '.': [[0, 0, 1]],
    '(': [[0, 1, 0]],
    ')': [[1, 0, 0]]
}

base_dict = {
    'A': [[0, 0, 0, 1]],
    'U': [[0, 0, 1, 0]],
    'G': [[0, 1, 0, 0]],
    'C': [[1, 0, 0, 0]]
}


class RNADataset(Dataset):
    def __init__(self, seq_path, notation_path):
        self.seq_names = []
        self.notation_names = []
        for curDir, Dir, files in os.walk(seq_path):
            for filename in files:
                self.seq_names += [os.path.join(curDir, filename)]
        for curDir, Dir, files in os.walk(notation_path):
            for filename in files:
                self.notation_names += [os.path.join(curDir, filename)]
        self.data_len = len(self.seq_names)

    def __getitem__(self, index):
        # valid = -1
        seq_info = pd.read_csv(self.seq_names[index], header=None)
        notation_info = pd.read_csv(self.notation_names[index], header=None)
        seq = seq_info.loc[:, 0]
        notations = notation_info.loc[:, 0]
        base_embed = torch.tensor(base_dict[seq[0]])
        notation_embed = torch.tensor(notation_dict[notations[0]])
        for base in seq[1:]:
            base_oh = base_dict[base]
            base_embed_tmp = torch.tensor(base_oh)
            base_embed = torch.cat((base_embed, base_embed_tmp), 0)
        for notation in notations[1:]:
            notation_oh = notation_dict[notation]
            notation_embed_tmp = torch.tensor(notation_oh)
            notation_embed = torch.cat((notation_embed, notation_embed_tmp), 0)
        # true_length = base_embed.shape[0]
        # while base_embed.shape[0] < 300:
        #     base_embed = torch.cat((base_embed, torch.tensor([[0, 0, 0, 0]])), 0)
        # while notation_embed.shape[0] < 300:
        #     notation_embed = torch.cat((notation_embed, torch.tensor([[0, 0, 0]])), 0)
        # if base_embed.shape[0] == 300:
        #     valid = 1
        # return valid,base_embed,notation_embed,true_length
        return base_embed,notation_embed

    def __len__(self):
        return self.data_len
