'''
Convert .st files to sequences needed
'''

import os
import pandas as pd
import csv
from tqdm import tqdm

# 遍历指定目录，显示目录下的所有文件名

list_Dir = 'new_public_set/new_public_set/'  # 修改为本地文件路径
stru_Dir = 'test_stru/'
seq_Dir = 'test_seq/'

cnt = 0

def write_csv(fileDir, filename):
    fopen = open(fileDir+filename, 'r')
    seq = fopen.readline()[:-1]
    struct = fopen.readline()[:-1]
    if not os.path.exists(stru_Dir):
        os.mkdir(stru_Dir)
    if not os.path.exists(seq_Dir):
        os.mkdir(seq_Dir)
    rnafile = filename[:-3]
    rnafile = rnafile+".csv"
    rnafile = stru_Dir + rnafile
    rnacsv = open(rnafile, 'w', newline="")
    writer = csv.writer(rnacsv)
    m = len(struct)
    for i in range(m):
        # print(rnastructure)
        writer.writerow(struct[i])
    rnaseqfile = seq_Dir + filename[:-3] + "_seq.csv"
    rnaseqcsv = open(rnaseqfile, 'w', newline="")
    seqwriter = csv.writer(rnaseqcsv)
    m = len(seq)
    for i in range(m):
        seqwriter.writerow(seq[i])
    

if __name__ == '__main__':
    pathDir = os.listdir(list_Dir)
    for i in tqdm(pathDir):
        if i.endswith(".st"):
            # print(i)
            write_csv(list_Dir,i)
