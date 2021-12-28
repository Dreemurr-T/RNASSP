import os
import pandas as pd
import csv
from tqdm import tqdm

# 遍历指定目录，显示目录下的所有文件名

list_Dir = 'archiveII/'  # 修改为本地文件路径
stru_Dir = 'data_stru/'
seq_Dir = 'data_seq/'

cnt = 0

def readfile(fileDir, filename):
    fopen = open(fileDir+filename, 'r')
    rnafile = fopen.readlines()
    del rnafile[0]
    with open("test.txt", 'w') as f:
        for i in rnafile:
            f.write(i)
    data = pd.read_csv("test.txt", sep='\t', header=None)
    return data, filename


def transform(data):
    rnaseq = data.loc[:, 1]
    rnadata1 = data.loc[:, 0]
    rnadata2 = data.loc[:, 4]
    rnastructure = []
    for i in range(len(rnadata2)):
        if rnadata2[i] == 0:
            rnastructure.append(".")
        else:
            if rnadata1[i] > rnadata2[i]:
                rnastructure.append(")")
            else:
                rnastructure.append("(")
    return rnaseq, rnastructure


def savefile(rnaseq, rnastructure, filename):
    global cnt
    if len(rnaseq) == len(rnastructure):
        cnt += 1
        if not os.path.exists(stru_Dir):
            os.mkdir(stru_Dir)
        rnafile = filename[:-3]
        rnafile = rnafile+".csv"
        rnafile = stru_Dir + rnafile
        rnacsv = open(rnafile, 'w', newline="")
        writer = csv.writer(rnacsv)
        m = len(rnastructure)
        for i in range(m):
            # print(rnastructure)
            writer.writerow(rnastructure[i])
        if not os.path.exists(seq_Dir):
            os.mkdir(seq_Dir)
        rnaseqfile = seq_Dir + filename[:-3] + "_seq.csv"
        rnaseqcsv = open(rnaseqfile, 'w', newline="")
        seqwriter = csv.writer(rnaseqcsv)
        m = len(rnastructure)
        for i in range(m):
            seqwriter.writerow(rnaseq[i])


if __name__ == '__main__':
    pathDir = os.listdir(list_Dir)
    for i in tqdm(pathDir):
        if i.endswith(".ct") and i.split('_')[0] != 'telomerase':
            # print(i)
            data, filename = readfile(list_Dir, i)
            rnaseq, rnastructure = transform(data)
            savefile(rnaseq, rnastructure, filename)
    print('extracted %s valid RNA files' % cnt)
