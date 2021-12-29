import numpy as np
import webbrowser
import pandas as pd
import numpy as np
import csv


# 判断碱基之间的配对是否符合规则
def is_paired(x, y):
    if x == 'A' and y == 'U':
        return True
    elif x == 'G' and y == 'C':
        return True
    elif x == "G" and y == 'U':
        return True
    elif x == 'U' and y == 'A':
        return True
    elif x == 'C' and y == 'G':
        return True
    elif x == "U" and y == 'G':
        return True
    else:
        return False

# 记录由碱基对配对关系决定的，参与计算时的取值,bases是碱基序列
def count_paired(prediction, bases):
    l = prediction.shape[0]
    r = np.zeros((l, l))
    for i in range(l):
        for j in range(l):
            if is_paired(bases[i], bases[j]) and abs(i-j)>4:
                if i < j :
                    r[i, j] = prediction[i][1] + prediction[j][0]
                else:
                    r[i, j] = prediction[i][0] + prediction[j][1]
            else:
                r[i, j] = prediction[i][2] + prediction[j][2]
    return r

# 借鉴Nussinov算法，迭代计算从i到j的序列最大概率和
def Nus_p(prediction, bases):
    r = count_paired(prediction, bases)
    l = prediction.shape[0]
    final_pre = ''
    n = np.zeros((l, l))
    result = [[''] * l for i in range(l)]
    for k in range(l):
        i = 0
        while i + k < l:
            # 对n(i,i+k)操作
            j = i + k
            # 对n(i,j)操作
            max1 = 0
            max3 = 0
            if i + 1 < l and j - 1 >= 0:
                max1 = n[i + 1][j] + prediction[i][2]
                max3 = n[i + 1][j - 1] + r[i][j]  # 只有这种情况对应配对
            max2 = n[i][j - 1] + prediction[j][2]
            max4 = 0
            for x in range(i + 1, j - 1, 1):
                if max4 < n[i][x] + n[x + 1][j]:
                    max4 = n[i][x] + n[x + 1][j]  # 可能为分割成两个子序列
                    point = x
            m = max(max1, max2, max3, max4)
            n[i][j] = m
            if m == max1:
                result[i][j] = '.' + result[i + 1][j]
            elif m == max2:
                result[i][j] = result[i][j - 1] + '.'
            elif m == max3:
                if len(result[i + 1][j - 1]) == j - i - 1:
                    if is_paired(bases[i], bases[j]):
                        result[i][j] = '(' + result[i + 1][j - 1] + ')'  # 可以保证产生的预测序列左右括号必然一一对应
                    else:
                        result[i][j] = '.' + result[i + 1][j - 1] + '.'
                else:
                    result[i][j] = '.'
            elif m == max4:
                result[i][j] = result[i][point] + result[point + 1][j]
            i = i + 1
    str = result[0][l - 1]
    for c in str:
        final_pre += c
    return final_pre