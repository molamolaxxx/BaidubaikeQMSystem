# -*- coding:utf-8 -*-
'''计算信息增益 , 以信息增益作为多决策向量的权重'''
import numpy as np
import pandas as pd

def get_entropy (tuple_list):

    total = len(tuple_list)

    data_1_num = sum(tuple_list)
    data_0_num = total - data_1_num

    #计算熵
    entropy = - ((0 if data_0_num == 0 else data_0_num / total * np.log2(data_0_num / total))
                                           + (0 if data_1_num == 0 else data_1_num / total * np.log2(data_1_num / total)))


    return entropy

if __name__ == '__main__':

    # 读数据
    data_0_df = pd.read_csv("../data/0.csv")
    data_1_df = pd.read_csv("../data/1.csv")
    data_0 = data_0_df.values
    data_1 = data_1_df.values

    #拼接
    data = np.concatenate([data_0,data_1],axis=0)

    # 熵
    entropy = - (len(data_0) / (len(data))*np.log2(len(data_0) / (len(data))) \
              + len(data_1) / (len(data))*np.log2(len(data_1) / (len(data))))

    #共有多少列
    column_num = data.shape[1] - 1
    label = data[:,-1]

    tuple_list = []

    entropy_list = []
    # 23
    for column_idx in range(column_num):
        column = data[:,column_idx]

        # 8000
        for idx,c in enumerate(column):
            tuple = (c,label[idx])
            tuple_list.append(tuple)

        tuple_list_sort = sorted(tuple_list)

        label_list = []
        for tuple in tuple_list_sort:
            label_list.append(tuple[1])

        #分别计算可能的信息增益
        gain_list = []

        # 8000
        for i in range(len(tuple_list_sort)-1) :
            #划分值

            i = i + 1
            pre_label = label_list[0:i]
            post_label = label_list[i:]

            #计算信息增益
            pre_entropy = get_entropy(pre_label)
            post_entropy = get_entropy(post_label)

            _gain = entropy - (len(pre_label)/len(tuple_list_sort))*pre_entropy - (len(post_label)/len(tuple_list_sort))*post_entropy
            gain_list.append(_gain)

        entropy_gain = max(gain_list)

        tuple_list.clear()
        print("----{}----".format(column_idx))
        entropy_list.append(entropy_gain)
    print(entropy_list)
