import numpy as np
import pandas as pd
from  model.datautils import data_preprocess,get_full_train_data

class BackThresholdFilter():
    '''回调阀值过滤器，将经过分类器的准确率经过函数计算，得到改变阈值的个数，改进数据集进行计算
    '''
    def __init__(self,raw_data=None,raw_target=None,back_result_tensor=None,top=None):
        '''
        :param raw_data : format:[[1,3,2,4],[2,4,6,3],...]
        :param raw_target : format:[0,1,...]
        :param back_result_tensor : format:[1,2,3,4]
        :param top: format : 2 将1,2即tensor的0,1位进行过滤
        '''
        self.threshold = None
        #选择前几个作为过滤的数据
        self.top = top

        #原始数据
        self.raw_data = raw_data
        self.raw_target = raw_target

        #结果值，单独分类的准确率
        self.back_result_tensor = back_result_tensor

    def parse_data(self):
        '''将原始数据转化成改变阈值的数据'''
        pass

    def get_column(self):
        '''获取需要过滤的列
        :return [[...],[...],[...]]
        '''
        column_list = []

        for i in range(1,self.top+1):
            #获得数据项位置
            index = self.back_result_tensor.index(i)
            #提取数据列
            column = np.array(self.raw_data)[:,index]

            column_list.append(column)

        return column_list

    def compute_threshold(self,tensor):
        '''计算阈值
        :param tensor : 某一列的所有值，如[3,3,2,6,1,3,5]对应[0,1,0,0,1,0,1]
        '''
        positive = [(t,1) for idx,t in enumerate(tensor) if self.raw_target[idx] == 1]
        negative = [(t,0) for idx,t in enumerate(tensor) if self.raw_target[idx] == 0]

        total = []
        total.extend(negative)
        total.extend(positive)

        sort_total = sorted(total)
        print(sort_total)
        #中位数
        #pos_median = np.median(positive)
        #neg_median = np.median(negative)

        #return (pos_min,pos_max),(neg_min,neg_max)


if __name__ == '__main__':
    # 读数据
    data_0_df = pd.read_csv("../data/0.csv")
    data_1_df = pd.read_csv("../data/1.csv")
    data_0 = data_0_df.values
    data_1 = data_1_df.values

    data = np.vstack((data_0, data_1))
    # 　打乱
    np.random.shuffle(data)

    Y = data[:, -1]
    X = np.delete(data, -1, axis=1)

    back_t = [22, 8, 21, 11, 10, 5, 1, 13, 17, 14, 6, 2, 20, 18, 9, 16, 3, 4, 19, 15, 12, 7]
    top = 1

    my_filter = BackThresholdFilter(raw_data=X,raw_target=Y,back_result_tensor=back_t,top=3)

    for c in my_filter.get_column():
        my_filter.compute_threshold(c)