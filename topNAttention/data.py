from config import ATNNConfig
from model.datautils import data_preprocess,get_full_train_data
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

class Data:

    def __init__(self):
        self.config = ATNNConfig()
        self.test_X_final = None
        self.test_Y_final = None

    def build(self):
        # 读数据
        data_0_df = pd.read_csv(self.config.normal_data_path)
        data_1_df = pd.read_csv(self.config.characteristic_data_path)
        data_0 = data_0_df.values
        data_1 = data_1_df.values

        # 数据预处理
        data_0, data_1 = data_preprocess(data_0, data_1)
        # 获得全部的数据
        self.X, self.Y = get_full_train_data(data_0, data_1)
        #划分测试集和训练集
        self.X, self.test_X_final, self.Y, self.test_Y_final = train_test_split(self.X, self.Y, test_size=0.3)

        assert len(self.X) == len(self.Y)

    def get_batches_data(self,batch_size):
        '''获得batch数据'''
        total_size = len(self.X)
        batch_num = total_size//batch_size

        batch_X = []
        batch_Y = []

        for idx in range(total_size):

            batch_X.append(self.X[idx])
            batch_Y.append(self.Y[idx])

            if len(batch_X) == batch_num:
                assert len(batch_X) == len(batch_Y)
                #划分验证集
                train_X, test_X, train_Y, test_Y = train_test_split(batch_X,batch_Y,test_size=0.3)
                yield train_X,train_Y ,test_X ,test_Y
                batch_X = []
                batch_Y = []

    def split_data(self,X):
        '''按照top划分数据集
        @:return top_data [none,topN]
        other data  [none,22-topN]
        '''
        top_data = None
        other_data = X
        index_list = []
        for i in range(self.config.topN):
            index = self.config.sort_list.index(i+1)

            #将X中的index列筛选出
            X = np.array(X)
            column = np.expand_dims(X[:,index],axis=1)

            if top_data is None:
                top_data = column
            else:
                top_data = np.concatenate((top_data,column),axis=1)

            index_list.append(index)
        #删除列
        for index in sorted(index_list,reverse=True):
            other_data = np.delete(other_data,index,axis=1)

        assert len(X) == len(top_data) == len(other_data)
        return top_data,other_data


if __name__ == '__main__':
    data = Data()
    data.build()
    for idx,(X_batch,Y_batch ,batch_X_test ,batch_Y_test) in enumerate(data.get_batches_data(21)):
        data.split_data(X_batch)
