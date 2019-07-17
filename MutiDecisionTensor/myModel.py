import sys
sys.path.append("/home/molamola/PycharmProjects/baidubaikeQMSystem/")

from myConfig import Config
from model.datautils import data_preprocess,get_full_train_data
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn import tree
import numpy as np
import pickle


class MultiplyDecisionTensor:

    def __init__(self):
        self.my_config = Config()

    def _build_data(self):
        '''搭建数据模型'''
        # 读数据
        data_0_df = pd.read_csv(self.my_config.normal_data_path)
        data_1_df = pd.read_csv(self.my_config.characteristic_data_path)
        data_0 = data_0_df.values
        data_1 = data_1_df.values

        # 数据预处理
        data_0, data_1 = data_preprocess(data_0, data_1)
        self.X, self.Y = get_full_train_data(data_0, data_1)
        self.X, self.X_test, self.Y, self.Y_test = train_test_split(self.X, self.Y, test_size=0.4)

    def _build_model(self):
        '''初始化模型'''
        #self.trainer = RandomForestClassifier(n_estimators=120)

        self.trainer = tree.DecisionTreeClassifier()

    def _get_MultiplyDecisionTensor(self):
        '''获得多决策向量,每个数据项训练N次获取f1评估值最高的'''
        result = None
        for idx in range(22):
            column = np.expand_dims(self._get_one_column(index=idx),axis=1)
            if result is None :
                result = column
            else:
                result = np.concatenate((result,column),axis=1)
        self._save(result)
        print("保存成功")

    def _get_one_column(self,index=None):
        #模型评价集合
        eva_list = []
        output_list = []
        #保存模型集合
        model_list = []
        '''获得一列'''
        if index is None:
            print("please get one index")
            return

        #提取训练集的一列
        X = np.expand_dims(np.array(self.X)[:,index],axis=1)
        X_test = np.expand_dims(np.array(self.X_test)[:, index], axis=1)

        for train_idx in range(self.my_config.train_time):
            print("train index:{} || train_idx :{}".format(index, train_idx))
            '''训练并测试一次'''

            #训练
            self.trainer.fit(X,self.Y)
            #验证
            result = self.trainer.predict(X_test)
            #验证结果加入
            eva_list.append(precision_score(y_true=self.Y_test,y_pred=result))

            output_list.append(result)
            model_list.append(self.trainer)

        best_index = eva_list.index(max(eva_list))

        #保存最好的模型
        best_model = model_list[best_index]
        file = open(self.my_config.model_save_path+'model-'+str(index)+'.pickle','wb')
        pickle.dump(best_model,file)

        return output_list[best_index]

    def _save(self,X):
        '''保存向量'''
        Y = np.expand_dims(self.Y_test,axis=1)
        data = np.concatenate((X,Y),axis=1)
        data_pd = pd.DataFrame(data)
        data_pd.to_csv(self.my_config.tensor_save_path+"MDT.csv")

    '''for user'''
    def build(self):
        self._build_data()
        self._build_model()

    def run(self):
        self._get_MultiplyDecisionTensor()
