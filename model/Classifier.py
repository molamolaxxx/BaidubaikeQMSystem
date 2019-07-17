from config import ClassifierConfig
#svm
from sklearn import svm
#rf
from sklearn.ensemble import RandomForestClassifier
#dt
from sklearn import tree
#lr
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from model.datautils import  to_array,to_df,data_preprocess,get_train_and_test_data,\
    cal_result,add_eva_data,get_average,get_max,split_data_label,get_full_train_data,split_data_average_1
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score , cross_validate , train_test_split

class Classifier():
    '''svm分类器，用于通过词条属性判断词条类别：'''
    def __init__(self):
        #配置类
        self.config=ClassifierConfig()

    def _build_data(self):
        '''搭建数据模型'''
        # 读数据
        data_0_df = pd.read_csv(self.config.normal_data_path)
        data_1_df = pd.read_csv(self.config.characteristic_data_path)
        data_0 = data_0_df.values
        data_1 = data_1_df.values

        #数据预处理
        data_0,data_1=data_preprocess(data_0,data_1)
        if self.config.use_cross_validation:
            self.X,self.Y = get_full_train_data(data_0,data_1)
        else:
            #获得训练和测试数据
            self.train,self.test=get_train_and_test_data(data_0, data_1)

            #把label分出来
            self.x_train,self.y_train\
                ,self.x_test,self.y_test=split_data_label(self.train,self.test)

    def _build_model(self):
        '''搭建机器学习模型框架'''
        if self.config.use_method=='svm':
            print("use method svm")
            self.trainer = svm.SVC(kernel=self.config.kernel,C=self.config.C)
        elif self.config.use_method=='rf':
            print("use method RandomForest")
            self.trainer = RandomForestClassifier(n_estimators=self.config.n_estimators)
        elif self.config.use_method=='dt':
            print("use method DecisionTree")
            self.trainer = tree.DecisionTreeClassifier()
        elif self.config.use_method=='lr':
            print("use method RegressionClassification")
            self.trainer = LogisticRegression(penalty=self.config.penalty)


    def train_model(self):
        '''训练模型'''
        #使用十字交叉验证
        if self.config.use_cross_validation:
            eva_list = self._train_cross_val()
            return eva_list
        else:
            # 总共运行多少次实例
            for t in range(self.config.total_num):
                # 总共训练多少次
                for epoch in range(self.config.epoch_num):
                    #获得一次的训练结果
                    print("t:{},epoch:{}".format(t,epoch))
                    eva_dict = self._train_epoch()
                    add_eva_data(eva_dict)
                print(get_average(self.config.epoch_num))
                get_max()

    def _train_epoch(self):
        '''训练一个poch的数据'''
        assert len(self.x_train)==len(self.y_train)
        assert len(self.x_test)==len(self.y_test)

        # 获得一次的训练结果
        self.trainer.fit(self.x_train, self.y_train)

        # 预测数据
        assert len(self.y_train) == len(self.x_train)
        assert len(self.y_test) == len(self.x_test)
        result = self.trainer.predict(self.x_test)
        #percent = cal_result(result, self.y_test)
        #评估数据
        acc = accuracy_score(self.y_test, result)
        p = precision_score(self.y_test, result)
        r = recall_score(self.y_test, result)
        f1 = f1_score(self.y_test, result)
        #输出日志
        #print({"acc": acc, "p": p, "r": r, "f1": f1})
        # 重新获取数据模型
        self._build_data()
        return {"acc": acc, "p": p, "r": r, "f1": f1}

    def _train_epoch_cross_val_score(self):
        scoring = ['precision_weighted', 'recall_weighted','f1_weighted','precision_macro', 'recall_macro','f1_macro']  # 设置评分项
        _score = cross_validate(self.trainer,self.x_train,self.y_train,cv=10,scoring=scoring,return_train_score=False)

        print("-----weighted")
        print("----------precision:{}".format(np.mean(_score["test_precision_weighted"])))

        print("----------recall:{}".format(np.mean(_score["test_recall_weighted"])))

        print("----------f1:{}".format(np.mean(_score["test_f1_weighted"])))

        print("-----macro")
        print("----------precision:{}".format(np.mean(_score["test_precision_macro"])))

        print("----------recall:{}".format(np.mean(_score["test_recall_macro"])))

        print("----------f1:{}".format(np.mean(_score["test_f1_macro"])))

        return _score

    def _train_cross_val(self):
        '''手动十字交叉 '''
        p_list_pos = []
        p_list_neg = []
        recall_list_pos = []
        recall_list_neg = []
        f1_list_pos = []
        f1_list_neg = []


        for X_each, Y_each,X_test,Y_test in split_data_average_1(self.X, self.Y):

            self.x_test = X_test
            self.y_test = Y_test

            self.trainer.fit(X_each,Y_each)

            result = self.trainer.predict(self.x_test)
            p_pos = precision_score(y_true=self.y_test,y_pred=result,pos_label=1)
            p_neg = precision_score(y_true=self.y_test, y_pred=result, pos_label=0)
            f1_pos = f1_score(y_true=self.y_test,y_pred=result,pos_label=1)
            f1_neg = f1_score(y_true=self.y_test,y_pred=result,pos_label=0)
            r_pos = recall_score(y_true=self.y_test,y_pred=result,pos_label=1)
            r_neg = recall_score(y_true=self.y_test,y_pred=result,pos_label=0)

            p_list_pos.append(p_pos)
            p_list_neg.append(p_neg)
            recall_list_pos.append(r_pos)
            recall_list_neg.append(r_neg)
            f1_list_pos.append(f1_pos)
            f1_list_neg.append(f1_neg)

        result_dict = {'precision_score_pos': np.average(p_list_pos),
                       'precision_score_neg': np.average(p_list_neg),
                       'recall_score_pos': np.average(recall_list_pos),
                       'recall_score_neg': np.average(recall_list_neg),
                       'f1_pos': np.average(f1_list_pos),
                       'f1_neg': np.average(f1_list_neg)}
        print(result_dict)
        return result_dict

    def build(self):
        self._build_data()
        self._build_model()






