from sklearn import tree
from model.datautils import  to_array,to_df,data_preprocess,get_train_and_test_data,\
    cal_result,add_eva_data,get_average,get_max,split_data_label,get_full_train_data,split_data_average,get_topN_column,split_textEditor_data
import pandas as pd
from config import ClassifierConfig
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn import svm
#lr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from collections import Counter
from runner import Runner
import matplotlib.pyplot as plt

text_topN = 8
editor_topN = 8
class DoubleDecideModel:
    '''
    双决策树算法，分别将文本属性和编辑者属性放到两个决策树中，使用&操作得出结果
    '''
    def __init__(self):
        # 配置类
        self.config = ClassifierConfig()
        self.runner = Runner()

    def build_model(self):
        '''搭建双决策树模型框架'''

        self.text_model = tree.DecisionTreeClassifier()
        self.editor_model = tree.DecisionTreeClassifier()

        # self.text_model = RandomForestClassifier(n_estimators=100)
        # self.editor_model = RandomForestClassifier(n_estimators=100)

        # self.text_model = RandomForestClassifier(n_estimators=100)
        #self.editor_model = LogisticRegression(penalty=self.config.penalty)

        #self.text_model = RandomForestClassifier(n_estimators=100)
        #self.editor_model = RandomForestClassifier(n_estimators=200)
        #self.editor_model = svm.SVC(kernel=self.config.kernel, C=self.config.C)



    def build_data(self):
        '''搭建数据模型'''
        # 读数据
        data_0_df = pd.read_csv('.'+self.config.normal_data_path)
        data_1_df = pd.read_csv('.'+self.config.characteristic_data_path)
        data_0 = data_0_df.values
        data_1 = data_1_df.values

        # 数据预处理
        data_0, data_1 = data_preprocess(data_0, data_1)
        self.X, self.Y = get_full_train_data(data_0, data_1)
        #划分训练集和测试集
        #self.x_train,self.x_test,self.y_train,self.y_test = train_test_split(self.x_train,self.y_train,test_size=0.15)

        self.X_text , self.X_editor = split_textEditor_data(self.X)

    def train_model(self):
        '''训练模型 ,使用N折交叉验证'''
        p_list_pos = []
        p_list_neg = []
        recall_list_pos = []
        recall_list_neg = []
        f1_list_pos = []
        f1_list_neg = []

        ''' 1.九份词条自身训练集　2.九份编辑者自身训练集　3.训练集对应的y 4.一份测试集　5.测试集对应的y'''

        for X_each_text,X_each_editor,Y_each , X_test ,Y_test in split_data_average(self.X_text,self.X_editor,self.X,self.Y):

            self.x_test = X_test
            self.y_test = Y_test

            try:
                print("train text model")
                self.text_model.fit(X_each_text,Y_each)

                print("train editor model")
                self.editor_model.fit(X_each_editor,Y_each)

            except ValueError as e:
                print("value error:{}".format(e))
                continue

            result_dict = self.evaluate_model()
            p_list_pos.append(result_dict['precision_score_pos'])
            p_list_neg.append(result_dict['precision_score_neg'])
            recall_list_pos.append(result_dict['recall_score_pos'])
            recall_list_neg.append(result_dict['recall_score_neg'])
            f1_list_pos.append(result_dict['f1_pos'])
            f1_list_neg.append(result_dict['f1_neg'])


        print('*'*60)
        print('----------------------final result-----------------------')

        result_dict = {'precision_score_pos': np.average(p_list_pos),
                       'precision_score_neg': np.average(p_list_neg),
                       'recall_score_pos': np.average(recall_list_pos),
                       'recall_score_neg': np.average(recall_list_neg),
                       'f1_pos': np.average(f1_list_pos),
                       'f1_neg': np.average(f1_list_neg)}

        print(result_dict)
        return result_dict


    def evaluate_model(self):

        x_test_text,x_test_editor = split_textEditor_data(self.x_test)

        print("evaluate text model")
        text_model_result = self.text_model.predict(x_test_text)
        p_text = precision_score(y_pred=text_model_result,y_true=self.y_test)
        print(p_text)

        print("evaluate editor model")
        editor_model_result = self.editor_model.predict(x_test_editor)
        p_editor = precision_score(y_pred=editor_model_result, y_true=self.y_test)
        print(p_editor)

        #确保两个模型的长度相等
        assert len(text_model_result) == len(editor_model_result)

        final_result = []
        # score_tuple_list = []
        # score_dict = {}
        for idx in range(len(text_model_result)):
            if  text_model_result[idx] and editor_model_result[idx]:
                final_result.append(1)
            else:
                #todo 很多高质量词条被错分成低质量词条，在这里筛选回来
                if text_model_result[idx] or editor_model_result[idx]:
                    #print("被错分成低质量的词条")
                    score = self.runner.predict_one_item(self.x_test[idx])

                    #元组列表用来画图和算出阈值，字典用来取出筛选过程中用到的score
                    score_tuple_list.append((score,self.y_test[idx]))
                    score_dict[idx] = score

                    #待过滤词条
                    final_result.append(2)
                    continue

                final_result.append(0)

        #元组列表排序
        score_tuple_list_sort = sorted(score_tuple_list,reverse=True)

        '''根据百分比得出阈值，筛选重新打分'''

        thresh = score_tuple_list_sort[int(len(score_tuple_list_sort)/2)][0]
        print("筛选百分比为：{} , thresh为{}".format(1 / 2,thresh))

        for idx,r in enumerate(final_result):
            if r == 2:

                score = score_dict[idx]

                if score > thresh:
                    final_result[idx] = 1
                else:
                    final_result[idx] = 0

        # #画图
        # fig = plt.figure(figsize=(14, 10))
        # ax = fig.add_subplot(111)
        #
        # blue_x = []
        # blue_y = []
        # red_x = []
        # red_y = []
        # for idx,tuple in enumerate(score_tuple_list_sort):
        #     if tuple[1] == 1:
        #         blue_x.append(idx)
        #         blue_y.append(tuple[0])
        #     else:
        #         red_x.append(idx)
        #         red_y.append(tuple[0])
        #
        # plt.bar(blue_x, blue_y, color='blue')
        # plt.bar(red_x, red_y, color='red')
        # plt.legend()
        # plt.show()

        print(np.sum(final_result))
        print(np.sum(self.y_test))

        print("evaluate final model")
        p_pos = precision_score(y_pred=final_result, y_true=self.y_test,pos_label=1)
        p_neg = precision_score(y_pred=final_result, y_true=self.y_test, pos_label=0)
        f1_pos = f1_score(y_pred=final_result, y_true=self.y_test,pos_label=1)
        f1_neg = f1_score(y_pred=final_result, y_true=self.y_test, pos_label=0)
        r_pos = recall_score(y_pred=final_result, y_true=self.y_test,pos_label=1)
        r_neg = recall_score(y_pred=final_result, y_true=self.y_test, pos_label=0)

        print('*'*60)

        result_dict = {'precision_score_pos':p_pos,
                           'precision_score_neg':p_neg,
                           'recall_score_pos':r_pos,
                           'recall_score_neg':r_neg,
                           'f1_pos':f1_pos,
                           'f1_neg':f1_neg}
        print(result_dict)
        return result_dict

if __name__ == '__main__':

    total_result_dict = {'precision_score_pos': 0,
                   'precision_score_neg': 0,
                   'recall_score_pos': 0,
                   'recall_score_neg': 0,
                   'f1_pos': 0,
                   'f1_neg': 0}

    for i in range(1):
        ddt = DoubleDecideModel()
        ddt.build_data()
        ddt.build_model()
        result_dict = ddt.train_model()
        #字典加和
        X = Counter(total_result_dict)
        Y = Counter(result_dict)
        total_result_dict = dict(X+Y)

    for i in total_result_dict.keys():
        total_result_dict[i] = total_result_dict[i]/10

    print("\n----------------------total result----------------------")
    print(total_result_dict)

