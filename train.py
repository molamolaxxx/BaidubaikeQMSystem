'''训练模型'''
from model.Classifier import Classifier
from collections import Counter

if __name__ == '__main__':
    total_result_dict = {'precision_score_pos': 0,
                         'precision_score_neg': 0,
                         'recall_score_pos': 0,
                         'recall_score_neg': 0,
                         'f1_pos': 0,
                         'f1_neg': 0}
    for i in range(10):
        # 获得svm类
        svm = Classifier()
        # 搭建svm模型
        svm.build()
        # 训练模型
        result_dict = svm.train_model()

        # 字典加和
        X = Counter(total_result_dict)
        Y = Counter(result_dict)
        total_result_dict = dict(X + Y)

    for i in total_result_dict.keys():
        total_result_dict[i] = total_result_dict[i] / 10

    print("\n----------------------total result----------------------")
    print(total_result_dict)
