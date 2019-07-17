'''按照影响因素大小，对数据项进行排序'''
from model.Classifier import Classifier
from config import ClassifierConfig
import numpy as np
key=['0:id','1:abstract','2:content','3:s_content','4:t_content'
    ,'5:img','6:ref','7:click','8:share','9:good',
     '10:edittime','11:tag','12:items','13:lemmaID','14:editor_goodVersionCount',
     '15:editor_commitPassedCount',
     '16:editor_level','17:editor_featuredLemmaCount'
    ,'18:editor_createPassedCount'
    ,'19:editor_commitTotalCount'
    ,'20:editor_experience'
    ,'21:editor_passRatio']
sort_dict={}

if __name__ == '__main__':

    for i in range(0,22):

        ClassifierConfig.indexs = {a for a in range(22)}
        ClassifierConfig.indexs.remove(i)

        # 获得分类器
        classifier = Classifier()
        # 搭建分类器
        classifier.build()
        # 训练模型
        dict = classifier.train_model()

        sort_dict.update({i:np.mean(dict['recall_score_pos'])})

    result = zip(sort_dict.values(), sort_dict.keys())

    rank_list = [r for r in range(22)]

    for idx,tuple in enumerate(sorted(result,reverse=True)):

        rank = int(key[tuple[1]].split(':')[0])
        #rank_list[rank] = idx+1
        rank_list[rank] = tuple[0]
        print("{}-------------------{}".format(key[tuple[1]],tuple[0]))

    print(rank_list)