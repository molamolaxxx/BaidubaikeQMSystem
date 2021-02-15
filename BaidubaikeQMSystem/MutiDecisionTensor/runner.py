import sys
sys.path.append("/home/molamola/PycharmProjects/baidubaikeQMSystem/")

from MutiDecisionTensor.myModel import MultiplyDecisionTensor
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import minmax_scale
from myConfig import Config
from predict import predict_use_Weighting
from myspider.run import Runner as SpRunner
from model.datautils import return_scaled_data

class Runner:
    def __init__(self):
        self.model_list = None
        self.config = Config()
        self.load_model()


    def predict_one_item(self,item):
        sort_list = [21, 7, 22, 12, 9, 4, 1, 13, 17, 15, 11, 2, 14, 19, 8, 18, 3, 5, 20, 16, 6, 10]
        predict_list = [0.30192381189192286, 0.5532114141114585, 0.3025585117977661, 0.583706696758526, 0.43291370631097414,
                        0.5697886262678413, 0.738438946069259, 0.43441379684486414, 0.3362204539374841, 0.36324345708469574,
                        0.5172107974293514, 0.7048391812865497, 0.34922884953833017, 0.30653098794840006,
                        0.4569829858950164, 0.3413951379731814, 0.5633179880014539, 0.5490270040464569, 0.3081263655307966,
                        0.3483470535738997, 0.4918375665799344, 0.47519785753662]
        recall_list = [0.2870431893687707, 0.4432258064516129, 0.30603174603174604, 0.1341296928327645, 0.22145214521452145,
                       0.23689320388349516, 0.7426621160409557, 0.4359322033898304, 0.2624137931034483, 0.34579124579124576,
                       0.4353135313531353, 0.10862619808306709, 0.055852842809364554, 0.31328671328671326,
                       0.4170491803278689, 0.35942492012779553, 0.5835483870967743, 0.523529411764706, 0.3022727272727273,
                       0.3541666666666667, 0.38637873754152824, 0.4474576271186441]
        entropy_list = [0.03084108954595821, 0.16352505388044114, 0.04882526724022174, 0.17473839125601265,
                        0.15749392656526007, 0.11909840466098781, 0.4123163166217875, 0.1224391689814765,
                        0.05038443813986626, 0.06491032060649232, 0.14468527400740477, 0.050148784324932916,
                        0.3115704967188957, 0.005311865750827649, 0.12953110009922997, 0.025407695776341277,
                        0.21364959923527893, 0.21309136111455734, 0.01966866389205979, 0.02615621778006494,
                        0.03850272200759208, 0.16572379167498968]

        weight_list = [0 for i in range(22)]
        for index in range(22):
            # todo 计算weight-list 投票模型　weight = predict^2 + recall^2
            weight_list[index] = pow(predict_list[index], 2) + pow(recall_list[index], 2)
            # todo 计算weight-list 熵模型　weight = predict^2 + recall^2
            #weight_list[index] = entropy_list[index]

        # 对权重进行自动化确定
        weight_list = minmax_scale(weight_list)

        '''预测一个词条分类结果'''
        weight = 0
        for idx,cl in enumerate(item):
            model = self.model_list[idx]
            result =model.predict([[cl]])

            result = int(result[0])
            weight += result*weight_list[idx]

        return weight

    def load_model(self):
        if self.model_list is None:
            self.model_list = []
            for idx in range(22):
                #以二进制读取
                file = open(self.config.model_save_path+'model-'+str(idx)+'.pickle','rb')
                model = pickle.load(file)
                self.model_list.append(model)
        else:
            pass

if __name__ == '__main__':

    #获得词条
    url = sys.argv[1]
    if url is None:
        print("请输入评价词条网址")

    spRunner = SpRunner()
    item = spRunner.get_one_page_data(url)

    input_data = [item.id,item.abstract,item.content,item.s_content,item.t_content
        ,item.img,item.ref,item.click,item.share,item.good,item.edit_time,item.tag,item.items,item.lemmaId,item.editor_goodVersionCount,
                  item.editor_commitPassedCount,item.editor_level,item.editor_featuredLemmaCount
                  ,item.editor_createPassedCount,item.editor_commitTotalCount,item.editor_experience,item.editor_passRatio,item.flag]

    input_data = [int(i) for i in input_data]


    r = Runner()
    result = r.predict_one_item(return_scaled_data(input_data)[0:21])

    print("词条最终质量评分为{}".format(result))

    pd_data = pd.read_csv(r.config.tensor_save_path + "MDT.csv")
    data = np.array(pd_data.values)[:, 1:].astype(int)

    X_data = data[:, :-1]
    Y_data = data[:, -1]

    high, good, mid, low = predict_use_Weighting(X_data, Y_data, 0.2,get_detail=True)

    if result >= high:
        print("词条质量为优秀,建议评为优质词条")
    elif result < high and result >= good:
        print("词条质量为良好,建议评为优质词条")
    elif result < good and result >= mid:
        print("词条质量为中等")
    elif result < mid and result >= low:
        print("词条质量为中下等")
    elif result < low:
        print("词条质量为下等")

