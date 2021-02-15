import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score ,confusion_matrix
from sklearn.preprocessing import minmax_scale ,maxabs_scale ,scale
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
from myConfig import Config

threshold = 6
biases = 1.5
key=['0:id','1:abstract','2:content','3:s_content','4:t_content'
    ,'5:img','6:ref','7:click','8:share','9:good',
     '10:edittime','11:tag','12:items','13:lemmaID','14:editor_goodVersionCount',
     '15:editor_commitPassedCount',
     '16:editor_level','17:editor_featuredLemmaCount'
    ,'18:editor_createPassedCount'
    ,'19:editor_commitTotalCount'
    ,'20:editor_experience'
    ,'21:editor_passRatio']

def predict_use_total(data):
    result = []
    '''根据优质总数判定'''
    for d in data:
        if sum(d) > threshold :
            result.append(1)
        else:
            result.append(0)
    return result

def predict_use_Weighting(data,y,p,get_detail = False,raw_score = -1,weighted=True):

    # sort_list = [21, 8, 19, 12, 9, 4, 1, 13, 18, 15, 7, 2, 22, 17, 10, 16, 3, 5, 20, 14, 11, 6]
    # sort_list = [17, 5, 16, 20, 19, 11, 1, 9, 15, 10, 7, 21, 22, 14, 8, 13, 2, 3, 18, 12, 6, 4]
    sort_list = [21, 7, 22, 12, 9, 4, 1, 13, 17, 15, 11, 2, 14, 19, 8, 18, 3, 5, 20, 16, 6, 10]
    predict_list = [0.30192381189192286, 0.5532114141114585, 0.3025585117977661, 0.583706696758526, 0.43291370631097414, 0.5697886262678413, 0.738438946069259, 0.43441379684486414, 0.3362204539374841, 0.36324345708469574, 0.5172107974293514, 0.7048391812865497, 0.34922884953833017, 0.30653098794840006, 0.4569829858950164, 0.3413951379731814, 0.5633179880014539, 0.5490270040464569, 0.3081263655307966, 0.3483470535738997, 0.4918375665799344, 0.47519785753662]
    recall_list =  [0.2870431893687707, 0.4432258064516129, 0.30603174603174604, 0.1341296928327645, 0.22145214521452145, 0.23689320388349516, 0.7426621160409557, 0.4359322033898304, 0.2624137931034483, 0.34579124579124576, 0.4353135313531353, 0.10862619808306709, 0.055852842809364554, 0.31328671328671326, 0.4170491803278689, 0.35942492012779553, 0.5835483870967743, 0.523529411764706, 0.3022727272727273, 0.3541666666666667, 0.38637873754152824, 0.4474576271186441]
    entropy_list = [0.03084108954595821, 0.16352505388044114, 0.04882526724022174, 0.17473839125601265, 0.15749392656526007,0.11909840466098781, 0.4123163166217875, 0.1224391689814765, 0.05038443813986626, 0.06491032060649232,0.14468527400740477, 0.050148784324932916, 0.3115704967188957, 0.005311865750827649, 0.12953110009922997,0.025407695776341277, 0.21364959923527893, 0.21309136111455734, 0.01966866389205979, 0.02615621778006494,0.03850272200759208, 0.16572379167498968]

    weight_list = [0 for i in range(22)]
    for index in range(22):
        #todo 计算weight-list 投票模型　weight = predict^2 + recall^2
        # weight_list[index] = (pow(predict_list[index],2) + pow(recall_list[index],2) )* entropy_list[index]

        # todo 计算weight-list 熵模型　
        if weighted:
            weight_list[index] = entropy_list[index]
        else:
            weight_list[index] = 0.1

    #对权重进行自动化确定
    if weighted:
        weight_list = minmax_scale(weight_list)

    weight_tuple_list = []
    result = []
    result_dict = []
    for i,d in enumerate(data):
        weight = 0

        for idx,r in enumerate(d):
            weight += r*weight_list[idx]
        weight += 0.1
        # #如果大于阈值再增加weight
        # if sum(d) > threshold and d[6]==1:
        #     weight *= biases
        weight_tuple_list.append((weight,i))

    #对评分进行排序
    weight_tuple_list = sorted(weight_tuple_list, reverse=True)
    #结果数组初始化
    result = [0 for i in weight_tuple_list]

    max_weight = -1
    min_weight = 999
    for tuple in weight_tuple_list:

        weight = tuple[0]
        if weight > max_weight:
            max_weight = weight
        if weight < min_weight:
            min_weight = weight


        #定义阈值 , 取整
        # threshold = weight_tuple_list[int(len(weight_tuple_list)*p)][0]
        threshold = p
        if weight > threshold:
            result[tuple[1]] = 1
        else:
            result[tuple[1]] = 0


    print("筛选率：{}".format(p))
    print("阈值:{}".format(threshold))
    print("max_score:{}".format(max_weight))
    print("min_score:{}".format(min_weight))

    if get_detail:
        idx = len(weight_tuple_list)
        if raw_score != -1:
            for _idx, tuple in enumerate(weight_tuple_list):
                if raw_score > tuple[0]:
                    idx = _idx
                    break


        high = weight_tuple_list[int(len(weight_tuple_list)*0.1)][0]
        good = weight_tuple_list[int(len(weight_tuple_list) * 0.2)][0]
        mid = weight_tuple_list[int(len(weight_tuple_list) * 0.6)][0]
        low = weight_tuple_list[int(len(weight_tuple_list) * 0.8)][0]
        # 百分比词条得分
        percent_score = (len(weight_tuple_list) - idx )/ len(weight_tuple_list)
        return high,good,mid,low,percent_score*100



    for idx,tuple in enumerate(weight_tuple_list):
        #print("score:{} level:{}".format(tuple[0],y[tuple[1]]))
        #print("data:{}".format(data[tuple[1]]))
        # 建立对象
        dict = {"idx":idx,"level":y[tuple[1]],"score":tuple[0]}

        result_dict.append(dict)

    return result , result_dict

'''统计各排名点前后数目'''
def statics(list, forward = True, points= [300,600,900,1500,2100,2700]):
    result_arr = np.zeros(len(points))
    if forward:
        point_idx = 0
    else:
        type = len(points) - 1
    for data in list:
        if point_idx >= len(points) : break
        if data <= points[point_idx]:
            result_arr[point_idx] += 1
        else:
            point_idx+=1
    for i in range(1, len(result_arr)):
        result_arr[i] += result_arr[i-1]
    print(str(result_arr))


if __name__ == '__main__':
    songTi = font_manager.FontProperties(fname='/usr/local/share/fonts/s/simhei.ttf')
    conf = Config()
    p_list = []
    r_list = []
    f1_list = []


    p = 0.28

    '''通过决策向量预测'''
    pd_data = pd.read_csv(conf.tensor_save_path+"MDT.csv")
    data = np.array(pd_data.values)[:,1:].astype(int)

    X_data = data[:,:-1]
    Y_data = data[:,-1]

    result,result_dict = predict_use_Weighting(X_data,Y_data,p)

    #混淆矩阵
    matrix = confusion_matrix(y_pred=result, y_true=Y_data)

    acc = accuracy_score(y_pred=result,y_true=Y_data)
    p_1 = precision_score(y_pred=result,y_true=Y_data,pos_label=1)
    r_1 = recall_score(y_pred=result,y_true=Y_data,pos_label=1)
    f1_1 = f1_score(y_pred=result,y_true=Y_data,pos_label=1)
    p_0 = precision_score(y_pred=result, y_true=Y_data,pos_label=0)
    r_0 = recall_score(y_pred=result, y_true=Y_data,pos_label=0)
    f1_0 = f1_score(y_pred=result,y_true=Y_data,pos_label=0)

    print("precise_pos:{}".format(p_1))
    print("recall_pos:{}".format(r_1))
    print("F1-Score_pos:{}".format(f1_1))
    print("\n")
    print("precise_neg:{}".format(p_0))
    print("recall_neg:{}".format(r_0))
    print("F1-Score_neg:{}".format(f1_0))
    #
    p_list.append(p_1)
    r_list.append(r_1)
    f1_list.append(f1_1)

    # 建立对象
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111)
    # ax.set_title('BaiduBaike items\' score quality ranking')
    plt.xlabel("词条索引", fontproperties=songTi,fontsize=26)
    plt.ylabel("词条评分", fontproperties=songTi,fontsize=26)


    blue_x = []
    blue_y = []
    red_x = []
    red_y = []
    for id, dict in enumerate(result_dict):
        if dict['level'] == 1:
            blue_x.append(dict['idx'])
            blue_y.append(dict['score'])
        else:
            red_x.append(dict['idx'])
            red_y.append(dict['score'])


    red_x = red_x[5:-1]
    red_y = red_y[5:-1]
    statics(blue_x)
    statics(red_x)
    # plt.bar(red_x,red_y,color='black',width = 1)
    plt.xticks(range(0, len(result_dict) + 1, 300))
    plt.tick_params(labelsize=22)
    plt.bar(blue_x, blue_y, color='black')

    # plt.legend()
    plt.show()