import pandas as pd
from config import ClassifierConfig
import numpy as np
from sklearn.preprocessing import minmax_scale ,maxabs_scale ,scale
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
'''1~12 12个文本属性
    14~21 8 个编辑者属性
'''
key=['0:id','1:abstract','2:content','3:s_content','4:t_content'
    ,'5:img','6:ref','7:click','8:share','9:good',
     '10:edittime','11:tag','12:items','13:lemmaID','14:editor_goodVersionCount',
     '15:editor_commitPassedCount',
     '16:editor_level','17:editor_featuredLemmaCount'
    ,'18:editor_createPassedCount'
    ,'19:editor_commitTotalCount'
    ,'20:editor_experience'
    ,'21:editor_passRatio']
#不使用列进行剔除
#选取10个文本特征 6影响较大
#indexs={0,13,14,15,16,17,18,19,20,21}
#选取10个编辑者特征 16、20影响
#indexs={0,1,2,3,4,5,6,7,8,9,10,11,12,13,16,20}

indexs={}

def to_df(data):
    '''转化成dataframe数据'''
    return pd.DataFrame(data)

def to_array(df):
    '''转化成numpy数据'''
    return df.values

def data_preprocess(data_0,data_1):
    '''数据预处理'''
    # 数据预处理
    len_0 = len(data_0)
    len_1 = len(data_1)

    # 拼接数据使用minmax_scale归一化/主成分分析法
    data = np.vstack((data_0, data_1))

    # 使用主成分分析
    if ClassifierConfig.use_PCA:
        print("use PCA")
        # 将标签与数据分开
        _label = data[:, -1]
        _data = data[:, :-1]
        pca = PCA(n_components=ClassifierConfig.n_components)
        # 训练PCA
        _data = pca.fit_transform(_data)
        # print("explained_variance_ratio:{}".format(pca.explained_variance_ratio_))
        #合并 维度[,n_components+1]
        data=np.c_[_data,_label]
    #归一化
    data_scaled = minmax_scale(data)
    data_0 = data_scaled[0:len_0,:]
    data_1 = data_scaled[len_0:len_0 + len_1,:]
    return data_0,data_1

def get_train_and_test_data(data_0,data_1):
    # 划分训练集和测试集
    train_0, test_0 = train_test_split(data_0, test_size=ClassifierConfig.normal_cut_size)
    train_1, test_1 = train_test_split(data_1, test_size=ClassifierConfig.characteristic_cut_size)
    print("训练数据：特色词条{}条,非特色词条:{}条".format(len(train_1), len(train_0)))
    print("测试数据：特色词条{}条,非特色词条:{}条".format(len(test_1), len(test_0)))
    # 拼接训练集和测试集
    train = np.vstack((train_0, train_1))
    test = np.vstack((test_0, test_1))
    return train,test

'''交叉验证下的数据处理'''
def get_full_train_data(data_0,data_1):

    print("特色词条{}条,非特色词条:{}条".format(len(data_1),len(data_0)))
    '''数据预处理'''
    # 数据预处理
    len_0 = len(data_0)
    len_1 = len(data_1)

    # 拼接数据使用minmax_scale归一化/主成分分析法
    data = np.vstack((data_0, data_1))

    # 使用主成分分析
    if ClassifierConfig.use_PCA:

        # 将标签与数据分开
        _label = data[:, -1]
        _data = data[:, :-1]
        pca = PCA(n_components=ClassifierConfig.n_components)
        # 训练PCA
        _data = pca.fit_transform(_data)
        # print("explained_variance_ratio:{}".format(pca.explained_variance_ratio_))
        # 合并 维度[,n_components+1]
        data = np.c_[_data, _label]
    # 归一化
    data_scaled = minmax_scale(data)
    #　打乱
    np.random.shuffle(data_scaled)

    # 删除部分条目
    for index in sorted(ClassifierConfig.indexs,reverse=True):
        data_scaled = np.delete(data_scaled, index, axis=1)

    # 获得训练标签
    Y = data_scaled[:, -1]
    X = np.delete(data_scaled, -1, axis=1)  # 删除flag
    return X,Y

def return_scaled_data(item):
    '''返回归一化好的数据'''
    # 读数据
    data_0_df = pd.read_csv("../data/0.csv")
    data_1_df = pd.read_csv("../data/1.csv")
    data_0 = data_0_df.values
    data_1 = data_1_df.values

    data = np.vstack((data_0, data_1))
    # 　打乱
    np.random.shuffle(data)

    item = np.expand_dims(item,axis=0)
    data = np.vstack((data, item))
    data_scaled = minmax_scale(data)

    item_scaled = data_scaled[-1]
    return item_scaled

def split_data_label(train,test):
    '''拆分label'''
    #打乱数据集
    np.random.shuffle(train)
    np.random.shuffle(test)
    # 转化成dataframe
    #train_df = pd.DataFrame(train)
    #test_df = pd.DataFrame(test)

    # 获得训练标签
    y_train = train[:,-1]
    x_train = np.delete(train,-1,axis=1)  # 删除flag
    # 获得测试标签
    y_test = test[:,-1]
    x_test = np.delete(test, -1, axis=1)  # 删除flag
    # 删除部分条目
    for index in indexs:
        x_train = np.delete(x_train, index, axis=1)
        x_test = np.delete(x_test, index, axis=1)

    return x_train,y_train,x_test,y_test

def cal_result(my_result,target):
    #准确率
    correct=0
    i=0
    for each in my_result :
        if each==target[i]:
            correct+=1
        i+=1
    return correct/len(my_result)

#获取模型评估统计数据
t_acc=0
t_p=0
t_r=0
t_f1=0
#最大评估参数
max_acc=-999
max_p=-999
max_r=-999
max_f1=-999

def add_eva_data(eva_dict):
    global t_acc, t_p, t_r, t_f1
    global max_acc, max_p, max_r, max_f1
    t_acc+=eva_dict['acc']
    t_p+=eva_dict['p']
    t_r+=eva_dict['r']
    t_f1+=eva_dict['f1']
    if eva_dict['acc']>max_acc:
        max_acc=eva_dict['acc']
    if eva_dict['p'] > max_p:
        max_p = eva_dict['p']
    if eva_dict['r'] > max_r:
        max_r = eva_dict['r']
    if eva_dict['f1'] > max_f1:
        max_f1 = eva_dict['f1']

def get_average(num):
    global t_acc, t_p, t_r, t_f1
    print("平均评估参数:精确度:{},准确度:{},召回率:{},f1值:{}"
          .format(t_p/num,t_acc/num,t_r/num,t_f1/num))
    acc=t_acc/num
    p=t_p/num
    r=t_r/num
    f1=t_f1/num
    t_acc = 0
    t_p = 0
    t_r = 0
    t_f1 = 0
    return {"acc": acc, "p": p, "r": r, "f1": f1}

def get_max():
    print("最大评估参数:精确度:{},准确度:{},召回率:{},f1值:{}"
          .format(max_p, max_acc, max_r, max_f1))

'''X_each, Y_each,X_test,Y_test'''
def split_data_average_1(a,b):
    a = list(a)
    b = list(b)
    assert len(a) == len(b)
    '''将数据划分成10折'''
    total_len = len(a)
    each_len = total_len // 10
    # print("each_len:{}".format(each_len))
    for i in range(0, total_len, each_len):
        # 若每一组不满each_len
        if i + each_len > total_len:
            break
        # 3
        r3 = a[i:i + each_len]
        # 4
        r4 = b[i:i + each_len]

        # 1
        if i == 0 :
            r1 = a[i+each_len:total_len]
        else:
            r1 = a[0:i]
            r1.extend(a[i+each_len:total_len])

        #2
        if i == 0 :
            r2 = b[i+each_len:total_len]
        else:
            r2 = b[0:i]
            r2.extend(b[i+each_len:total_len])

        yield r1, r2, r3, r4

'''十折交叉验证'''
''' @input  1.x词条 2.x编辑者　3.X 4.Y'''
''' @return 1.九份词条自身训练集　2.九份编辑者自身训练集　3.训练集对应的y 4.一份测试集　5.测试集对应的y'''
def split_data_average(a,b,c,d):
    a = list(a)
    b = list(b)
    c = list(c)
    d = list(d)

    assert len(a) == len(b) == len(c)
    '''将数据划分成10折'''
    total_len = len(a)
    each_len = total_len//10
    # print("each_len:{}".format(each_len))
    for i in range(0,total_len,each_len):
        #若每一组不满each_len
        if i +each_len > total_len:
            break

        #4
        r4 = c[i:i+each_len]
        #5
        r5 = d[i:i+each_len]
        #1

        if i == 0 :
            r1 = a[i+each_len:total_len]
        else:
            r1 = a[0:i]
            r1.extend(a[i+each_len:total_len])

        #2
        if i == 0 :
            r2 = b[i+each_len:total_len]
        else:
            r2 = b[0:i]
            r2.extend(b[i+each_len:total_len])

        #3
        if i == 0 :
            r3 = d[i+each_len:total_len]
        else:
            r3 = d[0:i]
            r3.extend(d[i+each_len:total_len])


        yield r1 , r2 , r3 ,r4 ,r5

def get_topN_column(topN,raw_data,index_top,add=None):
    target_data = None
    for i in range(topN):
        if add == None or add == False:
            index = index_top[i]
        elif add == True:
            index = index_top[i]+14

        if target_data is None:
            target_data = raw_data[:, index]
            target_data = np.expand_dims(target_data, axis=1)
        else:
            target_data = np.concatenate((target_data, np.expand_dims(raw_data[:, index], axis=1)),
                                               axis=1)
    return target_data

def split_textEditor_data(raw_data):
    '''将词条数据分为文本相关和编辑者相关'''
    data = np.array(raw_data)
    text_data = data[:,:14]
    editor_data = data[:,14:]

    return text_data , editor_data

if __name__ == '__main__':
    return_scaled_data([0, 714, 9075, 8, 12, 30, 279, 76336654, 309, 417384, 4825, 10, 1, 129156, 106.96, 189129.0, 8.36, 102.2, 3185.28, 216381.28, 33578.28, 94.52,1])