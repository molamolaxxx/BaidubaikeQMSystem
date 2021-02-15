'''评分阈值下的roc曲线'''
from predict import predict_use_Weighting,pd,np,Config,confusion_matrix

def run_auc(x,y):
    x.sort()
    y.sort()
    sum = 0
    num = len(x) - 1
    for i in range(num):
        sum += (y[i] + y[i+1])*(x[i+1]-x[i])/2
    print("auc:"+str(sum))

if __name__ == '__main__':
    conf = Config()
    '''通过决策向量预测'''
    pd_data = pd.read_csv(conf.tensor_save_path + "MDT.csv")
    data = np.array(pd_data.values)[:, 1:].astype(int)

    X_data = data[:, :-1]
    Y_data = data[:, -1]

    x = []
    y = []
    # 加权重
    for p in np.arange(0,5.6,0.1):
        '''Y_data:标签 result：预测值'''
        result, result_dict = predict_use_Weighting(X_data, Y_data, p)
        # 混淆矩阵
        matrix = confusion_matrix(y_pred=result, y_true=Y_data)
        tp_rate = matrix[1][1]/(matrix[1][0]+matrix[1][1])
        fp_rate = matrix[0][1]/(matrix[0][0]+matrix[0][1])
        print((fp_rate,tp_rate))
        x.append(fp_rate)
        y.append(tp_rate)

    x1 = []
    y1 = []
    # 不加权重
    for p in np.arange(0, 2.2, 0.05):
        '''Y_data:标签 result：预测值'''
        result, result_dict = predict_use_Weighting(X_data, Y_data, p,weighted=False)
        # 混淆矩阵
        matrix = confusion_matrix(y_pred=result, y_true=Y_data)
        tp_rate = matrix[1][1] / (matrix[1][0] + matrix[1][1])
        fp_rate = matrix[0][1] / (matrix[0][0] + matrix[0][1])
        print((fp_rate, tp_rate))
        x1.append(fp_rate)
        y1.append(tp_rate)

    run_auc(x,y)
    run_auc(x1,y1)
    import matplotlib.pyplot as plt
    from pylab import mpl
    import matplotlib.font_manager as font_manager

    songTi = font_manager.FontProperties(fname='/usr/local/share/fonts/s/simhei.ttf')
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111)
    # 字体
    # ax.set_title('评分模式下的ROC曲线', fontsize=26, fontproperties=songTi)
    plt.ylabel("True Positive Rate", fontproperties=songTi, fontsize=26)
    plt.xlabel("False Positive Rate", fontproperties=songTi, fontsize=26)
    plt.tick_params(labelsize=20)
    plt.xlim(-0.02,1.02)
    plt.ylim(-0.02,1.02)
    #
    # plt.ylim((0, 100))
    # plt.xlim((0,3000))
    # 画图
    # plt.plot(x,y1, 'o-',color="green", markersize=3, label="BERT-CRF")

    plt.plot(x, y, 'o-', markersize=3, label="weighted",color="blue")
    plt.plot(x1, y1, 'o-', markersize=3, label="not weighted",color="red")
    plt.plot([0,1], [0,1], 'o-', markersize=3, linestyle='--')
    # plt.plot(x, y4, 'o-', color="black", markersize=3, label="BERT+FCEncoder-CRF")
    plt.xticks(np.arange(0,1.1,0.1))
    plt.yticks(np.arange(0,1.1,0.1))
    plt.legend(prop={'family': 'Times New Roman',
                     'weight': 'normal',
                     'size': 18,
                     }, loc='lower right')
    plt.show()
