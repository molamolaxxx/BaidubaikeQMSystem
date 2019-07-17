class ClassifierConfig():
    '''svm的配置类'''
    def __init__(self):
        '''分类器配置'''
        #使用模块
        self.use_method='svm'

        #svm配置,name=svm
        self.C = 100
        self.kernel = 'rbf'

        #随机森林配置,name=rf
        self.n_estimators = 100

        #决策树配置name=dt

        #逻辑回归分类器配置 ,name=lr
        self.penalty = 'l2'


        '''训练参数配置'''
        self.epoch_num=20
        self.total_num=1
        #总共训练750次

        '''数据配置'''
        #普通词条数据
        self.normal_data_path="./data/0.csv"
        #特色词条数据
        self.characteristic_data_path = "./data/1.csv"

        #使用十字交叉验证
        self.use_cross_validation = True
        self.sort_list = [21, 8, 19, 12, 9, 4, 1, 13, 18, 15, 7, 2, 22, 17, 10, 16, 3, 5, 20, 14, 11, 6]

    #使用主成分分析法
    use_PCA = False
    #主成分个数
    n_components = 'mle'

    #划分特色测试集和训练集比例
    characteristic_cut_size=0.1
    # 划分普通测试集和训练集比例
    normal_cut_size = 0.1

    #删除的列
    indexs={}


'''爬虫的配置类'''
class SpiderConfig():
    '''爬虫配置'''
    #根地址
    base_url="http://baike.baidu.com/view/"
    #爬虫线程数
    thread_num=10

    '''数据库配置'''
    #用户名
    username="root"
    #密码
    password="314"
    #数据库地址
    host="127.0.0.1"
    #端口号
    port=3306
    #数据库名称
    db_name="baidu_db"


class ATNNConfig():
    def __init__(self):

        self.topN = 5

        # 普通词条数据
        self.normal_data_path = "./data/0.csv"
        # 特色词条数据
        self.characteristic_data_path = "./data/1.csv"

        #ATNN参数，各全连接层的参数
        self.topN_fc_num = 15
        self.other_fc_num = 10

        #超参
        self.batch_size = 10
        self.epochs = 100
        self.learning_rate = 0.1
        self.dropout = 0.2

        self.sort_list =[21, 8, 19, 12, 9, 4, 1, 13, 18, 15, 7, 2, 22, 17, 10, 16, 3, 5, 20, 14, 11, 6]

        '''for bpnn'''
        self.use_BP_NN = False
        self.bp_size = 10
