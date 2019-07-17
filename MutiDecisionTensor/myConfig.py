class Config:
    '''
    多决策向量：配置类
    '''
    def __init__(self):
        '''数据配置'''
        # 普通词条数据
        self.normal_data_path = "../data/0.csv"
        # 特色词条数据
        self.characteristic_data_path = "../data/1.csv"

        self.test_size = 0.1

        #每一列训练次数
        self.train_time = 10

        #模型保存地址
        self.model_save_path = "../data/model_path/"

        #多决策向量保存地址
        self.tensor_save_path = "../data/"

        #多决策向量权重
        self.weight = []