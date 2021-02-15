import tensorflow as tf
import pandas as pd
from config import ATNNConfig
from topNAttention.data import Data
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class ATNN:
    '''ATNN : attention Neural Networks 注意力神经网络
    '''
    def __init__(self):
        self.config = ATNNConfig()

    def build_data(self):
        self.data = Data()

    def build_model(self):
        tf.reset_default_graph()
        #添加占位器
        self._define_placeholder()
        #定义变量
        self._define_variable()
        #搭建网络结构
        self._define_NN_structure()
        #初始化训练器
        self.init_trainer()
        #初始化会话
        self._init_sess()
        self.log_trainable()

        self.total_temp = {"acc":0, "p":0, "r":0, "f1":0}
        self.batch_size = None

    def train_model(self):

        for epoch in range(self.config.epochs):
            print("build data")
            self.data.build()

            print("start training")
            print("epoch:{}".format(epoch))

            for X,Y,X_test,Y_test in self.data.get_batches_data(self.config.batch_size):
                if not self.config.use_BP_NN:
                    X = np.array(X)
                    X_top ,X_other = self.data.split_data(X)

                    #Y增加维度
                    Y = np.expand_dims(Y,axis=1)

                    #转化数据类型
                    Y = np.array(Y).astype(np.int32)

                    X_top = np.array(X_top).astype(np.float32)
                    X_other = np.array(X_other).astype(np.float32)


                    fd = self._get_feed_dict(topN_input=X_top,other_input=X_other,
                                             dropout=self.config.dropout,learning_rate=self.config.learning_rate,label=Y)

                else:
                    # Y增加维度
                    Y = np.expand_dims(Y, axis=1)

                    fd = self._get_feed_dict(input=X,
                                             dropout=self.config.dropout, learning_rate=self.config.learning_rate,
                                             label=Y)

                #开始训练
                self.sess.run(self.trainer,feed_dict=fd)
                loss = self.sess.run(self.loss ,feed_dict=fd)
                print("loss:{}".format(loss))

                '''测试训练结果'''
                result = self._predict_batch(X_test,Y_test)
                self.total_temp['p'] += result['p']
                self.total_temp['acc'] += result['acc']
                self.total_temp['f1'] += result['f1']
                self.total_temp['r'] += result['r']

            self.total_temp['p'] /= self.config.batch_size
            self.total_temp['acc'] /= self.config.batch_size
            self.total_temp['f1'] /= self.config.batch_size
            self.total_temp['r'] /= self.config.batch_size

            print('-'*50)
            print(self.total_temp)
            self.total_temp = {"acc": 0, "p": 0, "r": 0, "f1": 0}


        result = self._predict_batch(self.data.test_X_final,self.data.test_Y_final)
        print("final test result:{}".format(result))
    def init_trainer(self):
        loss = self.loss
        #学习率
        learning_rate = self.learning_rate
        self.trainer = tf.train.ProximalGradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
    '''inner function'''

    def log_trainable(self):
        """打印变量信息
        """
        variables_names = [v.name for v in tf.trainable_variables()]
        values = self.sess.run(variables_names)
        for k, v in zip(variables_names, values):
            print("Variable: {}".format(k))
            print("Shape: {}".format(v.shape))

    def _get_feed_dict(self,topN_input=None,other_input=None,input=None,dropout=None,learning_rate=None ,label=None):
        '''
        :param topN_layer:  [N,topn]
        :param other_layer: [N,22 - topn]
        :param dropout:
        :param learning_rate:
        :param label:  0/1
        :return:
        '''
        '''@:return feeddict'''
        if not self.config.use_BP_NN:
            feed_dict = {

                self.topN_input:topN_input,

                self.other_input:other_input,

                self.label :label,

                self.dropout: dropout,

                self.learning_rate: learning_rate
            }

        else:
            feed_dict = {

                self.input: input,

                self.label: label,

                self.dropout: dropout,

                self.learning_rate: learning_rate
            }

        return feed_dict

    def _predict_batch(self,X_test,Y_test):
        '''测试'''
        if not self.config.use_BP_NN:
            X_test = np.array(X_test)
            X_top_test , X_other_test =  self.data.split_data(X_test)

            y_true = Y_test
            # Y增加维度
            Y_test = np.expand_dims(Y_test, axis=1)

            # 转化数据类型
            Y_test = np.array(Y_test).astype(np.int32)
            X_top_test = np.array(X_top_test).astype(np.float32)
            X_other_test = np.array(X_other_test).astype(np.float32)

            fd_test = self._get_feed_dict(topN_input=X_top_test, other_input=X_other_test, dropout=self.config.dropout,learning_rate= self.config.learning_rate, label=Y_test)

        else:
            y_true = Y_test
            # Y增加维度
            Y_test = np.expand_dims(Y_test, axis=1)
            fd_test = self._get_feed_dict(input=X_test, dropout=self.config.dropout,
                                          learning_rate=self.config.learning_rate, label=Y_test)

        y_pred = self.sess.run(self.relations_pred, feed_dict=fd_test)
        y_true = np.array(y_true).astype(np.int32)

        #print(y_pred)
        #print(y_true)

        p = precision_score(y_true=y_true, y_pred=y_pred)
        r = recall_score(y_true=y_true,y_pred=y_pred)
        acc = accuracy_score(y_true=y_true, y_pred=y_pred)
        f1 = f1_score(y_true=y_true, y_pred=y_pred)

        return {"acc":acc, "p":p, "r":r, "f1":f1}


    def _define_variable(self):
        '''定义神经网络变量'''
        '''first layer'''
        if not self.config.use_BP_NN:
            # 添加第一层注意力权重 ,shape = [topN,topN_fc_num]
            self.top_full_connect_layer = tf.get_variable('W_top_1', dtype=tf.float32,
                                                          shape=[self.config.topN, self.config.topN_fc_num])
            # 偏置
            self.top_full_connect_biases = tf.get_variable('b_top_1', dtype=tf.float32,
                                                           shape=[self.config.topN_fc_num])

            # 添加第一层非注意力权重 ,shape = [22 - topN,other_fc_num]
            self.other_full_connect_layer = tf.get_variable('W_other_1', dtype=tf.float32,
                                                            shape=[22 - self.config.topN, self.config.other_fc_num])

            # 偏置
            self.other_full_connect_biases = tf.get_variable('b_other_1', dtype=tf.float32,
                                                             shape=[self.config.other_fc_num])

            '''end layer shape = [topN_fc_num+other_fc_num,2]'''
            #激活层
            self.activation_layer = tf.get_variable('activation_layer', dtype=tf.float32,
                                             shape=[self.config.topN_fc_num, 1])
            # 偏置
            self.activation_biases = tf.get_variable('b_activation', dtype=tf.float32,
                                                    shape=[1])
            #抑制层
            self.Inhibition_layer = tf.get_variable('Inhibition_layer', dtype=tf.float32,
                                                    shape=[self.config.other_fc_num, 1])
            # 偏置
            self.Inhibition_biases = tf.get_variable('b_Inhibition', dtype=tf.float32,
                                                    shape=[1])


            '''for bpnn'''
        else:
            self.bp_layer = tf.get_variable('bp_layer', dtype=tf.float32,shape=[22,self.config.bp_size])
            self.bp_biases = tf.get_variable('bp_biases', dtype=tf.float32,shape=[self.config.bp_size])

            self.logits_layer = tf.get_variable('logits_layer', dtype=tf.float32,shape=[self.config.bp_size,2])
            self.logits_biases = tf.get_variable('logits_biases', dtype=tf.float32, shape=[2])

    def _define_NN_structure(self):
        '''定义网络结构'''
        #第一层[,topN]   [,22-topN ]
        if not self.config.use_BP_NN:
            top_full_connect_result = tf.matmul(self.topN_input,self.top_full_connect_layer)+self.top_full_connect_biases
            top_full_connect_result = tf.nn.tanh(top_full_connect_result)

            #加入dropout层，筛选无用信息
            self.other_full_connect_layer = tf.nn.dropout(self.other_full_connect_layer,self.dropout)
            other_full_connect_result = tf.matmul(self.other_input,self.other_full_connect_layer)+self.other_full_connect_biases

            # [,1]
            activation_result = tf.matmul(top_full_connect_result,self.activation_layer)+self.activation_biases
            Inhibition_result = tf.matmul(other_full_connect_result,self.Inhibition_layer)+self.Inhibition_biases

            # #全连接 , 在第一个维度拼接 [,topN_fc_num+other_fc_num]
            # full_connect = tf.concat([top_full_connect_result,other_full_connect_result],1)
            logits = tf.concat([Inhibition_result,activation_result],1)
            #第二层 [,2]
            #logits = tf.matmul(full_connect,self.end_layer)+self.end_layer_biases
            #logits = tf.nn.relu(logits)
        else:
            '''use normal bp'''
            first_result = tf.matmul(self.input,self.bp_layer)+self.bp_biases

            second_result = tf.matmul(first_result,self.logits_layer)+self.logits_biases

            logits = tf.nn.tanh(second_result)

        '''softmax'''
        #[batchsize,2]
        softmax_tensor = tf.nn.softmax(logits)
        # [batchsize,1]
        _relations_pred = tf.argmax(softmax_tensor, axis=1)
        # [batchsize]
        self.relations_pred = tf.reshape(_relations_pred, [-1])

        # 扩充维数，将（5，）扩充为（5,1），里面的内容不变
        batch_size = tf.size(self.label)
        # 扩充维数
        indices = tf.expand_dims(tf.range(0, batch_size, 1), 1)
        # 将indices和labels在第二维连接
        concated = tf.concat([indices, self.label], 1)
        onehot_labels = tf.sparse_to_dense(
            concated, tf.stack([batch_size, 2]), 1.0, 0.0)

        '''交叉熵函数'''
        cross_entropy = -tf.reduce_sum(onehot_labels * tf.log(softmax_tensor), reduction_indices=[1])
        self.loss = tf.reduce_mean(cross_entropy)

    def _define_placeholder(self):
        '''添加placeholder'''

        if not self.config.use_BP_NN:
            #输入层，存放前n个分类准确率高的项
            self.topN_input = tf.placeholder(name="topN_input",dtype=tf.float32,shape=[None,self.config.topN])
            #输入层，存放其他的项
            self.other_input = tf.placeholder(name="other_input",dtype=tf.float32,shape=[None,22 - self.config.topN])

        else:
            '''for bpnn'''
            self.input = tf.placeholder(name="input", dtype=tf.float32, shape=[None, 22])

        # 输入标签值 0 或 1
        self.label = tf.placeholder(name="label",dtype=tf.int32, shape=[None,1])

        #超参
        self.dropout = tf.placeholder(name="dr",dtype=tf.float32,shape=[])
        self.learning_rate = tf.placeholder(dtype=tf.float32, shape=[],
                                            name="lr")

    def _init_sess(self):
        '''初始化tensorflow会话'''

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

if __name__ == '__main__':
    model = ATNN()
    model.build_data()
    model.build_model()
    model.train_model()