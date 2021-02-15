from myModel import MultiplyDecisionTensor

if __name__ == '__main__':
    '''训练多决策向量'''
    model = MultiplyDecisionTensor()
    model.build()
    model.run()