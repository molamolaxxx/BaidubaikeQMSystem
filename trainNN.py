from topNAttention.atnn import ATNN

if __name__ == '__main__':
    model = ATNN()

    model.build_data()
    model.build_model()

    model.train_model()