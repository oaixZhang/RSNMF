import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.externals import joblib

from util import update_puk, update_qik, calculate_rmse, calculate_delta


class RSNMF():
    def __init__(self, k=20, lmda=0.06, maxround=1000, maxdelay=10):
        """
        :param k: feature dimension
        :param lmda: lambda for l2 regularization
        """
        self.k = k
        self.lmda = lmda
        self.maxround = maxround
        self.minrmse = 100
        self.minround = 10000
        self.maxdelay = maxdelay

    def fit(self, ratings, test):
        """
        :param ratings: train set
        :param test: test or validate set
        :return arrar: RMSE on test set
        """
        self.max_userid = np.max(ratings[:, 0])
        self.max_itemid = np.max(ratings[:, 1])
        # +1 to keep array index consistent with userId(itemId)
        self.P = np.random.rand(self.max_userid + 1, self.k)
        self.Q = np.random.rand(self.max_itemid + 1, self.k)
        # item set rated by user u , index denotes userId
        self.Iu = np.zeros(self.max_userid + 1)
        # user set having rated item i
        self.Ui = np.zeros(self.max_itemid + 1)

        for rating in ratings:
            self.Iu[rating[0]] += 1
            self.Ui[rating[1]] += 1

        start_time = time.time()
        rmses = []
        f = open('./logs/{}_k-{}_lambda-{}.txt'.format(start_time, self.k, self.lmda), 'w')

        for round in range(self.maxround):
            userup = np.zeros([self.max_userid + 1, self.k])
            userdown = np.zeros([self.max_userid + 1, self.k])
            itemup = np.zeros([self.max_itemid + 1, self.k])
            itemdown = np.zeros([self.max_itemid + 1, self.k])

            calculate_delta(ratings, self.P, self.Q, userup, userdown, itemup, itemdown)

            # update puk
            update_puk(self.max_userid, self.k, self.Iu, self.lmda, self.P, userdown, userup)
            # update qik
            update_qik(self.max_itemid, self.k, self.Ui, self.lmda, self.Q, itemdown, itemup)

            # current RMSE on test set
            rmse = calculate_rmse(test, self.P, self.Q)
            print(rmse)
            f.write('{}\n'.format(rmse))
            rmses.append(rmse)

            # record
            if self.minrmse > rmse:
                self.minrmse = rmse
                self.minround = round
            elif round - self.minround == 1:  # RMSE first rises, backup current model params
                joblib.dump(self, './models/rsnmf_{}.pkl'.format(start_time))
            elif round - self.minround == self.maxdelay:  # if RMSE keeps rising beyond given maxdelay round , break and stop training
                break

        end_time = time.time()
        train_time = end_time - start_time

        f.write('-----------------------\n')
        f.write('train round: %d \n' % (self.minround))
        f.write('train time: %f s \n' % (train_time))
        f.flush()
        f.close()

        return rmses

    def predict(self, userid, itemid):

        """
        :param userid:
        :param itemid:
        :return:
        """
        return self.P[userid].dot(self.Q[itemid].T)


def plot_rmse():
    rmses = []
    with open('./logs/1556594371.2918932_k-20_lambda-0.06.txt', 'r') as f:
        for line in f.readlines():
            try:
                rmses.append(float(line.strip('\n')))
            except ValueError:
                break
    print(rmses)
    plt.plot(rmses, label='lambda = 0.06 , k = 20')
    plt.ylim(0.8, 1.8)
    plt.legend(loc='upper right')
    plt.show()


if __name__ == '__main__':
    # train = load_ratings('./data/train.txt')
    # test = load_ratings('./data/test.txt')
    df = pd.read_csv('./data/train_ratings_df.csv')
    train = df.values
    df = pd.read_csv('./data/test_ratings_df.csv')
    test = df.values

    rsnmf = RSNMF()
    rmses = rsnmf.fit(train, test)
    # plot_rmse()
