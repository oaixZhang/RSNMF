import time

import numpy as np
from sklearn.externals import joblib

from load_data import load_ratings

"""
my implementation of koren MF without bias
"""


class MF():

    def __init__(self, k=20, lmda=0.06, gamma=0.01, maxround=1000, maxdelay=10):
        """
        :param k: feature dimension
        :param gamma: learning rate
        :param lmda: lambda for l2 regularization
        """
        self.k = k

        self.lmda = lmda
        self.gamma = gamma
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

        # # Create a list of training samples
        # self.samples = [
        #     (i, j, self.R[i, j])
        #     for i in range(self.num_users)
        #     for j in range(self.num_items)
        #     if self.R[i, j] > 0
        # ]

        start_time = time.time()
        rmses = []
        f = open('./logs/{}_k-{}_lambda-{}.txt'.format(start_time, self.k, self.lmda), 'w')

        for round in range(self.maxround):
            for rating in ratings:
                userid = rating[0]
                itemid = rating[1]
                rui = rating[2]
                ruihat = self.P[userid].dot(self.Q[itemid].T)
                eui = (rui - ruihat)

                # update puk
                self.P[userid, :] += self.gamma * (eui * self.Q[itemid, :] - self.lmda * self.P[userid, :])

                # update qik
                self.Q[itemid, :] += self.gamma * (eui * self.P[userid, :] - self.lmda * self.Q[itemid, :])

            # current RMSE on test set
            sum, count = 0, 0
            for tr in test:  # tr short for test rating
                userid = tr[0]
                itemid = tr[1]
                rui = tr[2]
                ruihat = self.P[userid].dot(self.Q[itemid].T)
                sum += (rui - ruihat) ** 2
                count += 1

            rmse = np.sqrt(sum / count)
            print(rmse)
            f.write('{}\n'.format(rmse))
            f.flush()
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


if __name__ == '__main__':
    train = load_ratings('./train.txt')
    test = load_ratings('./test.txt')
    rsnmf = MF()
    rmses = rsnmf.fit(train, test)
