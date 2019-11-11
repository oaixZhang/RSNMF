import numpy as np
from load_data import load_ratings
from sklearn.externals import joblib
import time


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

            for rating in ratings:
                userid = rating[0]
                itemid = rating[1]
                rui = rating[2]
                ruihat = self.P[userid].dot(self.Q[itemid].T)

                for i in range(self.k):
                    userup[userid][i] += self.Q[itemid][i] * rui
                    userdown[userid][i] += self.Q[itemid][i] * ruihat
                    itemup[itemid][i] += self.P[userid][i] * rui
                    itemdown[itemid][i] += self.P[userid][i] * ruihat

            # update puk
            for uid in range(1, self.max_userid + 1):
                for i in range(self.k):
                    userdown[uid][i] += self.Iu[uid] * self.lmda * self.P[uid][i]
                    if userdown[uid][i] != 0:
                        self.P[uid][i] *= (userup[uid][i] / userdown[uid][i])
            # update qik
            for iid in range(1, self.max_itemid + 1):
                for i in range(self.k):
                    itemdown[iid][i] += self.Ui[iid] * self.lmda * self.Q[iid][i]
                    if itemdown[iid][i] != 0:
                        self.Q[iid][i] *= (itemup[iid][i] / itemdown[iid][i])

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
    rsnmf = RSNMF()
    rmses = rsnmf.fit(train, test)
