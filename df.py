import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

from gcforest import CascadeForestRegressor
from load_data import load_users, load_movies, concat_u_i
from util import calculate_delta, update_puk, update_qik, calculate_rmse


class RSNMFDF():
    def __init__(self, k=20, lmda=0.06, maxround=1000, maxdelay=10):
        self.k = k
        self.lmda = lmda
        self.maxround = maxround
        self.minrmse = 100
        self.minround = 10000
        self.maxdelay = maxdelay

    def traindf(self, X_train, y_train):
        regressor = CascadeForestRegressor(
            estimators_config=[{'estimator_class': ExtraTreesRegressor,
                                'estimator_params': {'n_estimators': 1000,
                                                     'min_samples_split': 0.1,
                                                     'max_features': 1,
                                                     'n_jobs': -1, }},
                               {'estimator_class': ExtraTreesRegressor,
                                'estimator_params': {'n_estimators': 1000,
                                                     'min_samples_split': 0.1,
                                                     'max_features': 'sqrt',
                                                     'n_jobs': -1, }},
                               {'estimator_class': RandomForestRegressor,
                                'estimator_params': {'n_estimators': 1000,
                                                     'min_samples_split': 0.1,
                                                     'max_features': 1,
                                                     'n_jobs': -1, }},
                               {'estimator_class': RandomForestRegressor,
                                'estimator_params': {'n_estimators': 1000,
                                                     'min_samples_split': 0.1,
                                                     'max_features': 'sqrt',
                                                     'n_jobs': -1, }}])
        regressor.fit(X_train[:10000], y_train[:10000])
        return regressor

    def _df_predict(self, user_id, movie_id):
        data = concat_u_i(self.users, self.movies, user_id, movie_id)
        return self.df.predict(data)

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

        self.users = load_users()
        self.movies = load_movies()

        # train DF first
        dftrain = pd.read_csv('./data/train_df.csv')
        y_dftrain = dftrain.pop('21').values
        X_dftrain = dftrain.values

        self.df = self.traindf(X_dftrain, y_dftrain)

        start_time = time.time()
        rmses = []
        f = open('./logs/DFNMF{}_k-{}_lambda-{}.txt'.format(start_time, self.k, self.lmda), 'w')

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


if __name__ == '__main__':
    df = pd.read_csv('./data/train_ratings_df.csv')
    train = df.values
    df = pd.read_csv('./data/test_ratings_df.csv')
    test = df.values

    rsnmf = RSNMFDF()
    rmses = rsnmf.fit(train, test)
