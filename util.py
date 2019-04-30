from numba import jit
import numpy as np


@jit(nopython=True)
def update_puk(max_userid, k, Iu, lmda, P, userdown, userup):
    for uid in range(1, max_userid + 1):
        for i in range(k):
            userdown[uid][i] += Iu[uid] * lmda * P[uid][i]
            if userdown[uid][i] != 0:
                P[uid][i] *= (userup[uid][i] / userdown[uid][i])


@jit(nopython=True)
def update_qik(max_itemid, k, Ui, lmda, Q, itemdown, itemup):
    for iid in range(1, max_itemid + 1):
        for i in range(k):
            itemdown[iid][i] += Ui[iid] * lmda * Q[iid][i]
            if itemdown[iid][i] != 0:
                Q[iid][i] *= (itemup[iid][i] / itemdown[iid][i])


@jit(nopython=True)
def calculate_rmse(test, P, Q):
    sum, count = 0, 0
    for i in range(len(test)):
        userid, itemid, rui = test[i]
        ruihat = P[userid].dot(Q[itemid].T)
        sum += (rui - ruihat) ** 2
        count += 1
    return np.sqrt(sum / count)


@jit(nopython=True)
def calculate_delta(ratings, P, Q, userup, userdown, itemup, itemdown):
    for i in range(len(ratings)):
        userid, itemid, rui = ratings[i]
        ruihat = P[userid].dot(Q[itemid].T)
        userup[userid] += Q[itemid] * rui
        userdown[userid] += Q[itemid] * ruihat
        itemup[itemid] += P[userid] * rui
        itemdown[itemid] += P[userid] * ruihat
