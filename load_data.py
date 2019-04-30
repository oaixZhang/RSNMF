import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_line(file, separator='::'):
    for line in file:
        yield line.strip().split(separator)


def load_ratings(file='./data/train.txt'):
    ratings = list()
    with open(file, 'r') as f:
        for line in read_line(f):
            user_id, movie_id, user_movie_rating = line
            ratings.append((int(user_id), int(movie_id), float(user_movie_rating)))
    return np.asarray(ratings, dtype=np.int)


def load_users(file='./data/users.dat'):
    users = list()
    with open(file, 'r') as f:
        for line in read_line(f):
            user_id, gender, age, occupation, zcode = line
            users.append((int(user_id), int(gender == 'M'), int(age), int(occupation)))
    return pd.DataFrame(users, columns=['user_id', 'gender', 'age', 'occupation'])


def load_movies_from_dat(file='./data/movies.dat'):
    movies = list()
    with open(file, 'r', encoding='ISO-8859-1') as f:
        for line in read_line(f):
            movie_id, title, genres = line
            genres = genres.split('|')
            genres = pd.Series(genres)
            genres = genres.map({'Action': 0, 'Adventure': 1, 'Animation': 2, "Children's": 3, 'Comedy': 4, 'Crime': 5,
                                 'Documentary': 6, 'Drama': 7, 'Fantasy': 8, 'Film-Noir': 9, 'Horror': 10,
                                 'Musical': 11, 'Mystery': 12, 'Romance': 13, 'Sci-Fi': 14, 'Thriller': 15, 'War': 16,
                                 'Western': 17})
            genres = genres.tolist()
            genre = np.zeros([18], dtype=int)
            for g in genres:
                genre[g] = 1
            temp = np.concatenate([[int(movie_id)], genre])
            movies.append(temp)
            # movies['genres'] = movies['genres'].
    movies = pd.DataFrame(movies,
                          columns=['movie_id', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy',
                                   'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
                                   'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'])
    movies.to_csv('./data/movies.csv', index=0)
    return movies


def load_movies():
    movies = pd.read_csv('./data/movies.csv')
    return movies


def convert2matrix():
    users = load_users()
    print(users.shape)
    movies = pd.read_csv('./data/movies.csv')
    print(movies.shape)

    ratings_train = load_ratings('./data/train.txt')
    print('train.txt shape:', ratings_train.shape)
    traindata = list()
    error_index = []
    index = 0
    for rating in ratings_train:
        user_id, movie_id, user_movie_rating = rating
        try:
            u = users[users.user_id == user_id].values[0]
            u = u[1:]
            # print(u)
            m = movies[movies.movie_id == movie_id].values[0]
            m = m[1:]
            # print(m)
            temp = np.concatenate([u, m])
            temp = np.concatenate([temp, [user_movie_rating]])
            # print(temp)
            traindata.append(temp)
            index += 1
        except IndexError:
            error_index.append(index)
            index += 1
            continue
    traindata = np.asarray(traindata)
    df = pd.DataFrame(traindata)
    df.to_csv('./data/train_df.csv', index=0)
    print('df traindata.shape =', df.shape)
    print('error count:', len(error_index))
    ratings_train = np.delete(ratings_train, error_index, axis=0)
    df_ratings = pd.DataFrame(ratings_train)
    print('ratings.shape =', df_ratings.shape)
    df_ratings.to_csv('./data/train_ratings_df.csv', index=0)

    # test data
    ratings_test = load_ratings('./data/test.txt')
    print('test.txt shape:', ratings_test.shape)
    testdata = list()
    error_index = []
    index = 0
    for rating in ratings_test:
        user_id, movie_id, user_movie_rating = rating
        try:
            u = users[users.user_id == user_id].values[0]
            u = u[1:]
            # print(u)
            m = movies[movies.movie_id == movie_id].values[0]
            m = m[1:]
            # print(m)
            temp = np.concatenate([u, m])
            temp = np.concatenate([temp, [user_movie_rating]])
            # print(temp)
            testdata.append(temp)
            index += 1
        except IndexError:
            error_index.append(index)
            index += 1
            continue
    testdata = np.asarray(testdata)
    df = pd.DataFrame(testdata)
    df.to_csv('./data/test_df.csv', index=0)
    print('df testdata.shape =', df.shape)
    print('error count:', len(error_index))
    ratings_test = np.delete(ratings_test, error_index, axis=0)
    df_ratings = pd.DataFrame(ratings_test)
    print('ratings.shape =', df_ratings.shape)
    df_ratings.to_csv('./data/test_ratings_df.csv', index=0)


def plot(file='./logs/1555943194.652819_k-20_lambda-0.06.txt'):
    rmses = list()
    with open(file, 'r') as f:
        for line in f:
            rmse = line.strip('\n')
            rmses.append(rmse)

    plt.plot(np.asarray(rmses))
    plt.show()


def concat_u_i(users, movies, user_id, movie_id):
    u = users[users.user_id == user_id].values[0]
    u = u[1:]
    # print(u)
    m = movies[movies.movie_id == movie_id].values[0]
    m = m[1:]
    # print(m)
    temp = np.concatenate([u, m])
    # print(temp)
    return temp.reshape(1, -1)


if __name__ == "__main__":
    # ratings = load_ratings()
    # users = load_users()
    # load_movies()
    # movies = pd.read_csv('./movies.csv')
    # convert2matrix()
    # print(ratings.shape)
    # convert2matrix()
    # plot()

    users = load_users()
    movies = pd.read_csv('./data/movies.csv')
    a = concat_u_i(users, movies, 1, 2)
    print(a.shape)
    print(a)

