import numpy as np


def read_line(file, separator='::'):
    for line in file:
        yield line.strip().split(separator)


def load_ratings(file='train.txt'):
    ratings = list()
    with open(file, 'r') as f:
        for line in read_line(f):
            user_id, movie_id, user_movie_rating = line
            ratings.append((int(user_id), int(movie_id), float(user_movie_rating)))
    return np.asarray(ratings, dtype=np.int)


if __name__ == "__main__":
    ratings = load_ratings()
    print(ratings.shape)
    print(np.unique(ratings[:, 0]))
    print(np.unique(ratings[:, 1]))
