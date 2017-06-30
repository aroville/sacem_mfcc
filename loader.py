import constants as C
from pymongo import MongoClient
from pickle import loads
from multiprocessing import Pool, Value, Manager

PROJ = dict(projection={C.LBL: 1, C.FT: 1, '_id': 0})
counter = Value('i', 0)
total = 0
all_data = Manager().list()


def __debinarize(row):
    global counter
    with counter.get_lock():
        counter.value += 1
    print('processed {:.2f}%'.format(100*counter.value/total), end='\r')
    label = row[C.LBL]

    for ft in loads(row[C.FT]):
        all_data.append((ft, label))


def get_data():
    with MongoClient() as c:
        global total
        total = c[C.DB][C.COL].count()
        with Pool() as p:
            p.map(__debinarize, c[C.DB][C.COL].find(**PROJ))
            print()
    return list(zip(*all_data))
