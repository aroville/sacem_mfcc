from pymongo import MongoClient
from pickle import loads
from multiprocessing import Pool, Value

DB = 'db_sacem_dev'
COL = 'yt_training_set_mfcc'
LBL = 'label'
FT = 'features'
PROJ_FT = dict(projection={FT: 1, '_id': 0})
PROJ_LBL = dict(projection={LBL: 1, '_id': 0})
counter = Value('i', 0)
total = 0


def __debinarize(ft_bin):
    global counter
    with counter.get_lock():
        counter.value += 100
    print('processed {:.2f}%'.format(counter.value/total), end='\r')
    return loads(ft_bin['features'])


def get_data():
    with MongoClient() as c:
        y = [d[LBL] for d in c[DB][COL].find(**PROJ_LBL)]
        global total
        total = len(y)
        with Pool() as p:
            x = p.map(__debinarize, c[DB][COL].find(**PROJ_FT))
        print()

    #
    # with Pool() as p:
    #     x = p.map(__debinarize, x_bin[:10])
    print(len(x))
    print(len(y))
