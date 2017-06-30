import constants as c
from utils import CustomCounter
from pymongo import MongoClient
from multiprocessing import Pool, Manager

_counter = CustomCounter({'done': MongoClient()[c.DB][c.COL_FT].count()})
all_data = Manager().list()


def __unwrap(row):
    all_data.extend([(ft, row[c.LBL]) for ft in row[c.FT]])
    _counter.update('done')


def get_data():
    with MongoClient() as cli:
        col = cli[c.DB][c.COL_FT]
        with Pool() as p:
            p.map(__unwrap, col.find(projection={c.LBL: 1, c.FT: 1}))
            print()
        cli.close()
    return list(zip(*all_data))
