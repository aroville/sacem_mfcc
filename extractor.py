import constants as c
from utils import CustomCounter
import librosa as lb
from os import listdir
from os.path import join
from numpy.random import choice
from multiprocessing import Pool
from pymongo import MongoClient
from collections import defaultdict, Counter
from itertools import repeat

_db = MongoClient()[c.DB]
_counter = CustomCounter({'done': 0, 'errors': 0})
_noises = defaultdict(list)


def __filter_processed_ids(files):
    processed_ids = Counter(list(_db[c.COL_FT].distinct(c.YT_ID)))
    print('already processed:', len(processed_ids))
    files = Counter(list(map(lambda f: f.replace('.mp3', ''), files)))
    return list((files - processed_ids).elements())


def extract_features_from_yt_audioset():
    q = {c.YT_ID: {'$in': __filter_processed_ids(listdir(c.PATH_YT_AUDIO))}}
    args = _db[c.COL_AUDIOSET].find(q, {c.YT_ID: 1, c.LBL: 1}).limit(c.LIMIT)
    Pool().starmap(__extract_one, zip(args, repeat(c.PATH_YT_AUDIO)))
    print('\n')


def extract_features_from_augmented_audioset():
    for cat in listdir(c.PATH_NOISES):
        for noise_id in listdir(join(c.PATH_NOISES, cat)):
            _noises[cat].append(noise_id.replace('.mp3', ''))
    args = [{c.YT_ID: f, c.LBL: 'music'}
            for f in __filter_processed_ids(listdir(c.PATH_Q3))[:c.LIMIT]]
    Pool().starmap(__extract_one, zip(args, repeat(c.PATH_Q3), repeat(True)))
    print('\n')


def __load_audio(audio_path, yt_id, noise_cat=None, noise_id=None):
    mp3_path = join(audio_path, yt_id + '.mp3')
    try:
        y_music, _ = lb.load(mp3_path, sr=c.EXTRACT_ARGS['sr'])
        _counter.update('done')
    except (EOFError, FileNotFoundError):
        _counter.update('errors')
        return None
    if noise_id is None:
        return y_music

    noise_path = join(c.PATH_NOISES, noise_cat, noise_id + '.mp3')
    y_noise, _ = lb.load(noise_path, sr=c.EXTRACT_ARGS['sr'])
    n = min(y_music.shape[0], y_noise.shape[0])
    return 0.35 * y_music[:n] + 0.65 * y_noise[:n]


def __extract_one(args, audio_path, add_noise=False):
    for noise_cat, noise_id in rand_noises() if add_noise else [(None, None)]:
        y = __load_audio(audio_path, args[c.YT_ID], noise_cat, noise_id)
        if y is not None:
            ft = c.FT_FUNC(y=y, **c.EXTRACT_ARGS).T.tolist()
            __to_mongo(args[c.YT_ID], args[c.LBL] == 'music', ft)


def rand_noises():
    return [(cat, choice(_noises[cat])) for cat in _noises]


def __to_mongo(yt_id, lbl, ft):
    with MongoClient() as cli:
        cli[c.DB][c.COL_FT].insert_one({c.YT_ID: yt_id, c.LBL: lbl, c.FT: ft})
