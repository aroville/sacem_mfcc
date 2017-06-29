import librosa as lb
from os import listdir
from os.path import join
from numpy.random import choice
from pickle import dumps
from bson import Binary
from multiprocessing import Pool, Value
from pymongo import MongoClient
from collections import defaultdict

DB = 'db_sacem_dev'
COL_SRC = 'yt_audioset'
COL_DST = 'yt_training_set_mfcc'
PROJ = {'_id': 0, 'youtube_id': 1, 'label': 1}

DATA = '/home/axel/sacem/data'
YT_AUDIO = join(DATA, 'youtube_audioset', 'mp3')
AUGMENTED = join(DATA, 'augmentation')
TOP_Q3 = join(AUGMENTED, 'top_q3')
NOISES = join(AUGMENTED, 'noises')

_db = MongoClient()[DB]
_params = dict(sr=22050, n_fft=880, hop_length=870, n_mfcc=30)
_errors = Value('i', 0)
_processed = Value('i', 0)
_noise_categories = []
_all_noises = defaultdict(list)


def __filter_processed_ids(files):
    processed_ids = list(_db[COL_DST].distinct('yt_id'))
    print('already processed: %d' % len(processed_ids))
    files = list(map(lambda f: f.replace('.mp3', ''), files))
    return [f for f in files if f not in processed_ids]


def extract_and_save_mfccs_from_yt_audioset():
    files = __filter_processed_ids(listdir(YT_AUDIO))
    query = {'youtube_id': {'$in': files}}
    args = list(_db[COL_SRC].find(query, projection=PROJ))
    with Pool() as p:
        p.map(__extract_mfcc_from_audio, args)


def extract_and_save_mfccs_from_augmented_audioset():
    __get_all_noises()
    files = __filter_processed_ids(listdir(TOP_Q3))
    args = [{'youtube_id': f, 'label': 'music', 'add_noise': True}
            for f in files]
    with Pool() as p:
        p.map(__extract_mfcc_from_audio, args)


def __load_audio(audio_path, yt_id, noise_cat=None, noise_id=None):
    mp3_path = join(audio_path, yt_id + '.mp3')
    try:
        y_music, _ = lb.load(mp3_path, sr=_params['sr'])
        with _processed.get_lock():
            _processed.value += 1
    except EOFError:
        with _errors.get_lock():
            _errors.value += 1
        return None
    print('processed: {:6}\terrors: {:6}'
          .format(_processed.value, _errors.value), end='\r')

    if noise_id is None:
        return y_music

    noise_path = join(NOISES, noise_cat, noise_id + '.mp3')
    y_noise, _ = lb.load(noise_path, sr=_params['sr'])
    n = min(y_music.shape[0], y_noise.shape[0])
    return 0.35 * y_music[:n] + 0.65 * y_noise[:n]


def __extract_mfcc_from_audio(args):
    yt_id = args['youtube_id']
    label = args['label'] == 'music'
    noises = []
    if args.get('add_noise'):
        audio_path = TOP_Q3
        noises.extend([(cat, choice(_all_noises[cat]))
                       for cat in _noise_categories])
    else:
        noises = [(None, None)]
        audio_path = YT_AUDIO

    for noise_cat, noise_id in noises:
        y = __load_audio(audio_path, yt_id, noise_cat, noise_id)
        if y is None:
            return
        mfccs = Binary(dumps(lb.feature.mfcc(y=y, **_params)))
        __to_mongo(yt_id, label, mfccs)


def __to_mongo(yt_id, label, mfccs):
    with MongoClient() as client:
        client[DB][COL_DST].insert_one({
            'yt_id': yt_id,
            'label': label,
            'features': mfccs
        })


def __get_all_noises():
    global _noise_categories
    _noise_categories = listdir(NOISES)
    for cat in _noise_categories:
        for noise_id in listdir(join(NOISES, cat)):
            _all_noises[cat].append(noise_id.replace('.mp3', ''))
