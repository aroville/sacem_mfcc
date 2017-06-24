from monary import Monary, MonaryParam
import librosa as lb
from os import listdir
from os.path import join, isfile
import numpy as np
from pickle import dumps
from numpy.ma import masked_array
# from bson import Binary
from multiprocessing import Pool, Value
from time import time
from pymongo import MongoClient

DB = 'db_sacem_dev'
SOURCE_COL = 'yt_audioset'
DEST_COL = 'yt_training_set_mfcc'
PROJECTION = {'_id': 0, 'youtube_id': 1, 'label': 1}
DATA_PATH = '/home/axel/sacem/data'
AUDIO_PATH = join(DATA_PATH, 'youtube_audioset', 'mp3')

AUGMENTATION_PATH = join(DATA_PATH, 'augmentation')
TOP_Q3 = join(AUGMENTATION_PATH, 'top_q3')
NOISES = join(AUGMENTATION_PATH, 'noises')
augmentation_noises = {'bar': [], 'beach': [], 'car': [], 'city': [], 'concert': [], 'road': []}

params = dict(sr=22050, n_fft=880, hop_length=870, n_mfcc=30)
errors = Value('i', 0)
processed = Value('i', 0)

def extract_mfccs():
    extract_mfccs_for_youtube_audioset()


def extract_mfccs_for_youtube_audioset():
    db = MongoClient()[DB]

    files = [f for f in listdir(AUDIO_PATH) if f.endswith('.mp3')]
    print('found %d files' % len(files))

    already_processed_ids = list(db[DEST_COL].distinct('yt_id'))
    print('already processed: %d' % len(already_processed_ids))
    files = [f for f in files if f.replace('.mp3', '') not in already_processed_ids]
    query = {'youtube_id': {'$in': [f.replace('.mp3', '') for f in files]}}

    projection = {'_id': 0, 'youtube_id': 1, 'label': 1}
    ids_labels = list(db[SOURCE_COL].find(query, projection=projection))
    print(len(ids_labels))

    with Pool(2) as p:
        mfccs_bin = p.map(extract_mfcc_from_yt_audioset, ids_labels[:50])
    print(mfccs_bin)


def load_audio(audio_path, yt_id, noise=None):
    mp3_path = join(audio_path, yt_id + '.mp3')

    sr = params['sr']
    y_music = None
    try:
        y_music, _ = lb.load(mp3_path, sr=sr)
        global processed
        with processed.get_lock():
            processed.value += 1
    except:
        print(mp3_path)
        with errors.get_lock():
            errors.value += 1

    # print('processed: {}\terrors: {}'.format(processed.value, errors.value))
    if y_music is None:
        return None

    if noise is None:
        return y_music

    noise_path = join(NOISES, noise + '.mp3')
    y_noise, _ = lb.load(noise_path, sr=sr)
    n = min(y_music.shape[0], y_noise.shape[0])
    return 0.35 * y_music[:n] + 0.65 * y_noise[:n]


def extract_mfcc_from_yt_audioset(id_label):
    try:
        yt_id = id_label['youtube_id']
        y = load_audio(AUDIO_PATH, yt_id)
        # if y is None:
        #     pass
        # return dumps(lb.feature.mfcc(y=y, **params))
    except:
        return None


# class FeaturesBuilder:
#     def __init__(self, processing_params):
#         # load youtube file names and labels
#         self.processing_params = processing_params
#         self._all_files = self.__load_file_names(AUDIO_PATH)
#         self._labels = self.__load_labels()
#
#     @staticmethod
#     def __load_file_names(path):
#         files = [f.replace('.mp3', '') for f in listdir(path)
#                  if f.endswith('.mp3')]
#         np.random.shuffle(files)
#         return files
#
#     def __load_labels(self):
#         query = {'youtube_id': {'$in': self._all_files}}
#         projection = {'_id': 0, 'youtube_id': 1, 'label': 1}
#         with MongoClient as c:
#             mg_labels = c[DB][SOURCE_COL].find(query, projection=projection)
#         return {l['youtube_id']: 1 if l['label'] == 'music' else 0 for l in mg_labels}
#
#     def __load_augmented_noises(self, augmentation_noise_path):
#         for noise_type in augmentation_noises:
#             path = join(augmentation_noise_path, noise_type)
#             files = self.__load_file_names(path)
#             augmentation_noises[noise_type] = files
#         return augmentation_noises
#
#     def singlefeature_tomongo(self, yt_id):
#         extractor = MFCCFeatureExtractor(AUDIO_PATH, yt_id, self.processing_params)
#         extractor.process()
#         extractor.to_mongo(db=DB, collection=DEST_COL, label=self._labels[yt_id])
#
#     def build_mfcc_features_to_mongo(self, n=-1):
#         t0 = time()
#         with Pool() as p:
#             p.map(self.singlefeature_tomongo, self._all_files[:n])
#         print('\nDONE extracting -- %ds' % int(time()-t0))
#
#
# class MFCCFeatureExtractor:
#     def __init__(self, audio_path, yt_id, processing_params):
#         self._audio_path = audio_path
#         self._yt_id = yt_id
#         self._processing_params = processing_params
#         self._features = None
#         self._feature_name = 'mfcc'
#
#     def load_audio(self):
#         mp3_path = join(self._audio_path, self._yt_id + '.mp3')
#         sr = self._processing_params['sr']
#         y, _ = lb.load(mp3_path, sr=sr)
#         return y
#
#     def process(self):
#         y = self.load_audio()
#         self._features = lb.feature.mfcc(y=y, **self._processing_params).T
#         del y
#
#     def to_mongo(self, db, collection, label):
#         self._features = self._features.astype('float16')
#         n = self._features.shape[0]
#
#         bin_feats = [Binary(dumps(f, protocol=2)) for f in self._features]
#         bin_len = max([len(f) for f in bin_feats])
#
#         z_n = np.zeros(n)
#         z_n_b = np.zeros(n, dtype=np.bool)
#
#         insert_params = MonaryParam.from_lists(
#             [masked_array([self._yt_id]*n, z_n, 'S%s' % len(self._yt_id)),
#              masked_array(bin_feats, z_n, '<V%s' % bin_len),
#              masked_array([np.bool(label)]*n, z_n_b, 'bool')],
#             ['yt_id', 'features', 'label'],
#             ['string:%s' % len(self._yt_id),
#              'binary:%s' % bin_len, 'bool'])
#
#         with Monary() as client:
#             client.insert(db, collection, insert_params)
