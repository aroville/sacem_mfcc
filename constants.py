from os.path import join
import librosa

# PATHS
DATA_ROOT = '/home/axel/sacem/data'
PATH_YT_AUDIO = join(DATA_ROOT, 'youtube_audioset', 'mp3')
AUGMENTED = join(DATA_ROOT, 'augmentation')
PATH_Q3 = join(AUGMENTED, 'top_q3')
PATH_NOISES = join(AUGMENTED, 'noises')


# MONGODB DATABASE
DB = 'db_sacem_dev'
COL_AUDIOSET = 'yt_audioset'
LBL = 'label'
FT = 'features'
YT_ID = 'youtube_id'


# EXTRACTION PARAMETERS
mode = 'mfcc'
EXTRACT_ARGS = dict(sr=22050, n_fft=880, hop_length=870)
if mode == 'mfcc':
    COL_FT = 'yt_training_set_mfcc'
    FT_FUNC = librosa.feature.mfcc
    EXTRACT_ARGS['n_mfcc'] = 30
elif mode == 'melspec':
    COL_FT = 'yt_training_set_melspec'
    FT_FUNC = librosa.feature.melspectrogram
