import extractor, loader
from sklearn.ensemble import RandomForestClassifier
from os.path import join
import os
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from sklearn.base import clone
from time import time

extractor.extract_features_from_yt_audioset()
extractor.extract_features_from_augmented_audioset()
x, y = loader.get_data()

print(len(x))
print(len(x[0]))
print(len(y))

n_components = 8
print('Using PCA on dataset: keeping %s features' % n_components)
pca = PCA(n_components=n_components)
pca.fit(x, y)

x = pca.transform(x)
print('Explained variance ratio: %s' % pca.explained_variance_ratio_)
print('\n\n')

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.3)


fix_params = dict(
    min_samples_split=3,
    n_jobs=-1,
    max_features='log2',
    criterion='entropy',
    # class_weight={0: 1, 1: 3},
    verbose=1,
    max_depth=38
)

param_grid = ParameterGrid(dict(
    n_estimators=list(range(35, 46, 5))
))

max_score = 0
max_params = None
max_rf = None
for params in param_grid:
    t0 = time()
    rf = RandomForestClassifier(**fix_params)
    rf.set_params(**params)

    params_s = '__'.join('%s_%s' % (k, v) for k, v in params.items())
    print('params: ' + params_s)
    print('fitting', end='...')
    rf.fit(x_train, y_train)
    print('done, {:.2f}s'.format(time()-t0))

    t0 = time()
    print('scoring', end='...')
    score = rf.score(x_val, y_val)
    print('done, score={:.5f}, {:.2f}s\n\n'.format(score, time()-t0))

    if score > max_score:
        max_score = score
        max_params = params
        try:
            max_rf = clone(rf, safe=True)
        except:
            pass
    del rf

max_params_s = '__'.join('%s_%s' % (k, v) for k, v in max_params.items())
print('BEST ===> ' + max_params_s)
folder = join('saved_models', 'RandomForest__' + str(int(time())))
os.makedirs(folder)

print('fitting best model on all data', end='...')
max_rf.fit(x, y)
if max_rf is None:
    max_rf = RandomForestClassifier(**fix_params)
    max_rf.set_params(**max_params)
    print('fitting', end='...')
    max_rf.fit(x_train, y_train)
    print('done')
f_name = join(folder, max_params_s)+'__score_%d.pkl' % int(10000*max_score)
print('dumping to file %s' % f_name)
joblib.dump(max_rf, f_name)

