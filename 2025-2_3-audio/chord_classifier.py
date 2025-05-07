import os
import time
from glob import glob
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.decomposition
import sklearn.ensemble
import sklearn.feature_selection
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing
import soundfile as sf
import sklearn
import joblib

def get_chord_path_dict(dir_path, extention: str = '.wav'):
    ''' Returns a dictionary with keys being chord names, and values being paths to sample files'''
    path_dict = dict()
    chord_dirs = glob(pathname=dir_path)
    #print(chord_dirs)
    for dir in chord_dirs:
        path_dict[dir.split('\\')[-1]] = glob(dir + r'\*' + extention)
    return path_dict
    
def load_samples(chord_path_dict):
    ''' Returns the data from the sample files in the form of a dictionary of dictionaries, being keys chord names, sample names, and value being loaded data '''
    chord_sample_data = dict()
    for chord in chord_path_dict:
        sample_data = dict()
        for path in chord_path_dict[chord]:
            data, _ = sf.read(path)
            sample_data[path.split('\\')[-1].rstrip('.wav')] = data.T
        chord_sample_data[chord] = sample_data
    return chord_sample_data
            
def make_signal_df(category_sample_data_dict:dict):
    ''' Returns a dataframe with multiindex containing sample data, dropping missing value columns'''
    df_data = []
    sample_index = []
    channel_index = []
    chord_index = []
    for category, sample_data in category_sample_data_dict.items():
        for sample, data in sample_data.items():
            #print(sample, data.shape)
            for channel, channel_data in enumerate(data):
                sample_index.append(sample)
                channel_index.append(channel)
                chord_index.append(category)
                df_data.append(channel_data)

    index = pd.MultiIndex.from_arrays([chord_index, sample_index, channel_index], names=('chord', 'sample', 'channel'))
    df = pd.DataFrame(data=df_data, index=index)
    df = df.dropna(axis='columns')
    return df

def get_signal_data():
    '''Combines the above three functions to get a dataframe'''
    chord_path_dict = get_chord_path_dict(os.path.abspath(r"2025-2_3-audio\samples\*"))
    chord_sample_data = load_samples(chord_path_dict)
    df = make_signal_df(chord_sample_data)
    return df

def get_freq_amp_data(df, freq_columns=False):
    df_data = []
    chord_index = []
    sample_index = []
    channel_index = []
    for index, row in df.iterrows():
        #print(index)
        df_data.append(np.abs(np.fft.rfft(row)))
        chord, sample, channel = index
        chord_index.append(chord)
        sample_index.append(sample)
        channel_index.append(channel)
    index = pd.MultiIndex.from_arrays([chord_index, sample_index, channel_index], names=('chord', 'sample', 'channel'))
    cols = None
    if freq_columns:
        cols = np.fft.rfftfreq(len(row), 1/freq_columns)
    df_amp = pd.DataFrame(data=df_data, index=index, columns=cols)
    return df_amp


def train_classifier_model(X_train, y_train):
    clf = sklearn.ensemble.RandomForestClassifier(n_estimators = 1000, max_depth = 4, min_samples_split=0.1)
    clf.fit(X_train, y_train)
    return clf

def get_chord_classifier():
    df = get_signal_data()
    df_freq_amp = get_freq_amp_data(df)
    df = df.reset_index(level=('chord'))
    df_freq_amp = df_freq_amp.reset_index(level=('chord'))
    # print(df)
    # print(df_freq_amp)
    X = df_freq_amp.loc[:, df_freq_amp.columns != 'chord']
    X = sklearn.preprocessing.minmax_scale(X)
    y = df_freq_amp['chord']
    #print(X, y)
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X,y, test_size=0.33,random_state=0)
    # print (X_train, y_train)
    clf = train_classifier_model(X_train, y_train)
    proba_results = clf.predict_proba(X_test)
    # print(proba_results)
    # print(y_test)
    print(clf.score(X_test,y_test))
    for i in range(len(proba_results)):
        print(y_test.iloc[i], proba_results[i])

    return clf

def hamming_window(df):
    return np.hamming(df.shape[-1])

def apply_window(df):
    window_function = hamming_window(df)
    return df.mul(window_function, axis='columns') # multiply elementwise filter onto data columns

def chord_pipeline():
    df = get_signal_data()
    df = apply_window(df)
    df_freq_amp = get_freq_amp_data(df)
    df = df.reset_index(level=('chord'))
    df_freq_amp = df_freq_amp.reset_index(level=('chord'))

    pipe = sklearn.pipeline.Pipeline([('feature_reduction', sklearn.feature_selection.SelectPercentile()),
                                      ('scaler', sklearn.preprocessing.MinMaxScaler()),
                                      ('pca', sklearn.decomposition.PCA()),
                                      ('classifier', sklearn.ensemble.RandomForestClassifier(random_state=0))])
                                      #('svc', sklearn.svm.SVC(probability=True))])
    
    params = {
        'feature_reduction__score_func' : [sklearn.feature_selection.f_classif, sklearn.feature_selection.chi2, sklearn.feature_selection.mutual_info_classif],
        'feature_reduction__percentile' : [*list(range(8,16))],
        'pca__n_components' : [*list(np.linspace(0.9,0.98,9))],
        'classifier__n_estimators' : [10, 25, 50, 100, 200],
        'classifier__max_depth' : [None, 10, 30],
        'classifier__min_samples_split' : [2, 4, 0.2],
        'classifier__max_features' : ['sqrt', 'log2', None]
    }

    grid_search = sklearn.model_selection.GridSearchCV(estimator=pipe,
                                                       param_grid=params,
                                                       cv=5,
                                                       scoring='accuracy',
                                                       verbose=1,
                                                       n_jobs=-1)
    
    X = df_freq_amp.loc[:, df_freq_amp.columns != 'chord']
    X = sklearn.preprocessing.minmax_scale(X)
    y = df_freq_amp['chord']
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X,y, test_size=0.33,random_state=0)

    grid_search.fit(X_train, y_train)
    model = grid_search.best_estimator_

    # pipe.fit(X_train, y_train)
    proba_results = model.predict_proba(X_test)

    print(model.score(X_test,y_test))
    for i in range(len(proba_results)):
        print(y_test.iloc[i], proba_results[i])

    joblib.dump(model, 'pipe_selected.joblib')
    model_loaded = get_pipe_optimized()

    print(model.score(X_test,y_test))
    #print(model_loaded.score(X_test,y_test))
    proba_loaded = model_loaded.predict_proba(X_test)
    for i in range(len(proba_results)):
        print(y_test.iloc[i], proba_results[i], proba_loaded[i])

    return model

def get_pipe_optimized():
    return joblib.load(os.path.abspath('.\\pipe_optimized.joblib'))

def get_pipe_windowed():
    return joblib.load(os.path.abspath('.\\pipe_windowed.joblib'))

def get_pipe_selected():
    return joblib.load(os.path.abspath('.\\pipe_selected.joblib'))

def get_pipe_svc():
    return joblib.load(os.path.abspath('.\\pipe_svc.joblib'))

if __name__ == '__main__':
    sns.set_style("whitegrid")
    #get_chord_classifier()
    start = time.time()
    # try:
    #     chord_pipeline()
    # except Exception as E:
    #     print(E)
    pipe = get_pipe_selected()
    for param in pipe.get_params():
        print(param, pipe.get_params()[param])
    end = time.time()
    print(end-start)