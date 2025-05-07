import sklearn.ensemble
import sklearn.model_selection
import sklearn.pipeline
import sklearn.svm
import chord_classifier
import sklearn
import numpy as np
import time
import soundfile as sf
from tqdm import tqdm
from collections import Counter

def pipe_gradient_bosting_classifier():
    pipe = sklearn.pipeline.Pipeline([('scaler', sklearn.preprocessing.MinMaxScaler()),
                                      ('pca', sklearn.decomposition.PCA()),
                                      ('classifier', sklearn.ensemble.GradientBoostingClassifier(random_state=0))])
    params = {
        'pca__n_components' : [0.95, 0.98],
        'classifier__n_estimators' : [10, 50, 100],
        'classifier__max_depth' : [None, 3],
    }
    return sklearn.model_selection.GridSearchCV(estimator=pipe,  
                                                param_grid=params,
                                                cv=5,
                                                verbose=1,
                                                n_jobs=-1)
    
def pipe_svc():
    pipe = sklearn.pipeline.Pipeline([('scaler', sklearn.preprocessing.StandardScaler()),
                                      ('classifier', sklearn.svm.SVC(random_state=0, probability=True))])
    params = {
        'classifier__C' : [0.5, 1, 2]
    }
    return sklearn.model_selection.GridSearchCV(estimator=pipe,  
                                                param_grid=params,
                                                cv=5,
                                                verbose=1,
                                                n_jobs=-1)   

def pipe_random_forrest_classifier():
    pipe = sklearn.pipeline.Pipeline([('feature_reduction', sklearn.feature_selection.SelectPercentile()),
                                      ('scaler', sklearn.preprocessing.MinMaxScaler()),
                                      ('pca', sklearn.decomposition.PCA()),
                                      ('classifier', sklearn.ensemble.RandomForestClassifier(random_state=0))])
    
    params = {
        'pca__n_components' : [0.88, 0.90, 0.92, 0.95], #[0.95],#
        'classifier__n_estimators' : [100, 150, 200],
        'classifier__max_depth' : [None, 3],# 5],
        'classifier__min_samples_split' : [2, 4, 0.2],
        'classifier__max_features' : ['sqrt', 'log2', None]
        }
    
    return sklearn.model_selection.GridSearchCV(estimator=pipe,
                                                param_grid=params,
                                                cv=5,
                                                scoring='accuracy',
                                                verbose=1,
                                                n_jobs=-1)

def pipe_decision_tree_classifier():
    pipe = sklearn.pipeline.Pipeline([('feature_reduction', sklearn.feature_selection.SelectPercentile()),
                                      ('scaler', sklearn.preprocessing.StandardScaler()),
                                      #('pca', sklearn.decomposition.PCA()),
                                      ('classifier', sklearn.tree.DecisionTreeClassifier(random_state=0))])
    
    params = {
        #'pca__n_components' : [0.88, 0.90, 0.92, 0.95], #[0.95],#
        'classifier__max_depth' : [None, 3, 5, 10, 16],
        'classifier__min_samples_split' : [2, 4, 0.2],
        'classifier__max_features' : ['sqrt', 'log2']
        }
    
    return sklearn.model_selection.GridSearchCV(estimator=pipe,
                                                param_grid=params,
                                                cv=5,
                                                scoring='accuracy',
                                                verbose=1,
                                                n_jobs=-1)


def get_data():
    df = chord_classifier.get_signal_data()
    df = chord_classifier.apply_window(df)
    df_freq_amp = chord_classifier.get_freq_amp_data(df, 44100)
    df_freq_amp = df_freq_amp.reset_index(level=('chord'))
    print(df_freq_amp)
    X = df_freq_amp.loc[:, df_freq_amp.columns != 'chord']
    y = df_freq_amp['chord']
    return X, y

def split_data(X, y):
    return sklearn.model_selection.train_test_split(X, y, test_size=0.25)

def get_model(model: str):
    models = {'gradBoostClf' : pipe_gradient_bosting_classifier,
              'svc' : pipe_svc,
              'randomForrestClf': pipe_random_forrest_classifier,
              'decisionTreeClf' : pipe_decision_tree_classifier}
    return models[model]()

def get_models(models:list[str]) -> list[sklearn.model_selection.GridSearchCV]:
    for model in models:
        model = get_model(model)
    return models

def fit_model(X_train, y_train, model: sklearn.model_selection.GridSearchCV):
    return model.fit(X_train, y_train)
    
def fit_models(X_train, y_train, models: list[sklearn.model_selection.GridSearchCV]):
    for model in models:
        model.append(fit_model(X_train,y_train, model))
    return models


def score_model(X_test, y_test, model_fitted: sklearn.model_selection.GridSearchCV):
    model = model_fitted.best_estimator_
    proba = model.predict_proba(X_test)
    print(model.score(X_test, y_test))
    for i in range(len(proba)):
        print(y_test.iloc[i], proba[i])
    print(model_fitted.best_params_)

def score_models(X_train, X_test, y_train, y_test, models):
    start = time.time()
    for model in models:
        model = get_model(model=model)
        model = fit_model(X_train, y_train, model)
        score_model(X_test, y_test, model)
        print(f'finished in {time.time()- start}s')

def get_clf() -> sklearn.base.BaseEstimator:
    X, y = get_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = get_model('decisionTreeClf')
    model = fit_model(X_train, y_train, model)
    return model.best_estimator_

def classify_amplitudes(amplitudes, clf):
    amps = amplitudes.reshape(1,-1)
    #print(clf.predict(amps), clf.predict_proba(amps))
    return clf.predict(amps)

def score_rolling_array():
    c = Counter()
    clf = get_clf()
    rec, _ = sf.read(r'2025-2_3-audio\chords\acde.wav')
    data = np.zeros((clf.steps[0][1].n_features_in_*2 -1, # *2 due to clf taking rfft result -1 to make it odd and get right number samples
                     rec.shape[1])) # number of channels
    print(data.shape, data[0])
    old_res = ''
    for sample in tqdm(rec): #rec:#
        data = np.roll(data, -1)
        data[-1] = sample
        for channel in data.copy().T:
            #print(channel.shape, np.max(channel))
            channel = np.multiply(channel, np.hamming(channel.shape[0]))
            if np.max(channel) >= 0.05:
                amplitudes = np.abs(np.fft.rfft(channel))
                res = classify_amplitudes(amplitudes, clf)
                try:
                    c[res[0]] += 1
                except Exception as E:
                    print(f'Exception {E}, from {res}')
                    print(c)
                if old_res != res:
                    old_res = res
                    print(res)

    print(c)

def main():
    X, y = get_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    models = [#'gradBoostClf', 
              #'svc', 
              #'randomForrestClf',
              'decisionTreeClf']
    models = fit_models(models)
    score_models(X_train, X_test, y_train, y_test, models)

if __name__ == '__main__':
    #main()
    score_rolling_array()