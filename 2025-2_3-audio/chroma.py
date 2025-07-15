import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.ensemble
import sklearn.model_selection
import sklearn.multiclass
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.tree
from chord_classifier import get_signal_data
import joblib

def get_notes() -> list[str]:
    '''Returns a list of notes in the chromatic scale'''
    return ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def sample_chroma(sample: pd.Series):
    '''Calculates the mean chroma value accross an entire time series'''
    chromas = librosa.feature.chroma_stft(y= sample.to_numpy(), sr=44100)
    return pd.Series(np.mean(chromas, axis = 1), index = get_notes())

def df_chroma(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(sample_chroma, axis=1)

def plot_chord_chromas(chroma_df: pd.DataFrame):
    ax = chroma_df.groupby(level='chord').mean().T.plot(layout='tight')
    ax.set_xticks([i for i in range(len(get_notes()))])
    ax.set_xticklabels(get_notes())
    plt.show(block=False)
    
def chroma_clf(chroma_df: pd.DataFrame):
    chroma_df = chroma_df.reset_index(level=('chord'))
    X = chroma_df.loc[:, chroma_df.columns != 'chord']
    y = chroma_df['chord']
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X,y, test_size=0.33,random_state=0)
    clf = sklearn.ensemble.RandomForestClassifier(random_state=0)
    param_grid = {'n_estimators' : [5, 12, 20, 100]}
    grid_search = sklearn.model_selection.GridSearchCV(estimator=clf,
                                                       param_grid=param_grid,
                                                       cv=5,
                                                       scoring='accuracy',
                                                       verbose=1,
                                                       n_jobs=-1)
    
    model = grid_search.fit(X_train, y_train)

    score_model(model.best_estimator_, X_test, y_test)
    return model.best_estimator_

def chroma_pipe(signal_data: pd.DataFrame):
    signal_data = signal_data.reset_index(level=('chord'))
    X = signal_data.loc[:, signal_data.columns != 'chord']
    y = signal_data['chord']
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X,y, test_size=0.33,random_state=0)
    pipe = sklearn.pipeline.Pipeline([
        ('chroma_transformer', sklearn.preprocessing.FunctionTransformer(func=df_chroma,
                                                                         validate=False)),
        ('ovr_dtree_clf', sklearn.multiclass.OneVsRestClassifier(estimator=sklearn.tree.DecisionTreeClassifier(random_state=0)))
    ])
    param_grid = {'ovr_dtree_clf__estimator__criterion':         ['gini', 'entropy', 'log_loss'],
                  'ovr_dtree_clf__estimator__max_depth':         [None, 4, 6],
                  'ovr_dtree_clf__estimator__min_samples_split': [2, 3, 6]}
    grid_search = sklearn.model_selection.GridSearchCV(estimator=pipe,
                                                       param_grid=param_grid,
                                                       cv=5,
                                                       scoring='accuracy',
                                                       verbose=1,
                                                       n_jobs=-1)
    model = grid_search.fit(X_train, y_train)

    score_model(model.best_estimator_, X_test, y_test)
    return model.best_estimator_

def score_model(clf: sklearn.model_selection.GridSearchCV, X_test, y_test):
    proba = clf.predict_proba(X_test)
    print(clf.score(X_test, y_test))
    for i in range(len(proba)):
        print(y_test.iloc[i], proba[i])

def get_chroma_clf():
    df = get_signal_data()
    chroma_df = df.apply(sample_chroma, axis=1)

    clf = chroma_clf(chroma_df)
    return df.shape[-1], clf

def save_model():
    window_size, clf = get_chroma_clf()
    joblib.dump((window_size, clf), r".\2025-2_3-audio\models\chroma.joblib")

def get_chroma_pipe_clf():
    df = get_signal_data()
    clf = chroma_pipe(df)
    return clf

def save_pipe_model():
    clf = get_chroma_pipe_clf()
    joblib.dump(clf, r".\2025-2_3-audio\models\chroma_pipe.joblib")

def load_model():
    ''' Returns a touple of original window size and model in shape (window_size, model) '''
    return joblib.load(r".\2025-2_3-audio\models\chroma.joblib")

def main():
    # df = get_signal_data()
    # chroma_df = df.apply(sample_chroma, axis=1)
    # print(chroma_df)
    # print(chroma_df.groupby(level='chord').mean())
    # plot_chord_chromas(chroma_df)
    # clf = chroma_clf(chroma_df)
    # print(clf)
    #save_model()
    #print(load_model())
    save_pipe_model()



if __name__ == '__main__':
    main()

