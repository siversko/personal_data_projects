import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA

class WinePCA():
    def __init__(self):
        self.data          = load_wine(as_frame=True)['frame']
        self.feature_names = load_wine()['feature_names']
        self.target_names  = load_wine()['target_names']
        self.X = self.data[self.feature_names]
        self.y = self.data.target

    def perform_PCA(self, scaleX = True):
        self.pca = PCA(n_components=0.95)
        X = self.X.to_numpy() # Copy of X limited to this function 
        if scaleX:
            X = scale(X)
        self.pca_scores = self.pca.fit_transform(X)

    def score_plot(self, ax, pcA=0, pcB=1):
        scatter = ax.scatter(self.pca_scores[: , pcA], 
                             self.pca_scores[: , pcB],
                             c = self.data.target)
        ax.set(xlabel= f"Principal Component {pcA+1}",
               ylabel= f"Principal Component {pcB+1}",
               )
        legend1 = plt.legend(scatter.legend_elements()[0], self.target_names,
                             title = 'Producer')

        ax.legend(scatter.legend_elements()[0], self.target_names, title = 'Producer')

        #ax.add_artist(legend1)
        ax.set_title('Score plot')
        return ax

    def plot_loadnings_scatter(self, ax, pcA=0, pcB=1, scale=1):
        pca_loadings = self.pca.components_ # rows = pc loadings, columns = feature weight
        ax.axhline(y = 0, ls=':', color = 'black')
        ax.axvline(x = 0, ls=':', color = 'black')
        for i, feature in enumerate(self.feature_names):
            x_offset = 0.03
            if pca_loadings[pcA][i] > 0:
                x_offset *= -1

            y_offset = -0.03
            if pca_loadings[pcB][i] > -0.01:
                y_offset *= -1

            scatter = ax.scatter(pca_loadings[pcA][i], pca_loadings[pcB][i], label=feature.replace('_', ' '))
            ax.annotate(feature.replace('_', ' '), (pca_loadings[pcA][i], pca_loadings[pcB][i]),
                                                   (pca_loadings[pcA][i]+x_offset, pca_loadings[pcB][i]+y_offset), 
                                                   arrowprops=dict(width=1,
                                                                   headwidth=1,
                                                                   shrink=0.1))
        #ax.legend(loc='center left', bbox_to_anchor=(1,0.5))
        ax.set(xlabel= f"Principal Component {pcA+1}",
               ylabel= f"Principal Component {pcB+1}",
               )
        ax.set_title('Loadings plot')
        return ax

    def loadings_plot(self, ax, pcA=0, pcB=1, scale=1):

        ''' Produces loadings plot
        numberical values has no significance, only relative positions
        '''
        pca_loadings = self.pca.components_ # rows = pc loadings, columns = feature weight
        pca_loadings *= scale
        for i, feature in enumerate(self.feature_names):
            x_offset = 0.3 
            if pca_loadings[pcA][i] > 0:
                x_offset *= 0

            y_offset = 0.15
            if pca_loadings[pcB][i] > -0.5:
                y_offset *= -1
            ax.arrow(0,0, pca_loadings[pcA][i], pca_loadings[pcB][i], head_width=0.05, alpha=0.5, color = 'red')
            ax.annotate(feature.replace('_', '\n'), (pca_loadings[pcA][i], pca_loadings[pcB][i]),
                                                   (pca_loadings[pcA][i]-x_offset, pca_loadings[pcB][i]-y_offset))

    
    def find_scaling(self, pc):
        score = max(self.pca_scores[: , pc], key=abs)
        loading = max(self.pca.components_[pc], key=abs)
        return abs(score)/abs(loading)

    def biplot(self, ax, pcA=0, pcB=1, scale=0.0):
        self.score_plot(ax, pcA, pcB)
        if not scale:
            scale = max(self.find_scaling(pcA), self.find_scaling(pcB))
        self.loadings_plot(ax, pcA, pcB, scale=scale)
        ax.axhline(y = 0, ls=':', color = 'black')
        ax.axvline(x = 0, ls=':', color = 'black')
        ax.set(xlabel= f"Principal Component {pcA+1}",
               ylabel= f"Principal Component {pcB+1}")
        ax.set_title('Biplot')
        return ax

    def scree_plot(self,ax):
        expalined_variance_ratio = self.pca.explained_variance_ratio_
        cumulative_variance = [0]
        bar_labels = []
        for i, variance_ratio in enumerate(expalined_variance_ratio):
            bar_labels.append(f'PC{i+1}')
            cumulative_variance.append(cumulative_variance[-1] + variance_ratio)
        cumulative_variance.pop(0)
        ax.grid(alpha=0.25)
        ax.set_axisbelow(True)
        ax.bar(bar_labels, expalined_variance_ratio, label=bar_labels)
        ax.plot(range(0,len(cumulative_variance)), cumulative_variance, color = 'black', label='cumulative explained variance')

    def pairplot(self, features=[]):
        sns.color_palette("tab10")
        if not features:
            print(':(')
            sns.pairplot(self.data, hue="target", palette='bright')
        else:
            features.append('target')
            print(':)')
            sns.pairplot(self.data[features], hue="target", palette='bright')
        sns.reset_defaults()

if __name__ == '__main__':
    winepca = WinePCA()
    fig = plt.figure(1, figsize=(8,6))
    ax = fig.add_subplot(111)
    winepca.perform_PCA()
    #winepca.score_plot(ax)
    #winepca.plot_loadnings_scatter(ax)
    #winepca.biplot(ax,0,1,scale=0)
    winepca.pairplot()
    #winepca.scree_plot(ax)
    plt.tight_layout()
    plt.show()


