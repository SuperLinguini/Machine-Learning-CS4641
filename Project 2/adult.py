import tensorflow as tf
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.ensemble import AdaBoostClassifier
from sklearn import preprocessing as preprocessing
import time
import os

import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
# matplotlib.interactive(True)

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.decomposition import FastICA

int_vars = ['Age', 'fnlwgt', 'Education Num', 'Capital Gain', 'Capital Loss', 'Hours Per Week']
dtypes = {x: np.int32 for x in int_vars}

df = pd.read_csv('adult.data.txt',
                   names=['Age', 'Workclass', 'fnlwgt', 'Education', 'Education Num',
                          'Marital Status', 'Occupation', 'Relationship', 'Race',
                          'Sex', 'Capital Gain', 'Capital Loss', 'Hours Per Week', 'Native Country', 'Target'],
                   sep=r'\s*,\s*', engine='python', na_values='?', dtype=dtypes
                )
df = df[df['Native Country'] == 'United-States']
df = df.dropna(axis=0)

del df['Education']
del df['Native Country']
# df.fillna('', inplace=True)

df.head(10)

df['Sex'] = df['Sex'].apply(lambda x: 1 if x == 'Female' else 0)
def convert_labels(x):
    if x == '<=50K':
        return 0
    else:
        return 1
df['Target'] = df['Target'].apply(convert_labels)
df.head(10)

df = pd.get_dummies(df, columns=['Workclass', 'Marital Status', 'Occupation', 'Relationship', 'Race'])

X_train, X_test, y_train, y_test = train_test_split(
    df[list(set(df.columns) - set(['Target']))],
    df['Target'], train_size=0.70)
# scaler = preprocessing.StandardScaler()
# X_train = pd.DataFrame(scaler.fit_transform(X_train.astype('float32')), columns=X_train.columns)
# X_test = scaler.transform(X_test.astype('float32'))

def kmeans(train_data, test_data, output_str, n_splits=5):
    # if type(train_data) is np.ndarray:
    #     train_set = pd.DataFrame(train_data)
    # if type(test_data) is np.ndarray:
    #     test_set = pd.DataFrame(test_data)
    n_clusters = [x for x in range(2,11)]

    columns = ['Num Clusters', 'Score', 'Train Time', 'Test Time']
    results = pd.DataFrame(columns=columns)

    # kf = KFold(n_splits=n_splits, shuffle=True)
    for n in n_clusters:
        # end_train = 0
        # end_test = 0
        # cv_score = 0
        # test_score = 0
        #
        # for train_indices, test_indices in kf.split(train_set):
        #     train = train_set.iloc[train_indices]
        #     test = train_set.iloc[test_indices]
        start_train = time.time()
        k_means = KMeans(n_clusters=n, n_jobs=-1).fit(train_data)
        print(k_means)
        end_train = time.time() - start_train

        start_test = time.time()
        # cv_score = k_means.score(test)
        test_score = k_means.score(test_data)
        end_test = time.time() - start_test

        values = [n, test_score / n_splits, end_train/n_splits, end_test/n_splits]
        results.loc[len(results)] = values

        print(' '.join(str(col) for col in columns))
        print(' '.join(str(val) for val in values))

    # plt.show(results['Num Clusters'], results['Test Score'])
    ax = results.plot(x='Num Clusters', y='Score', title='KMeans ' + output_str)
    fig = ax.get_figure()
    fig.savefig(os.path.join('Images','adult_kmeans_{}.png'.format(output_str)))

    results.to_excel('adult_kmeans_{}.xls'.format(output_str))


def expectation_maximization(train_data, test_data, output_str, n_splits=5):
    # if type(train_data) is np.ndarray:
    #     train_set = pd.DataFrame(train_data)
    # if type(test_data) is np.ndarray:
    #     test_set = pd.DataFrame(test_data)

    n_clusters = [x for x in range(2,11)]

    columns = ['Num Clusters', 'Score', 'Train Time', 'Test Time']
    results = pd.DataFrame(columns=columns)

    # kf = KFold(n_splits=n_splits, shuffle=True)
    for n in n_clusters:
        # end_train = 0
        # end_test = 0
        # cv_score = 0
        # test_score = 0

        # for train_indices, test_indices in kf.split(train_set):
        #     train = train_set.iloc[train_indices]
        #     test = train_set.iloc[test_indices]
        start_train = time.time()
        gm = GaussianMixture(n_components=n).fit(train_data)
        print(gm)
        end_train = time.time() - start_train

        start_test = time.time()
        # cv_score += gm.score(test)
        test_score = gm.score(test_data)
        end_test = time.time() - start_test

        values = [n, test_score / n_splits, end_train / n_splits, end_test / n_splits]
        results.loc[len(results)] = values

        print(' '.join(str(col) for col in columns))
        print(' '.join(str(val) for val in values))

    ax = results.plot(x='Num Clusters', y='Score', title='Expectation Maximization ' + output_str)
    fig = ax.get_figure()
    fig.savefig(os.path.join('Images','adult_expectation_maximization_{}.png'.format(output_str)))

    results.to_excel('adult_expectation_maximization_{}.xls'.format(output_str))

def overall():
    kmeans(X_train, X_test, 'Overall')
    expectation_maximization(X_train, X_test, 'Overall')

def principal_component_analysis():
    pca = PCA(n_components=20)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    kmeans(X_train_pca, X_test_pca, 'PCA')
    expectation_maximization(X_train_pca, X_test_pca, 'PCA')

def independent_component_analysis():
    ica = FastICA(n_components=20)
    X_train_ica = ica.fit_transform(X_train)
    X_test_ica = ica.transform(X_test)
    kmeans(X_train_ica, X_test_ica, 'ICA')
    # expectation_maximization(X_train_ica, X_test_ica, 'ICA')

def randomized_projection():
    rp = GaussianRandomProjection(n_components=20)
    X_train_rp = rp.fit_transform(X_train)
    X_test_rp = rp.transform(X_test)
    kmeans(X_train_rp, X_test_rp, 'RandomizedProjection')
    expectation_maximization(X_train_rp, X_test_rp, 'RandomizedProjection')

def latent_dirichlet_allocation():
    lda = LatentDirichletAllocation(n_topics=20, learning_method='batch', n_jobs=-1)
    X_train_lda = lda.fit_transform(X_train)
    X_test_lda = lda.transform(X_test)
    kmeans(X_train_lda, X_test_lda, 'LDA')
    expectation_maximization(X_train_lda, X_test_lda, 'LDA')

if __name__ == '__main__':
    overall()

    principal_component_analysis()
    # independent_component_analysis()
    # randomized_projection()
    # latent_dirichlet_allocation()
