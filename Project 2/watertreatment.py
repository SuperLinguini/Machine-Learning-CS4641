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


URL = 'water-treatment.data.txt'

df = pd.read_csv(
    URL,
    encoding='latin-1',
    skipinitialspace=True,
    na_values=['?'],
    index_col=None,
    header=None,
)

del df[df.columns[0]]

def get_features(frame):
    '''
    Transforms and scales the input data and returns a numpy array that
    is suitable for use with scikit-learn.

    Note that in unsupervised learning there are no labels.
    '''

    # Replace missing values with 0.0
    # or we can use scikit-learn to calculate missing values below
    # frame[frame.isnull()] = 0.0

    # Convert values to floats
    arr = np.array(frame, dtype=np.float)

    # Impute missing values from the mean of their entire column
    from sklearn.preprocessing import Imputer
    imputer = Imputer(strategy='mean')
    arr = imputer.fit_transform(arr)

    # Normalize the entire data set to mean=0.0 and variance=1.0
    from sklearn.preprocessing import scale
    arr_scaled = scale(arr)

    return arr_scaled, arr

df_scaled, df = get_features(df)

X_train_unscaled, X_test_unscaled, X_train, X_test = train_test_split(
    df, df_scaled, train_size=0.70)
# scaler = preprocessing.StandardScaler()
# X_train = pd.DataFrame(scaler.fit_transform(X_train_unscaled.astype('float32')), columns=X_train_unscaled.columns)
# X_test = scaler.transform(X_test_unscaled.astype('float32'))

def kmeans(train_data, test_data, output_str, i=0):
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

        values = [n, test_score, end_train, end_test]
        results.loc[len(results)] = values

        print(' '.join(str(col) for col in columns))
        print(' '.join(str(val) for val in values))

    # plt.show(results['Num Clusters'], results['Test Score'])
    ax = results.plot(x='Num Clusters', y='Score', title='KMeans ' + output_str)
    fig = ax.get_figure()
    fig.savefig(os.path.join('Images','water_kmeans_{}{}.png'.format(output_str, i)))

    results.to_excel('water_kmeans_{}.xls'.format(output_str))


def expectation_maximization(train_data, test_data, output_str, i=0):
    # if type(train_data) is np.ndarray:
    #     train_set = pd.DataFrame(train_data)
    #     test_set = pd.DataFrame(test_data)
    #     train_set = train_set.append(test_set)
    # else:
    #     train_set = train_data

    n_clusters = [x for x in range(2,11)]

    columns = ['Num Clusters', 'Score', 'Train Time', 'Test Time']
    results = pd.DataFrame(columns=columns)

    # kf = KFold(n_splits=n_splits, shuffle=True)
    for n in n_clusters:
        # end_train = 0
        # end_test = 0
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

        values = [n, test_score, end_train, end_test]
        results.loc[len(results)] = values

        print(' '.join(str(col) for col in columns))
        print(' '.join(str(val) for val in values))

    ax = results.plot(x='Num Clusters', y='Score', title='Expectation Maximization ' + output_str)
    fig = ax.get_figure()
    fig.savefig(os.path.join('Images','water_expectation_maximization_{}{}.png'.format(output_str, i)))

    results.to_excel('water_expectation_maximization_{}.xls'.format(output_str))

def overall(i):
    kmeans(X_train, X_test, 'Overall', i)
    expectation_maximization(X_train, X_test, 'Overall', i)
    # neural_network(X_train, y_train, X_test, y_test, 'Overall', i)

def principal_component_analysis(i):
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    kmeans(X_train_pca, X_test_pca, 'PCA', i)
    expectation_maximization(X_train_pca, X_test_pca, 'PCA', i)
    # neural_network(X_train_pca, y_train, X_test_pca, y_test, 'PCA', i)

def independent_component_analysis(i):
    ica = FastICA(n_components=2)
    X_train_ica = ica.fit_transform(X_train)
    X_test_ica = ica.transform(X_test)
    kmeans(X_train_ica, X_test_ica, 'ICA', i)
    expectation_maximization(X_train_ica, X_test_ica, 'ICA', i)
    # neural_network(X_train_ica, y_train, X_test_ica, y_test, 'ICA', i)

def randomized_projection(i):
    rp = GaussianRandomProjection(n_components=2)
    X_train_rp = rp.fit_transform(X_train)
    X_test_rp = rp.transform(X_test)
    kmeans(X_train_rp, X_test_rp, 'RandomizedProjection', i)
    expectation_maximization(X_train_rp, X_test_rp, 'RandomizedProjection', i)
    # neural_network(X_train_rp, y_train, X_test_rp, y_test, 'RandomizedProjection', i)

def latent_dirichlet_allocation(i):
    lda = LatentDirichletAllocation(n_topics=2, learning_method='batch', n_jobs=-1)
    X_train_lda = lda.fit_transform(X_train_unscaled)
    X_test_lda = lda.transform(X_test_unscaled)
    kmeans(X_train_lda, X_test_lda, 'LDA', i)
    expectation_maximization(X_train_lda, X_test_lda, 'LDA', i)
    # neural_network(X_train_lda, y_train, X_test_lda, y_test, 'LDA', i)

def neural_network(X_train, y_train, X_test, y_test, output_str, i):
    # X_train2 = X_train.as_matrix()
    # train_validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
    #     X_train,
    #     y_train,
    #     every_n_steps=100)
    test_validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
        X_test,
        y_test,
        every_n_steps=50)


    tf.logging.set_verbosity(tf.logging.INFO)
    feature_columns = [tf.contrib.layers.real_valued_column("", dimension=X_train.shape[1])]
    classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                                hidden_units=[10, 10],
                                                # dropout=0.5,
                                                model_dir="/tmp/water_nn_{}{}".format(output_str, i),
                                                config=tf.contrib.learn.RunConfig(save_checkpoints_steps=100))
    if i % 2 == 0:
        setattr(classifier, 'dropout', 0.5)
    classifier.fit(x=X_train, y=y_train, steps=1500, monitors=[test_validation_monitor])
    accuracy_score = classifier.evaluate(x=X_test, y=y_test, steps=50)["accuracy"]

    print("\nTest Accuracy: {0:f}\n".format(accuracy_score))

if __name__ == '__main__':

    for i in range(1):
        overall(i)

        principal_component_analysis(i)
        independent_component_analysis(i)
        randomized_projection(i)
        latent_dirichlet_allocation(i)
