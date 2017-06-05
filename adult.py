#!/users/mlingineni3-gtri/mil/bin python2.7

import pandas as pd
import numpy as np
import time

# import tensorflow as tf
import sklearn.preprocessing as preprocessing
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree.tree import DecisionTreeClassifier

def encode_discrete_features(df):
    result = df.copy()
    encoders = {}
    for column in result.columns:
        if result.dtypes[column] == np.object:
            encoders[column] = preprocessing.LabelEncoder()
            result[column] = encoders[column].fit_transform(result[column])
    return result, encoders

data = pd.read_csv('adult.data.txt',
                   names=['Age', 'Workclass', 'fnlwgt', 'Education', 'Education Num',
                          'Marital Status', 'Occupation', 'Relationship', 'Race',
                          'Sex', 'Capital Gain', 'Capital Loss', 'Hours Per Week', 'Native Country', 'Target'],
                   sep=r'\s*,\s*', engine='python', na_values='?')

del data['Education']
data.fillna('', inplace=True)

encoded_data, encoders = encode_discrete_features(data)

X_train, X_test, y_train, y_test = train_test_split(
    encoded_data[list(set(encoded_data.columns) - set(['Target']))],
    encoded_data['Target'], train_size=0.70)
scaler = preprocessing.StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train.astype('float32')), columns=X_train.columns)
X_test = scaler.transform(X_test.astype('float32'))

# X_train = X_train.as_matrix()
#
# def main():
#     train_validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
#         X_train,
#         y_train,
#         every_n_steps=50)
#     test_validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
#         X_test,
#         y_test,
#         every_n_steps=50)
#
#
#     tf.logging.set_verbosity(tf.logging.INFO)
#     feature_columns = [tf.contrib.layers.real_valued_column("", dimension=13)]
#     classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
#                                                 hidden_units=[10, 10],
#                                                 n_classes=2,
#                                                 model_dir="/tmp/adult_model")
#     classifier.fit(x=X_train, y=y_train, steps=1000, monitors=[train_validation_monitor, test_validation_monitor])
#     accuracy_score = classifier.evaluate(x=X_test, y=y_test, steps=1)["accuracy"]
#
#     print("\nTest Accuracy: {0:f}\n".format(accuracy_score))


# X_train = X_train.as_matrix()
# # X_test = X_test.as_matrix()
#
#
# neural_network = tflearn.input_data(shape=[None, 13])
# neural_network = tflearn.fully_connected(neural_network, 16)
# neural_network = tflearn.fully_connected(neural_network, 16)
# neural_network = tflearn.fully_connected(neural_network, 1, activation='softmax')
# neural_network = tflearn.regression(neural_network)
#
# model = tflearn.DNN(neural_network)
#
# model.fit(X_train, y_train, n_epoch=10, batch_size=16, show_metric=True)
#
# pred = model.predict(X_test)


# svm = SVC()
# svm.fit(X_train, y_train)
# y_pred = svm.predict(X_test)
# score = svm.score(X_test, y_test)



# svm_linear = SVC(kernel='linear')
# svm_linear.fit(X_train, y_train)
# y_pred_lin = svm_linear.predict(X_test)
# score_lin = svm_linear.score(X_test, y_test)

# neighbors = [1, 5, 10, 20, 40]
# weights = ['uniform', 'distance']
#
# df = pd.DataFrame(columns=['K', 'Weight', 'Training Score', 'Test Score', 'Train Time', 'Test Time'])
# for weight in weights:
#     for neighbor in neighbors:
#         start_train = time.time()
#         knn = KNeighborsClassifier(n_jobs=-1, weights=weight, n_neighbors=neighbor)
#         knn.fit(X_train, y_train)
#         end_train = time.time() - start_train
#
#         train_score = knn.score(X_train, y_train)
#         start_test = time.time()
#         test_score = knn.score(X_test, y_test)
#         end_test = time.time() - start_test
#         df.loc[len(df)] = [neighbor, weight, train_score, test_score, end_train, end_test]
#         print('KNN: N-', neighbor, ' Weight-', weight, ' Training Score- ', train_score, ' Test Score-', test_score, ' Train Time- ', end_train, 'Test Time- ', end_test)
# df.to_excel('adult.xls')



# dt = DecisionTreeClassifier()
# dt.fit(X_train, y_train)
# dt.score(X_test, y_test)
#

def decision_tree_depths():
    max_depths = [2, 4, 6, 8, 10, 12, 16, 18, 20, 25, 30, 40]

    columns = ['Max Depths', 'Training Score', 'Test Score', 'Train Time', 'Test Time']
    df = pd.DataFrame(columns=columns)

    for depth in max_depths:
        start_train = time.time()
        dt = DecisionTreeClassifier(max_depth=depth)
        print(dt)
        dt.fit(X_train, y_train)
        end_train = time.time() - start_train

        train_score = dt.score(X_train, y_train)
        start_test = time.time()
        test_score = dt.score(X_test, y_test)
        end_test = time.time() - start_test

        values = [depth, train_score, test_score, end_train, end_test]
        df.loc[len(df)] = values

        print(' '.join(str(col) for col in columns))
        print(' '.join(str(val) for val in values))

    df.to_excel('adult_dt.xls')

def decision_tree_training_sets():
    training_set_sizes = [.1,.25,.5,.75,.9]

    columns = ['Training Set Size', 'Training Score', 'Test Score', 'Train Time', 'Test Time']
    df = pd.DataFrame(columns=columns)

    for training_set_size in training_set_sizes:
        X_train, X_test, y_train, y_test = train_test_split(
            encoded_data[list(set(encoded_data.columns) - set(['Target']))],
            encoded_data['Target'], train_size=training_set_size)
        scaler = preprocessing.StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train.astype('float32')), columns=X_train.columns)
        X_test = scaler.transform(X_test.astype('float32'))

        start_train = time.time()
        dt = DecisionTreeClassifier(max_depth=8)
        print(dt)
        dt.fit(X_train, y_train)
        end_train = time.time() - start_train

        train_score = dt.score(X_train, y_train)
        start_test = time.time()
        test_score = dt.score(X_test, y_test)
        end_test = time.time() - start_test

        values = [training_set_size, train_score, test_score, end_train, end_test]
        df.loc[len(df)] = values

        print(' '.join(str(col) for col in columns))
        print(' '.join(str(val) for val in values))

    df.to_excel('adult_dt_training_sets.xls')

def boosting_estimators():
    max_estimators = [2, 4, 6, 8, 10, 12, 16, 18, 20, 25, 30, 40]

    columns = ['Max Estimators', 'Training Score', 'Test Score', 'Train Time', 'Test Time']
    df = pd.DataFrame(columns=columns)

    for estimator in max_estimators:
        start_train = time.time()
        dt = AdaBoostClassifier(n_estimators=estimator)
        print(dt)
        dt.fit(X_train, y_train)
        end_train = time.time() - start_train

        train_score = dt.score(X_train, y_train)
        start_test = time.time()
        test_score = dt.score(X_test, y_test)
        end_test = time.time() - start_test

        values = [estimator, train_score, test_score, end_train, end_test]
        df.loc[len(df)] = values

        print(' '.join(str(col) for col in columns))
        print(' '.join(str(val) for val in values))

    df.to_excel('adult_adaboost_estimators.xls')

def boosting_training_sets():
    training_set_sizes = [.1,.25,.5,.75,.9]

    columns = ['Training Set Size', 'Training Score', 'Test Score', 'Train Time', 'Test Time']
    df = pd.DataFrame(columns=columns)

    for training_set_size in training_set_sizes:
        X_train, X_test, y_train, y_test = train_test_split(
            encoded_data[list(set(encoded_data.columns) - set(['Target']))],
            encoded_data['Target'], train_size=training_set_size)
        scaler = preprocessing.StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train.astype('float32')), columns=X_train.columns)
        X_test = scaler.transform(X_test.astype('float32'))

        start_train = time.time()
        dt = AdaBoostClassifier(asdasd)  # TODO
        print(dt)
        dt.fit(X_train, y_train)
        end_train = time.time() - start_train

        train_score = dt.score(X_train, y_train)
        start_test = time.time()
        test_score = dt.score(X_test, y_test)
        end_test = time.time() - start_test

        values = [training_set_size, train_score, test_score, end_train, end_test]
        df.loc[len(df)] = values

        print(' '.join(str(col) for col in columns))
        print(' '.join(str(val) for val in values))

    df.to_excel('adult_adaboost_training_sets.xls')


def knn():
    neighbors = [1, 5, 10, 20, 40]
    weights = ['uniform', 'distance']

    for weight in weights:
        for neighbor in neighbors:
            start_train = time.time()
            knn = KNeighborsClassifier(n_jobs=-1, weights=weight, n_neighbors=neighbor)
            knn.fit(X_train, y_train)
            end_train = time.time() - start_train

            train_score = knn.score(X_train, y_train)
            start_test = time.time()
            test_score = knn.score(X_test, y_test)
            end_test = time.time() - start_test
            print('KNN: N-', neighbor, ' Weight-', weight, ' Training Score- ', train_score, ' Test Score-', test_score,
                  ' Train Time- ', end_train, 'Test Time- ', end_test)

def svm():
    training_set_size = [.1,.25,.5,.75,.9]
    kernels = ['rbf', 'poly']

    columns = ['Kernel', 'Training Set Size', 'Training Score', 'Test Score', 'Train Time', 'Test Time']
    df = pd.DataFrame(columns=columns)

    for kernel in kernels:
        for tset_size in training_set_size:
            X_train, X_test, y_train, y_test = train_test_split(
                encoded_data[list(set(encoded_data.columns) - set(['Target']))],
                encoded_data['Target'], train_size=tset_size)
            scaler = preprocessing.StandardScaler()
            X_train = pd.DataFrame(scaler.fit_transform(X_train.astype('float32')), columns=X_train.columns)
            X_test = scaler.transform(X_test.astype('float32'))

            start = time.time()
            bagging_svm = BaggingClassifier(SVC(kernel=kernel, cache_size=1000), n_jobs=-1)
            print(bagging_svm)
            bagging_svm.fit(X_train, y_train)
            end_train = time.time() - start

            # y_pred = bagging_svm.predict(X_test)
            train_score = bagging_svm.score(X_train, y_train)
            start_test = time.time()
            test_score = bagging_svm.score(X_test, y_test)
            end_test = time.time() - start_test
            values = [kernel, tset_size, train_score, test_score, end_train, end_test]
            df.loc[len(df)] = values
            print(' '.join(str(col) for col in columns))
            print(' '.join(str(val) for val in values))
    df.to_excel('adult_svm.xls')

def main():
    boosting_estimators()


if __name__ == '__main__':
    main()
