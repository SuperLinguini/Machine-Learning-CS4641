import pandas as pd
import numpy as np
import time
import tensorflow as tf

import sklearn.preprocessing as preprocessing
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree.tree import DecisionTreeClassifier

def encode_categorical_features(df):
    result = df.copy()
    encoders = {}
    for column in result.columns:
        if result.dtypes[column] == np.object:
            encoders[column] = preprocessing.LabelEncoder()
            result[column] = encoders[column].fit_transform(result[column])
    return result, encoders

data = pd.read_csv('diabetic_data.csv', na_values='?', engine='python')

data.fillna('', inplace=True)

encoded_data, encoders = encode_categorical_features(data)

encoded_data.rename(columns={'readmitted': 'Target'}, inplace=True)

del data['encounter_id']
del data['patient_nbr']

X_train, X_test, y_train, y_test = train_test_split(
    encoded_data[list(set(encoded_data.columns) - set(['Target']))],
    encoded_data['Target'], train_size=0.70)
scaler = preprocessing.StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train.astype('float64')), columns=X_train.columns)
X_test = scaler.transform(X_test.astype('float64'))



# svm = SVC(cache_size=2000)
# print(svm)
# svm.fit(X_train, y_train)
# y_pred = svm.predict(X_test)
# score = svm.score(X_test, y_test)
# print(score)
#
# svm_linear = SVC(kernel='linear')
# svm_linear.fit(X_train, y_train)
# y_pred_lin = svm_linear.predict(X_test)
# score_lin = svm_linear.score(X_test, y_test)
#
# knn = KNeighborsClassifier(n_jobs=-1)
# knn.fit(X_train, y_train)
# y_pred_knn_3 = knn.predict(X_test)
# score_knn_3 = knn.score(X_test, y_test)
#
# knn_5 = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
# knn_5.fit(X_train, y_train)
# y_pred_knn_5 = knn.predict(X_test)
# score_knn_5 = knn.score(X_test, y_test)
#
# dt = DecisionTreeClassifier()
# dt.fit(X_train, y_train)
# dt_score = dt.score(X_test, y_test)


# neighbors = [1, 5, 10, 20, 40]
# weights = ['uniform', 'distance']
#
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
#         print('KNN: N-', neighbor, ' Weight-', weight, ' Training Score- ', train_score, ' Test Score-', test_score,
#               ' Train Time- ', end_train, 'Test Time- ', end_test)

# def main():
#     start = time.time()
#     bagging_svm = BaggingClassifier(SVC(), n_jobs=-1)
#     print(bagging_svm)
#     bagging_svm.fit(X_train, y_train)
#     # y_pred = bagging_svm.predict(X_test)
#     bagging_svm_score = bagging_svm.score(X_test, y_test)
#     print(bagging_svm_score)
#     print(time.time() - start)

X_train = X_train.as_matrix()

def main():
    train_validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
        X_train,
        y_train,
        every_n_steps=50)
    test_validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
        X_test,
        y_test,
        every_n_steps=50)

    tf.logging.set_verbosity(tf.logging.INFO)
    feature_columns = [tf.contrib.layers.real_valued_column("", dimension=49)]
    classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                                hidden_units=[10, 10],
                                                n_classes=3,
                                                model_dir="/tmp/diabetes_model")
    classifier.fit(x=X_train, y=y_train, steps=8000)
    accuracy_score = classifier.evaluate(x=X_test, y=y_test, steps=1)["accuracy"]

    print("\nTest Accuracy: {0:f}\n".format(accuracy_score))

if __name__ == '__main__':
    main()