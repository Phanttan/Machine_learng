from sklearn.datasets import load_iris
iris_dataset = load_iris()
# Check some information from iris_data
    # print(iris_dataset)
    # print(iris_dataset.keys())
    # # print(iris_dataset.values())
    # print(iris_dataset.keys()[3])
    # # print(iris_dataset.values()[3])
    # print('Target names:{}'.format(iris_dataset['target_names']))
    # print("Feature names: \n{}".format(iris_dataset['feature_names']))
    # print('Shape of data:{}'.format(iris_dataset['data'].shape))
    # print('First five columns of data: \n{}'.format(iris_dataset['data'][:5]))
    # print("Type of target: {}".format(type(iris_dataset['target'])))
    # print("Target:\n{}".format(iris_dataset['target']))

################
# What is tran_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(iris_dataset['data'],iris_dataset['target'], random_state = 0)
    # print('X_train shape:{}'.format(X_train.shape))
    # print('Y_train shape:{}'.format(Y_train.shape))
    # print('X_test shape:{}'.format(X_test.shape))
    # print('Y_test shape:{}'.format(Y_test.shape))
# Scatter Plot
import pandas as pd
import mglearn
# creat dataframe from data in X_train
# label the columns using the strings in iris_dataset.feature_names
iris_dataframe = pd.DataFrame(X_train,columns= iris_dataset.feature_names)
# create a scatter matrix from the dataframe, color by y_train
grr = pd.plotting.scatter_matrix(iris_dataframe, c=Y_train, figsize=(15, 15), marker='o', hist_kwds={'bins':20}, s=60, alpha=.8 ,cmap=mglearn.cm3)

##########################

