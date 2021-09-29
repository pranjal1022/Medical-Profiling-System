# importing libraries

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


import pickle

from feature_engineering import FeatEng


class BestModel():

    def __init__(self, path, target):

        self.x_train = FeatEng(path=path, target=target).splitting()[2]
        self.x_test = FeatEng(path=path, target=target).splitting()[3]
        self.y_train = FeatEng(path=path, target=target).splitting()[4]
        self.y_test = FeatEng(path=path, target=target).splitting()[5]
        self.filename = (path.split('/')[1].split('.')[0] + '.pkl')

# Logistic Regression Algorithm

    def logistic_regression(self):

        try:

            # Model Training
            # hyper_parameters = {'n_estimators': np.arange(100, 2100, 100),
            #                     'max_depth': np.arange(2, 21, 1),
            #                     'max_features': ['auto', 'sqrt', 'log2']}

            model1 = LogisticRegression()
            model1.fit(self.x_train, self.y_train)

            # Model Prediction
            y_predict = model1.predict(self.x_test)
            a = accuracy_score(self.y_test, y_predict)

            return [a, model1]

        except Exception as e:
            print(e)



    def model(self):
        model = self.logistic_regression()[1]
        print('Logistic Regression chosen')

        pickle.dump(model, open('Models/' + self.filename, 'wb'))


# Dataset Names with respect to Target variables

info = [['diabetes.csv', 'Outcome'], ['heart.csv', 'target'],['liver.csv', 'Dataset']]

# Creating Pickle File

for i in info:
    BestModel(path=('Data/' + i[0]), target=i[1]).model()
