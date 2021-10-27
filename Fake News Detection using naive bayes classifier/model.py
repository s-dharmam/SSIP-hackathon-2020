import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import os


class predict:
    def get_result(message):

        THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
        my_file = os.path.join(THIS_FOLDER, 'data.csv')
        df = pd.read_csv(my_file, encoding="latin-1")
        label = df['v1']
        title = df['v2']
        label = label.map({'ham': 0, 'spam': 1})
        cv = CountVectorizer()
        X = cv.fit_transform(title)
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            label,
                                                            test_size=0.33,
                                                            random_state=42)
        clf = MultinomialNB()
        y_train = y_train.astype('int')
        y_test = y_test.astype('int')
        clf.fit(X_train, y_train)

        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
        return my_prediction