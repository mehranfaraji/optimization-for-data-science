import numpy as np
import pandas as pd

A4A_PATH = 'datasets/a4a.txt'
A4A_PATH_T = 'datasets/a4a_t.txt'
SVMGUIDE1_PATH = 'datasets/svmguide1.txt'
SVMGUIDE1_PATH_T = 'datasets/svmguide1_t.txt'
LIVER_DISORDER = 'datasets/liver-disorders.txt'
LIVER_DISORDER_T = 'datasets/liver-disorders_t.txt'

def load_a4a_dataset(path):
        num_cols = 123  
        X_data = []
        y_data = []
        with open(path, 'r') as file:
            for line in file:
                words = line.strip().split()

                target = int(words[0])

                row = [0] * 123

                for word in words[1:]:
                    feature, value = word.split(':')
                    col_index = int(feature) - 1  
                    row[col_index] = float(value)

                X_data.append(row)
                y_data.append(target)

        columns = [f'Feature_{i}' for i in range(1, num_cols + 1)]
        data = pd.DataFrame(data=np.column_stack([X_data, y_data]), columns=columns + ['Target'])

        data = data.fillna(0)

        X = data[data.columns[:-1]].to_numpy()
        y = data[data.columns[-1]].to_numpy()
        X = X[:,:-1]

        return X, y

def load_dataset(path, scaled_normalize):
        y_data = []
        X_data = []
        with open(path, 'r') as file:
            for line in file:
                words = line.strip().split()

                target = int(words[0])
                line_values = []
                for word in words[1:]:
                    _ , value = word.split(':')
                    line_values.append(float(value))
                X_data.append(np.array(line_values))
                y_data.append(target)
        
        X = np.array(X_data)
        y = np.array(y_data)
        y[y== 0] = -1

        if scaled_normalize:
            normalized_X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
            X = 2 * normalized_X - 1

        return X, y

def a4a():
    X_train, y_train = load_a4a_dataset(path= A4A_PATH)
    X_test, y_test = load_a4a_dataset(path= A4A_PATH_T)
    return X_train, y_train, X_test, y_test

def svm_guid(scaled_normalize= True):
    X_train, y_train = load_dataset(path= SVMGUIDE1_PATH, scaled_normalize= scaled_normalize)
    X_test, y_test = load_dataset(path= SVMGUIDE1_PATH_T, scaled_normalize= scaled_normalize)
    return X_train, y_train, X_test, y_test

def liver_disorder(scaled_normalize= True):
    X_train, y_train = load_dataset(path= LIVER_DISORDER, scaled_normalize= scaled_normalize)
    X_test, y_test = load_dataset(path= LIVER_DISORDER_T, scaled_normalize= scaled_normalize)
    return X_train, y_train, X_test, y_test