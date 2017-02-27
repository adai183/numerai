import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from sklearn.preprocessing import StandardScaler


# Set seed for reproducibility
seed = 0
np.random.seed(seed)


# from keras.models import Sequential
# from keras.layers.core import Dense, Activation, Flatten, Dropout
# from keras.layers import BatchNormalization
# from keras.optimizers import Adam
# from keras.regularizers import l2


# def process(x, y):
#     """
#     do preprocessing
#     """
#     return x, y


# def generator(iterable, batch_size=512):
#     """
#     @iterable: pd.DataFrame
#     """

#     data_num = iterable.shape[0]

#     while True:
#         # shuffle Data before creating batches
#         iterable = iterable.sample(frac=1)

#         for ndx in range(0, data_num, batch_size):
#             batch = iterable.iloc[ndx:min(ndx + batch_size, data_num)]

#             x_train, y_train = process(batch)

#             yield (x_train, y_train)


def main():
    # Set seed for reproducibility
    np.random.seed(0)

    print("Loading data...")
    # Load the data from the CSV files
    training_data = pd.read_csv(
        'numerai_datasets/numerai_training_data.csv', header=0)

    # Transform the loaded CSV data into numpy arrays
    Y = training_data['target'].as_matrix()

    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(Y)
    Y = encoder.transform(Y)

    X = training_data.drop('target', axis=1).as_matrix()
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)
    # t_id = prediction_data['t_id'].as_matrix()
    # x_prediction = prediction_data.drop('t_id', axis=1).as_matrix()

    # larger model
    def Model():
        # create model
        model = Sequential()
        model.add(Dense(1000, input_dim=50, init='normal', activation='relu'))
        model.add(Dense(500, init='normal', activation='relu'))
        model.add(Dense(250, init='normal', activation='relu'))
        model.add(Dense(100, init='normal', activation='relu'))
        model.add(Dropout(.3))
        model.add(Dense(10, init='normal', activation='relu'))
        model.add(Dropout(.3))
        model.add(Dense(1, init='normal', activation='sigmoid'))
        # Compile model
        model.compile(loss='binary_crossentropy',
                      optimizer='adam', metrics=['accuracy'])
        return model

    # estimators = []
    # estimators.append(('standardize', StandardScaler()))
    # estimators.append(('mlp', KerasClassifier(
    #     build_fn=create_larger,
    #     nb_epoch=100,
    #     batch_size=2000,
    #     verbose=1)))

    # pipeline = Pipeline(estimators)
    # kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    # results = cross_val_score(pipeline, X, Y, cv=kfold)
    # print(results)
    # print("Performance: %.2f%% (%.2f%%)" %
    #       (results.mean() * 100, results.std() * 100))

    model = Model()
    model.fit(
        X,
        Y,
        batch_size=1000,
        nb_epoch=15000,
        shuffle=True,
        validation_split=0.2,
        callbacks=[ModelCheckpoint('model.h5',
                                   monitor='val_acc',
                                   verbose=1,
                                   save_best_only=True,
                                   mode='auto',
                                   period=1),
                   CSVLogger('train_stats.csv')
                   ],)


if __name__ == '__main__':
    main()
