import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger


# from keras.models import Sequential
# from keras.layers.core import Dense, Activation, Flatten, Dropout
# from keras.layers import BatchNormalization
# from keras.optimizers import Adam
# from keras.regularizers import l2
# from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger


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
    prediction_data = pd.read_csv(
        'numerai_datasets/numerai_tournament_data.csv', header=0)

    # Transform the loaded CSV data into numpy arrays
    Y = training_data['target'].as_matrix()
    X = training_data.drop('target', axis=1).as_matrix()
    t_id = prediction_data['t_id'].as_matrix()
    x_prediction = prediction_data.drop('t_id', axis=1).as_matrix()

    print(X.shape)

    model = Sequential()

    # 0. Dense 2128
    model.add(Dense(16000, input_dim=50))
    model.add(Activation('elu'))

    # 0. Dense 2128
    model.add(Dense(8000))
    model.add(Activation('elu'))

    # 0. Dense 2128
    model.add(Dense(4000))
    model.add(Activation('elu'))

    # 1. Dense 1064
    model.add(Dense(2000, init='uniform'))
    model.add(Activation('elu'))

    # 2. Dense 532
    model.add(Dense(1000, init='uniform'))
    model.add(Activation('elu'))

    # 3. Dense 256
    model.add(Dense(500, init='uniform'))
    model.add(Activation('elu'))

    # 4. Dense 128
    model.add(Dense(250, init='uniform'))
    model.add(Activation('elu'))

    # 5. Dense 64
    model.add(Dense(100, init='uniform'))
    model.add(Activation('elu'))

    # 6. Dense 32
    model.add(Dense(50, init='uniform'))
    model.add(Activation('elu'))

    # 7. Dense 16
    model.add(Dense(10, init='uniform'))
    model.add(BatchNormalization())
    model.add(Activation('elu'))
    model.add(Dropout(.5))

    # 8. Dense 8
    model.add(Dense(6, init='uniform'))
    model.add(Activation('elu'))
    model.add(Dropout(.5))

    # 9. Output
    model.add(Dense(1))

    # Configures the learning process and metrics
    model.compile(optimizer=Adam(lr=0.0001),
                  loss='mean_squared_logarithmic_error', metrics=['accuracy'])

    # Train the model
    # History is a record of training loss and metrics
    model.fit(
        X,
        Y,
        batch_size=2000,
        nb_epoch=100,
        shuffle=True,
        validation_split=0.2,
        callbacks=[EarlyStopping(monitor='val_acc',
                                 min_delta=0.001,
                                 patience=10,
                                 verbose=2,
                                 mode='auto'),
                   ModelCheckpoint('model.h5',
                                   monitor='val_acc',
                                   verbose=1,
                                   save_best_only=True,
                                   mode='auto',
                                   period=1),
                   CSVLogger('train_stats.csv')
                   ],)


if __name__ == '__main__':
    main()
