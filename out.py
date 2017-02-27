import pandas as pd
import numpy as np
from keras.models import load_model

model = load_model('model.h5')

# Set seed for reproducibility
np.random.seed(0)

print("Loading data...")
# Load the data from the CSV files
prediction_data = pd.read_csv(
    'numerai_datasets/numerai_tournament_data.csv', header=0)


# Transform the loaded CSV data into numpy arrays
t_id = prediction_data['t_id']
x_prediction = prediction_data.drop('t_id', axis=1)

pred = model.predict(x_prediction.as_matrix(), verbose=1)

df = pd.DataFrame(pred, index=t_id)

print('writing csv ...')
df.to_csv('numerai_datasets/predictions.csv')
