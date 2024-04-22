from sklearn.model_selection import train_test_split
from misc import standardize, reverse
import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import PolynomialFeatures
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

# Sliding window approaches
def create_sequences_X(data, sequence_length):
    X = []
    for i in range(len(data) - sequence_length):
        sequence = data[i:i + sequence_length + 1]
        X.append(sequence)
    return np.array(X)

def create_sequences_y(data, sequence_length):
    y = []
    for i in range(len(data) - sequence_length):
        target = data[i + sequence_length]
        y.append(target)
    return np.array(y)



df = pd.read_csv("AAPL_data_train.csv")
df = df.drop('index', axis=1)
df = df.drop('time', axis=1)

mean = df['close'].mean()
std = df['close'].std()

temp = pd.DataFrame()

temp['temp'] = (df['close'] - df['open'] > 0).astype(int)
#print(df.head())
df = standardize(df)

df['close'] = temp['temp']


df = df.drop('high', axis=1)
df = df.drop('low', axis=1)
#print(df.head())
# This is the previous hours volume (technically time lag one)
df['volume'] = df['volume'].shift(1)
#print(df.head())
df.dropna(inplace=True)
#print(df.head())


for i in range(2):
    df[f'volume lag {i+1}'] = df['volume'].shift(i+1)

df.dropna(inplace=True)
X = df.drop('close', axis=1)
y = df['close']

poly = PolynomialFeatures(degree=1)
X_poly = poly.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, shuffle=False)

sequence_length = 12  # Number of timesteps
#print(len(y_test))
X_train = create_sequences_X(X_train, sequence_length)
X_test = create_sequences_X(X_test, sequence_length)
y_train = create_sequences_y(y_train.to_numpy(), sequence_length)
y_test = create_sequences_y(y_test.to_numpy(), sequence_length)

checkpoint_path = "model_checkpoint.keras"
checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

model = Sequential()
model.add(LSTM(units=50, return_sequences=False, input_shape=(12, 5)))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=30, batch_size=1, validation_data=(X_test, y_test), callbacks=[checkpoint], verbose=2)
model.summary()

model = load_model('model_checkpoint.keras')

y_pred = model.predict(X_test)

# Calculate metrics
#mse = mean_squared_error(y_test, y_pred)
#r2 = r2_score(y_test, y_pred)

#actual_y = np.array([reverse(x, mean=mean, std=std) for x in y_test])
#predicted_y = np.array([reverse(x, mean=mean, std=std) for x in y_pred])
predicted_classes = (y_pred > 0.5).astype(int)
#print(predicted_classes)


accuracy = accuracy_score(y_test, predicted_classes)
precision = precision_score(y_test, predicted_classes)
recall = recall_score(y_test, predicted_classes)
f1 = f1_score(y_test, predicted_classes)
conf_matrix = confusion_matrix(y_test, predicted_classes)

errors = 0
correct = 0
error_curve = []
for i in range(len(y_test)):
    if y_test[i] != predicted_classes[i]:
        errors += 1
    else:
        correct += 1
    error_curve.append(errors)


plt.figure(figsize=(10, 6))

plt.plot(range(len(error_curve)), error_curve, label=f'RF')

plt.xlabel('Time (hours')
plt.ylabel('Errors made')
plt.title('Plot of Sub-arrays')
plt.legend()
plt.show()

print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1: ", f1)
print("CM: ", conf_matrix)

      
#print("Correct %: ", correct / len(y_pred))