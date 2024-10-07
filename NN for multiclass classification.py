import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

df = pd.read_csv(r'C:\Users\bhara\Desktop\Neural-Network-for-Multiclass-classification-main\Customer Data\train.csv')

def prepareY(df):
    df.dropna(subset=['Var_1'], inplace=True)
    Y = df["Var_1"]
    yencoder = LabelEncoder()
    yencoder.fit(Y)
    encoded_Y = yencoder.transform(Y)
    return to_categorical(encoded_Y), encoded_Y, yencoder

hot_y, Y, yencoder = prepareY(df)
df = df.drop(["Var_1"], axis=1)

def fillmissing(df, feature, method):
    if method == "mode":
        df[feature] = df[feature].fillna(df[feature].mode()[0])
    elif method == "median":
        df[feature] = df[feature].fillna(df[feature].median())
    else:
        df[feature] = df[feature].fillna(df[feature].mean())

def prepareFeatures(df):
    columns = df.select_dtypes(include=['object']).columns
    for feature in columns:
        le = LabelEncoder()
        df[feature] = le.fit_transform(df[feature])
    features_missing = df.columns[df.isna().any()]
    for feature in features_missing:
        fillmissing(df, feature=feature, method="mean")
    scaler = MinMaxScaler(feature_range=(-1, 1))
    return scaler.fit_transform(df)

df = df.drop(["Segmentation", "ID"], axis=1)
X = prepareFeatures(df)

def baseline_model():
    model = Sequential()
    model.add(Input(shape=(X.shape[1],)))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(hot_y.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = baseline_model()
history = model.fit(X, hot_y, validation_split=0.33, epochs=200, batch_size=100, verbose=0)
_, accuracy = model.evaluate(X, hot_y, verbose=0)
print(f'Accuracy: {accuracy*100:.2f}%')
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
y_pred = np.argmax(model.predict(X), axis=1)

cm = confusion_matrix(Y, y_pred)
plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=yencoder.classes_, yticklabels=yencoder.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()

report = classification_report(Y, y_pred, target_names=yencoder.classes_)
print("Classification Report:")
print(report)
print("Model Summary:")
model.summary()
