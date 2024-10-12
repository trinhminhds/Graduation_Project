import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from Model.SVM import SVC
import numpy as np
from tensorflow.keras import datasets

(X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()

print(X_train.shape)

X_train_flat = X_train.reshape(-1, 28 * 28)
X_test_flat = X_test.reshape(-1, 28 * 28)

print(X_test_flat.shape, X_train_flat.shape)

#svc = SVC(0.001, 1000, 0.01)
#svc.fit(X_train_flat, y_train)

#y_pred = svc.predict(X_test_flat)

#print(y_pred[:5])


