import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# Get the data
x = np.load('tinyX.npy') # this should have shape (26344, 3, 64, 64)
trainY = np.load('tinyY.npy') 
xTest = np.load('tinyX_test.npy') # (6600, 3, 64, 64)

# Reshape the data into 1D arrays
trainX = x.reshape(len(x), -1)
testX = xTest.reshape(len(xTest), -1)

# Train the model
pipe = Pipeline(steps=[
    ('logistic', LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=None, solver='liblinear', tol=0.001))])
pipe.fit(trainX, trainY)

# Predict the test data
results = pipe.predict(testX)

# Print the results report
report = classification_report(trainY, results)
print(report)
