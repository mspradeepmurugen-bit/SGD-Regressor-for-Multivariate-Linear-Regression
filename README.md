# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Import libraries — Import SGDRegressor, MultiOutputRegressor, NumPy, and Matplotlib.

Create the dataset — Define the input features X with two features and target Y with two output variables.

Build and train the model — Create an SGDRegressor with a learning rate of 0.01, wrap it using MultiOutputRegressor, and train the model using model.fit(X, Y).

Make predictions — Predict the outputs for the training data and for a new sample such as [8, 7]. Display the actual and predicted output values.

Visualize performance — Plot Actual vs Predicted values separately for Output 1 and Output 2. The closer the points are to the diagonal reference line, the better the predictions.

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: Pradeep.M
RegisterNumber:212225220071  
*/
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
import numpy as np
import matplotlib.pyplot as plt

X = np.array([
    [1, 2],
    [2, 1],
    [3, 4],
    [4, 3],
    [5, 5],
    [6, 7],
    [7, 6]
])


Y = np.array([
    [5, 8],
    [6, 9],
    [9,12],
    [10,13],
    [13,16],
    [16,20],
    [17,21]
])

sgd = SGDRegressor(
    max_iter=1000,
    eta0=0.01,
    learning_rate='constant',
    random_state=42
)

model = MultiOutputRegressor(sgd)

model.fit(X, Y)

Y_pred = model.predict(X)

print("\nActual Outputs")
print(Y)

print("\nPredicted Outputs")
print(np.round(Y_pred,2))

new_sample = np.array([[8, 7]])
prediction = model.predict(new_sample)

print("\nPrediction for", new_sample)
print(prediction)


plt.figure(figsize=(6,4))
plt.scatter(Y[:,0], Y_pred[:,0], color='blue')
plt.plot([Y[:,0].min(), Y[:,0].max()],
         [Y[:,0].min(), Y[:,0].max()],
         'r--')
plt.xlabel("Actual Output 1")
plt.ylabel("Predicted Output 1")
plt.title("Output 1: Actual vs Predicted")
plt.grid(True)
plt.show()

plt.figure(figsize=(6,4))
plt.scatter(Y[:,1], Y_pred[:,1], color='green')
plt.plot([Y[:,1].min(), Y[:,1].max()],
         [Y[:,1].min(), Y[:,1].max()],
         'r--')
plt.xlabel("Actual Output 2")
plt.ylabel("Predicted Output 2")
plt.title("Output 2: Actual vs Predicted")
plt.grid(True)
plt.show()
```

## Output:
<img width="580" height="392" alt="Screenshot 2026-08-20 213915" src="https://github.com/user-attachments/assets/2e943da4-b68e-4edb-846e-db983196bdb7" />

<img width="700" height="799" alt="image" src="https://github.com/user-attachments/assets/a930802f-e1fd-4eeb-b9c2-845ae99e56bc" />

## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
