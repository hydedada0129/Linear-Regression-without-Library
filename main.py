import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 0. preparing data, using a simple training set
np.random.seed(100)

X = 2 * np.random.rand(500, 1) # m = 500, d(features) = 1 , random 0~1
y = 3 * X + 0 + np.random.randn(500, 1)   #y=ax+intercept+noise. randn: normal distribution

# drawing the plot
fig = plt.figure(figsize=(8,6))
plt.scatter(X, y)
plt.title("Dataset")
plt.xlabel("X")
plt.ylabel("y")
# plt.show()

# 1. Split the data into a training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y)

print(f'Shape X_train: {X_train.shape}')
print(f'Shape y_train: {y_train.shape}')
print(f'Shape X_test: {X_test.shape}')
print(f'Shape y_test: {y_test.shape}')

# Linear Regression Class
class LinearRegression:
  def __init__():
    pass

  def train_gradient_descent(self, X, y, learning_rate=0.01, n_iters=100):
    n_samples, n_features = X.shape   # X is X_train
    self.weights = np.zeros(shape=(n_features, 1))
    self.bias = 0
    costs = []                        # loss
    for i in range(n_iters):      
      # linear regression model. y=Xw + b
      y_predict = np.dot(X, self.weights) + self.bias

      # Loss Function
      cost = np.mean((np.square(y-y_predict)))
      # also, (1/n_samples)* np.sum((y_predict-y)**2)
      costs.append(cost)

      if i % 10 == 0:
        print(f'cost at iteration {i}: {cost}')

      # weight derivative 
      dj_dw = (2 / n_samples) * np.dot(X.T, (y_predict - y))
      '''
      X is (m, d)    
      X.T is (d, m), (y_predict - y) is (m, 1), dj_dw is (d, 1)
      np.dot : matrix multiplication
      '''

      # bias derivative
      dj_db = (2 / n_samples) * np.sum((y_predict - y))

      # update the parameters w, b
      self.weights -= learning_rate * dj_dw
      self.bias -= learning_rate * dj_db
    return self.weights, self.bias, costs

  def prefict(self, X):
    return np.dot(X, self.weights) + self.bias

# 3. Using class for training linear regression model
regressor = LinearRegression()
w_trained, b_trained, costs = regressor.train_gradient_descent(X_train, y_train, learning_rate=0.001, n_iters=101)

fig = plt.figure(figsize=(8,6))
plt.plot(np.arange(101), costs)
plt.title("Development of cost during training")
plt.xlabel("Number of iterations")
plt.ylabel("Cost")
# plt.show()

# 4. Testing
n_samples_test, _ = X_test.shape
y_predict_test = regressor.predict(X_test)   
#y=Xw + b, input_test(X_test), output_test(y_test)

error_test = (1/n_samples_test) * np.sum((y_predict_test - y_test) ** 2)
print(f"MSE on the test set: {np.round(error_test, 4)}")
#loss function is prediciton ability

# Plot the test predictions
fig = plt.figure(figsize=(8,6))
plt.title("Dataset in blue, predictions for test set in orange")
plt.scatter(X_test, y_test) # blue
plt.scatter(X_test, y_predict_test) # orange
plt.xlabel("X")
plt.ylabel("y")
plt.show()

