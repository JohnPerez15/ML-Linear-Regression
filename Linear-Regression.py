import matplotlib
import numpy as np
import sklearn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#%matplotlib inline


class BostonHousingDataset:
    def __init__(self):
        self.url = "http://lib.stat.cmu.edu/datasets/boston"
        self.feature_names = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"]

    def load_dataset(self):
        # Fetch data from URL
        raw_df = pd.read_csv(self.url, sep="\s+", skiprows=22, header=None)
        data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
        target = raw_df.values[1::2, 2]

        # Create the dictionary in sklearn format
        dataset = {
            'data': [],
            'target': [],
            'feature_names': self.feature_names,
            'DESCR': 'Boston House Prices dataset'
        }

        dataset['data'] = data
        dataset['target'] = target

        return dataset

# Load the Boston Housing Dataset from sklearn
boston_housing = BostonHousingDataset()
boston_dataset = boston_housing.load_dataset()
boston_dataset.keys(), boston_dataset['DESCR']

# Create the dataset
boston = pd.DataFrame(boston_dataset['data'], columns=boston_dataset['feature_names'])
boston['MEDV'] = boston_dataset['target']
boston.head()

# Introductory Data Analysis
# First, let us make sure there are no missing values or NANs in the dataset
print(boston.isnull().sum())

# Next, let us plot the target vaqriable MEDV

sns.set(rc={'figure.figsize': (11.7, 8.27)})
sns.distplot(boston['MEDV'], bins=30)
plt.show()

# Finally, let us get the correlation matrix
correlation_matrix = boston.corr().round(2)
# annot = True to print the values inside the square
sns.heatmap(data=correlation_matrix, annot=True)
# Let us take few of the features and see how they relate to the target in a 1D plot
plt.figure(figsize=(20, 5))

features = ['LSTAT', 'RM', 'CHAS', 'NOX', 'AGE', 'DIS']
target = boston['MEDV']

for i, col in enumerate(features):
    plt.subplot(1, len(features), i + 1)
    x = boston[col]
    y = target
    plt.scatter(x, y, marker='o')
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel('MEDV')

#load data points and splits the data
from sklearn.model_selection import train_test_split
X = boston.to_numpy()
X = np.delete(X, 13, 1)
y = boston['MEDV'].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# Lets now train the model
from sklearn.linear_model import LinearRegression
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)

# Model Evaluation
# Lets first evaluate on training set
from sklearn.metrics import r2_score

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

y_pred_train = lin_model.predict(X_train)
rmse_train = rmse(y_pred_train, y_train)
r2_train = r2_score(y_train, y_pred_train)

print("")
print("sklearn's implementation on Linear Regression")
print("Training RMSE = " + str(rmse_train))
print("Training R2 = " + str(r2_train))

# Let us now evaluate on the test set
y_pred_test = lin_model.predict(X_test)
rmse_test = rmse(y_pred_test, y_test)
r2_test = r2_score(y_test, y_pred_test)
print("Test RMSE = " + str(rmse_test))
print("Test R2 = " + str(r2_test))

# Finally, let us see the learnt weights!
np.set_printoptions(precision=3)
print(lin_model.coef_)



print("")
print("My results on linear regression")

class MyLinearRegression:
    def __init__(self, l_rate = .000001, num_iterate = 3000):
        self.l_rate = l_rate
        self.num_iterate = num_iterate
        self.w = None
        self.b = 0

    def my_gradient_descent(self, X, y):
        num_points, num_features = X.shape
        self.w = np.zeros(num_features)
        self.b = 0
        M = len(X)

        for i in range(self.num_iterate):
            y_predicted = np.dot(X, self.w) + self.b

            # compute the gradients

            # 1/M * 2 * x^(m) * (y_predicted - y_actual)
            dw = (2 / M) * np.dot(X.T, (y_predicted - y))

            # 1/M * 2 * y_predicted - y_actual
            db = (2 / M) * np.sum(y_predicted - y)

            self.w = self.w - dw * self.l_rate
            self.b = self.b - db * self.l_rate

    def my_predict(self, X):
        return np.dot(X, self.w) + self.b


def my_r2_score(y_predicted, y_actual):
    y_actual_mean = np.mean(y_actual)
    my_SSR = np.sum((y_actual - y_predicted) ** 2)
    my_SST = np.sum((y_actual - y_actual_mean) ** 2)
    r2 = 1 - (my_SSR / my_SST)
    return r2


# function used to normalize my data
# using min max normalization
def my_min_max_normalization(X):
    min_num = np.min(X, axis = 0)
    max_num = np.max(X, axis = 0)
    x_normalized = (X - min_num) / (max_num - min_num)
    return x_normalized


my_lin_model = MyLinearRegression()
my_lin_model.my_gradient_descent(X_train, y_train)

my_y_pred_train = my_lin_model.my_predict(X_train)
my_rmse_train = rmse(my_y_pred_train, y_train)
my_r2_train = my_r2_score(my_y_pred_train, y_train)

print("My Training RMSE = " + str(my_rmse_train))
print("My Training R2 = " + str(my_r2_train))

my_y_pred_test = my_lin_model.my_predict(X_test)
my_rmse_test = rmse(my_y_pred_test, y_test)
my_r2_test = my_r2_score(my_y_pred_test, y_test)

print("My Test RMSE = " + str(my_rmse_test))
print("My Test R2 = " + str(my_r2_test))

# used for normalized output
X_train_norm = my_min_max_normalization(X_train)
X_test_norm = my_min_max_normalization(X_test)

my_lin_model_norm = MyLinearRegression()
my_lin_model_norm.my_gradient_descent(X_train_norm, y_train)

my_y_pred_train_norm = my_lin_model_norm.my_predict(X_train_norm)
my_rmse_train_norm = rmse(my_y_pred_train_norm, y_train)
my_r2_train_norm= my_r2_score(my_y_pred_train_norm, y_train)

print("My Training RMSE w/ norm = " + str(my_rmse_train_norm))
print("My Training R2 w/ norm = " + str(my_r2_train_norm))

my_y_pred_test_norm = my_lin_model_norm.my_predict(X_test_norm)
my_rmse_test_norm = rmse(my_y_pred_test_norm, y_test)
my_r2_test_norm= my_r2_score(my_y_pred_test_norm, y_test)

print("My Test RMSE w/ norm = " + str(my_rmse_test_norm))
print("My Test R2 w/ norm = " + str(my_r2_test_norm))