from preprocess import preprocess
from sklearn.linear_model import Lasso
import numpy as np
import pandas as pd

seed = 1
np.random.seed(seed)
train_data_name = 'train.csv'
test_data_name = 'test.csv'
train_data, test_data = preprocess(train_data_name, test_data_name)
train_data = np.mat(train_data)

x_train = train_data[:, 0:-1]
y_train = train_data[:, -1]
x_test = np.mat(test_data)

lasso_lambda = 1e-4
reg = Lasso(alpha=lasso_lambda, max_iter=-1).fit(x_train, y_train)

y_test_predict = np.exp(reg.predict(x_test)) - 1

test_predict_df = pd.DataFrame({'Id':test_data.index, 'SalePrice':y_test_predict})

submit_file_name = 'lasso_submission.csv'
test_predict_df.to_csv(path_or_buf=submit_file_name, index = False)