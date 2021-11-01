import pandas as pd
import numpy as np

def preprocess(train_data_name, test_data_name):
    train_data = pd.read_csv(train_data_name, index_col='Id')
    test_data = pd.read_csv(test_data_name, index_col='Id')

    ntrain = train_data.shape[0]
    ntest = test_data.shape[0]

    encoding_columns = ['MSSubClass']
    response_columns = ['SalePrice']

    for column_name in train_data.columns:
        if train_data[column_name].dtype == 'object':
            encoding_columns.append(column_name)

    map_dict = {}
    for column_name in train_data.columns:
        if column_name in response_columns:
            # log transformation of response variable
            train_data[column_name] = np.log(train_data[column_name] + 1)
        elif column_name not in encoding_columns:
            temp_mean = train_data.loc[~np.isnan(train_data[column_name]), column_name].mean()
            temp_std = train_data.loc[~np.isnan(train_data[column_name]), column_name].std()
            # normalize train data
            train_data.loc[np.isnan(train_data[column_name]), column_name] = temp_mean
            train_data[column_name] = (train_data[column_name] - temp_mean) / temp_std
            # normalize test data
            test_data.loc[np.isnan(test_data[column_name]), column_name] = temp_mean
            test_data[column_name] = (test_data[column_name] - temp_mean) / temp_std
        else:

            distinct_value = [str(value) for value in pd.concat([train_data[column_name], test_data[column_name]]).unique()]
            distinct_value.sort()
            num_distinct = len(distinct_value)
            map_dict[column_name] = {}
            for i, value in enumerate(distinct_value):
                map_dict[column_name][value] = [0 for _ in range(num_distinct - 1)]
                if i < num_distinct - 1:
                    map_dict[column_name][value][i] = 1

            temp_train_data = np.zeros([ntrain, num_distinct - 1])
            for i, index in enumerate(train_data.index):
                temp_train_data[i] = map_dict[column_name][str(train_data.loc[index, column_name])]

            temp_test_data = np.zeros([ntest, num_distinct - 1])
            for i, index in enumerate(test_data.index):
                temp_test_data[i] = map_dict[column_name][str(test_data.loc[index, column_name])]

            col_loc = train_data.columns.get_loc(column_name)
            for i in range(num_distinct - 1):
                train_data.insert(col_loc + i + 1, column_name + '{}'.format(i + 1), temp_train_data[:, i])
                test_data.insert(col_loc + i + 1, column_name + '{}'.format(i + 1), temp_test_data[:, i])
            train_data = train_data.drop(columns=column_name)
            test_data = test_data.drop(columns=column_name)

    return train_data, test_data



