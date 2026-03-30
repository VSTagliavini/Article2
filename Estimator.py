import types

import Executer
import ANN

import numpy as np
import ast
import sklearn
import torch

available_estimators = ('SVM', 'DT', 'SGD', 'KNN', 'GP', 'GB', 'RF', 'MLP', 'RNN', 'GRU', 'LSTM')

class Estimator:
    def __init__(self, function, num_parameters, regressor):
        if type(function) != types.FunctionType:
            raise Exception(f'Expected function but received {type(function)}')
        self.function = function

        if not isinstance(num_parameters, int):
            raise Exception(f'num_parameters must be integer but received {type(num_parameters)}')
        if num_parameters <= 0:
            raise Exception(f'num_parameters must be greater than 0')
        self.num_parameters = num_parameters

        if regressor not in available_estimators:
            raise Exception(f'Regressor {regressor} is not implemented')
        self.regressor = regressor
    def _readfile(self, size):
        #reads a training set file and stores its entries
        self.parameters = []
        self.values = []

        with open(f'Estimations/{self.function.__name__}', 'r') as save_file:
            line = save_file.readline().rstrip()
            current_line = 0

            while line != "" and current_line < size:
                current_line = current_line + 1
                parameters, value = ast.literal_eval(f'{line}')

                self.parameters.append(parameters)
                self.values.append(value)

                line = save_file.readline().rstrip()
            if current_line < size:
                raise Exception('Not enough data')
    def BuildTrainingSet(self, size):
        #check if build set of that size exists, if not, creates it
        with open(f'Estimations/{self.function.__name__}', 'w') as save_file:
            for _ in range(size):
                parameters = np.random.rand(self.num_parameters)
                value = self.function(parameters)
                save_file.write(f'{parameters.tolist()}, {value}\n')
    def Fit(self, size):
        #fits the specified regressor to a specific number of function entries from a training set
        self._readfile(size)

        if self.regressor == 'SVM':
            self.reg = sklearn.pipeline.make_pipeline(
                    sklearn.preprocessing.StandardScaler(),
                    sklearn.svm.SVR(
                        kernel='linear',       #‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
                        degree=3,           #Only important for poly kernel
                        gamma='scale',      #Only important to rbf, poly and sigmoid kernels
                        coef0=0.0,          #Only important for poly and sigmoid kernels
                        tol=1e-3,
                        C=100,
                        epsilon=0.001
                    ),
                )

            self.reg.fit(self.parameters, self.values)
        elif self.regressor == 'SGD':
            self.reg = sklearn.pipeline.make_pipeline(
                    sklearn.preprocessing.StandardScaler(),
                    sklearn.linear_model.SGDRegressor(
                        max_iter=100,
                        penalty='l2',               #'l2', 'l1', 'elasticnet', None
                        alpha=0.001,
                        shuffle=True,
                        learning_rate='invscaling'  #constant, optimal, invscaling, adaptative, pa1, pa2
                    )
                )
            self.reg.fit(self.parameters, self.values)
        elif self.regressor == 'KNN':
            self.reg = sklearn.neighbors.KNeighborsRegressor(
                n_neighbors=5,
                weights='distance',         #uniform, distance
                algorithm='auto',           #BallTree, KDTree, brute, auto
                leaf_size=5,
                )
            self.reg.fit(self.parameters, self.values)
        elif self.regressor == 'DT':
            self.reg = sklearn.tree.DecisionTreeRegressor(
                criterion='poisson',
                max_depth=None,

            )
            self.reg.fit(self.parameters, self.values)
        elif self.regressor == 'GP':
            self.reg = sklearn.gaussian_process.GaussianProcessRegressor()
            self.reg.fit(self.parameters, self.values)
        elif self.regressor == 'GB':
            self.reg = sklearn.ensemble.GradientBoostingRegressor()
            self.reg.fit(self.parameters, self.values)
        elif self.regressor == 'RF':
            self.reg = sklearn.ensemble.RandomForestRegressor()
            self.reg.fit(self.parameters, self.values)
        elif self.regressor in ['MLP', 'RNN', 'GRU', 'LSTM']:
            tensor_parameters = torch.tensor(self.parameters, dtype=torch.float32)
            tensor_objective = torch.tensor(self.values, dtype=torch.float32)

            self.normalize_mean = tensor_objective.mean()
            self.normalize_std = tensor_objective.std()

            tensor_objective = (tensor_objective - tensor_objective.mean()) / tensor_objective.std()

            tensor_dataset = torch.utils.data.TensorDataset(tensor_parameters, tensor_objective)
            data_loader = torch.utils.data.DataLoader(tensor_dataset, batch_size=20, shuffle=True)

            if self.regressor == 'MLP':
                self.reg = ANN.MLP(self.num_parameters)
            elif self.regressor == 'RNN':
                self.reg = ANN.RNN(self.num_parameters)
            elif self.regressor == 'GRU':
                self.reg = ANN.GRU(self.num_parameters)
            elif self.regressor == 'LSTM':
                self.reg = ANN.LSTM(self.num_parameters)

            loss_func = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(self.reg.parameters(), lr=1e-4)

            if torch.cuda.is_available():
                self.reg.cuda()

            for i in range(100):
                for batch, (X, y) in enumerate(data_loader):
                    if torch.cuda.is_available():
                        X, y = X.cuda(), y.cuda()

                    optimizer.zero_grad()

                    estimate = self.reg(X)
                    estimate = estimate.squeeze(1)
                    loss = loss_func(estimate, y)

                    loss.backward()
                    optimizer.step()
    def Estimate(self, parameters):
        #uses the fitted regressor to estimate the function value in the given point
        parameters = np.clip(parameters, a_min=0, a_max=1)

        if self.regressor in ['SVM', 'DT', 'SGD', 'KNN', 'GP', 'GB', 'RF']:
            parameters = parameters.reshape(1, -1)
            return self.reg.predict(parameters)[0]
        elif self.regressor in ['MLP', 'RNN', 'GRU', 'LSTM']:
            X = torch.tensor(parameters, dtype=torch.float32)
            self.reg.eval()
            estimate = 0
            with torch.no_grad():
                if torch.cuda.is_available():
                    X = X.cuda()
                estimate = self.reg(X)
            return float(((estimate * self.normalize_std) + self.normalize_mean).item())