import types
import sys
import json

import numpy as np
import pygad
import pyswarms
import scipy
import skopt

import Estimator

class Optimizer:
    def __init__(self, function, num_parameters, algorithm, use_estimator=False, estimation_points=0, estimation_alg='DT'):
        if type(function) != types.FunctionType:
            raise Exception(f'Expected function but received {type(function)}')
        self.func = function

        if not isinstance(num_parameters, int):
            raise Exception(f'num_parameters must be integer but received {type(num_parameters)}')
        if num_parameters <= 0:
            raise Exception(f'num_parameters must be greater than 0')
        self.num_parameters = num_parameters

        self.algorithm = algorithm

        if use_estimator:
            if not isinstance(estimation_points, int):
                raise Exception(f'estimation_points must be integer but received {type(estimation_points)}')
            if estimation_points <= 0:
                raise Exception(f'estimation_points must be greater than 0')
            if estimation_alg not in Estimator.available_estimators:
                raise Exception(f'Algorithm {estimation_alg} is not supported')
        self.use_estimator = use_estimator
        self.estimation_points = estimation_points
        self.estimation_alg = estimation_alg

        self.history = []
        self.best_result = sys.maxsize
        self.best_parameters = None

        if self.use_estimator:
            self.estimator = Estimator.Estimator(self.func, self.num_parameters, self.estimation_alg)

            try:
                self.estimator.Fit(self.estimation_points)
            except Exception as e:
                self.estimator.BuildTrainingSet(self.estimation_points)
                self.estimator.Fit(self.estimation_points)

class BO(Optimizer):
    def __init__(self, function, num_parameters, algorithm, use_estimator=False, estimation_points=0, estimation_alg='DT'):
        super().__init__(function, num_parameters, algorithm, use_estimator, estimation_points, estimation_alg)

        self.bounds = {}
        for i in range(num_parameters):
            self.bounds[f'param_{i}'] = (0, 1)

    def optimization_function(self, x):
        if len(self.history) > self.total_executions:
            raise Exception('max executions')
        if self.use_estimator:
            value = self.estimator.Estimate(x)
        else:
            value = self.func(x)
        self.history.append(value)

        if value < self.best_result:
            self.best_result = value
            self.best_parameters = x

        return value

    def Optimize(self, init_points, search_points):
        self.total_executions = search_points

        try:
            res = skopt.gp_minimize(
                self.optimization_function,
                dimensions=[skopt.space.Real(0, 1)] * self.num_parameters,
                n_calls=init_points + search_points,
                n_random_starts=init_points
            )
        except Exception as e:
        #self.optimizer = bayes_opt.BayesianOptimization(f=self.optimization_function, pbounds=self.bounds, verbose=0)
        #self.optimizer.maximize(n_iter=search_points, init_points=init_points)

            model = {
                    'sample': self.total_executions,
                    'best_time': self.best_result,
                    'best_solution': self.best_parameters,
                    'history': self.history
                }

            with open(f'./Saves/BO/{self.algorithm}_{self.func.__name__}-a{self.estimation_alg}{self.estimation_points}.json', 'w') as save_file:
                json.dump(model, save_file)

class PS(Optimizer):
    def __init__(self, function, num_parameters, algorithm, use_estimator=False, estimation_points=0, estimation_alg='DT'):
        super().__init__(function, num_parameters, algorithm, use_estimator, estimation_points, estimation_alg)
    
    def optimization_function(self, x):
        X = np.array(x)

        if self.use_estimator:
            if X.ndim == 2:
                costs = []
                for particle in X:
                    value = self.estimator.Estimate(particle)
                    self.history.append(value)
                    costs.append(value)

                    if value < self.best_result:
                        self.best_result = value
                        self.best_parameters = particle.tolist()
                return costs
            else:
                value = self.estimator.Estimate(X)
                self.history.append(value)
                costs.append(value)

                if value < self.best_result:
                    self.best_result = value
                    self.best_parameters = particle.tolist()
                return value
        else:
            if X.ndim == 2:
                costs = []
                for particle in X:
                    value = self.func(particle)
                    self.history.append(value)
                    costs.append(value)

                    if value < self.best_result:
                        self.best_result = value
                        self.best_parameters = particle.tolist()
                return costs
            else:
                value = self.func(X)
                self.history.append(value)
                costs.append(value)

                if value < self.best_result:
                    self.best_result = value
                    self.best_parameters = particle.tolist()
                return value
        

    def Optimize(self, num_particles, num_generations):
        self.total_executions = num_particles * num_generations
        optimizer =  pyswarms.single.LocalBestPSO(
                n_particles=num_particles,
                dimensions=self.num_parameters,
                options={
                    'c1': 0.6,
                    'c2': 0.5,
                    'w': 0.7,
                    'k': 2,
                    'p': 2
                },
                bounds=(np.array([0]*self.num_parameters), np.array([1]*self.num_parameters),)
            )
        cost, pos = optimizer.optimize(self.optimization_function, num_generations)

        model = {
                'sample': self.total_executions,
                'best_time': self.best_result,
                'best_solution': self.best_parameters,
                'history': self.history
            }

        with open(f'./Saves/PS/{self.algorithm}_{self.func.__name__}-a{self.estimation_alg}{self.estimation_points}.json', 'w') as save_file:
            json.dump(model, save_file)

class GA(Optimizer):
    def __init__(self, function, num_parameters, algorithm, use_estimator=False, estimation_points=0, estimation_alg='DT'):
        super().__init__(function, num_parameters, algorithm, use_estimator, estimation_points, estimation_alg)
        self.gene_space = []
        for _ in range(num_parameters):
            self.gene_space.append({'low': 0, 'high':1})

    def optimization_function(self, GA, x, idx):
        if self.use_estimator:
            value = self.estimator.Estimate(x)
        else:
            value = self.func(x)      
        self.history.append(value)

        if value < self.best_result:
            self.best_result = value
            self.best_parameters = x
        return 1000000/value
    
    def Optimize(self, num_individuals, num_generations):
        self.total_executions = num_individuals * num_generations
        if num_individuals % 2 == 0:
            mating = num_individuals/2
        else:
            mating = round((num_individuals+1)/2)
        optimizer = pygad.GA(
            fitness_func=self.optimization_function,
            num_generations=num_generations,
            sol_per_pop=num_individuals,
            num_parents_mating=mating,
            num_genes=self.num_parameters, 
            gene_space=self.gene_space, 
            gene_type=float
        )
        optimizer.run()

        model = {
            'sample': self.total_executions,
            'best_time': self.best_result,
            'best_solution': self.best_parameters.tolist(),
            'history': self.history
        }

        with open(f'./Saves/GA/{self.algorithm}_{self.func.__name__}-a{self.estimation_alg}{self.estimation_points}.json', 'w') as save_file:
            json.dump(model, save_file)

class BH(Optimizer):
    def __init__(self, function, num_parameters, algorithm, use_estimator=False, estimation_points=0, estimation_alg='DT'):
        super().__init__(function, num_parameters, algorithm, use_estimator, estimation_points, estimation_alg)

        self.bounds = [[0,1]] * self.num_parameters
    
    def optimization_function(self, x):
        if len(self.history) > self.total_executions:
            raise Exception('max executions')
        if self.use_estimator:
            value = self.estimator.Estimate(x)
        else:
            value = self.func(x)
        self.history.append(value)

        if value < self.best_result:
            self.best_result = value
            self.best_parameters = x
        return value

    def Optimize(self, global_search, local_search):
        self.total_executions = global_search*local_search
        try:
            optimal = scipy.optimize.basinhopping(
                self.optimization_function,
                np.random.rand(self.num_parameters),
                niter=global_search,
                minimizer_kwargs={
                    'method': 'L-BFGS-B',
                    'bounds': [[0,1]]*self.num_parameters,
                    'options': {
                        'maxiter': local_search,
                        'maxfun': local_search
                    }
                })
            raise Exception('finished execution')
        except Exception as e:
            model = {
                'sample': self.total_executions,
                'best_time': self.best_result,
                'best_solution': self.best_parameters.tolist(),
                'history': self.history
            }

            with open(f'./Saves/BH/{self.algorithm}_{self.func.__name__}-a{self.estimation_alg}{self.estimation_points}.json', 'w') as save_file:
                json.dump(model, save_file)

class DE(Optimizer):
    def __init__(self, function, num_parameters, algorithm, use_estimator=False, estimation_points=0, estimation_alg='DT'):
        super().__init__(function, num_parameters, algorithm, use_estimator, estimation_points, estimation_alg)

        self.bounds = [[0,1]] * self.num_parameters

    def optimization_function(self, x):
        if len(self.history) > self.total_executions:
            raise Exception('max executions')
        if self.use_estimator:
            value = self.estimator.Estimate(x)
        else:
            value = self.func(x)
        self.history.append(value[0].item())

        if value < self.best_result:
            self.best_result = value
            self.best_parameters = x
        return value
    
    def Optimize(self, num_individuals, num_generations):
        self.total_executions = num_individuals * num_generations

        initial_population = np.random.rand(num_individuals, self.num_parameters).tolist()

        try:
            optimal = scipy.optimize.differential_evolution(
                self.optimization_function,
                self.bounds,
                maxiter=num_generations,
                popsize=num_individuals,
                init=initial_population,
                mutation=[0.1, 1.5],
                recombination=[0.3])
            raise Exception('finished execution')
        except Exception as e:
            model = {
                'sample': self.total_executions,
                'best_time': self.best_result,
                'best_solution': self.best_parameters,
                'history': self.history
            }

            with open(f'./Saves/DE/{self.algorithm}_{self.func.__name__}-a{self.estimation_alg}{self.estimation_points}.json', 'w') as save_file:
                json.dump(model, save_file)

class SA(Optimizer):
    def __init__(self, function, num_parameters, algorithm, use_estimator=False, estimation_points=0, estimation_alg='DT'):
        super().__init__(function, num_parameters, algorithm, use_estimator, estimation_points, estimation_alg)

        self.bounds = [[0,1]] * self.num_parameters

    def optimization_function(self, x):
        if len(self.history) > self.total_executions:
            raise Exception('max executions')
        if self.use_estimator:
            value = self.estimator.Estimate(x)
        else:
            value = self.func(x)
        self.history.append(value[0].item())

        if value < self.best_result:
            self.best_result = value
            self.best_parameters = x
        return value
    
    def Optimize(self, iterations):
        self.total_executions = iterations
        try:
            optimal = scipy.optimize.dual_annealing(
                self.optimization_function,
                self.bounds,
                maxfun=self.total_executions,
                no_local_search=True
            )
            raise Exception('finished execution')
        except Exception as e:
            model = {
                'sample': self.total_executions,
                'best_time': self.best_result,
                'best_solution': self.best_parameters,
                'history': self.history
            }

            with open(f'./Saves/SA/{self.algorithm}_{self.func.__name__}-a{self.estimation_alg}{self.estimation_points}.json', 'w') as save_file:
                json.dump(model, save_file)

class NM(Optimizer):
    def __init__(self, function, num_parameters, algorithm, use_estimator=False, estimation_points=0, estimation_alg='DT'):
        super().__init__(function, num_parameters, algorithm, use_estimator, estimation_points, estimation_alg)

        self.bounds = [[0,1]] * self.num_parameters

    def optimization_function(self, x):
        if len(self.history) > self.total_executions:
            raise Exception('max executions')
        if self.use_estimator:
            value = self.estimator.Estimate(x)
        else:
            value = self.func(x)
        self.history.append(value[0].item())

        if value < self.best_result:
            self.best_result = value
            self.best_parameters = x
        return value
    
    def Optimize(self, iterations):
        self.total_executions = iterations
        try:
            optimal = optimal = scipy.optimize.minimize(
                method='Nelder-Mead',
                fun=self.optimization_function,
                x0=np.random.random(self.num_parameters),
                bounds=self.bounds,
                options={
                     'maxiter': self.total_executions,
                     'maxfev': self.total_executions
                }
        )
            raise Exception('finished execution')
        except Exception as e:
            model = {
                'sample': self.total_executions,
                'best_time': self.best_result,
                'best_solution': self.best_parameters,
                'history': self.history
            }

            with open(f'./Saves/NM/{self.algorithm}_{self.func.__name__}-a{self.estimation_alg}{self.estimation_points}.json', 'w') as save_file:
                json.dump(model, save_file)

