import ExecuterCPU
import Estimator
import Optimizer

import os

for algorithm in ExecuterCPU.paths.keys():
    polybench = ExecuterCPU.PolyBench(algorithm)

    def execution_time(parameters):
        compilation = polybench.Compile(parameters, algorithm)
        if compilation['error'] != 0:
            return 1e12
        execution = polybench.Run()
        if execution['error'] != 0:
            return 1e12
        return execution['time']
    
    try:
        os.remove('Estimations/execution_time')
    except:
        pass

    for estimator in Estimator.available_estimators:
        a = Estimator.Estimator(execution_time, 168, estimator)
        try:
            a._readfile(50)
        except Exception as e:
            a.BuildTrainingSet(50)
        del a
        for executions in [500, 250, 100, 50]:
            print(f'Algorithm: {algorithm} - Estimator: {estimator} - Executions: {executions}')
        
            bo = Optimizer.BO(execution_time, 168, algorithm, use_estimator=True, estimation_points=executions, estimation_alg=estimator)
            bo.Optimize(25, 1250)

            ps = Optimizer.PS(execution_time, 168, algorithm, use_estimator=False, estimation_points=executions, estimation_alg=estimator)
            ps.Optimize(25, 50)
        
            ga = Optimizer.GA(execution_time, 168, algorithm, use_estimator=False, estimation_points=executions, estimation_alg=estimator)
            ga.Optimize(25, 50)

            bh = Optimizer.BH(execution_time, 168, algorithm, use_estimator=False, estimation_points=executions, estimation_alg=estimator)
            bh.Optimize(50, 25)

            de = Optimizer.DE(execution_time, 168, algorithm, use_estimator=False, estimation_points=executions, estimation_alg=estimator)
            de.Optimize(25, 50)

            sa = Optimizer.SA(execution_time, 168, algorithm, use_estimator=False, estimation_points=executions, estimation_alg=estimator)
            sa.Optimize(1250)

            nm = Optimizer.NM(execution_time, 168, algorithm, use_estimator=False, estimation_points=executions, estimation_alg=estimator)
            nm.Optimize(1250)
