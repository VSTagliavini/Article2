import ExecuterGPU
import Estimator
import Optimizer

import os

a = ExecuterGPU.CLBlast('./GPUMeasurement')
a.SetDimensions(1024, 1024, 1024)

def clblast(parameters):
    result = a.Run(parameters)
    if result == -1:
        return 1e12
    return result

try:
    os.remove('Estimations/clblast')
except:
    pass

for estimator in Estimator.available_estimators:
    for executions in [500, 250, 100, 50]:
        print(f'Estimator: {estimator} - Executions: {executions}')
    
        bo = Optimizer.BO(clblast, 16, 'CLBlast', use_estimator=True, estimation_points=executions, estimation_alg=estimator)
        bo.Optimize(25, 50)

        ps = Optimizer.PS(clblast, 16, 'CLBlast', use_estimator=True, estimation_points=executions, estimation_alg=estimator)
        ps.Optimize(25, 50)
        
        ga = Optimizer.GA(clblast, 16, 'CLBlast', use_estimator=True, estimation_points=executions, estimation_alg=estimator)
        ga.Optimize(25, 50)

        bh = Optimizer.BH(clblast, 16, 'CLBlast', use_estimator=True, estimation_points=executions, estimation_alg=estimator)
        bh.Optimize(50, 25)

        de = Optimizer.DE(clblast, 16, 'CLBlast', use_estimator=True, estimation_points=executions, estimation_alg=estimator)
        de.Optimize(25, 50)

        sa = Optimizer.SA(clblast, 16, 'CLBlast', use_estimator=True, estimation_points=executions, estimation_alg=estimator)
        sa.Optimize(1250)

        nm = Optimizer.NM(clblast, 16, 'CLBlast', use_estimator=True, estimation_points=executions, estimation_alg=estimator)
        nm.Optimize(1250)