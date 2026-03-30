import subprocess
import os

import numpy as np
import pyRAPL

#EXECUTAR PRIMEIRO
#sudo chmod o+r -R /sys/class/powercap/intel-rapl/

paths = {
    'correlation':      ['datamining/correlation', '/correlation.c'],
    'covariance':       ['datamining/covariance',  '/covariance.c'],
    'gemm':             ['linear-algebra/blas/gemm', '/gemm.c'],
    'gemver':           ['linear-algebra/blas/gemver', '/gemver.c'],
    'gesummv':          ['linear-algebra/blas/gesummv', '/gesummv.c'],
    'symm':             ['linear-algebra/blas/symm', '/symm.c'],
    'syr2k':            ['linear-algebra/blas/syr2k', '/syr2k.c'],
    'syrk':             ['linear-algebra/blas/syrk', '/syrk.c'],
    'trmm':             ['linear-algebra/blas/trmm', '/trmm.c'],
    '2mm':              ['linear-algebra/kernels/2mm', '/2mm.c'],
    '3mm':              ['linear-algebra/kernels/3mm', '/3mm.c'],
    'atax':             ['linear-algebra/kernels/atax', '/atax.c'],
    'bicg':             ['linear-algebra/kernels/bicg', '/bicg.c'],
    'doitgen':          ['linear-algebra/kernels/doitgen', '/doitgen.c'],
    'mvt':              ['linear-algebra/kernels/mvt', '/mvt.c'],
    'cholesky':         ['linear-algebra/solvers/cholesky', '/cholesky.c'],
    'durbin':           ['linear-algebra/solvers/durbin', '/durbin.c'],
    'gramschmidt':      ['linear-algebra/solvers/gramschmidt', '/gramschmidt.c'],
    'lu':               ['linear-algebra/solvers/lu', '/lu.c'],
    'ludcmp':           ['linear-algebra/solvers/ludcmp', '/ludcmp.c'],
    'trisolv':          ['linear-algebra/solvers/trisolv', '/trisolv.c'],
    'deriche':          ['medley/deriche', '/deriche.c'],
    'floyd-warshall':   ['medley/floyd-warshall', '/floyd-warshall.c'],
    'nussinov':         ['medley/nussinov', '/nussinov.c'],
    'adi':              ['stencils/adi', '/adi.c'],
    'fdtd-2d':          ['stencils/fdtd-2d', '/fdtd-2d.c'],
    'heat-3d':          ['stencils/heat-3d', '/heat-3d.c'],
    'jacobi-1d':        ['stencils/jacobi-1d', '/jacobi-1d.c'],
    'jacobi-2d':        ['stencils/jacobi-2d', '/jacobi-2d.c'],
    'seidel-2d':        ['stencils/seidel-2d', '/seidel-2d.c']
}

parameters_limits = [16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16,
                     3, 2, 2, 2, 1, 2, 1, 2, 1] + [1]*150
#19
M = [0, 2, 6, 14]
variable_flags = {
    '-falign-functions': lambda n,m,n2,m2: '-falign-functions=' + str(2**n) + ':' + str(2**m) + ':' + str(2**n2) + ':' + str(2**m2),
    '-falign-jumps': lambda n,m,n2,m2: '-falign-jumps=' + str(2**n) + ':' + str(2**m) + ':' + str(2**n2) + ':' + str(2**m2),
    '-falign-labels': lambda n,m,n2,m2: '-falign-labels=' + str(2**n) + ':' + str(2**m) + ':' + str(2**n2) + ':' + str(2**m2),
    '-falign-loops': lambda n,m,n2,m2: '-falign-loops=' + str(2**n) + ':' + str(2**m) + ':' + str(2**n2) + ':' + str(2**m2), 
    '-fsched-stalled-insns': lambda n: '-fsched-stalled-insns' if n == 0 else '-fsched-stalled-insns=' + str(n-1),
    '-fsched-stalled-insns-dep': lambda n: '-fsched-stalled-insns-dep=' + str(n),
    '-ftree-parallelize-loops': lambda n: '-ftree-parallelize-loops=' + str(n)
    }

#9
multivalue_flags = {
    '-fvect-cost-model': {0: '-fvect-cost-model=unlimited', 1: '-fvect-cost-model=dynamic', 2: '-fvect-cost-model=cheap', 3: '-fvect-cost-model=very-cheap'},
    '-flifetime-dse': {0: '-flifetime-dse=0', 1: '-flifetime-dse=1', 2: '-flifetime-dse=2'},
    '-ftrivial-auto-var-init': {0: '-ftrivial-auto-var-init=uninitialized', 1: '-ftrivial-auto-var-init=pattern', 2: '-ftrivial-auto-var-init=zero'},
    '-ffp-contract': {0: '-ffp-contract=off', 1: '-ffp-contract=on', 2: '-ffp-contract=fast'},
    '-fira-algorithm': {0: '-fira-algorithm=CB', 1: '-fira-algorithm=priority'},
    '-fexcess-precision': {0: '-fexcess-precision=fast', 1: '-fexcess-precision=standard', 2: '-fexcess-precision=16'},
    '-freorder-blocks-algorithm': {0: '-freorder-blocks-algorithm=simple', 1: '-freorder-blocks-algorithm=stc'},
    '-fira-region': {0: '-fira-region=one', 1: '-fira-region=all', 2: '-fira-region=mixed'},
    '-flive-patching': {0: '-flive-patching=inline-only-static', 1: '-flive-patching=inline-clone'},
}

#150
#ffoo or fno-foo
binary_flags = ['-ftree-vrp', '-fsched-group-heuristic', '-floop-nest-optimize', '-fsplit-wide-types', '-fira-loop-pressure', '-fpeel-loops', '-fsemantic-interposition', '-ftree-vectorize', '-fisolate-erroneous-paths-dereference', '-fauto-inc-dec', '-fstrict-aliasing', '-fprefetch-loop-arrays', '-fpredictive-commoning', '-ftree-copy-prop', '-fsplit-paths', '-fsplit-wide-types-early', '-ftree-ch', '-fsched-spec-insn-heuristic', '-ffinite-math-only', '-ftree-tail-merge', '-fthread-jumps', '-fassociative-math', '-ftree-pta', '-ftree-loop-im', '-ffloat-store', '-funsafe-math-optimizations', '-fssa-phiopt', '-fsched-spec-load-dangerous', '-fdelete-null-pointer-checks', '-fweb', '-fschedule-insns2', '-fshrink-wrap-separate', '-ffinite-loops', '-fsignaling-nans', '-ffp-int-builtin-inexact', '-fearly-inlining', '-fira-hoist-pressure', '-fomit-frame-pointer', '-fhoist-adjacent-loads', '-frerun-cse-after-loop', '-fipa-strict-aliasing', '-ftree-dse', '-fcx-fortran-rules', '-ftree-loop-if-convert', '-ftree-loop-distribute-patterns', '-ftree-ccp', '-fcode-hoisting', '-finline-atomics', '-fsched-pressure', '-fisolate-erroneous-paths-attribute', '-fprofile-partial-training', '-fgcse-lm', '-ftree-forwprop', '-ftree-dce', '-fif-conversion', '-findirect-inlining', '-fvariable-expansion-in-unroller', '-fsplit-loops', '-fivopts', '-fdevirtualize-speculatively', '-frename-registers', '-fdce', '-ftree-partial-pre', '-fmove-loop-invariants', '-fvpt', '-freorder-blocks-and-partition', '-fmove-loop-stores', '-finline-small-functions', '-ftree-sink', '-ftree-builtin-call-dce', '-ftree-loop-ivcanon', '-fshrink-wrap', '-fprofile-reorder-functions', '-ftree-loop-optimize', '-freorder-functions', '-fcprop-registers', '-fdevirtualize', '-funroll-loops', '-fallow-store-data-races', '-ftree-fre', '-fcompare-elim', '-ftree-loop-distribution', '-fcx-limited-range', '-funconstrained-commons', '-free', '-funswitch-loops', '-ftree-dominator-opts', '-fsel-sched-pipelining-outer-loops', '-fsingle-precision-constant', '-fselective-scheduling', '-fsched-spec-load', '-ftree-cselim', '-fsched-dep-count-heuristic', '-freorder-blocks', '-fgcse-after-reload', '-finline-functions-called-once', '-ftracer', '-fgcse-las', '-fmodulo-sched-allow-regmoves', '-fmodulo-sched', '-ftree-switch-conversion', '-fssa-backprop', '-freschedule-modulo-scheduled-loops', '-fsave-optimization-record', '-ftree-scev-cprop', '-fcaller-saves', '-fcse-follow-jumps', '-fipa-profile', '-frounding-math', '-fgcse', '-faggressive-loop-optimizations', '-foptimize-sibling-calls', '-fcombine-stack-adjustments', '-fgraphite-identity', '-ftree-loop-vectorize', '-floop-parallelize-all', '-fconserve-stack', '-fsel-sched-pipelining', '-fsched-last-insn-heuristic', '-ftree-bit-ccp', '-finline-functions', '-flra-remat', '-fselective-scheduling2', '-fsched2-use-superblocks', '-fdse', '-ftree-coalesce-vars', '-flive-range-shrinkage', '-fsched-rank-heuristic', '-funroll-all-loops', '-fschedule-fusion', '-fgcse-sm', '-fstdarg-opt', '-ftree-slsr', '-ftree-sra', '-fcrossjumping', '-fsplit-ivs-in-unroller', '-ftree-ter', '-floop-interchange', '-fforward-propagate', '-fsched-critical-path-heuristic', '-fstore-merging', '-ftree-pre', '-floop-unroll-and-jam', '-freciprocal-math', '-ftree-phiprop', '-fschedule-insns', '-fif-conversion2', '-flimit-function-alignment', '-ftree-reassoc', '-fexpensive-optimizations']
# '-fbranch-probabilities',

    #'-fipa-sra',
    #'-fipa-cp',
    #'-fipa-pta',
    #'-fipa-reference',
    #'-fipa-ra',
    #'-fipa-icf',
    #'-fipa-icf-variables',
    #'-fipa-bit-cp',
    #'-fipa-modref',
    #'-fipa-cp-clone',
    #'-fipa-icf-functions',
    #'-fipa-vrp',
    #'-fipa-stack-alignment',
    #'-fipa-reference-addressable',
    #'-fipa-pure-const',
    #'-fpartial-inlining',

class PolyBench:
    def __init__(self, program):
        pyRAPL.setup(socket_ids=[0])
        self.path = paths[program]

    def save_parameters(self, parameters, file):
        #Convert parameters to strings and add them to options.txt
        parameters = [round(parameters[i]*parameters_limits[i]) for i in range(168)]

        with open('options/'+file, 'w') as FILE:

            flags = ''

            #Variable flags
            flags += variable_flags['-falign-functions']         (parameters[0], parameters[1], parameters[2], parameters[3]) + '\n'
            flags += variable_flags['-falign-jumps']             (parameters[4], parameters[5], parameters[6], parameters[7]) + '\n'
            flags += variable_flags['-falign-labels']            (parameters[8], parameters[9], parameters[10], parameters[11]) + '\n'
            flags += variable_flags['-falign-loops']             (parameters[12], parameters[13], parameters[14], parameters[15]) + '\n'
            flags += variable_flags['-fsched-stalled-insns']     (parameters[19]) + '\n'
            flags += variable_flags['-fsched-stalled-insns-dep'] (parameters[20]) + '\n'
            flags += variable_flags['-ftree-parallelize-loops']  (parameters[21]) + '\n'

            flags += '\n'
            #Multivalue flags
            i = 19
            for flag in multivalue_flags:
                try: 
                    flags += multivalue_flags[flag][parameters[i]] + '\n'
                except:
                    print(f'Error: flag:{flag} parameter:{parameters[i]}')
                i += 1

            flags += '\n'

            #Binary flags
            for flag in binary_flags:
                if parameters[i-19] == 0:
                    flags += flag[0:2]+ 'no-' + flag[2:] + '\n'
                else:
                    flags += flag + '\n'
                i += 1
            
            print(flags, file=FILE)

    #Exemplo de comando:
    #gcc @options.txt -I PolyBenchC/datamining/correlation PolyBenchC/datamining/correlation/correlation.c -lm -o atax_time



    #gcc -O3 -I PolyBenchC/utilities -I PolyBenchC/' + self.path[0] + ' PolyBenchC/utilities/polybench.c PolyBenchC/' + self.path[0] + self.path[1] + ' -DPOLYBENCH_TIME -DPOLYBENCH_DUMP_ARRAYS -LARGE_DATASET -lm -o atax_time
    def Compile(self, parameters, file):
        self.save_parameters(parameters, file)

        measure = pyRAPL.Measurement('bar')
        measure.begin()
        result = os.system('gcc @options/' + file + ' -I PolyBenchC/utilities -I PolyBenchC/' + self.path[0] + ' PolyBenchC/utilities/polybench.c PolyBenchC/' + self.path[0] + self.path[1] + ' -DPOLYBENCH_TIME -DPOLYBENCH_DUMP_ARRAYS -LARGE_DATASET -lm -o atax_time' )

        measure.end()
        
        if result != 0:
            return {'energy': -1, 'time': -1, 'error': result}
        try:
            a = {'energy': measure.result.pkg[0], 'time': measure.result.duration, 'error': 0}
            return a
        except Exception as e:
            return {'energy': -1, 'time': -1, 'error': -1}
    
    def Run(self):
        measure = pyRAPL.Measurement('bar')
        measure.begin()
        result = subprocess.run('./atax_time', capture_output=True)
        measure.end()

        time = float(result.stdout.decode("utf-8").rstrip())*1000

        if result.returncode != 0:
            return {'energy': -1, 'time': -1, 'error': result.stderr.decode("utf-8").rstrip()}

        if time == 0:
            return {'energy': -1, 'time': -1, 'error': -1}

        try:
            a = {'energy': measure.result.pkg[0], 'time': time, 'error': 0}
            return a
        except Exception as e:
            return {'energy': -1, 'time': -1, 'error': -1}

if __name__ == "__main__":

    program = 'correlation'
    a = PolyBench(program)
    params = [0.019488291766242116, 0.36648171038771127, 0.15555989666469616, 0.5213252080632927, 0.016157305714852743, 0.45077787929931346, 0.04171719109234273, 0.6413681483555388, 0.1914818983973111, 0.38916793248169856, 0.7204147998187098, 0.6439882496314782, 0.12423793297307051, 0.2251390122891076, 0.5897462872050593, 0.32343856727155695, 0.1267904315070595, 0.7667766590642346, 0.676889098455943, 0.4139257230673057, 0.6319040826276116, 0.5510223688564028, 0.8328716873103708, 0.5046206103154274, 0.22932280822942153, 0.3394060812593044, 0.016988759403256948, 0.15579139446766677, 0.3600371701979852, 0.6826941680425688, 0.5541191908519759, 0.562208382349828, 0.46584939446788, 0.24323206805848085, 0.3155810545982739, 0.8113677450466139, 0.10487557555820837, 0.1139914520115165, 0.5971407969721824, 0.0848153639945336, 0.9414584916245698, 0.11530342124134052, 0.819998535043559, 0.6278352762199695, 0.32089125768338245, 0.5303621174511071, 0.6559005128313389, 0.07495061017824467, 0.8980583436641184, 0.987410593541849, 0.6411891502630332, 0.5546893586502032, 0.7874108572048246, 0.036505728583342, 0.3302318915600214, 0.9358770984862544, 0.919268395545782, 0.04361179492162526, 0.25681010121899983, 0.14452897678514187, 0.9829623295633672, 0.860194252964159, 0.36843908075097187, 0.8068029053034312, 0.003642699999827248, 0.8278861664336492, 0.04250690440855631, 0.07702082754449768, 0.3234749042084346, 0.7859267515544471, 0.5864679835008738, 0.6286392007806012, 0.09668496090405088, 0.3863489692490596, 0.4158430089505847, 0.03141740735424747, 0.4640228590272566, 0.22490215592818552, 0.7280937053297983, 0.7393070241907773, 0.5604161875775732, 0.42351502693349197, 0.08689159223703014, 0.05670280453601939, 0.1582666679484056, 0.7664877974526924, 0.7868085120444156, 0.803869646571499, 0.991409524507494, 0.7124684579351577, 0.8303079540691964, 0.8172454249561675, 0.567824336992123, 0.004049279068064626, 0.5590074854593086, 0.9760786846460793, 0.4032864704082114, 0.48619992467246076, 0.6678440790274373, 0.5066325149813867, 0.21621865347881708, 0.5587670057398697, 0.43157724403909725, 0.17615395505041853, 0.3353788999143561, 0.5006323061008621, 0.6626577955897615, 0.6795915605698619, 0.431819886360695, 0.6865695248245556, 0.4604982973175691, 0.42141775939997894, 0.7426126801665032, 0.4523714189709195, 0.7720030230592185, 0.05090531802248921, 0.7407882726451854, 0.6817804376025349, 0.09845279822430653, 0.6426035908908343, 0.4240268101405995, 0.1951681694772609, 0.6707197783192452, 0.6477991659446172, 0.3442496503831526, 0.7831476149275233, 0.7037061260319017, 0.47816661375279224, 0.29648129135213497, 0.29474856879316114, 0.02366357865664359, 0.2314132623135966, 0.03878037495661668, 0.32641280560146424, 0.5329615944782691, 0.16061656862714024, 0.7739687639333712, 0.6880495217897034, 0.005425538263667096, 0.650695690819546, 0.9000389015360359, 0.7786406200257803, 0.8304699457789545, 0.8484660035968645, 0.5062661501684996, 0.25155840708514565, 0.7822062768943755, 0.2936458640414885, 0.5326333652312601, 0.3839915726859947, 0.8516469012145724, 0.9297496089716112, 0.24938943814730063, 0.20408104868823684, 0.6659869439870799, 0.8275434181576111, 0.9079305116054641, 0.588664517983385, 0.19076485174878655, 0.009322189023380778, 0.5143801223356796, 0.22656199837964996, 0.04832185242675224, 0.4104977996994956, 0.9298376030708283, 0.2741993339634008, 0.39745762719022937, 0.24178254080516937]
    compilation = a.Compile(params, program)
    print(compilation)
    print(a.Run())


    for program in paths.keys():
        print(program, end=': ')
        a = PolyBench(program)
        
        params = np.random.rand(168)
        compilation = a.Compile(params, program)
        print(f'Compilation energy: {compilation["energy"]} uJ, Compilation duration: {compilation["time"]} us')
        
        # Now run multiple times with optimized binary
        for i in range(4):
            run = a.Run()
            print(run, end='; ')
        print()
