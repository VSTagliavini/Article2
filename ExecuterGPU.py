import subprocess
import numpy as np

#Arguments order
#GEMMK 0    - [0, 1]
#VWM 1      - [1, 2, 4, 8, 16]
#VWN 4      - [1, 2, 4, 8, 16]
#STRN 0     - [0, 1]
#STRM 0     - [0, 1]
#MWG <1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024>
#NWG <1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024>
#KWG <1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024>
#MDIMA <1, 2, 4, 8, 16, 32, 64, 128>
#MDIMC <1, 2, 4, 8, 16, 32, 64, 128>
#NDIMB <1, 2, 4, 8, 16, 32, 64, 128>
#NDIMC <1, 2, 4, 8, 16, 32, 64, 128>
#SA 0       - [0, 1]
#SB 0       - [0, 1]
#KWI 2      - [1, 2, 4, 8, 16, 32, 64]
#KREG 1     - [1, 2, 4, 8, 16, 32, 64]

parameters_limits = [1, 5, 5, 1, 1, 10, 10, 10, 7, 7, 7, 7, 1, 1, 6, 6]
class CLBlast():
    def __init__(self, file):
        self.file = file

    def SetDimensions(self, M, N, K):
        self.M = M
        self.N = N
        self.K = K

    def _convert_parameters(self, parameters):
        params = [int(round(parameters[i] * parameters_limits[i])) for i in range(len(parameters))]

        params[1] = 2 ** params[1]
        params[2] = 2 ** params[2]
        params[5] = 2 ** params[5]
        params[6] = 2 ** params[6]
        params[7] = 2 ** params[7]
        params[8] = 2 ** params[8]
        params[9] = 2 ** params[9]
        params[10]= 2 ** params[10]
        params[11]= 2 ** params[11]
        params[14] = 2 ** params[14]
        params[15] = 2 ** params[15]

        return params

    def Run(self, parameters):
        if (len(parameters) != 16):
            return -1
        if all(isinstance(parameter, (int, float, np.int64)) for parameter in parameters):
            parameters = self._convert_parameters(parameters)
            command = self.file + ' ' + str(self.M) + ' ' + str(self.N) + ' ' + str(self.K)
            for parameter in parameters:
                command = command + ' ' + str(parameter)

            result = subprocess.run(command.split(" "), capture_output=True)
            if result.returncode == 0:
                output = result.stdout.decode("utf-8").rstrip()
                output = output.split(" ")
                output = [float(a) for a in output]

                if output[0]:
                    return sum(output[1:]) / (len(output)-1)
                else:
                    return -1
            else:
                return -1


a = CLBlast('./GPUMeasurement')
a.SetDimensions(1024, 1024, 1024)

for _ in range(512):
    parameters = np.random.rand(16)
    print(parameters, end=' - ')
    print(a.Run(parameters))
