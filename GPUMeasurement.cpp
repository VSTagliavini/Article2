#include <CL/cl.h>
#include <clblast.h>
#include <unordered_map>
#include <string>
#include <iostream>
#include <chrono>
#include <vector>
//g++ GPUMeasurement.cpp -o GPUMeasurement -lclblast -lOpenCL

float alpha, beta;

cl_platform_id platform;
cl_device_id device;
cl_context context;
cl_command_queue queue;

std::vector<float> A;
std::vector<float> B;
std::vector<float> C;
std::vector<float> R;

std::unordered_map<std::string, size_t> params = {
    {"GEMMK", 0},
    {"VWM", 2},
    {"VWN", 4},
    {"STRN", 0},
    {"STRM", 0},
    {"MWG", 32},
    {"NWG", 64},
    {"KWG", 32},
    {"MDIMA", 8},
    {"MDIMC", 8},
    {"NDIMB", 8},
    {"NDIMC", 8},
    {"SA", 0},
    {"SB", 0},
    {"KWI", 2},
    {"KREG", 1}};

void InitializeOpenCL() {
    clGetPlatformIDs(1, &platform, nullptr);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, 1, &device, nullptr);
    context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);
    queue = clCreateCommandQueue(context, device, 0, nullptr);
}

void InitializeMatrixes(int M, int N, int K) {
    alpha = rand() % 16;
    beta = rand() % 16;
    for (size_t row = 0; row < M; ++row) {
        for (size_t col = 0; col < K; ++col) {
            A[row * K + col] = rand() % 16;
        }
    }
    for (size_t row = 0; row < K; ++row) {
        for (size_t col = 0; col < N; ++col) {
            B[row * N + col] = rand() % 16;
        }
    }
    for (size_t row = 0; row < M; row++) {
        for (size_t col = 0; col < N; col++) {
            C[row*N+col] = rand() % 16;
        }
    }
}

int main(int argc, char **argv) {
    if (argc == 1) {
        std::cout << "No arguments provided.\n";
        return 1;
    }

    //Matrix sizes
    size_t M = std::stoi(argv[1]);
    size_t N = std::stoi(argv[2]);
    size_t K = std::stoi(argv[3]);
    srand(0);

    A.resize(M * K, 0.0f);
    B.resize(K * N, 0.0f);
    C.resize(M * N, 0.0f);
    R.resize(M * N, 0.0f);
    

    InitializeMatrixes(M, N, K);
    InitializeOpenCL();

    //Copies parameters specified
    std::vector<std::string> parameter_names = {"GEMMK", "VWM", "VWN", "STRN", "STRM", "MWG", "NWG", "KWG", "MDIMA", "MDIMC", "NDIMB", "NDIMC", "SA", "SB", "KWI", "KREG"};
    for (int i=4; i<argc; i++) {
        params.find(parameter_names[i-4])->second = std::stoi(argv[i]);
    }
    
    cl_mem dA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * A.size(), A.data(), nullptr);
    cl_mem dB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * B.size(), B.data(), nullptr);
    cl_mem dC = clCreateBuffer(context, CL_MEM_READ_WRITE| CL_MEM_COPY_HOST_PTR, sizeof(float) * C.size(), C.data(), nullptr);

    clEnqueueWriteBuffer(queue, dA, CL_TRUE, 0, sizeof(float) * A.size(), (void *) A.data(), 0, NULL, NULL );
    clEnqueueWriteBuffer(queue, dB, CL_TRUE, 0, sizeof(float) * B.size(), (void *) B.data(), 0, NULL, NULL );
    clEnqueueWriteBuffer(queue, dC, CL_TRUE, 0, sizeof(float) * C.size(), (void *) C.data(), 0, NULL, NULL );

    //Executes kernel once with default parameters to get a correct value for C
    auto st = clblast::Gemm(
            clblast::Layout::kRowMajor,
            clblast::Transpose::kNo,
            clblast::Transpose::kNo,
            M, N, K,
            alpha, dA, 0, K, dB, 0, N,
            beta, dC, 0, N,
            &queue);
    clEnqueueReadBuffer(queue, dC, CL_TRUE, 0, sizeof(float) * R.size(), R.data(), 0, nullptr, nullptr);

    auto status = clblast::OverrideParameters(device, "Xgemm", clblast::Precision::kSingle, params);
    if (status != clblast::StatusCode::kSuccess) {
        std::cerr << "OverrideParameters failed: " << static_cast<int>(status) << "\n";
        return 1;
    }

    //Runs the kernel 64 times to obtain the average run time
    for (int i = 0; i < 15; i++) {
        
        clEnqueueWriteBuffer(queue, dC, CL_TRUE, 0, sizeof(float) * C.size(), (void *) C.data(), 0, NULL, NULL );

        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

        auto status = clblast::Gemm(
            clblast::Layout::kRowMajor,
            clblast::Transpose::kNo,
            clblast::Transpose::kNo,
            M, N, K,
            alpha, dA, 0, K, dB, 0, N,
            beta, dC, 0, N,
            &queue);

        if (status != clblast::StatusCode::kSuccess) {
            std::cerr << "GEMM failed\n";
            return 2;
        }
        clFinish(queue);
        
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

        clEnqueueReadBuffer(queue, dC, CL_TRUE, 0, sizeof(float) * C.size(), C.data(), 0, nullptr, nullptr);

        //The first execution is discarded for calculating run time as the GPU is 'warming up'
        //Instead, it's result compared to the default's result to check if it generates a valid result
        if (i == 0) {
            auto a = R == C;
            std::cout << a << " ";

            if (!a) {
                break;
            }
        } else {
            std::cout << std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin).count() << " ";
        }        
    }
    std::cout << std::endl;

    clReleaseMemObject(dA);
    clReleaseMemObject(dB);
    clReleaseMemObject(dC);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return 0;
}