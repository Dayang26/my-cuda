#include <stdio.h>

// __global__ 表示这个函数在 GPU 上运行，由 CPU 调用
__global__ void helloFromGPU() {
    printf("Hello from GPU! [Block: %d, Thread: %d]\n", blockIdx.x, threadIdx.x);
}

int main() {
    printf("Hello from CPU!\n");

    // 启动 1 个线程块，包含 4 个 GPU 线程并发执行
    helloFromGPU<<<1, 4>>>();

    // 等待 GPU 计算完成并同步输出缓冲区
    cudaDeviceSynchronize();

    return 0;
}