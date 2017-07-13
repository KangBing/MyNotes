
现代计算机，要求更快的计算速度，那么
## 什么方法可以使计算机运行更快
1、更快的时钟。2、每个时钟周日执行更多指令。3、增加处理器

##  选择公牛还是小鸡
GPU的设计思想就是增加处理器个数。但GPU处理器和CPU不同。
假如需要耕地，有2中选择：1、两头公牛。2、1024只小鸡。你会怎么选择？CPU计算单元就像公牛一样，单个核很强大。而GPU的核就像小鸡一样，虽然单个核比较弱，但是核的个数超级多。
现代GPU设计：
* Thousands of ALUs
* Hundreds of Processors
* Tens of Thousands of Concurrent Threads

## CPU处理速度很难再增长
随着时间发展，晶体管尺寸变得更小，处理速度更快，能耗更低。
在CPU发展刚开始的几十年，时钟频率的提高是处理速度增快的主要因素。最近几年，CPU处理速度的增加主要在于集成了更多的芯片。

## 为什么CPU处理速度不能一直增长
不是因为晶体管不能再变小。即使晶体管更小、能耗更小；但是单位面积产生的热能太多，无法使得芯片保持冷却。
那么增快计算的方法为：
* 单个更小的、更高效的处理器
* 许多这样的处理器的集成

## 我们要设计什么类型的处理器
传统的CPU有着复杂的控制硬件，它有着比较好的灵活性和性能。在固定用电情况下，它的计算效益不高。
GPU控制硬件简单，计算效益高。但是编程模型复杂，要把复杂问题分解为简单问题来解决。

## 设计高能效芯片的技术
即集成更多的简单的处理器

## 设计高能效处理器
需要搞清楚两个概念：延迟和吞吐量
* 延迟：指的是时间
* 吞吐量：指的是单位时间内的处理任务数量。
两者单位不同。

## 延迟和吞吐量
A、B两地相距4500km。轿车载客2人，时速200km/h；客车载客40人，时速50km/h。
轿车延迟：22.5h，吞吐量0.089people/h
客车延迟：90h，吞吐量0.45people/h

## GPU设计核心
* 大量简单的处理单元，控制单元简单，更多为了计算
* 显式并行编程模型
* 优化吞吐量，而不是延迟

## CUDA编程
CUDA支持多种语言，这里主要讲解C语言。使用CUDA编程，有些代码运行在CPU，有些代码运行在GPU。
CPU叫做HOST，GPU叫做DEVICE。CPU来控制GPU，向GPU发送指令。CPU和GPU有各自独立的内存。
数据可以在GPU和CPU之间传递，使用`cudaMemcpy`函数；在GPU上分配内存使用`cudaMalloc`函数；CPU可以启动核函数。

## GPU程序流程
1、CPU调用`cudaMalloc`在GPU上分配内存
2、CPU调用`cudaMemcpy`把数据拷贝到GPU
3、CPU启动核函数
4、CPU调用`cudaMalloc`把结果从GPU拷贝到CPU

## GPU擅长
1、启动大量线程
2、并行运行这些线程

## 数组内数字求平方
CPU代码
```
for(i = 0; i < 64; ++i){
	out[i] = in[i] * in[i];
}
```
CPU只有一个线程执行上面代码。总共循环64次。

GPU完整代码
```language
#include <stdio.h>

__global__ void square(float* d_out, float* d_in){
    int idx = threadIdx.x;
    float f = d_in[idx];
    d_out[idx] = f * f;
}

int main(int argc, char* argv[]){
    const int ARRAY_SIZE = 64;
    const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

    float h_in[ARRAY_SIZE];
    float h_out[ARRAY_SIZE];
    for(int i = 0; i < ARRAY_SIZE; ++i){
        h_in[i] = float(i);
    }

    float* d_in;
    float* d_out;

    cudaMalloc((void**)&d_in, ARRAY_BYTES);
    cudaMalloc((void**)&d_out, ARRAY_BYTES);

    cudaMemcpy(d_in, h_in, ARRAY_BYTES, cudaMemcpyHostToDevice);

    square<<<1, ARRAY_SIZE>>>(d_out, d_in);

    cudaMemcpy(h_out, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);

    for(int i = 0; i < ARRAY_SIZE; ++i){
        printf("%f", h_out[i]);
        printf(  ((i % 4 ) != 3)? "\t": "\n");
    }

    cudaFree(d_in);
    cudaFree(d_out);

    return 0;


}
```
上面代码中`square<<<1, ARRAY_SIZE>>>(d_out, d_in);`是启动了`ARRAY_SIZE`个线程来执行核函数。

## 配置启动核函数的参数
上面代码中，启动核函数部分为`square<<<1, 64>>>(d_out, d_in)`，第一个参数`1`代表number of blocks，第二个参数表示threads per blocks。所以启动核函数时形式为：
```language
KERNEL<<<GRID OF BLOCKS, BLOCKS OF THREADSS>>>(...)
```
启动的两个参数其实都是三维的`dim3(x, y, z)`，因为`dim3(w, 1, 1) == dim3(w) == w`，所以可以简写。即`square<<<1, 64>>> == square<dim3(1,1,1), dim3(64, 1, 1)>>>`。

启动核函数完整形式为
```language
KERNEL<<<dim3(bx, by, bz), dim3(tx, ty, tz), shmem, S>>>(...)
```
其中`shmem`表示shared memory per block in bytes，`stream`，即动态分配的共享内存的大小，单位是byte，不需要时为0或者不写;`S`类型为`cudaStream_t`，表示核函数在哪个流之中，默认为0。

## MAP操作
MAP操作是指有很多数据都要处理，核函数单独处理每一个数据

## Problem Set
这个作业要求是把一副彩色图片转换成灰度图片。彩色图片的一个像素对应RGB（alpha没有使用），转换过程就是把RGB三个值转换为一个值，具体公式为 output = .299f * R + .587f * G + .114f * B。做法很简单，就是GPU一个像素点对应一个GPU核函数。
图像实际是一个矩阵，按照行优先来存储。因此启动核函数时的参数`dim3`只需要使用前2个维度即可。x对应列，y对应行；要求bx * tx >= 列数， by * ty >= 行数。
实现就比较容易写出来了，`student_func.cu`如下
```language
#include "utils.h"

__global__
void rgba_to_greyscale(const uchar4* const rgbaImage,
                       unsigned char* const greyImage,
                       int numRows, int numCols)
{
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if(col >= numCols || row >= numRows){ //确保不越界
        return;
  }
  
  int offset = row * numCols + col; //行优先存储
  uchar4 rgba_pixel = rgbaImage[offset];
  float greyness = 0.299f * rgba_pixel.x + 0.587f * rgba_pixel.y + 0.114f * rgba_pixel.z;
  greyImage[offset] = static_cast<unsigned char>(greyness);
}

void your_rgba_to_greyscale(const uchar4 * const h_rgbaImage, uchar4 * const d_rgbaImage,
                            unsigned char* const d_greyImage, size_t numRows, size_t numCols)
{
  const dim3 blockSize(32, 32, 1);
  const dim3 gridSize( 1 + numCols / blockSize.x, 1 + numRows / blockSize.y, 1); 
  rgba_to_greyscale<<<gridSize, blockSize>>>(d_rgbaImage, d_greyImage, numRows, numCols);
  
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}
```



