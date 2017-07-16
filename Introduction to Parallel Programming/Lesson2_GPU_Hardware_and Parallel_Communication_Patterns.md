# GPU Hardware and Parallel Communication Patterns

## 通信方式
GPU多线程执行任务时，线程之间共同协作主要是线程之间如何通信。线程通信包括
* 从同一块内存读取数据
* 把数据写到同一块内存
* 多个线程读取数据，运行过程中交换数据

## Map和Gather
Map:从特定内存读数据，运算后写到内存，读取的数据和写入的数据有一一对应关系。
Gather:多个输入对应一个输出，例如模糊图像操作（求图像某一像素及其周围像素的平均值）。

## Scatter
Scatter：一个或多个线程将结果写入到一个或多个位置

## Stencli-Quiz
Stencil:模板操作，从固定邻居位置读取数据，数据有重用，例如卷积核滑过图像。

## Transpose
转置操作是指重新排序内存中的元素。
一个矩阵
```
1  2  3  4  5
6  7  8  9  10
11 12 13 14 15 
```
转置后
```
1  6  11
2  7  12
3  8  13
4  9  14
5  10  15
```
转之前后在内存存储表示
```
// 转置前
1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
// 转置后
1 6 11 2 7 12 3 8 13 4 8 14 5 10 15
```
转置不仅仅用于数组、矩阵、图片，还可以用于数据结构，例如一个结构体
```
struct foo{
	float f;
    int i;
};
foo array[1000]
```
array of structures(AOS)内存中排序
```
f|i|f|i|f|i......
```
有时为了计算方便，将相同类型数据排列在一起作为一个数组
structures of arrays(SOA)
````
f|f|f|f|f.........|i|i|i|i|i|........
````

##  一个例子
```
float out[], in[];
int i = threadIdx.x;
int j = threadIdx.y;

const float pi = 3.14.15;

out[i] = pi * in[i]; // map操作

out[i + j * 128] = in[j + i * 128]; // transpose

if(i % 2){
	out[i-1] += pi * in[i]; out[i+1] = pi * in[i]; // scatter
    
    out[i]  = (in[i] + in[i - 1]  + in[i+1]) * pi / 3.0f; // gather，也看是看做stencil
}
```

## Parallel Communication Patterns
Map：one - to - one
Transpose: ont - to - one
Gather: many - to - one
Scatter：one - to - many
Stencil:several - to - one

Reduce: all - to - one
scan / fort: all - to - all

## Programmer View of the GPU
GPU中kernel是指C/C++中的核函数。
GPU中线程执行核函数。即使执行相同函数，不同线程执行路径可能不同，因为内部有if、for等控制语句。
thread blocks:只是一组线程，这组线程共同协作来解决计算任务（子任务）。

## Thread Blocks and GPU Hardware
为什么把问题分成blocks？
blocks什么时候运行？
block以什么顺序运行？
blocks怎么协作，有哪些限制？

一堆线程可以称为流处理器（Streaming Multiprocessors, SMs)。不同GPU有不同数量的SMs。
一个SM可以有很多简单的处理器，还有其他资源例如内存。GPU负责给SM分配block，分配的block在SM上运行。


