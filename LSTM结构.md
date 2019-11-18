# LSTM

> lstm 通过三个门（遗忘门，输入门，输出门）来维护细胞状态，
> 细胞状态在整条链上运行，只有线性交互，故保持以前的信息

**所谓门**就是一个sigmod和一个按位乘法操作，如图

![gate](gate.png)


![lstm](lstm.jpg)

参考：https://www.jianshu.com/p/9dc9f41f0b29

