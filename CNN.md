# CNN

![卷积的数学定义](卷积的数学定义.png)

自变量为卷积核上的t，x是x轴上的值，t再x轴上移动

表达式：3x3 的 kernel

$$
O(i,j) = \sum_{m=-1}^{1} \sum_{n=-1}^{1} I(i-m,j-n) K(m,n)
$$

# [CNN 感受野计算](https://www.jianshu.com/p/e875117e5372)

$$
l_{k} = l_{k-1}+ \left [ (f_{k}-1)*\prod_{i=1}^{k-1}s_{i} \right ]
$$


其中 $l_{k-1}$ 为第 $k-1$ 层对应的感受野大小，$f_k$ 为第 $k$ 层的卷积核大小，或者是池化层的池化尺寸大小; $s_i$ 是第i层 stride
