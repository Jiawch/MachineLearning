# BP

[pytorch的梯度计算以及backward方法](https://blog.csdn.net/f156207495/article/details/88727860)

[Pytorch 自动求梯度（autograd）](https://zhuanlan.zhihu.com/p/82077506)

model.parameters() 只包含模型的参数，.zero() 也只是把模型参数梯度置0，  
如果想引入input梯度，那么每次input需要单独置零。

![BP](BP.jpg)

参考：周志华《机器学习》
