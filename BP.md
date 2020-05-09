# BP

[pytorch的梯度计算以及backward方法](https://blog.csdn.net/f156207495/article/details/88727860)

[Pytorch 自动求梯度（autograd）](https://zhuanlan.zhihu.com/p/82077506)

[【pytorch】筛选冻结部分网络层参数同时设置有参数组的时候该怎么办？](https://blog.csdn.net/lingzhou33/article/details/88977700)

optimizer 里面只能放需要被更新的参数  
optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, net.parameters()),  # 记住一定要加上filter()，不然会报错
            lr=0.01, weight_decay=1e-5, momentum=0.9)

model.parameters() 只包含模型的参数，.zero() 也只是把模型参数梯度置0，  
如果想引入input梯度，那么每次input需要单独置零。

![BP](BP.jpg)

参考：周志华《机器学习》
