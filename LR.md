# 手推 LR

## 1. 初始条件

m个**独立同分布**的训练集${(x_1,y_1), (x_2,y_2),...,(x_n,y_n)}$
$y\in{0,1}$

## 2. 先验概率

> sigmod 函数可以表示样本为正例的概率

$P(y=1|x) = \frac{1}{1 + e^{-\omega x} }=p_1$
$P(y=0|x) =1 - P(y=1|x) = 1 - \frac{1}{1 + e^{-\omega x}} = \frac{1}{1+e^{\omega x} } = p_0$

> 目标：P(y|x)，每个样本属于其真实的概率越大越好

## 3. 似然函数

$L(\omega) = \prod_1^n P(y_i=1|x)^{y_i} P(y_i=0|x)^{1-y_i} $

## 4. 对数似然

$l(\omega) = lnL(\omega) = \sum_1^n (y_i lnp_1+(1-y_i)lnp_0)$
$= \sum (y_iln(\frac{p_1}{p_0})+lnp_0)$
$=\sum (y_i \omega x-ln(1+e^{\omega x}))$

## 5. 梯度下降法

$J(\omega) = -l(\omega) = \sum (ln(1+e^{\omega x})-y_i \omega x)$

> $l(\omega)$最大，等价于求$J(\omega)$最小

$\omega = \omega - \alpha \frac{\partial{J(\omega)}}{\partial \omega}$
$\frac{\partial{J(\omega)}}{\partial \omega} = \sum(\frac{xe^{\omega x}}{1+e^{\omega x}} - y_i x) = \sum (x(p_1 - y_i))$


