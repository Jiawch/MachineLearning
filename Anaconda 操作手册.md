# Anaconda 操作手册

## Miniconda
[安装教程](https://zhuanlan.zhihu.com/p/63897033)  
Anaconda之外还有一个精简版, 不需要图形的环境时简直不要太配

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  
sudo chmod 777 Miniconda3-latest-Linux-x86_64.sh    
./Miniconda3-latest-Linux-x86_64.sh  
```

最终会默认安装在 ```/home/t-jiawch/miniconda3```

可能需要  
```
chmod 777 ~/miniconda3/  
chmod 777 ~/.conda/  
```

## 环境配置

```bash
conda create -n py36 python=3.6  
conda activate py36  
conda deactivate 用以退出环境  
```

注：环境会建在 ```~/.conda/envs/py36```

## 设置源

```python
conda config --add channels https://mirrors.cloud.tencent.com/anaconda/pkgs/free/

conda config --add channels https://mirrors.cloud.tencent.com/anaconda/pkgs/main/

conda config --set show_channel_urls yes
```

## 查看源

	conda config --show channels

## 移除源

	conda config --remove channels https网站

## 创建环境

```python
cd envs
# 可能要sudo
conda create --prefix=./py36gpu python=3.6
```

## 删除环境

	conda remove -n py36 --all

## 查看环境

```python
conda info -e 
```

## 退出环境

	source deactivate

## jupyter 添加 kernel

	pip install ipykernel 
	python -m ipykernel install --name XXXX

如果不行

	/opt/anaconda3/envs/环境名/bin/python -m ipykernel install --name XXXX

## PermissionError(13, 'Permission denied')

	sudo chown -R jawei:jawei /opt/anaconda3
