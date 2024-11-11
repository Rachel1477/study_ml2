# 一些相关的工具
这篇文档会介绍一些重要的工具
# 目录
- [part1. git](#part1-git)
- [part2. GPU加速工具链](#part2-gpu加速工具链)
- [part3-pytorch](#part3-pytorch)
- [part4.验证gpu加速是否与pytorch相匹配](#part4验证gpu加速是否与pytorch相匹配)
- [开始猫狗分类器](#猫狗分类器)
## part1. git

**无论你以后从事什么计算机相关的工作，一定会用到git，它是一个很重要的工具。校园项目，公司实习，真正的就业，都离不开git** 

### 什么是 Git？
Git 是一个 版本控制系统。它就像是一个强大的记录本，用来记录你对一个文件或者项目所做的每一个修改。通过 Git，你可以：

>下载:可以下载他人的开源项目

>保存:你工作的每一步，不怕丢失。

>回溯:到过去的版本，查看曾经的工作成果。

>与别人协作:多人可以一起做同一个项目，而 Git 会帮助你们管理每个人的修改。

### GitHub 和 Git
GitHub 是一个基于 Git 的平台，它就像是一个云端仓库，可以让你存放代码、分享和协作。就像是你可以把本地保存的文件上传到 Google Drive 或 百度网盘 一样，GitHub 提供了一个在线存储的地方，方便多人一起管理和修改代码。你可以从中找到很多优秀的项目，比如**你们下个学期的C++课程设计。**

### 在Windows系统中安装git

点击[这里](https://github.com/git-for-windows/git/releases/download/v2.47.0.windows.2/Git-2.47.0.2-64-bit.exe)下载安装包

之后双击安装，如果你第一次接触git,就一路点击next之后安装。如果你对git有一定了解，可以选择自己想要的设置

之后需要把git添加到系统环境变量，找到你安装git的目录，以此选择Git-cmd,然后把完整的地址复制下来添加到系统环境变量，如下图：
![图1](https://i-blog.csdnimg.cn/direct/2a4ed93cfe28487cab6079ecabf0486c.png)


![图2](https://i-blog.csdnimg.cn/direct/04e2e8f2b8da4903a977252b49a9c9e8.png)


随后在控制台输入git,如果你得到以下结果，说明你的git已经配置完成
![图3](https://i-blog.csdnimg.cn/direct/2db9a8c35e084d7d9a6db7674007e9ff.png)

更多关于git的知识请查看[官方文档](https://git-scm.com/docs)


## part2. GPU加速工具链
>**（如果你还不知道GPU，CPU，显卡，内存这些概念，请认真学习<计算机科学概论>这门大一上学期的课，或者务必请教这门课的老师！！）**

在深度学习中，模型训练需要处理大量数据，且计算量非常非常大。如果只用电脑的 CPU 来完成这些计算，速度会非常慢，可能需要好几天甚至更长时间。而 GPU在架构设计上的多核并行处理机制在进行tensor运算的时候可以**极大**加快计算速度（我希望大家对于这里的”极大“有个具象的认识：原本使用CPU需要2个小时的训练任务放到GPU上只需要6分钟！！！）所以GPU加速是必要的

### GPU工具链的作用
为了让深度学习框架（比如 PyTorch 或 TensorFlow）能有效利用 GPU 的强大性能，需要安装一套 GPU 加速工具链，包括 NVIDIA 驱动（nvidia-driver）、CUDA 工具包（CUDA Toolkit） 和 cuDNN（CUDA 深度神经网络库）。它们各自的作用如下：

>NVIDIA 驱动 (nvidia-driver)： 就像是电脑上的“驱动程序”，负责让操作系统能识别 GPU 并与之通讯。安装驱动后，系统才知道 GPU 的存在，并可以发送任务给它处理。没有驱动，电脑就无法利用 GPU 的任何功能。

>CUDA 工具包 (CUDA Toolkit)： CUDA 是 NVIDIA 专门为 GPU 编程开发的一套软件框架。它提供了基本的计算工具，让深度学习框架能够利用 GPU 进行加速运算。CUDA 工具包包含许多数学计算的底层功能，比如矩阵运算、向量运算，这些都是深度学习模型训练中常用的运算。


>cuDNN (CUDA 深度神经网络库): cuDNN 是一个专门为深度学习设计的加速库。它包含了各种优化好的深度学习基础算法，比如卷积、池化、激活函数等。cuDNN 是在 CUDA 的基础上进一步优化的，让深度学习模型的训练速度更快。

**下面我们来安装他们**
### 1.NVIDIA 驱动 (nvidia-driver)

如果你的电脑用的是Windows操作系统，一般电脑会默认自动安装nvidia驱动，当然也不排除个别情况下商家偷懒

输入以下命令查看你的显卡状态
```
nvidia-smi
```
正常情况下会显示如下图片，我们可以从中发现一些有用的信息，如下图：
![图5](https://i-blog.csdnimg.cn/direct/116c377eed124e2b80a1a6bea10f0629.png)

如果没有，请根据你购买电脑时候商家给的显卡信息，去[英伟达官网](https://www.nvidia.cn/geforce/drivers/)搜索相应型号的驱动下载，或者联系电脑卖家给你安装

### 2.CUDA toolkit
上面的图片中，显示了你的显卡最大支持的CUDA版本，在选择的时候不要超过这个版本
在[nvidia官网](https://developer.nvidia.com/cuda-toolkit)选择你想要的版本，**注意：考虑到与pytorch兼容，只有11.8，12.1，12.4三个版本可以选择**

我个人推荐CUDA toolkit 11.8，这是目前最稳定的版本，下载链接在下面
https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_522.06_windows.exe

下载安装后，别忘记把以下目录添加到环境变量，具体操作如part 1里添加环境变量的操作
![图6](https://i-blog.csdnimg.cn/direct/0d0141da15114685b5d90f03deb4443e.png)

要验证是否安装成功，可以运行以下指令
```
nvcc -V
```
正常情况下会显示这些
![7](https://i-blog.csdnimg.cn/direct/1461899b314a4cb2b6c40688081ea312.png)

### 3.cudNN
在[nvidia官方网页](https://docs.nvidia.com/deeplearning/cudnn/latest/reference/support-matrix.html)可以查看每个CUDA toolkit版本对应的cudNN版本
![8](https://i-blog.csdnimg.cn/direct/9cd8ce68f25b421082243ec0702e1431.png)

例如我下载了CUDA toolkit 11.8，对应的cuDNN 版本为 9.5.1 [点击这里下载cudNN9.5.1](https://developer.download.nvidia.com/compute/cudnn/9.5.1/local_installers/cudnn_9.5.1_windows.exe)

在[这里](https://docs.nvidia.com/deeplearning/cudnn/latest/)找到每个CUDA toolkit对应的cudNN下载

>**注意**这里可能会让你注册一个nvidia的账号，就正常的注册就行，像注册游戏账号一样

安装好之后会自动添加环境变量

如何验证我们下面再说


## part3. pytorch
PyTorch 是一个流行的 深度学习框架，它是用来帮助人们更快、更方便地创建和训练神经网络模型的工具。
在人工智能和机器学习中，我们需要用复杂的数学运算来让模型从数据中“学习”。PyTorch 就是专门为这种任务设计的，它能让我们方便地使用 GPU 加速运算，让训练模型的过程更高效。而且，PyTorch 的设计让它使用起来像在写简单的 Python 代码，这让许多新手很快就能上手。
### 如何安装
首先在控制台激活你的conda环境（不会conda环境的，或者还没有安装minconda的，看[这篇文档](https://github.com/Rachel1477/study_ml-/tree/master)第二部分-安装python的IDE）

>**请务必激活conda环境再进行下面的操作!!!**

>**请务必激活conda环境再进行下面的操作!!!**

>**请务必激活conda环境再进行下面的操作!!!**

重要的事情说三遍，不然把pytorch装到系统环境里有你好哭的

之后进入[pytorch官网](https://pytorch.org/)，可以看到如下图的界面
![8](https://i-blog.csdnimg.cn/direct/cb5012182a5c4a6bb354e7966c812578.png)
然后依次选择，按图片上的选择也可以，唯一要注意的就是你的CUDA版本要与之前安装的版本对应
最后把Run this Command 中的内容复制到控制台，运行

这里把11.8版本的安装命令放在这里
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```


pytorch有大概5G，下载有点慢需要等一会

如何验证是否安装完成？
```
python
import torch
print(torch.__version__)
```
如下图所示就是安装完成了，会打印出你的pytorch版本号
![9](https://i-blog.csdnimg.cn/direct/5bb6c82b5faa419b9435290ff3d24fe6.png)

## part4.验证GPU加速是否与pytorch相匹配
还是在上面的conda环境中，输入以下指令
```
python
import torch
print(torch.cuda.is_available())  # 输出 True 表示 CUDA toolkit成功匹配
print(torch.backends.cudnn.enabled)  # 如果为 True 表示 cuDNN 成功匹配
```
然后哪里出问题了重装哪里，如果都是正确的，恭喜你，基本的深度学习环境你已经配置完成，可以开始你的第一个项目了--猫狗分类器


## 猫狗分类器
github地址在这里 https://github.com/Rachel1477/CNN-catvsdog

使用git可以把项目下载到你的电脑中
```
git clone https://github.com/Rachel1477/CNN-catvsdog.git
```
为了方便大家，我把数据集加入到这个项目中，无需额外下载,同时我上次训练的模型文件也在这里（Resnet34.pth）
![10](https://i-blog.csdnimg.cn/direct/f09dba75b15448ae9d38bd88261b7328.png)

开始训练前，请先安装依赖，在这个目录下打开控制台输入
```
pip install -r requirements.txt
```
之后就可以开始了
进入train.py,运行即可开始训练，完成后会有一张图片显示训练结果，如图
![11](https://i-blog.csdnimg.cn/direct/9c391c06cfeb4846afc69f8f5daa2db2.png)
(我这是试运行，只训练了2个epoch，大家训练的时候可以把参数调高一些)

train.py中的一些参数都是可以调整的
### 这里解释一些参数的含义
>--Epoch：表示遍历整个训练集的次数。在每个 epoch 中，模型会遍历一次全部训练数据并更新其参数。一轮训练完成后，模型会再用同一数据集重新训练，以进一步优化。更高的 epoch 数通常意味着更充分的训练，但过多可能导致模型过拟合。

>--batch_size：指每次训练所使用的样本数量。将训练集分成若干批次（batch）有助于减少内存占用。较小的 batch size 会使模型更频繁地更新参数，但训练速度较慢；较大的 batch size 提升速度，但需要更多内存，并可能导致模型更易陷入局部最优解。

>--Lr：学习率控制。模型参数更新的步长，值越大则参数调整幅度越大，训练速度更快，但可能不稳定；值过小则模型收敛更慢。

### 训练结果的解释
>-- loss：是衡量模型预测结果与数据集之间差距的指标，越小表示模型给出的结果与数据集中的结果相近，当loss逐渐降低，表示模型已经基本学习完了数据集中的知识，可以停止训练了。

>-- Accuracy：衡量模型好坏的基本指标，表示模型给出的预测值的准确程度，越高说明模型效果越好

这两者是不一样的，并不是loss越低，accuracy就越高，就好比虽然你完全地学习完了一本《高等数学》教材，但是老师出的题目是线性代数的题目，此时loss很低（你已经精通《高等数学》）但是你的accuracy也很低（题目中出现《高等数学》中没有的知识）

模型的好坏不仅仅取决于训练情况，还取决于数据集，测试环境，算力等很多因素

## 使用你训练的模型
进入evaluate.py,运行即可使用模型做出预测结果，并保存在result中

理论知识看这里：
https://zhuanlan.zhihu.com/p/360550845

至此，你的第一个深度学习项目就完成了，是不是很有成就感（doge）








