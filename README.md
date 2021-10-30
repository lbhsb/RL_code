# 经典强化学习算法系列

#### 此项目用于实现经典强化学习算法，首先是**DQN**系列(包括DDQN,D3QN等)，后续其他算法会陆续更新。

***

- 注意事项：
  - DQN,DDQN,D3QN和noisynet DQN算法都是基于pytorch实现，priority replay DQN是基于tensorflow1.实现
  - 在不同的环境(比如Atari)中实现，只需要修改env的名称。
  - 其他超参数的修改，需要手动修改
  - 若需观看reward曲线，只需要在当前目录下打开终端，输入**tensorboard --logdir = /logs**

