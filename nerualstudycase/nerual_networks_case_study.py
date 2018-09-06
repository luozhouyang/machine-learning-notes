import numpy as np
import matplotlib.pyplot as plt

N = 100
D = 2
K = 3
X = np.zeros((N * K, D))
y = np.zeros(N * K, dtype='uint8')

for j in range(K):
  ix = range(N * j, N * (j + 1))
  r = np.linspace(0.0, 1, N)
  t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2
  X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
  y[ix] = j

# plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
# plt.show()

# 初始化权重和偏置
W = 0.01 * np.random.randn(D, K)
b = np.zeros((1, K))

step_size = 1e-0
reg = 1e-3  # regularization strength

# 获取训练样本数量
num_examples = X.shape[0]

for i in range(200):

  # 计算分类得分
  scores = np.dot(X, W) + b

  # 计算 softmax得分
  exp_scores = np.exp(scores)
  probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

  # 使用交叉熵损失
  correct_log_probs = -np.log(probs[range(num_examples), y])

  # 计算训练集的data loss，总的损失除以样本数量
  data_loss = np.sum(correct_log_probs) / num_examples
  # 计算正则项损失reg loss，使用L2正则
  # reg就是lambda
  reg_loss = 0.5 * reg * np.sum(W * W)
  # 计算总的损失函数
  loss = data_loss + reg_loss

  if i % 10 == 0:
    print("iteration %4d loss: %f" % (i, loss))

  # 计算梯度，反向传播
  # 为什么 dscores = probs ??
  dscores = probs
  dscores[range(num_examples), y] -= 1
  dscores /= num_examples

  dW = np.dot(X.T, dscores)
  db = np.sum(dscores, axis=0, keepdims=True)
  dW += reg * W  # 正则项的梯度，dW不是第一次出现，必须累加

  # 更新参数
  W += -step_size * dW
  b += -step_size * db

# 训练结束，估算准确率
scores = np.dot(X, W) + b
# 在第二个维度（类别维度）取出概率最高的分类
predicted_class = np.argmax(scores, axis=1)
print("Training accuracy: %.2f" % (np.mean(predicted_class == y)))
