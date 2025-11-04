# mymodels/baseline.py
import numpy as np

class CustomModel:
    """
    最小自定义模型示例
    - 分类: 预测训练集中出现最多的类
    - 回归: 预测训练集目标的均值
    接口:
      - train(X, y)
      - predict(X)
      - predict_proba(X)  # 如果是分类问题，可以选择实现
    """
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.is_classifier = True
        self.major_class = None
        self.reg_mean = None
        self.classes_ = None

    def train(self, X, y):
        y = np.array(y)
        # 判断是分类还是回归（简单判断：是否为整数/离散标签）
        # 这里采用简单启发式：如果 y 中的唯一值数量小于样本总数的一定比例，视为分类
        unique = np.unique(y)
        if y.dtype.kind in ('i','u') or len(unique) < max(10, len(y) * 0.1):
            self.is_classifier = True
            # 取众数
            vals, counts = np.unique(y, return_counts=True)
            self.major_class = vals[np.argmax(counts)]
            self.classes_ = vals.tolist()
        else:
            self.is_classifier = False
            self.reg_mean = float(np.mean(y))
        return self

    def predict(self, X):
        X = np.asarray(X)
        n = X.shape[0]
        if self.is_classifier:
            return np.array([self.major_class] * n)
        else:
            return np.array([self.reg_mean] * n)

    def predict_proba(self, X):
        if not self.is_classifier:
            raise NotImplementedError("回归模型没有 predict_proba")
        X = np.asarray(X)
        n = X.shape[0]
        # 返回每个样本对每个类的概率（简单：多数类概率 1.0，其它 0）
        probs = np.zeros((n, len(self.classes_)), dtype=float)
        major_idx = self.classes_.index(self.major_class)
        probs[:, major_idx] = 1.0
        return probs
