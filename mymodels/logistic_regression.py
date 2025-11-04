import numpy as np

class CustomModel:
    """
    手写逻辑回归模型（二分类任务）
    核心逻辑：基于sigmoid激活的线性模型 + 梯度下降优化交叉熵损失
    接口：train(X, y)、predict(X)、predict_proba(X)
    """
    def __init__(self, learning_rate=0.01, max_iter=1000, tol=1e-4):
        self.learning_rate = learning_rate  # 学习率（从框架动态传入）
        self.max_iter = max_iter            # 最大迭代次数（防止过拟合）
        self.tol = tol                      # 收敛阈值（损失变化小于阈值则停止）
        self.weights = None                 # 模型参数：特征权重（shape=(n_features, 1)）
        self.bias = 0.0                     # 模型参数：偏置项（ scalar）
        self.is_classifier = True           # 标记为分类模型（框架识别用）
        self.classes_ = [0, 1]              # 二分类固定类别（0和1）
        self.feature_importances_ = None    # 特征重要性（基于权重绝对值）

    def _sigmoid(self, z):
        """sigmoid激活函数：将线性输出映射到[0,1]概率区间，避免数值溢出"""
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def _cross_entropy_loss(self, y_true, y_pred_proba):
        """计算交叉熵损失（二分类专用），避免log(0)导致计算错误"""
        # 限制概率在[1e-10, 1-1e-10]区间
        y_pred_proba = np.clip(y_pred_proba, 1e-10, 1 - 1e-10)
        return -np.mean(y_true * np.log(y_pred_proba) + (1 - y_true) * np.log(1 - y_pred_proba))

    def train(self, X, y):
        """
        训练逻辑回归模型（梯度下降优化）
        参数：X（特征矩阵，shape=(n_samples, n_features)）、y（标签，shape=(n_samples,)）
        """
        # 数据格式转换与初始化
        X = np.asarray(X, dtype=np.float64)  # 转为浮点型矩阵
        y = np.asarray(y, dtype=np.float64).reshape(-1, 1)  # 标签转为列向量（n_samples, 1）
        n_samples, n_features = X.shape
        self.weights = np.zeros((n_features, 1), dtype=np.float64)  # 初始化权重为0

        # 梯度下降迭代训练
        prev_loss = float('inf')  # 初始损失设为无穷大
        for iter in range(self.max_iter):
            # 1. 前向传播：计算线性输出与概率
            z = np.dot(X, self.weights) + self.bias  # 线性部分：z = X*w + b
            y_pred_proba = self._sigmoid(z)         # 激活：得到类别1的概率

            # 2. 计算损失与收敛判断
            current_loss = self._cross_entropy_loss(y, y_pred_proba)
            # 损失变化小于阈值，提前收敛（避免无效迭代）
            if abs(prev_loss - current_loss) < self.tol:
                print(f"训练提前收敛：迭代{iter+1}次，损失变化<{self.tol}")
                break
            prev_loss = current_loss

            # 3. 反向传播：计算梯度（交叉熵损失对权重和偏置的导数）
            error = y_pred_proba - y  # 误差项（概率 - 真实标签）
            grad_weights = (1 / n_samples) * np.dot(X.T, error)  # 权重梯度（平均梯度）
            grad_bias = (1 / n_samples) * np.sum(error)          # 偏置梯度（平均梯度）

            # 4. 更新参数（梯度下降：向梯度反方向移动）
            self.weights -= self.learning_rate * grad_weights
            self.bias -= self.learning_rate * grad_bias

            # 日志：每100次迭代打印一次损失（便于观察训练进度）
            if (iter + 1) % 100 == 0:
                print(f"逻辑回归训练迭代{iter+1}/{self.max_iter}，损失：{current_loss:.6f}")

        # 计算特征重要性（基于权重绝对值）
        self.feature_importances_ = np.abs(self.weights.reshape(-1))
        # 归一化特征重要性
        if np.sum(self.feature_importances_) > 0:
            self.feature_importances_ /= np.sum(self.feature_importances_)

        return self

    def predict_proba(self, X):
        """
        预测类别概率（二分类）
        返回：shape=(n_samples, 2)，每一行对应[类别0概率, 类别1概率]
        """
        if not self.is_classifier:
            raise NotImplementedError("逻辑回归仅支持分类任务")
            
        X = np.asarray(X, dtype=np.float64)
        # 计算类别1的概率
        z = np.dot(X, self.weights) + self.bias
        proba_1 = self._sigmoid(z).reshape(-1)  # 转为1D数组（n_samples,）
        # 类别0概率 = 1 - 类别1概率
        proba_0 = 1 - proba_1
        # 拼接为2列概率矩阵
        return np.column_stack((proba_0, proba_1))

    def predict(self, X):
        """
        预测类别（二分类）
        逻辑：类别1概率>0.5则预测为1，否则为0
        """
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)  # 沿行取最大值索引（0或1）