import numpy as np


class CustomModel:
    """
    手写多层感知器（MLP）模型（支持二分类/多分类/回归）
    核心逻辑：全连接层+激活函数（ReLU/sigmoid/softmax）+梯度下降优化损失
    接口：train(X, y)、predict(X)、predict_proba(X)（分类任务可选）
    """

    def __init__(self, learning_rate=0.01, hidden_layers=[32, 16], max_iter=200, tol=1e-4):
        self.learning_rate = learning_rate  # 学习率（框架动态传入）
        self.hidden_layers = hidden_layers  # 隐藏层结构（优化：[32,16]更快）
        self.max_iter = max_iter  # 最大迭代次数（优化：200次足够且更快）
        self.tol = tol  # 收敛阈值（损失变化<阈值则停止）
        self.is_classifier = None  # 任务类型：True（分类）/False（回归）
        self.classes_ = None  # 分类任务的类别列表
        self.n_classes = None  # 分类任务的类别数量
        self.weights = []  # 权重参数：[W1, W2, ...]（W.shape=(输入维度, 输出维度)）
        self.biases = []  # 偏置参数：[b1, b2, ...]（b.shape=(1, 输出维度)）
        self.feature_importances_ = None  # 特征重要性（基于第一层权重）

    # --------------------------
    # 激活函数与导数（核心组件）
    # --------------------------
    def _sigmoid(self, z):
        """sigmoid激活函数（分类输出层/隐藏层），避免数值溢出"""
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def _sigmoid_deriv(self, a):
        """sigmoid导数（a为激活输出，避免重复计算）"""
        return a * (1 - a)

    def _relu(self, z):
        """ReLU激活函数（隐藏层专用，缓解梯度消失）"""
        return np.maximum(0, z)

    def _relu_deriv(self, z):
        """ReLU导数（z为激活前输入）"""
        return (z > 0).astype(float)

    def _softmax(self, z):
        """softmax激活函数（多分类输出层），避免数值溢出"""
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    # --------------------------
    # 损失函数与导数（任务适配）
    # --------------------------
    def _mse_loss(self, y_true, y_pred):
        """MSE损失（回归任务）"""
        return np.mean((y_true - y_pred) ** 2)

    def _mse_loss_deriv(self, y_true, y_pred):
        """MSE损失导数（回归任务）"""
        return 2 * (y_pred - y_true) / len(y_true)

    def _cross_entropy_loss(self, y_true, y_pred_proba):
        """交叉熵损失（分类任务），避免log(0)"""
        y_pred_proba = np.clip(y_pred_proba, 1e-10, 1 - 1e-10)
        if self.n_classes == 2:
            # 二分类（sigmoid输出）
            return -np.mean(y_true * np.log(y_pred_proba) + (1 - y_true) * np.log(1 - y_pred_proba))
        else:
            # 多分类（softmax输出，y_true为独热编码）
            return -np.mean(np.sum(y_true * np.log(y_pred_proba), axis=1))

    # --------------------------
    # 任务类型判断与数据格式化
    # --------------------------
    def _judge_task_type(self, y):
        """自动判断任务类型：分类（离散标签）/回归（连续标签）"""
        y = np.asarray(y)
        # 分类任务：标签为整数且唯一值少（<样本数10%且<10个）
        if y.dtype.kind in ('i', 'u'):
            unique_vals = np.unique(y)
            if len(unique_vals) < max(10, len(y) * 0.1):
                self.is_classifier = True
                self.classes_ = unique_vals.tolist()
                self.n_classes = len(unique_vals)
                return
        # 默认回归任务
        self.is_classifier = False
        self.n_classes = 1

    def _format_labels(self, y):
        """格式化标签：分类转独热编码，回归转列向量"""
        y = np.asarray(y, dtype=np.float64)
        if self.is_classifier:
            if self.n_classes == 2:
                return y.reshape(-1, 1)  # 二分类：列向量（0/1）
            else:
                # 多分类：独热编码（n_samples, n_classes）
                n_samples = len(y)
                y_onehot = np.zeros((n_samples, self.n_classes))
                y_onehot[np.arange(n_samples), y.astype(int)] = 1.0
                return y_onehot
        else:
            return y.reshape(-1, 1)  # 回归：列向量

    # --------------------------
    # 参数初始化（He/Xavier适配激活函数）
    # --------------------------
    def _init_params(self, input_dim):
        """初始化权重和偏置，隐藏层用He初始化（ReLU），输出层用对应初始化"""
        self.weights = []
        self.biases = []
        prev_dim = input_dim  # 前一层维度（输入层为特征数）

        # 初始化隐藏层参数（ReLU+He初始化）
        for hidden_dim in self.hidden_layers:
            W = np.random.randn(prev_dim, hidden_dim) * np.sqrt(2 / prev_dim)  # He初始化
            b = np.zeros((1, hidden_dim))  # 偏置初始化为0
            self.weights.append(W)
            self.biases.append(b)
            prev_dim = hidden_dim

        # 初始化输出层参数（根据任务类型选择初始化方式）
        if self.is_classifier:
            output_dim = 1 if self.n_classes == 2 else self.n_classes
            # 分类输出层：Xavier初始化（sigmoid/softmax）
            W_out = np.random.randn(prev_dim, output_dim) * np.sqrt(1 / prev_dim)
        else:
            output_dim = 1  # 回归输出层：1个神经元（无激活）
            # 回归输出层：He初始化（无激活）
            W_out = np.random.randn(prev_dim, output_dim) * np.sqrt(2 / prev_dim)
        self.weights.append(W_out)
        self.biases.append(np.zeros((1, output_dim)))

    # --------------------------
    # 前向传播（计算各层输出）
    # --------------------------
    def _forward_propagation(self, X):
        """前向传播：返回各层激活输出（activations）和激活前输入（z_list）"""
        activations = [X]  # 第0层：输入层输出（特征矩阵）
        z_list = []  # 各层激活前输入（用于反向传播求导）
        a_prev = X

        # 隐藏层前向传播（ReLU激活）
        for i in range(len(self.hidden_layers)):
            z = np.dot(a_prev, self.weights[i]) + self.biases[i]
            z_list.append(z)
            a_prev = self._relu(z)  # 隐藏层固定用ReLU
            activations.append(a_prev)

        # 输出层前向传播（根据任务类型选择激活函数）
        z_out = np.dot(a_prev, self.weights[-1]) + self.biases[-1]
        z_list.append(z_out)
        if self.is_classifier:
            a_out = self._sigmoid(z_out) if self.n_classes == 2 else self._softmax(z_out)
        else:
            a_out = z_out  # 回归：无激活（直接输出连续值）
        activations.append(a_out)

        return activations, z_list

    # --------------------------
    # 反向传播（计算梯度）
    # --------------------------
    def _backward_propagation(self, X, y_true, activations, z_list):
        """反向传播：计算各层权重和偏置的梯度，返回梯度列表"""
        n_samples = len(X)
        grads_W = [np.zeros_like(W) for W in self.weights]  # 权重梯度
        grads_b = [np.zeros_like(b) for b in self.biases]  # 偏置梯度

        # 1. 计算输出层梯度
        a_out = activations[-1]
        z_out = z_list[-1]
        if self.is_classifier:
            if self.n_classes == 2:
                # 二分类（sigmoid输出）：损失导数 * sigmoid导数
                delta_out = (a_out - y_true) * self._sigmoid_deriv(a_out)
            else:
                # 多分类（softmax输出）：直接用预测-真实（交叉熵+softmax导数简化）
                delta_out = a_out - y_true
        else:
            # 回归（无激活）：MSE损失导数
            delta_out = self._mse_loss_deriv(y_true, a_out)

        # 输出层参数梯度（平均梯度，避免样本数影响）
        a_prev = activations[-2]
        grads_W[-1] = np.dot(a_prev.T, delta_out) / n_samples
        grads_b[-1] = np.mean(delta_out, axis=0, keepdims=True)

        # 2. 计算隐藏层梯度（反向遍历隐藏层）
        delta = delta_out  # 初始为输出层误差
        for i in reversed(range(len(self.hidden_layers))):
            z = z_list[i]
            a_prev = activations[i]
            # 隐藏层误差：delta = 下一层误差 * 下一层权重.T * ReLU导数
            delta = np.dot(delta, self.weights[i + 1].T) * self._relu_deriv(z)
            # 隐藏层参数梯度
            grads_W[i] = np.dot(a_prev.T, delta) / n_samples
            grads_b[i] = np.mean(delta, axis=0, keepdims=True)

        return grads_W, grads_b

    # --------------------------
    # 模型核心接口（框架调用）
    # --------------------------
    def train(self, X, y):
        """
        训练MLP模型（梯度下降优化）
        参数：X（特征矩阵，shape=(n_samples, n_features)）、y（标签，shape=(n_samples,)）
        """
        # 数据格式转换与任务判断
        X = np.asarray(X, dtype=np.float64)
        self._judge_task_type(y)
        y_formatted = self._format_labels(y)
        input_dim = X.shape[1]

        # 初始化参数
        self._init_params(input_dim)

        # 梯度下降迭代训练
        prev_loss = float('inf')  # 初始损失设为无穷大
        for iter in range(self.max_iter):
            # 1. 前向传播：计算各层输出
            activations, z_list = self._forward_propagation(X)
            y_pred = activations[-1]

            # 2. 计算损失与收敛判断
            if self.is_classifier:
                current_loss = self._cross_entropy_loss(y_formatted, y_pred)
            else:
                current_loss = self._mse_loss(y_formatted, y_pred)
            # 损失变化小于阈值，提前收敛
            if abs(prev_loss - current_loss) < self.tol:
                print(f"训练提前收敛：迭代{iter + 1}次，损失变化<{self.tol}")
                break
            prev_loss = current_loss

            # 3. 反向传播：计算梯度
            grads_W, grads_b = self._backward_propagation(X, y_formatted, activations, z_list)

            # 4. 更新参数（梯度下降：向梯度反方向移动）
            for i in range(len(self.weights)):
                self.weights[i] -= self.learning_rate * grads_W[i]
                self.biases[i] -= self.learning_rate * grads_b[i]

            # 日志：每100次迭代打印损失
            if (iter + 1) % 100 == 0:
                print(f"MLP训练迭代{iter + 1}/{self.max_iter}，损失：{current_loss:.6f}")

        # 计算特征重要性（基于第一层权重的绝对值）
        if len(self.weights) > 0:
            first_layer_weights = self.weights[0]  # 第一层权重 (n_features, n_hidden)
            # 计算每个特征对所有隐藏神经元的影响
            feature_importance = np.sum(np.abs(first_layer_weights), axis=1)
            # 归一化特征重要性
            if np.sum(feature_importance) > 0:
                feature_importance /= np.sum(feature_importance)
            self.feature_importances_ = feature_importance
        else:
            self.feature_importances_ = np.zeros(X.shape[1])

        return self

    def predict_proba(self, X):
        """
        预测类别概率（仅分类任务）
        返回：shape=(n_samples, n_classes)，每行和为1
        """
        if not self.is_classifier:
            raise NotImplementedError("回归任务不支持predict_proba")

        X = np.asarray(X, dtype=np.float64)
        activations, _ = self._forward_propagation(X)
        y_pred_proba = activations[-1]

        # 二分类：扩展为2列（类别0概率=1-类别1概率）
        if self.n_classes == 2:
            proba_0 = 1 - y_pred_proba
            return np.column_stack((proba_0, y_pred_proba))
        else:
            return y_pred_proba  # 多分类：直接返回softmax结果

    def predict(self, X):
        """
        预测结果：分类返回类别，回归返回连续值
        """
        X = np.asarray(X, dtype=np.float64)
        activations, _ = self._forward_propagation(X)
        y_pred = activations[-1]

        if self.is_classifier:
            if self.n_classes == 2:
                return (y_pred >= 0.5).astype(int).reshape(-1)  # 二分类：阈值0.5
            else:
                return np.argmax(y_pred, axis=1)  # 多分类：取概率最大类别
        else:
            return y_pred.reshape(-1)  # 回归：直接返回连续值