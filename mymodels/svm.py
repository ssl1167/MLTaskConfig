import numpy as np


class CustomModel:
    """
    自定义支持向量机（SVM）实现
    - 支持线性核函数
    - 支持分类和回归任务
    - 分类：支持软间隔（通过C参数控制正则化）
    - 回归：支持ε-不敏感损失函数
    接口：train(X, y)、predict(X)、predict_proba(X)（仅分类）
    """

    def __init__(self, learning_rate=0.01, C=1.0, max_iter=1000, tol=1e-3, epsilon=0.1):
        self.learning_rate = learning_rate  # 学习率
        self.C = C  # 正则化参数
        self.max_iter = max_iter  # 最大迭代次数
        self.tol = tol  # 收敛阈值
        self.epsilon = epsilon  # ε-不敏感损失的参数（回归任务）
        self.w = None  # 权重向量
        self.b = 0.0  # 偏置项
        self.is_classifier = None  # 标记任务类型（True=分类，False=回归）
        self.classes_ = None  # 存储类别（分类任务）
        self.feature_importances_ = None  # 特征重要性（基于权重绝对值）
        self.support_vectors_ = None  # 支持向量
        self.support_vector_indices_ = None  # 支持向量索引

    def _judge_task_type(self, y):
        """判断任务类型：分类（离散标签）/回归（连续标签）"""
        y = np.array(y)
        # 规则：标签为整数类型 或 唯一值数量<样本数10% → 分类
        if y.dtype.kind in ('i', 'u') or len(np.unique(y)) < max(10, len(y) * 0.1):
            self.is_classifier = True
            self.classes_ = list(np.unique(y))
            self.n_classes = len(self.classes_)
        else:
            self.is_classifier = False
            self.classes_ = None
            self.n_classes = None
    
    def _convert_labels(self, y):
        """将原始标签转换为±1（适配SVM目标函数）"""
        unique_labels = np.unique(y)
        # 映射标签：较小值→-1，较大值→1
        label_map = {unique_labels[0]: -1, unique_labels[1]: 1}
        self.classes_ = [unique_labels[0], unique_labels[1]]  # 记录原始类别
        return np.array([label_map[label] for label in y])
    
    def _one_vs_rest_predict(self, X):
        """多分类预测：One-vs-Rest策略"""
        if self.n_classes <= 2:
            return self.predict(X)
        
        # 为每个类别训练一个二分类器
        predictions = np.zeros((X.shape[0], self.n_classes))
        
        for i, class_label in enumerate(self.classes_):
            # 创建二分类标签：当前类别 vs 其他类别
            y_binary = (y == class_label).astype(int)
            if len(np.unique(y_binary)) < 2:
                continue  # 跳过只有一个类别的数据
                
            # 训练二分类器
            binary_model = CustomModel(self.learning_rate, self.C, self.max_iter, self.tol, self.epsilon)
            binary_model.train(X, y_binary)
            
            # 获取概率预测
            if hasattr(binary_model, 'predict_proba'):
                proba = binary_model.predict_proba(X)
                predictions[:, i] = proba[:, 1]  # 正类概率
        
        # 选择概率最高的类别
        return np.array([self.classes_[np.argmax(predictions[i])] for i in range(X.shape[0])])
    
    def _compute_feature_importance(self):
        """计算特征重要性：基于权重向量的绝对值"""
        if self.w is not None:
            # 特征重要性 = 权重绝对值的归一化
            feature_importance = np.abs(self.w)
            if np.sum(feature_importance) > 0:
                feature_importance /= np.sum(feature_importance)
            self.feature_importances_ = feature_importance
        else:
            self.feature_importances_ = None
    
    def _identify_support_vectors(self, X, y):
        """识别支持向量：满足|w·x + b| ≤ 1的样本"""
        if self.w is None:
            return
        
        # 计算样本到超平面的距离
        distances = np.abs(np.dot(X, self.w) + self.b)
        
        # 支持向量：距离超平面最近的样本（阈值可调）
        threshold = 1.0 + 0.1  # 允许微小误差
        support_mask = distances <= threshold
        
        self.support_vector_indices_ = np.where(support_mask)[0]
        self.support_vectors_ = X[support_mask] if np.any(support_mask) else None
    
    def _classifier_gradients(self, X, y):
        """计算分类任务的梯度（合页损失）"""
        # 转换标签为[-1,1]
        y_converted = self._convert_labels(y)
        # 计算预测值：wx + b
        y_pred = np.dot(X, self.w) + self.b
        # 计算合页损失的梯度
        gradients_w = self.w.copy()  # L2正则化部分的梯度
        gradient_b = 0
        
        for i in range(len(y)):
            if y_converted[i] * y_pred[i] < 1:  # 违反软间隔约束
                gradients_w -= self.C * y_converted[i] * X[i]
                gradient_b -= self.C * y_converted[i]
                
        return gradients_w, gradient_b
    
    def _regressor_gradients(self, X, y):
        """计算回归任务的梯度（ε-不敏感损失）"""
        # 计算预测值：wx + b
        y_pred = np.dot(X, self.w) + self.b
        # 计算ε-不敏感损失的梯度
        gradients_w = self.w.copy()  # L2正则化部分的梯度
        gradient_b = 0
        
        for i in range(len(y)):
            error = y_pred[i] - y[i]
            if error > self.epsilon:  # 预测值过高
                gradients_w += self.C * X[i]
                gradient_b += self.C
            elif error < -self.epsilon:  # 预测值过低
                gradients_w -= self.C * X[i]
                gradient_b -= self.C
                
        return gradients_w, gradient_b

    def train(self, X, y):
        """训练SVM模型：通过梯度下降最小化目标函数"""
        # 数据格式转换
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).flatten()
        
        # 判断任务类型
        self._judge_task_type(y)
        
        # 获取特征数量并初始化权重
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)  # 初始化权重向量
        self.b = 0.0  # 初始化偏置项

        # 梯度下降优化
        for iter in range(self.max_iter):
            # 根据任务类型计算梯度
            if self.is_classifier:
                gradients_w, gradient_b = self._classifier_gradients(X, y)
            else:
                gradients_w, gradient_b = self._regressor_gradients(X, y)
                
            # 更新权重和偏置
            self.w -= self.learning_rate * gradients_w
            self.b -= self.learning_rate * gradient_b
            
            # 收敛检查：梯度范数小于阈值
            gradient_norm = np.sqrt(np.sum(gradients_w ** 2) + gradient_b ** 2)
            if gradient_norm < self.tol:
                print(f"SVM训练在迭代 {iter+1} 次时收敛")
                break
                
            # 日志：每100次迭代打印一次训练进度
            if (iter + 1) % 100 == 0:
                if self.is_classifier:
                    # 计算训练准确率
                    y_pred = self.predict(X)
                    accuracy = np.mean(y_pred == y)
                    print(f"SVM分类训练进度：{iter+1}/{self.max_iter}次迭代，训练准确率：{accuracy:.4f}")
                else:
                    # 计算训练MSE
                    y_pred = self.predict(X)
                    mse = np.mean((y_pred - y) ** 2)
                    print(f"SVM回归训练进度：{iter+1}/{self.max_iter}次迭代，训练MSE：{mse:.6f}")

        # 训练完成后计算特征重要性和识别支持向量
        self._compute_feature_importance()
        if self.is_classifier:
            self._identify_support_vectors(X, y)
        
        return self

    def predict(self, X):
        """预测：分类返回类别标签，回归返回连续值"""
        if self.w is None:
            raise ValueError("模型未训练，请先调用train()方法")
        
        X = np.asarray(X, dtype=np.float64)
        
        if self.is_classifier:
            # 多分类：使用One-vs-Rest策略
            if self.n_classes > 2:
                return self._one_vs_rest_predict(X)
            else:
                # 二分类：根据符号预测类别
                linear_output = np.dot(X, self.w) + self.b
                pred_converted = np.where(linear_output >= 0, 1, -1)
                # 映射回原始标签
                label_map = {-1: self.classes_[0], 1: self.classes_[1]}
                return np.array([label_map[pred] for pred in pred_converted])
        else:
            # 回归：直接返回线性输出
            linear_output = np.dot(X, self.w) + self.b
            return linear_output

    def predict_proba(self, X):
        """预测概率（近似计算）：基于样本到超平面的距离归一化"""
        if not self.is_classifier:
            raise NotImplementedError("回归模型无predict_proba接口")

        X = np.asarray(X, dtype=np.float64)
        
        if self.n_classes > 2:
            # 多分类：使用One-vs-Rest策略计算概率
            probabilities = np.zeros((X.shape[0], self.n_classes))
            
            for i, class_label in enumerate(self.classes_):
                # 为每个类别创建二分类器
                y_binary = (y == class_label).astype(int)
                if len(np.unique(y_binary)) < 2:
                    continue  # 跳过只有一个类别的数据
                    
                # 训练二分类器
                binary_model = CustomModel(self.learning_rate, self.C, self.max_iter, self.tol, self.epsilon)
                binary_model.train(X, y_binary)
                
                # 获取概率预测
                if hasattr(binary_model, 'predict_proba'):
                    proba = binary_model.predict_proba(X)
                    probabilities[:, i] = proba[:, 1]  # 正类概率
            
            # 归一化概率
            row_sums = np.sum(probabilities, axis=1)
            probabilities = probabilities / row_sums[:, np.newaxis]
            return probabilities
        else:
            # 二分类：使用sigmoid函数
            scores = np.dot(X, self.w) + self.b
            prob_pos = 1 / (1 + np.exp(-scores))  # 正类（classes_[1]）概率
            prob_neg = 1 - prob_pos  # 负类（classes_[0]）概率
            # 返回形状为(n_samples, 2)的概率矩阵，列对应classes_顺序
            return np.column_stack((prob_neg, prob_pos))