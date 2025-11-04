import numpy as np


class CustomModel:
    """
    自定义K最近邻（KNN）实现
    - 分类任务：近邻类别投票（加权/普通）
    - 回归任务：近邻目标值均值（加权/普通）
    - 距离度量：欧氏距离
    接口：train(X, y)、predict(X)、predict_proba(X)（分类专属）
    """

    def __init__(self, learning_rate=0.01, n_neighbors=5, weights='uniform'):
        # 超参数（learning_rate兼容框架，KNN暂不使用）
        self.learning_rate = learning_rate
        self.n_neighbors = n_neighbors  # K值：近邻数量
        self.weights = weights  # 权重模式：'uniform'（等权）/'distance'（距离加权）

        # 模型核心属性（KNN为惰性学习，训练阶段仅存储数据）
        self.X_train = None  # 训练集特征
        self.y_train = None  # 训练集标签
        self.is_classifier = None  # 任务类型标记（True=分类，False=回归）
        self.classes_ = None  # 分类任务：存储所有类别
        self.feature_importances_ = None  # 特征重要性（基于距离贡献）

    def _judge_task_type(self, y):
        """判断任务类型：分类（离散标签）/回归（连续标签）"""
        y = np.array(y)
        # 规则：标签为整数类型 或 唯一值数量<样本数10% → 分类
        if y.dtype.kind in ('i', 'u') or len(np.unique(y)) < max(10, len(y) * 0.1):
            self.is_classifier = True
            self.classes_ = list(np.unique(y))
        else:
            self.is_classifier = False

    def _euclidean_distance(self, x1, x2):
        """计算欧氏距离：d = √Σ(x1_i - x2_i)²"""
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def _get_neighbors(self, x):
        """获取单个测试样本的K个近邻：计算与所有训练样本距离，排序后取前K个"""
        # 计算当前测试样本与所有训练样本的距离
        distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]
        # 按距离排序，获取前K个训练样本的索引
        neighbor_indices = np.argsort(distances)[:self.n_neighbors]
        # 返回近邻的距离和标签
        neighbor_distances = [distances[i] for i in neighbor_indices]
        neighbor_labels = [self.y_train[i] for i in neighbor_indices]
        return neighbor_distances, neighbor_labels

    def _predict_single_classification(self, x):
        """单样本分类预测：基于近邻投票（支持加权）"""
        neighbor_distances, neighbor_labels = self._get_neighbors(x)

        if self.weights == 'uniform':
            # 等权投票：统计近邻中各类别出现次数，取最多的类别
            class_counts = {cls: 0 for cls in self.classes_}
            for label in neighbor_labels:
                class_counts[label] += 1
            return max(class_counts, key=class_counts.get)

        elif self.weights == 'distance':
            # 距离加权投票：权重=1/距离（距离越小权重越大，避免距离为0的除以0）
            class_weights = {cls: 0.0 for cls in self.classes_}
            for dist, label in zip(neighbor_distances, neighbor_labels):
                weight = 1.0 / (dist + 1e-6)  # 加小值避免除以0
                class_weights[label] += weight
            return max(class_weights, key=class_weights.get)

    def _predict_single_regression(self, x):
        """单样本回归预测：基于近邻均值（支持加权）"""
        neighbor_distances, neighbor_labels = self._get_neighbors(x)
        neighbor_labels = np.array(neighbor_labels, dtype=float)

        if self.weights == 'uniform':
            # 等权均值：近邻标签的算术平均
            return np.mean(neighbor_labels)

        elif self.weights == 'distance':
            # 距离加权均值：权重=1/距离，加权平均
            weights = 1.0 / (np.array(neighbor_distances) + 1e-6)
            weighted_sum = np.sum(neighbor_labels * weights)
            total_weight = np.sum(weights)
            return weighted_sum / total_weight

    def train(self, X, y):
        """KNN训练（惰性学习：仅存储训练数据，不进行参数学习）"""
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        # 判断任务类型
        self._judge_task_type(self.y_train)
        # 初始化特征重要性（将在预测时计算）
        self.feature_importances_ = np.zeros(X.shape[1])
        return self

    def predict(self, X):
        """批量预测：对每个测试样本调用单样本预测逻辑"""
        if self.X_train is None or self.y_train is None:
            raise ValueError("模型未训练，请先调用train()方法存储训练数据")

        X = np.array(X)
        if self.is_classifier:
            return np.array([self._predict_single_classification(x) for x in X])
        else:
            return np.array([self._predict_single_regression(x) for x in X])

    def predict_proba(self, X):
        """分类任务概率预测：返回每个类别在近邻中的占比（支持加权）"""
        if not self.is_classifier:
            raise NotImplementedError("回归模型不支持predict_proba接口")
        if self.X_train is None or self.y_train is None:
            raise ValueError("模型未训练，请先调用train()方法")

        X = np.array(X)
        n_samples = len(X)
        n_classes = len(self.classes_)
        probs = np.zeros((n_samples, n_classes))  # 形状：(样本数, 类别数)

        for i, x in enumerate(X):
            neighbor_distances, neighbor_labels = self._get_neighbors(x)
            class_idx_map = {cls: idx for idx, cls in enumerate(self.classes_)}  # 类别→索引映射

            if self.weights == 'uniform':
                # 等权概率：类别占比=近邻中该类别数量/K
                for label in neighbor_labels:
                    cls_idx = class_idx_map[label]
                    probs[i, cls_idx] += 1.0
                probs[i] /= self.n_neighbors  # 归一化到0-1

            elif self.weights == 'distance':
                # 距离加权概率：类别权重占比=该类别总权重/所有类别总权重
                class_weights = {cls: 0.0 for cls in self.classes_}
                for dist, label in zip(neighbor_distances, neighbor_labels):
                    weight = 1.0 / (dist + 1e-6)
                    class_weights[label] += weight
                total_weight = sum(class_weights.values())
                for cls, idx in class_idx_map.items():
                    probs[i, idx] = class_weights[cls] / total_weight  # 归一化到0-1

        return probs

    def _calculate_feature_importance(self, X_test, y_test):
        """计算特征重要性：基于特征对距离计算的影响"""
        if self.X_train is None or self.y_train is None:
            return None
            
        n_features = self.X_train.shape[1]
        feature_importance = np.zeros(n_features)
        
        # 对每个测试样本计算特征重要性
        for i, x_test in enumerate(X_test):
            # 计算完整距离
            full_distances = [self._euclidean_distance(x_test, x_train) for x_train in self.X_train]
            
            # 对每个特征，计算移除该特征后的距离变化
            for feat_idx in range(n_features):
                # 创建移除该特征的测试样本
                x_test_reduced = np.delete(x_test, feat_idx)
                
                # 计算移除该特征后的距离
                reduced_distances = []
                for x_train in self.X_train:
                    x_train_reduced = np.delete(x_train, feat_idx)
                    dist = self._euclidean_distance(x_test_reduced, x_train_reduced)
                    reduced_distances.append(dist)
                
                # 特征重要性 = 距离变化的方差（变化越大，特征越重要）
                distance_change = np.array(full_distances) - np.array(reduced_distances)
                feature_importance[feat_idx] += np.var(distance_change)
        
        # 归一化特征重要性
        if np.sum(feature_importance) > 0:
            feature_importance /= np.sum(feature_importance)
        
        return feature_importance