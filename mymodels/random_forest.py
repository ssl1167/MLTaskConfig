import numpy as np
import math
from collections import Counter

class DecisionTree:
    """决策树基学习器（CART算法），支持分类与回归"""
    def __init__(self, max_depth=5, min_samples_split=2, task_type=None):
        self.max_depth = max_depth  # 树最大深度（防止过拟合）
        self.min_samples_split = min_samples_split  # 节点分裂最小样本数
        self.task_type = task_type  # 'classification'/'regression'（自动推断）
        self.root = None  # 根节点

    def _infer_task_type(self, y):
        """根据标签推断任务类型：离散标签为分类，连续标签为回归"""
        y = np.array(y)
        unique_ratio = len(np.unique(y)) / len(y)
        if unique_ratio < 0.05 or len(np.unique(y)) < 10:
            return 'classification'
        else:
            return 'regression'


    def _calculate_impurity(self, y):
        """计算不纯度：分类用Gini系数，回归用MSE"""
        y = np.array(y)
        if self.task_type == 'classification':
            # Gini系数：1 - Σ(p_i²)，p_i为类占比
            counts = Counter(y)
            total = len(y)
            gini = 1.0 - sum((count / total) ** 2 for count in counts.values())
            return gini
        else:
            # MSE：均方误差
            mean = np.mean(y)
            mse = np.mean((y - mean) ** 2)
            return mse

    def _split_data(self, X, y, feature_idx, threshold):
        """根据特征索引和阈值分裂数据"""
        X = np.array(X)
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask
        return (X[left_mask], y[left_mask]), (X[right_mask], y[right_mask])

    def _find_best_split(self, X, y):
        """在当前子数据集上寻找最优分裂点（使用局部特征索引）"""
        X = np.array(X)
        y = np.array(y)
        best_impurity_gain = -np.inf
        best_feature_idx = None
        best_threshold = None
        parent_impurity = self._calculate_impurity(y)

        n_features_local = X.shape[1]

        # 遍历所有局部特征索引（0..k-1）
        for feature_idx in range(n_features_local):
            feature_values = X[:, feature_idx]
            unique_thresholds = np.unique(feature_values)

            for threshold in unique_thresholds:
                (X_left, y_left), (X_right, y_right) = self._split_data(X, y, feature_idx, threshold)

                if len(y_left) < self.min_samples_split or len(y_right) < self.min_samples_split:
                    continue

                weight_left = len(y_left) / len(y)
                weight_right = len(y_right) / len(y)
                child_impurity = (
                    weight_left * self._calculate_impurity(y_left)
                    + weight_right * self._calculate_impurity(y_right)
                )
                impurity_gain = parent_impurity - child_impurity

                if impurity_gain > 1e-10 and impurity_gain > best_impurity_gain:
                    best_impurity_gain = impurity_gain
                    best_feature_idx = feature_idx
                    best_threshold = threshold

        return best_feature_idx, best_threshold

    def _build_tree(self, X, y, depth=0):
        """递归构建决策树"""
        X = np.array(X)
        y = np.array(y)
        n_samples, n_features = X.shape

        # 终止条件：达到最大深度 / 样本数不足 / 标签纯（不纯度为0）
        if (depth >= self.max_depth 
            or len(y) < self.min_samples_split 
            or self._calculate_impurity(y) == 0):
            # 叶子节点：分类返回多数类，回归返回均值
            if self.task_type == 'classification':
                return Counter(y).most_common(1)[0][0]
            else:
                return np.mean(y)

        # 寻找最优分裂点
        best_feature_idx, best_threshold = self._find_best_split(X, y)

        # 若无法找到有效分裂（如所有特征值相同），返回叶子节点
        if best_feature_idx is None:
            if self.task_type == 'classification':
                return Counter(y).most_common(1)[0][0]
            else:
                return np.mean(y)

        # 递归分裂左右子树
        (X_left, y_left), (X_right, y_right) = self._split_data(X, y, best_feature_idx, best_threshold)
        left_subtree = self._build_tree(X_left, y_left, depth + 1)
        right_subtree = self._build_tree(X_right, y_right, depth + 1)

        # 返回内部节点（字典形式：特征索引、阈值、左右子树）
        return {
            'feature_idx': best_feature_idx,  # 局部特征索引（针对传入的子特征集）
            'threshold': best_threshold,
            'left': left_subtree,
            'right': right_subtree
        }

    def _predict_single(self, x):
        """单样本预测（递归遍历决策树）"""
        node = self.root
        while isinstance(node, dict):  # 若节点是字典（内部节点），继续遍历
            if x[node['feature_idx']] <= node['threshold']:
                node = node['left']
            else:
                node = node['right']
        return node  # 叶子节点（预测值）

    def train(self, X, y):
        """训练随机森林：生成多棵决策树并训练"""
        X = np.array(X)
        y = np.array(y)
        n_samples, n_features = X.shape

        # 推断任务类型（所有树共享同一任务类型）
        self.task_type = DecisionTree()._infer_task_type(y)
        if self.task_type == 'classification':
            self.classes_ = list(np.unique(y))
        self.is_classifier = (self.task_type == 'classification')  # 框架兼容字段

        # 计算每棵树使用的特征数
        max_features_per_tree = self._get_max_features(n_features)

        # 生成 n_estimators 棵决策树
        self.estimators = []
        for _ in range(self.n_estimators):
            # 1️⃣ Bagging 采样：生成子数据集
            X_bag, y_bag = self._bagging_sample(X, y)

            # 2️⃣ 随机选择特征（特征子集）
            feature_indices = self.rng.choice(n_features, size=max_features_per_tree, replace=False)

            # 3️⃣ 使用子数据集 + 特征子集训练决策树（树内部使用局部特征索引）
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                task_type=self.task_type
            )
            X_subset = X_bag[:, feature_indices]
            tree.root = tree._build_tree(X_subset, y_bag)

            # 4️⃣ 保存树和它对应的特征子集
            self.estimators.append((tree, feature_indices))

        return self



    def predict(self, X):
        """批量预测：对每个样本调用单样本预测"""
        X = np.array(X)
        return np.array([self._predict_single(x) for x in X])

    def predict_proba(self, X):
        """分类任务概率预测（仅支持分类）：返回[样本数×类别数]的概率矩阵"""
        if self.task_type != 'classification':
            raise NotImplementedError("回归任务不支持predict_proba")
        
        X = np.array(X)
        n_samples = len(X)
        n_classes = len(self.classes_)
        probs = np.zeros((n_samples, n_classes))  # 概率矩阵初始化

        # 遍历每个样本，预测类别并设置概率（硬概率：预测类为1，其他为0）
        for i, x in enumerate(X):
            pred_class = self._predict_single(x)
            class_idx = self.classes_.index(pred_class)
            probs[i, class_idx] = 1.0
        return probs


class CustomModel:
    """随机森林模型（Bagging + 多棵决策树）"""
    def __init__(self, n_estimators=10, max_depth=5, min_samples_split=2, 
                 max_features='sqrt', random_state=42, learning_rate=0.01):
        # 森林参数
        self.n_estimators = n_estimators  # 决策树数量
        self.max_depth = max_depth  # 单棵树最大深度
        self.min_samples_split = min_samples_split  # 单棵树分裂最小样本数
        self.max_features = max_features  # 每棵树使用的最大特征数（'sqrt'/'log2'/int）
        self.random_state = random_state  # 随机种子（保证可复现）
        self.learning_rate = learning_rate  # 预留参数（兼容框架）
        
        # 内部变量
        self.estimators = []  # 存储所有决策树
        self.task_type = None  # 'classification'/'regression'
        self.classes_ = None  # 分类任务的类别（用于predict_proba）
        self.rng = np.random.RandomState(random_state)  # 随机数生成器

    def _get_max_features(self, n_features):
        """根据max_features参数计算每棵树使用的特征数"""
        if isinstance(self.max_features, int):
            return min(self.max_features, n_features)
        elif self.max_features == 'sqrt':
            return int(math.sqrt(n_features))
        elif self.max_features == 'log2':
            return int(math.log2(n_features) + 1) if n_features > 1 else 1
        else:
            return n_features  # 默认使用所有特征

    def _bagging_sample(self, X, y):
        """Bagging采样：有放回采样生成子数据集"""
        n_samples = len(X)
        sample_indices = self.rng.choice(n_samples, size=n_samples, replace=True)
        return X[sample_indices], y[sample_indices]

    def train(self, X, y):
        """训练随机森林：生成多棵决策树并训练"""
        X = np.array(X)
        y = np.array(y)
        n_samples, n_features = X.shape

        # 推断任务类型（所有树共享同一任务类型）
        self.task_type = DecisionTree()._infer_task_type(y)
        if self.task_type == 'classification':
            self.classes_ = list(np.unique(y))
        self.is_classifier = (self.task_type == 'classification')  # 框架兼容字段

        # 计算每棵树使用的特征数
        max_features_per_tree = self._get_max_features(n_features)

        # 生成n_estimators棵决策树
        self.estimators = []
        for _ in range(self.n_estimators):
            # 1. Bagging采样：生成子数据集
            X_bag, y_bag = self._bagging_sample(X, y)
            
            # 2. 随机选择特征（特征子集）
            feature_indices = self.rng.choice(n_features, size=max_features_per_tree, replace=False)
            
            # 3. 训练单棵决策树（使用子数据集和特征子集）
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                task_type=self.task_type
            )
            # 直接构建单棵决策树（树内使用局部特征索引）
            tree.root = tree._build_tree(X_bag[:, feature_indices], y_bag)
            
            # 4. 保存树和其使用的特征子集
            self.estimators.append((tree, feature_indices))
        
        return self

    def predict(self, X):
        """随机森林预测：分类用投票，回归用均值"""
        X = np.array(X)
        n_samples = len(X)

        # 收集所有树的预测结果（n_estimators × n_samples）
        all_predictions = []
        for tree, feature_indices in self.estimators:
            # 仅使用树训练时的特征子集进行预测
            X_subset = X[:, feature_indices]
            tree_pred = tree.predict(X_subset)
            all_predictions.append(tree_pred)
        all_predictions = np.array(all_predictions)

        # 聚合预测结果
        if self.task_type == 'classification':
            # 分类：投票（每样本取多数类）
            return np.array([Counter(preds).most_common(1)[0][0] for preds in all_predictions.T])
        else:
            # 回归：均值（每样本取所有树预测的平均值）
            return np.mean(all_predictions, axis=0)

    def predict_proba(self, X):
        """分类任务概率预测：按树投票比例计算概率（不依赖树内的predict_proba）"""
        if self.task_type != 'classification':
            raise NotImplementedError("回归任务不支持predict_proba")
        
        X = np.array(X)
        n_samples = len(X)
        classes = list(self.classes_)
        n_classes = len(classes)
        vote_counts = np.zeros((n_samples, n_classes), dtype=float)

        # 对每棵树进行预测并累加投票
        for tree, feature_indices in self.estimators:
            X_subset = X[:, feature_indices]
            tree_preds = tree.predict(X_subset)
            # 将预测类别映射到列索引
            for i, pred in enumerate(tree_preds):
                try:
                    cls_idx = classes.index(pred)
                except ValueError:
                    # 罕见情况：树预测了未在训练集中出现的类别，跳过
                    continue
                vote_counts[i, cls_idx] += 1.0

        # 归一化为概率
        probs = vote_counts / max(1, self.n_estimators)
        return probs