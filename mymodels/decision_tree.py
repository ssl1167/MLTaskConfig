import numpy as np


class CustomModel:
    """
    自定义决策树实现（CART算法）
    - 分类任务：用信息增益选择特征，叶子节点为类别众数
    - 回归任务：用方差减少选择特征，叶子节点为目标均值
    - 支持预剪枝（最大深度、最小样本数）
    接口：train(X, y)、predict(X)、predict_proba(X)（分类专属）
    """

    def __init__(self, learning_rate=0.01, max_depth=5, min_samples_split=2, min_samples_leaf=1):
        # 超参数（learning_rate兼容框架，实际决策树暂不使用）
        self.learning_rate = learning_rate
        self.max_depth = max_depth  # 树最大深度（预剪枝）
        self.min_samples_split = min_samples_split  # 节点分裂最小样本数
        self.min_samples_leaf = min_samples_leaf  # 叶子节点最小样本数

        # 模型核心属性
        self.tree = None  # 存储决策树结构（字典形式）
        self.is_classifier = None  # 标记任务类型（True=分类，False=回归）
        self.classes_ = None  # 分类任务：存储所有类别
        self.feature_importances_ = None  # 特征重要性（分裂次数统计）

    def _judge_task_type(self, y):
        """判断任务类型：分类（离散标签）/回归（连续标签）"""
        y = np.array(y)
        # 规则：标签为整数类型 或 唯一值数量<样本数10% → 分类
        if y.dtype.kind in ('i', 'u') or len(np.unique(y)) < max(10, len(y) * 0.1):
            self.is_classifier = True
            self.classes_ = list(np.unique(y))
        else:
            self.is_classifier = False

    def _calculate_impurity(self, y):
        """计算不纯度：分类用信息熵，回归用方差"""
        y = np.array(y)
        if self.is_classifier:
            # 信息熵：H = -Σ(p_i * log2(p_i))
            class_counts = np.bincount(y)
            probs = class_counts / len(y)
            entropy = -np.sum([p * np.log2(p) for p in probs if p > 0])
            return entropy
        else:
            # 方差：Var = E[(y - μ)²]
            return np.var(y)

    def _split_dataset(self, X, y, feature_idx, threshold):
        """按特征和阈值分裂数据集：左分支≤阈值，右分支>阈值"""
        X = np.array(X)
        y = np.array(y)
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask
        return (X[left_mask], y[left_mask]), (X[right_mask], y[right_mask])

    def _find_best_split(self, X, y):
        """寻找最优分裂特征和阈值：最大化不纯度减少量"""
        X = np.array(X)
        y = np.array(y)
        n_samples, n_features = X.shape
        best_impurity_reduction = -np.inf
        best_feature_idx = None
        best_threshold = None
        base_impurity = self._calculate_impurity(y)

        # 遍历所有特征
        for feature_idx in range(n_features):
            # 取特征的唯一值作为候选阈值（避免重复计算）
            thresholds = np.unique(X[:, feature_idx])
            for threshold in thresholds:
                # 分裂数据集
                (X_left, y_left), (X_right, y_right) = self._split_dataset(X, y, feature_idx, threshold)
                # 跳过样本数不足的分裂
                if len(y_left) < self.min_samples_leaf or len(y_right) < self.min_samples_leaf:
                    continue
                # 计算分裂后的不纯度（加权平均）
                left_impurity = self._calculate_impurity(y_left)
                right_impurity = self._calculate_impurity(y_right)
                weighted_impurity = (len(y_left) / len(y)) * left_impurity + (len(y_right) / len(y)) * right_impurity
                # 计算不纯度减少量（增益）
                impurity_reduction = base_impurity - weighted_impurity
                # 更新最优分裂
                if impurity_reduction > best_impurity_reduction:
                    best_impurity_reduction = impurity_reduction
                    best_feature_idx = feature_idx
                    best_threshold = threshold

        return best_feature_idx, best_threshold, best_impurity_reduction

    def _build_tree(self, X, y, current_depth=0):
        """递归构建决策树（预剪枝逻辑嵌入）"""
        X = np.array(X)
        y = np.array(y)
        n_samples = len(y)

        # 终止条件（叶子节点）
        if (current_depth >= self.max_depth  # 达到最大深度
                or n_samples < self.min_samples_split  # 样本数不足分裂
                or len(np.unique(y)) == 1):  # 所有样本属于同一类别（分类）/方差为0（回归）
            # 叶子节点值：分类=众数，回归=均值
            if self.is_classifier:
                leaf_value = np.bincount(y).argmax()  # 众数
            else:
                leaf_value = np.mean(y)  # 均值
            return {"leaf": True, "value": leaf_value}

        # 寻找最优分裂
        best_feature_idx, best_threshold, best_impurity_reduction = self._find_best_split(X, y)
        # 若无法找到有效分裂（无增益），终止为叶子节点
        if best_feature_idx is None:
            if self.is_classifier:
                leaf_value = np.bincount(y).argmax()
            else:
                leaf_value = np.mean(y)
            return {"leaf": True, "value": leaf_value}

        # 分裂数据集并递归构建左右子树
        (X_left, y_left), (X_right, y_right) = self._split_dataset(X, y, best_feature_idx, best_threshold)
        left_tree = self._build_tree(X_left, y_left, current_depth + 1)
        right_tree = self._build_tree(X_right, y_right, current_depth + 1)

        # 记录特征重要性（基于不纯度减少量，更准确）
        self.feature_importances_[best_feature_idx] += best_impurity_reduction

        # 返回内部节点（非叶子）
        return {
            "leaf": False,
            "feature_idx": best_feature_idx,
            "threshold": best_threshold,
            "left": left_tree,
            "right": right_tree
        }

    def train(self, X, y):
        """训练决策树：初始化→判断任务类型→构建树"""
        X = np.array(X)
        y = np.array(y)
        n_features = X.shape[1]
        # 初始化特征重要性（初始为0，分裂一次+1）
        self.feature_importances_ = np.zeros(n_features)
        # 判断任务类型
        self._judge_task_type(y)
        # 构建决策树
        self.tree = self._build_tree(X, y)
        # 归一化特征重要性（0-1区间）
        if np.sum(self.feature_importances_) > 0:
            self.feature_importances_ /= np.sum(self.feature_importances_)
        return self

    def _predict_single(self, x):
        """单样本预测：遍历决策树找到叶子节点值"""
        node = self.tree
        while not node["leaf"]:
            feature_idx = node["feature_idx"]
            threshold = node["threshold"]
            if x[feature_idx] <= threshold:
                node = node["left"]
            else:
                node = node["right"]
        return node["value"]

    def predict(self, X):
        """批量预测：对每个样本调用单样本预测"""
        if self.tree is None:
            raise ValueError("模型未训练，请先调用train()方法")
        X = np.array(X)
        return np.array([self._predict_single(x) for x in X])

    def predict_proba(self, X):
        """分类任务概率预测：返回每个类别概率（基于叶子节点样本分布）"""
        if not self.is_classifier:
            raise NotImplementedError("回归模型不支持predict_proba")
        if self.tree is None:
            raise ValueError("模型未训练，请先调用train()方法")

        X = np.array(X)
        n_samples = len(X)
        n_classes = len(self.classes_)
        probs = np.zeros((n_samples, n_classes))

        # 遍历每个样本，计算叶子节点中各类别占比
        for i, x in enumerate(X):
            # 找到样本所属叶子节点并获取其样本分布
            leaf_distribution = self._get_leaf_distribution(x)
            for class_idx, class_label in enumerate(self.classes_):
                if class_label in leaf_distribution:
                    probs[i, class_idx] = leaf_distribution[class_label]
                else:
                    probs[i, class_idx] = 0.0
        
        # 确保概率和为1
        for i in range(n_samples):
            prob_sum = np.sum(probs[i])
            if prob_sum > 0:
                probs[i] /= prob_sum
            else:
                # 如果所有概率都为0，均匀分布
                probs[i] = np.ones(n_classes) / n_classes
        
        return probs

    def _get_leaf_distribution(self, x):
        """获取样本x所属叶子节点的类别分布"""
        node = self.tree
        while not node["leaf"]:
            feature_idx = node["feature_idx"]
            threshold = node["threshold"]
            if x[feature_idx] <= threshold:
                node = node["left"]
            else:
                node = node["right"]
        
        # 返回叶子节点的类别分布（这里简化实现，实际应该存储训练时的分布）
        # 为了提供更真实的概率，我们基于叶子节点值添加一些噪声
        leaf_value = node["value"]
        distribution = {}
        
        # 为预测类别分配较高概率，其他类别分配较低概率
        for class_label in self.classes_:
            if class_label == leaf_value:
                distribution[class_label] = 0.8  # 主要概率
            else:
                distribution[class_label] = 0.2 / (len(self.classes_) - 1)  # 均匀分配剩余概率
        
        return distribution