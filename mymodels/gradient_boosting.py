import numpy as np

class CustomModel:
    """
    手写梯度提升树（GBDT）模型（支持分类和回归任务）
    核心逻辑：基于CART树的梯度提升
    - 分类：拟合对数损失的负梯度
    - 回归：拟合平方损失的负梯度
    接口：train(X, y)、predict(X)、predict_proba(X)（仅分类）
    """
    def __init__(self, learning_rate=0.01, n_estimators=30, max_depth=3, min_samples_split=2, max_thresholds=15):
        self.learning_rate = learning_rate    # 学习率（框架动态传入）
        self.n_estimators = n_estimators      # 基学习器（CART树）数量（优化默认值平衡速度和精度）
        self.max_depth = max_depth            # 单棵树最大深度（防止过拟合）
        self.min_samples_split = min_samples_split  # 树分裂最小样本数
        self.max_thresholds = max_thresholds  # 每个特征最多尝试的阈值数量（性能优化）
        self.trees = []                       # 存储所有训练好的基学习器
        self.is_classifier = None             # 标记任务类型（True=分类，False=回归）
        self.classes_ = None                  # 分类任务：存储所有类别
        self.feature_importances_ = None      # 特征重要性

    # --------------------------
    # 内部类：CART树（支持分类和回归）
    # --------------------------
    class _CARTTree:
        def __init__(self, max_depth, min_samples_split, is_classifier=False, max_thresholds=10):
            self.max_depth = max_depth
            self.min_samples_split = min_samples_split
            self.is_classifier = is_classifier  # 是否为分类树
            self.max_thresholds = max_thresholds  # 每个特征最多尝试的阈值数量
            self.tree = {}  # 树结构：{'feature_idx': 特征索引, 'threshold': 阈值, 'left': 左子树, 'right': 右子树}
            self.value = None  # 叶子节点值

        def _split_data(self, X, y, feature_idx, threshold):
            """根据特征索引和阈值分裂数据为左右子树"""
            left_mask = X[:, feature_idx] <= threshold
            right_mask = ~left_mask
            return (X[left_mask], y[left_mask]), (X[right_mask], y[right_mask])

        def _find_best_split(self, X, y):
            """遍历所有特征和阈值，寻找最优分裂（性能优化版）"""
            n_samples, n_features = X.shape
            best_impurity = float('inf')
            best_feature = -1
            best_threshold = None

            # 遍历每个特征
            for feature_idx in range(n_features):
                # 性能优化：限制候选阈值数量
                unique_vals = np.unique(X[:, feature_idx])
                if len(unique_vals) <= self.max_thresholds:
                    # 唯一值较少，使用所有值
                    thresholds = unique_vals
                else:
                    # 唯一值太多，使用分位数采样
                    percentiles = np.linspace(0, 100, self.max_thresholds + 2)[1:-1]
                    thresholds = np.percentile(X[:, feature_idx], percentiles)
                    thresholds = np.unique(thresholds)  # 去重
                
                for threshold in thresholds:
                    # 分裂数据
                    (X_left, y_left), (X_right, y_right) = self._split_data(X, y, feature_idx, threshold)
                    # 跳过样本数不足的无效分裂
                    if len(y_left) < 1 or len(y_right) < 1:
                        continue
                    
                    # 根据任务类型选择不纯度计算方法
                    if self.is_classifier:
                        # 分类树：基尼系数
                        gini_left = self._gini(y_left)
                        gini_right = self._gini(y_right)
                        weighted_impurity = (len(y_left)/len(y)) * gini_left + (len(y_right)/len(y)) * gini_right
                    else:
                        # 回归树：方差
                        var_left = np.var(y_left)
                        var_right = np.var(y_right)
                        weighted_impurity = (len(y_left)/len(y)) * var_left + (len(y_right)/len(y)) * var_right
                    
                    # 更新最优分裂
                    if weighted_impurity < best_impurity:
                        best_impurity = weighted_impurity
                        best_feature = feature_idx
                        best_threshold = threshold
            return best_feature, best_threshold

        def _gini(self, y):
            """计算基尼系数（分类树分裂准则）"""
            classes, counts = np.unique(y, return_counts=True)
            prob = counts / len(y)
            return 1 - np.sum(prob ** 2)

        def _build_tree(self, X, y, depth=0, feature_importances=None):
            """递归构建CART树"""
            n_samples = len(y)
            # 终止条件1：达到最大深度或样本数小于分裂阈值
            if depth >= self.max_depth or n_samples < self.min_samples_split:
                # 叶子节点值：拟合残差的均值
                self.value = np.mean(y)
                return

            # 寻找最优分裂
            best_feature, best_threshold = self._find_best_split(X, y)
            # 终止条件2：无法找到有效分裂（所有特征值相同）
            if best_feature == -1:
                self.value = np.mean(y)
                return

            # 记录特征重要性（分裂次数）
            if feature_importances is not None:
                feature_importances[best_feature] += 1

            # 递归构建左右子树
            (X_left, y_left), (X_right, y_right) = self._split_data(X, y, best_feature, best_threshold)
            self.tree['feature_idx'] = best_feature
            self.tree['threshold'] = best_threshold
            self.tree['left'] = CustomModel._CARTTree(self.max_depth, self.min_samples_split, self.is_classifier, self.max_thresholds)
            self.tree['right'] = CustomModel._CARTTree(self.max_depth, self.min_samples_split, self.is_classifier, self.max_thresholds)
            self.tree['left']._build_tree(X_left, y_left, depth + 1, feature_importances)
            self.tree['right']._build_tree(X_right, y_right, depth + 1, feature_importances)

        def fit(self, X, y, feature_importances=None):
            """训练CART树（拟合输入数据的残差）"""
            self._build_tree(X, y, feature_importances=feature_importances)

        def predict_single(self, x):
            """单样本预测（遍历树结构找到对应叶子节点值）"""
            if self.value is not None:
                return self.value
            # 根据当前节点特征和阈值选择子树
            feature_idx = self.tree['feature_idx']
            threshold = self.tree['threshold']
            if x[feature_idx] <= threshold:
                return self.tree['left'].predict_single(x)
            else:
                return self.tree['right'].predict_single(x)

        def predict(self, X):
            """批量样本预测（对每个样本调用单样本预测）"""
            return np.array([self.predict_single(x) for x in X])

    # --------------------------
    # GBDT核心方法
    # --------------------------
    def _judge_task_type(self, y):
        """判断任务类型：分类（离散标签）/回归（连续标签）"""
        y = np.array(y)
        # 规则：标签为整数类型 或 唯一值数量<样本数10% → 分类
        if y.dtype.kind in ('i', 'u') or len(np.unique(y)) < max(10, len(y) * 0.1):
            self.is_classifier = True
            self.classes_ = list(np.unique(y))
        else:
            self.is_classifier = False
            self.classes_ = None

    def _sigmoid(self, z):
        """sigmoid激活函数（将logit值映射为[0,1]概率，避免数值溢出）"""
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def _compute_negative_gradient(self, y_true, y_pred):
        """计算负梯度（作为残差，GBDT拟合目标）"""
        if self.is_classifier:
            # 分类：对数损失的负梯度 = y - p
            y_pred_proba = self._sigmoid(y_pred)
            return y_true - y_pred_proba
        else:
            # 回归：平方损失的负梯度 = y - y_pred
            return y_true - y_pred

    def train(self, X, y):
        """
        训练GBDT模型（梯度下降迭代集成基学习器）
        参数：X（特征矩阵，shape=(n_samples, n_features)）、y（标签，shape=(n_samples,)）
        """
        # 数据格式转换
        X = np.asarray(X, dtype=np.float64)
        y_orig = np.asarray(y, dtype=np.float64)
        
        # 判断任务类型
        self._judge_task_type(y_orig)
        
        n_samples = len(y_orig)
        n_features = X.shape[1]
        # 初始化特征重要性
        self.feature_importances_ = np.zeros(n_features)
        
        if self.is_classifier and len(self.classes_) > 2:
            # 多分类：使用One-vs-All策略，为每个类别训练一组树
            self.trees = []  # 存储所有类别的树：[[class0_trees], [class1_trees], ...]
            for class_idx, class_label in enumerate(self.classes_):
                print(f"\n训练类别 {class_label} ({class_idx+1}/{len(self.classes_)})...")
                # 转换为二分类问题
                y_binary = (y_orig == class_label).astype(np.float64).reshape(-1, 1)
                f_prev = np.zeros(n_samples)
                class_trees = []
                
                for iter in range(self.n_estimators):
                    # 计算负梯度
                    residual = self._compute_negative_gradient(y_binary, f_prev)
                    
                    # 训练树
                    tree = self._CARTTree(
                        max_depth=self.max_depth,
                        min_samples_split=self.min_samples_split,
                        is_classifier=True,
                        max_thresholds=self.max_thresholds
                    )
                    tree.fit(X, residual, feature_importances=self.feature_importances_)
                    class_trees.append(tree)
                    
                    # 更新预测值
                    tree_pred = tree.predict(X)
                    f_prev += self.learning_rate * tree_pred
                    
                    # 打印进度
                    if (iter + 1) % 20 == 0 or (iter + 1) == self.n_estimators:
                        train_proba = self._sigmoid(f_prev)
                        train_acc = np.mean((train_proba >= 0.5).astype(int) == y_binary)
                        print(f"  进度：{iter+1}/{self.n_estimators}棵树，准确率：{train_acc:.4f}")
                
                self.trees.append(class_trees)
        else:
            # 二分类或回归
            y = y_orig.reshape(-1, 1)
            f_prev = np.zeros(n_samples)
            
            for iter in range(self.n_estimators):
                # 1. 计算当前迭代的负梯度（残差）
                residual = self._compute_negative_gradient(y, f_prev)

                # 2. 训练CART树拟合残差
                tree = self._CARTTree(
                    max_depth=self.max_depth, 
                    min_samples_split=self.min_samples_split,
                    is_classifier=self.is_classifier,
                    max_thresholds=self.max_thresholds
                )
                tree.fit(X, residual, feature_importances=self.feature_importances_)
                self.trees.append(tree)

                # 3. 更新模型预测值（累加基学习器预测结果，乘以学习率控制步长）
                tree_pred = tree.predict(X)
                f_prev += self.learning_rate * tree_pred

                # 日志：每20棵树或最后一棵打印训练进度（减少打印次数提升性能）
                if (iter + 1) % 20 == 0 or (iter + 1) == self.n_estimators:
                    if self.is_classifier:
                        train_proba = self._sigmoid(f_prev)
                        train_acc = np.mean((train_proba >= 0.5).astype(int) == y)
                        print(f"GBDT分类训练进度：{iter+1}/{self.n_estimators}棵树，训练准确率：{train_acc:.4f}")
                    else:
                        mse = np.mean((f_prev - y.flatten()) ** 2)
                        print(f"GBDT回归训练进度：{iter+1}/{self.n_estimators}棵树，训练MSE：{mse:.6f}")

        # 归一化特征重要性
        if np.sum(self.feature_importances_) > 0:
            self.feature_importances_ /= np.sum(self.feature_importances_)

        return self

    def predict_proba(self, X):
        """
        预测类别概率（仅分类任务）
        返回：shape=(n_samples, n_classes)，每一行对应各类别概率
        """
        if not self.is_classifier:
            raise NotImplementedError("回归模型不支持predict_proba")

        X = np.asarray(X, dtype=np.float64)
        n_samples = len(X)
        n_classes = len(self.classes_)
        
        if n_classes == 2:
            # 二分类：使用sigmoid激活
            f_final = np.zeros(n_samples)
            for tree in self.trees:
                f_final += self.learning_rate * tree.predict(X)
            proba_1 = self._sigmoid(f_final)
            proba_0 = 1 - proba_1
            return np.column_stack((proba_0, proba_1))
        else:
            # 多分类：One-vs-All策略
            # self.trees是二维列表：[[class0_trees], [class1_trees], ...]
            probas = np.zeros((n_samples, n_classes))
            
            for class_idx in range(n_classes):
                # 对每个类别，累加其所有树的预测
                f_final = np.zeros(n_samples)
                for tree in self.trees[class_idx]:
                    f_final += self.learning_rate * tree.predict(X)
                # 使用sigmoid得到该类别的得分
                probas[:, class_idx] = self._sigmoid(f_final)
            
            # Softmax归一化确保概率和为1
            exp_scores = np.exp(probas - np.max(probas, axis=1, keepdims=True))
            probas = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
            
            return probas

    def predict(self, X):
        """
        预测结果：
        - 分类：返回类别标签
        - 回归：返回连续值
        """
        X = np.asarray(X, dtype=np.float64)
        n_samples = len(X)
        
        if self.is_classifier:
            # 分类：使用概率预测结果
            probas = self.predict_proba(X)
            if len(self.classes_) == 2:
                # 二分类：概率>0.5则预测为1，否则为0
                predictions = (probas[:, 1] >= 0.5).astype(int)
                # 映射回原始类别
                if set(self.classes_) != {0, 1}:
                    class_map = {0: self.classes_[0], 1: self.classes_[1]}
                    predictions = np.array([class_map[p] for p in predictions])
            else:
                # 多分类：选择概率最大的类别
                predictions = np.argmax(probas, axis=1)
                # 映射回原始类别
                predictions = np.array([self.classes_[p] for p in predictions])
            return predictions
        else:
            # 回归：累加所有基学习器的预测结果
            f_final = np.zeros(n_samples)
            for tree in self.trees:
                f_final += self.learning_rate * tree.predict(X)
            return f_final