import numpy as np


class CustomModel:
    """
    手写朴素贝叶斯模型（支持二分类/多分类，自动适配连续/离散特征）
    核心逻辑：基于贝叶斯定理+特征条件独立假设，支持两种概率估计：
    - 高斯分布：用于连续特征（假设特征服从正态分布）
    - 多项式分布：用于离散特征（含拉普拉斯平滑避免零概率）
    接口：train(X, y)、predict(X)、predict_proba(X)
    """

    def __init__(self, learning_rate=0.01, alpha=1.0):
        self.learning_rate = learning_rate  # 框架兼容参数（朴素贝叶斯无学习率，仅占位）
        self.alpha = alpha  # 拉普拉斯平滑系数（离散特征专用）
        self.is_classifier = True  # 标记为分类模型（框架识别用）
        self.classes_ = None  # 类别列表（训练时从数据提取）
        self.n_classes = None  # 类别数量
        self.feature_type = None  # 特征类型：'continuous'（连续）/'discrete'（离散）
        # 高斯分布参数（连续特征）：{类别: {'mean': 特征均值, 'std': 特征标准差}}
        self.gaussian_params = {}
        # 多项式分布参数（离散特征）：{类别: {特征索引: {特征值: 条件概率 P(X_i=v|Y=c)}}}
        self.poly_params = {}
        self.feature_importances_ = None  # 特征重要性（基于条件概率差异）

    # --------------------------
    # 特征类型判断（自动适配）
    # --------------------------
    def _judge_feature_type(self, X):
        """判断特征类型：连续特征（默认）或离散特征（整数类型+唯一值占比低）"""
        X = np.asarray(X)
        # 非整数类型直接视为连续特征
        if not np.issubdtype(X.dtype, np.integer):
            return 'continuous'

        # 离散特征判断：所有特征的唯一值占比<10%（避免将连续特征误判为离散）
        n_samples = X.shape[0]
        for col in range(X.shape[1]):
            unique_ratio = len(np.unique(X[:, col])) / n_samples
            if unique_ratio > 0.1:
                return 'continuous'
        return 'discrete'

    # --------------------------
    # 类先验概率计算（基于样本频率）
    # --------------------------
    def _compute_class_priors(self, y):
        """计算 P(Y=c)：类别c在训练集中的出现频率"""
        self.classes_, class_counts = np.unique(y, return_counts=True)
        self.n_classes = len(self.classes_)
        total_samples = len(y)
        self.class_priors = class_counts / total_samples  # 形状：(n_classes,)

    # --------------------------
    # 高斯朴素贝叶斯（连续特征）
    # --------------------------
    def _gaussian_pdf(self, x, mean, std):
        """高斯概率密度函数：计算 P(X=x|Y=c)，避免标准差为0导致计算错误"""
        std = np.maximum(std, 1e-8)  # 防止除以0（使用np.maximum处理数组）
        exponent = np.exp(-((x - mean) ** 2) / (2 * std ** 2))
        return (1 / (np.sqrt(2 * np.pi) * std)) * exponent

    def _fit_gaussian(self, X, y):
        """拟合高斯分布参数：计算每个类别下各特征的均值和标准差"""
        for cls in self.classes_:
            # 提取当前类别的所有样本
            X_cls = X[y == cls]
            # 存储均值和标准差（形状：(n_features,)）
            self.gaussian_params[cls] = {
                'mean': np.mean(X_cls, axis=0),
                'std': np.std(X_cls, axis=0)
            }

    def _predict_proba_gaussian(self, X):
        """高斯模型预测后验概率：用对数概率避免数值下溢"""
        n_samples = X.shape[0]
        log_probs = np.zeros((n_samples, self.n_classes))  # 形状：(n_samples, n_classes)

        for cls_idx, cls in enumerate(self.classes_):
            # 1. 先验概率的对数：log(P(Y=c))
            log_prior = np.log(self.class_priors[cls_idx])
            # 2. 条件概率的对数：log(P(X|Y=c)) = sum(log(P(X_i|Y=c)))（特征独立假设）
            mean = self.gaussian_params[cls]['mean']
            std = self.gaussian_params[cls]['std']
            # 计算每个样本的条件概率对数（形状：(n_samples,)）
            log_likelihood = np.sum(np.log(self._gaussian_pdf(X, mean, std)), axis=1)
            # 3. 后验概率的对数：log(P(Y=c|X)) ∝ log(P(Y=c)) + log(P(X|Y=c))
            log_probs[:, cls_idx] = log_prior + log_likelihood

        # 转换为概率（softmax归一化，避免指数溢出）
        exp_log_probs = np.exp(log_probs - np.max(log_probs, axis=1, keepdims=True))
        return exp_log_probs / np.sum(exp_log_probs, axis=1, keepdims=True)

    # --------------------------
    # 多项式朴素贝叶斯（离散特征）
    # --------------------------
    def _fit_polynomial(self, X, y):
        """拟合多项式分布参数：计算每个类别下各特征值的条件概率（含拉普拉斯平滑）"""
        n_features = X.shape[1]
        for cls in self.classes_:
            # 提取当前类别的所有样本
            X_cls = X[y == cls]
            self.poly_params[cls] = {}  # 存储当前类别的特征参数

            for feat_idx in range(n_features):
                # 提取当前特征的所有值
                feat_vals = X_cls[:, feat_idx]
                # 统计特征值的出现次数
                unique_vals, val_counts = np.unique(feat_vals, return_counts=True)
                # 拉普拉斯平滑：分子+alpha，分母+alpha*特征值数量（避免零概率）
                total_count = len(feat_vals)
                val_probs = (val_counts + self.alpha) / (total_count + self.alpha * len(unique_vals))
                # 存储 {特征值: 条件概率}
                self.poly_params[cls][feat_idx] = dict(zip(unique_vals, val_probs))

    def _predict_proba_polynomial(self, X):
        """多项式模型预测后验概率：用对数概率避免数值下溢"""
        n_samples = X.shape[0]
        log_probs = np.zeros((n_samples, self.n_classes))  # 形状：(n_samples, n_classes)

        for cls_idx, cls in enumerate(self.classes_):
            # 1. 先验概率的对数：log(P(Y=c))
            log_prior = np.log(self.class_priors[cls_idx])
            # 2. 条件概率的对数：log(P(X|Y=c)) = sum(log(P(X_i|Y=c)))
            log_likelihood = np.zeros(n_samples)  # 每个样本的条件概率对数

            for feat_idx in range(X.shape[1]):
                for i in range(n_samples):
                    # 获取特征值（保持原始类型以匹配字典键）
                    val = X[i, feat_idx]
                    # 确保是Python标量而不是numpy数组
                    if hasattr(val, 'item'):
                        val = val.item()
                    
                    # 处理训练集中未出现的特征值（用平滑后的最小概率）
                    feat_probs = self.poly_params[cls][feat_idx]
                    if val in feat_probs:
                        log_likelihood[i] += np.log(feat_probs[val])
                    else:
                        # 未见过的特征值，使用拉普拉斯平滑的最小概率
                        unique_vals_count = len(feat_probs)
                        smooth_prob = self.alpha / (self.alpha * (unique_vals_count + 1))
                        log_likelihood[i] += np.log(smooth_prob)

            # 3. 后验概率的对数
            log_probs[:, cls_idx] = log_prior + log_likelihood

        # 转换为概率（softmax归一化）
        exp_log_probs = np.exp(log_probs - np.max(log_probs, axis=1, keepdims=True))
        return exp_log_probs / np.sum(exp_log_probs, axis=1, keepdims=True)

    # --------------------------
    # 模型核心接口（框架调用）
    # --------------------------
    def train(self, X, y):
        """
        训练朴素贝叶斯模型：自动判断特征类型，选择对应概率模型拟合
        参数：X（特征矩阵，shape=(n_samples, n_features)）、y（标签，shape=(n_samples,)）
        """
        # 数据格式转换
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        # 1. 判断特征类型（连续/离散）
        self.feature_type = self._judge_feature_type(X)
        print(
            f"朴素贝叶斯自动识别特征类型：{self.feature_type}，使用{['多项式', '高斯'][self.feature_type == 'continuous']}模型")
        # 2. 计算类先验概率
        self._compute_class_priors(y)
        # 3. 拟合对应概率模型
        if self.feature_type == 'continuous':
            self._fit_gaussian(X, y)
        else:
            self._fit_polynomial(X, y)
        
        # 4. 计算特征重要性
        self._compute_feature_importance()
        return self

    def _compute_feature_importance(self):
        """计算特征重要性：基于条件概率在不同类别间的差异"""
        n_features = len(self.gaussian_params[list(self.gaussian_params.keys())[0]]['mean']) if self.feature_type == 'continuous' else len(self.poly_params[list(self.poly_params.keys())[0]])
        feature_importance = np.zeros(n_features)
        
        if self.feature_type == 'continuous':
            # 连续特征：基于高斯分布参数差异
            for feat_idx in range(n_features):
                # 计算不同类别间该特征的差异
                means = [self.gaussian_params[cls]['mean'][feat_idx] for cls in self.classes_]
                stds = [self.gaussian_params[cls]['std'][feat_idx] for cls in self.classes_]
                
                # 特征重要性 = 类别间均值的方差 + 类别间标准差的方差
                mean_var = np.var(means)
                std_var = np.var(stds)
                feature_importance[feat_idx] = mean_var + std_var
        else:
            # 离散特征：基于条件概率差异
            for feat_idx in range(n_features):
                # 计算不同类别间该特征的条件概率差异
                prob_diffs = []
                for i, cls1 in enumerate(self.classes_):
                    for j, cls2 in enumerate(self.classes_):
                        if i < j:  # 避免重复计算
                            # 计算两个类别在该特征上的概率差异
                            prob1 = self.poly_params[cls1][feat_idx]
                            prob2 = self.poly_params[cls2][feat_idx]
                            
                            # 计算所有特征值的概率差异
                            all_vals = set(prob1.keys()) | set(prob2.keys())
                            diff_sum = 0
                            for val in all_vals:
                                p1 = prob1.get(val, 0.0)
                                p2 = prob2.get(val, 0.0)
                                diff_sum += abs(p1 - p2)
                            prob_diffs.append(diff_sum)
                
                feature_importance[feat_idx] = np.mean(prob_diffs) if prob_diffs else 0.0
        
        # 归一化特征重要性
        if np.sum(feature_importance) > 0:
            feature_importance /= np.sum(feature_importance)
        
        self.feature_importances_ = feature_importance

    def predict_proba(self, X):
        """
        预测类别概率：返回每个样本属于各类别的概率
        返回：shape=(n_samples, n_classes)，每行和为1
        """
        if not self.is_classifier:
            raise NotImplementedError("朴素贝叶斯仅支持分类任务")
            
        X = np.asarray(X, dtype=np.float64)
        # 根据特征类型选择概率预测方法
        if self.feature_type == 'continuous':
            return self._predict_proba_gaussian(X)
        else:
            return self._predict_proba_polynomial(X)

    def predict(self, X):
        """
        预测类别：选择后验概率最大的类别
        返回：shape=(n_samples,)，每个元素为类别标签
        """
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]