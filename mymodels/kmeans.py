import numpy as np


class CustomModel:
    """
    纯Numpy实现的K-Means聚类（不依赖第三方ML库）

    接口兼容本项目：
    - train(X, y): 忽略y，基于X无监督训练
    - predict(X): 预测每个样本的簇标签（0..k-1）
    - predict_proba(X): 返回到各簇中心的相对相似度（归一化为0-1），用于可视化/评估占位

    参数:
    - n_clusters: 簇数量
    - max_iter: 最大迭代次数
    - tol: 目标函数(inertia)的相对改进阈值，小于阈值则提前停止
    - init: 初始化方式，支持 'k-means++' 或 'random'
    - random_state: 随机种子
    """

    def __init__(self, learning_rate=0.01, n_clusters=3, max_iter=300, tol=1e-4, init='k-means++', random_state=42):
        # 与框架兼容的占位参数
        self.learning_rate = learning_rate

        # K-Means参数
        self.n_clusters = int(n_clusters)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.init = init
        self.random_state = int(random_state) if random_state is not None else None

        # 训练得到的属性
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = 0

        # 与框架接口对齐（用于前端/可视化）
        self.is_classifier = True  # 以分类模型形式暴露（标签为簇ID）
        self.classes_ = None       # [0..k-1]
        self.feature_importances_ = None  # 对K-Means不适用，置为None

        # 随机数生成器
        self._rng = np.random.RandomState(self.random_state)

    # --------------------------
    # 内部工具
    # --------------------------
    def _euclidean_distances(self, X, centers):
        """计算X到各簇中心的欧氏距离矩阵 (n_samples, n_clusters)。"""
        # 利用 (x - c)^2 = x^2 + c^2 - 2 x·c 的展开，向量化高效计算
        X_sq = np.sum(X * X, axis=1, keepdims=True)                 # (n_samples, 1)
        C_sq = np.sum(centers * centers, axis=1, keepdims=True).T   # (1, n_clusters)
        cross = np.dot(X, centers.T)                                # (n_samples, n_clusters)
        d2 = np.maximum(X_sq + C_sq - 2.0 * cross, 0.0)
        return np.sqrt(d2)

    def _init_centroids_random(self, X):
        indices = self._rng.choice(X.shape[0], size=self.n_clusters, replace=False)
        return X[indices].copy()

    def _init_centroids_kpp(self, X):
        """k-means++ 初始化。"""
        n_samples = X.shape[0]
        centers = np.empty((self.n_clusters, X.shape[1]), dtype=X.dtype)
        # 1) 首个中心随机选
        first_idx = self._rng.randint(n_samples)
        centers[0] = X[first_idx]
        # 2) 后续中心按与最近中心的距离平方加权采样
        closest_dist_sq = self._euclidean_distances(X, centers[[0]]).reshape(-1) ** 2
        for c in range(1, self.n_clusters):
            probs = closest_dist_sq / np.sum(closest_dist_sq)
            next_idx = self._rng.choice(n_samples, p=probs)
            centers[c] = X[next_idx]
            dist_sq_new_center = self._euclidean_distances(X, centers[[c]]).reshape(-1) ** 2
            closest_dist_sq = np.minimum(closest_dist_sq, dist_sq_new_center)
        return centers

    def _assign_labels(self, X, centers):
        dists = self._euclidean_distances(X, centers)
        labels = np.argmin(dists, axis=1)
        return labels, dists

    def _compute_centers(self, X, labels):
        """基于当前分配更新簇中心；若簇为空，重置为随机样本。"""
        n_features = X.shape[1]
        centers = np.zeros((self.n_clusters, n_features), dtype=X.dtype)
        for k in range(self.n_clusters):
            mask = (labels == k)
            if np.any(mask):
                centers[k] = np.mean(X[mask], axis=0)
            else:
                # 处理空簇：随机重启一个样本为中心
                rand_idx = self._rng.randint(X.shape[0])
                centers[k] = X[rand_idx]
        return centers

    def _compute_inertia(self, dists, labels):
        """计算惯性(inertia)：各样本到其簇中心距离平方和。"""
        # dists为到各中心距离矩阵；取选定簇的距离平方求和
        rows = np.arange(dists.shape[0])
        chosen = dists[rows, labels]
        return float(np.sum(chosen * chosen))

    # --------------------------
    # 对外接口
    # --------------------------
    def train(self, X, y=None):
        """训练K-Means（忽略y）。"""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[0] < self.n_clusters:
            raise ValueError("样本数必须≥簇数，且X为二维数组")

        # 初始化中心
        if self.init in ('k-means++', 'kmeans++', 'kpp'):
            centers = self._init_centroids_kpp(X)
        elif self.init == 'random':
            centers = self._init_centroids_random(X)
        else:
            raise ValueError("init 仅支持 'k-means++' 或 'random'")

        inertia_prev = np.inf
        for it in range(self.max_iter):
            labels, dists = self._assign_labels(X, centers)
            inertia = self._compute_inertia(dists, labels)

            # 更新中心
            new_centers = self._compute_centers(X, labels)

            # 收敛判定：惯性相对改进不足tol
            if inertia_prev - inertia <= self.tol * max(1.0, inertia_prev):
                centers = new_centers
                self.n_iter_ = it + 1
                break

            centers = new_centers
            inertia_prev = inertia
            self.n_iter_ = it + 1

        # 最终赋值
        self.cluster_centers_ = centers
        self.labels_ = labels
        self.inertia_ = inertia
        self.classes_ = list(range(self.n_clusters))
        self.feature_importances_ = None
        return self

    def predict(self, X):
        if self.cluster_centers_ is None:
            raise ValueError("模型未训练，请先调用train()方法")
        X = np.asarray(X, dtype=np.float64)
        labels, _ = self._assign_labels(X, self.cluster_centers_)
        return labels

    def predict_proba(self, X):
        """
        返回基于到簇中心的相对相似度（按逆距离或高斯核进行归一化）。
        - 这不是严格意义上的概率，仅用于可视化或需要概率矩阵的接口兼容。
        """
        if self.cluster_centers_ is None:
            raise ValueError("模型未训练，请先调用train()方法")
        X = np.asarray(X, dtype=np.float64)
        dists = self._euclidean_distances(X, self.cluster_centers_)  # (n_samples, k)

        # 使用高斯核将距离转为相似度，然后按行归一化
        # sigma 设为各样本到最近中心距离的中位数的尺度
        min_d = np.minimum.reduce(dists.T).reshape(-1, 1)  # (n_samples,1)
        sigma = np.median(min_d) if np.median(min_d) > 0 else 1.0
        sim = np.exp(- (dists ** 2) / (2 * (sigma ** 2)))
        row_sums = np.sum(sim, axis=1, keepdims=True)
        # 避免除零
        row_sums[row_sums == 0] = 1.0
        probs = sim / row_sums
        return probs


