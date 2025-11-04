import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class CustomVisualization:
    """
    K-Means 聚类可视化：
    - 二维投影聚类散点图（PCA/SVD降维至2D），标注簇中心
    - 簇大小柱状图
    - 若提供真实标签：基于多数映射的“伪监督”混淆矩阵
    - 真实vs聚类标签分布对比
    """

    def __init__(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        try:
            plt.rcParams['axes.unicode_minus'] = False
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'Arial Unicode MS', 'sans-serif']
            self.font_params = {"fontsize": 10}
        except Exception:
            self.font_params = {"fontsize": 10}

    def _save(self, fig, filename):
        save_path = os.path.join(self.output_dir, filename)
        plt.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return '/static/' + os.path.join('images', os.path.basename(self.output_dir), filename).replace('\\', '/').replace('\\', '/')

    # --------------------------
    # 工具：PCA/SVD 降维到2D
    # --------------------------
    def _pca_2d(self, X):
        X = np.asarray(X, dtype=np.float64)
        X_centered = X - np.mean(X, axis=0, keepdims=True)
        # 使用SVD做PCA投影到前2主成分
        U, S, VT = np.linalg.svd(X_centered, full_matrices=False)
        components = VT[:2].T  # (n_features, 2)
        X_2d = np.dot(X_centered, components)  # (n_samples, 2)
        return X_2d, components, np.mean(X, axis=0, keepdims=True)

    def _project_centers(self, centers, components, mean_vec):
        centers = np.asarray(centers, dtype=np.float64)
        centers_centered = centers - mean_vec
        return np.dot(centers_centered, components)

    # --------------------------
    # 图1：二维投影聚类散点 + 簇中心
    # --------------------------
    def _plot_cluster_scatter(self, X, labels, centers):
        X_2d, components, mean_vec = self._pca_2d(X)
        centers_2d = self._project_centers(centers, components, mean_vec)
        k = centers.shape[0]

        fig, ax = plt.subplots(figsize=(10, 7))
        palette = sns.color_palette('tab10', n_colors=max(k, 2))
        for c in range(k):
            mask = (labels == c)
            if np.any(mask):
                ax.scatter(X_2d[mask, 0], X_2d[mask, 1], s=40, alpha=0.7, color=palette[c], label=f'簇 {c}', edgecolors='black', linewidths=0.3)

        # 绘制中心
        ax.scatter(centers_2d[:, 0], centers_2d[:, 1], s=200, color='yellow', edgecolors='red', marker='*', linewidths=1.5, label='簇中心')
        ax.set_title('K-Means：二维投影聚类散点图（含簇中心）', fontsize=14, fontweight='bold')
        ax.set_xlabel('主成分1', **self.font_params)
        ax.set_ylabel('主成分2', **self.font_params)
        ax.legend(**self.font_params)
        ax.grid(True, alpha=0.3)
        return self._save(fig, 'kmeans_cluster_scatter.png')

    # --------------------------
    # 图2：簇大小分布
    # --------------------------
    def _plot_cluster_sizes(self, labels, k):
        sizes = np.array([np.sum(labels == c) for c in range(k)], dtype=int)
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(range(k), sizes, color='#3498DB', alpha=0.85)
        for i, bar in enumerate(bars):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5, str(int(sizes[i])), ha='center', va='bottom', fontsize=9)
        ax.set_xlabel('簇ID', **self.font_params)
        ax.set_ylabel('样本数量', **self.font_params)
        ax.set_title('K-Means：簇大小分布', fontsize=14, fontweight='bold')
        ax.set_xticks(range(k))
        ax.grid(axis='y', alpha=0.3)
        return self._save(fig, 'kmeans_cluster_sizes.png')

    # --------------------------
    # 图3：真实标签 vs 聚类簇（若y存在）
    # --------------------------
    def _plot_contingency_matrix(self, y_true, labels):
        y_true = np.asarray(y_true)
        labels = np.asarray(labels)
        classes = np.unique(y_true)
        k = np.max(labels) + 1
        table = np.zeros((len(classes), k), dtype=int)
        for i, cls in enumerate(classes):
            for c in range(k):
                table[i, c] = int(np.sum((y_true == cls) & (labels == c)))

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(table, annot=True, fmt='d', cmap='Blues', xticklabels=[f'簇 {c}' for c in range(k)], yticklabels=[str(c) for c in classes], ax=ax)
        ax.set_xlabel('聚类簇', **self.font_params)
        ax.set_ylabel('真实类别', **self.font_params)
        ax.set_title('真实类别 vs 聚类簇（列联表）', fontsize=14, fontweight='bold')
        return self._save(fig, 'kmeans_contingency.png')

    # --------------------------
    # 图4：真实vs聚类标签分布对比（仅展示分布差异）
    # --------------------------
    def _plot_truth_vs_cluster_distribution(self, y_true, labels):
        y_true = np.asarray(y_true)
        labels = np.asarray(labels)
        true_vals, true_counts = np.unique(y_true, return_counts=True)
        clus_vals, clus_counts = np.unique(labels, return_counts=True)
        # 对齐x轴
        x_true = [str(v) for v in true_vals]
        x_clus = [f'簇{v}' for v in clus_vals]
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(max(len(x_true), len(x_clus)))
        width = 0.4
        # 扩展到同长度以便可视化
        t_counts = np.zeros_like(x, dtype=int)
        c_counts = np.zeros_like(x, dtype=int)
        t_counts[:len(true_counts)] = true_counts
        c_counts[:len(clus_counts)] = clus_counts
        ax.bar(x - width/2, t_counts, width, label='真实标签分布', color='#2ECC71', alpha=0.85)
        ax.bar(x + width/2, c_counts, width, label='聚类簇分布', color='#E74C3C', alpha=0.85)
        ax.set_title('真实标签分布 vs 聚类簇分布', fontsize=14, fontweight='bold')
        ax.set_xlabel('类别/簇', **self.font_params)
        ax.set_ylabel('样本数量', **self.font_params)
        ax.legend(**self.font_params)
        ax.grid(True, alpha=0.3)
        return self._save(fig, 'kmeans_truth_vs_clusters.png')

    # --------------------------
    # 入口
    # --------------------------
    def generate_visualizations(self, model, X_train, X_test, y_train, y_test, y_pred, y_pred_proba, **kwargs):
        results = {}
        try:
            # 基础变量
            X = np.asarray(X_test if X_test is not None else X_train)
            labels = np.asarray(y_pred)
            centers = getattr(model, 'cluster_centers_', None)
            if centers is None:
                return {}

            # 1) 聚类二维散点
            try:
                results['cluster_scatter'] = self._plot_cluster_scatter(X, labels, centers)
            except Exception:
                pass

            # 2) 簇大小
            try:
                results['cluster_sizes'] = self._plot_cluster_sizes(labels, centers.shape[0])
            except Exception:
                pass

            # 3) 若有真实标签：列联表 + 分布对比
            if y_test is not None:
                try:
                    results['contingency_matrix'] = self._plot_contingency_matrix(y_test, labels)
                except Exception:
                    pass
                try:
                    results['truth_vs_clusters'] = self._plot_truth_vs_cluster_distribution(y_test, labels)
                except Exception:
                    pass

        except Exception:
            return {}

        return results


