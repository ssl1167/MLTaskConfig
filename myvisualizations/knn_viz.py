import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.model_selection import learning_curve
from sklearn.decomposition import PCA
import seaborn as sns


class CustomVisualization:
    def __init__(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir  # 可视化保存目录
        self._set_fonts()
        self.colors = {
            "train": "#3498DB",  # 训练样本颜色（蓝色）
            "test": "#E74C3C",  # 测试样本颜色（红色）
            "neighbor": "#F39C12"  # 近邻样本颜色（橙色）
        }  # 统一颜色方案
        
    def _set_fonts(self):
        """设置中文字体支持"""
        try:
            plt.rcParams['axes.unicode_minus'] = False
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'Arial Unicode MS', 'sans-serif']
            self.font_params = {"fontsize": 10}
        except Exception as e:
            print(f"字体设置失败: {str(e)}")
            self.font_params = {"fontsize": 10}

    def _get_2d_features(self, X, feature_names):
        """提取二维特征（优先用前两维，适配决策边界可视化）"""
        X = np.array(X)
        if X.shape[1] < 2:
            raise ValueError("KNN决策边界可视化需要至少2个特征（当前仅支持二维特征）")
        # 取前两维特征用于绘图
        X_2d = X[:, :2]
        # 处理特征名（无则用默认名）
        if feature_names and len(feature_names) >= 2:
            return X_2d, feature_names[:2]
        else:
            return X_2d, ["Feature 0", "Feature 1"]

    def _generate_neighbor_heatmap(self, model, X_test_sample, feature_names):
        """生成单个测试样本的近邻分布热力图：展示与所有训练样本的距离，标记K个近邻"""
        # 提取二维特征
        X_train_2d, feat_names = self._get_2d_features(model.X_train, feature_names)
        X_test_sample_2d = np.array(X_test_sample)[:2].reshape(1, -1)  # 测试样本取前两维

        # 计算测试样本与所有训练样本的距离
        distances = [model._euclidean_distance(X_test_sample_2d[0], x_train) for x_train in X_train_2d]
        distances = np.array(distances)
        # 获取K个近邻的索引
        neighbor_indices = np.argsort(distances)[:model.n_neighbors]

        # 绘制热力图（距离越小颜色越深）
        fig, ax = plt.subplots(figsize=(10, 8))
        # 绘制所有训练样本（颜色深浅表示距离）
        scatter_all = ax.scatter(
            X_train_2d[:, 0], X_train_2d[:, 1],
            c=distances, cmap="Blues_r", alpha=0.7, s=60, edgecolors="gray", linewidth=0.5
        )
        # 标记K个近邻（橙色突出）
        ax.scatter(
            X_train_2d[neighbor_indices, 0], X_train_2d[neighbor_indices, 1],
            c=self.colors["neighbor"], marker="*", s=200, edgecolors="black", linewidth=1,
            label=f"K={model.n_neighbors} 近邻"
        )
        # 标记当前测试样本（红色星形）
        ax.scatter(
            X_test_sample_2d[:, 0], X_test_sample_2d[:, 1],
            c=self.colors["test"], marker="D", s=150, edgecolors="black", linewidth=1,
            label="测试样本"
        )

        # 添加颜色条（表示距离）
        cbar = plt.colorbar(scatter_all, ax=ax)
        cbar.set_label(f"与测试样本的欧氏距离", **self.font_params)

        # 设置图表样式
        ax.set_xlabel(feat_names[0], **self.font_params)
        ax.set_ylabel(feat_names[1], **self.font_params)
        ax.set_title(f"KNN近邻分布（测试样本与训练样本距离）", fontsize=14, fontweight="bold")
        ax.legend(**self.font_params)
        ax.grid(alpha=0.3)

        # 保存图片
        heatmap_path = os.path.join(self.output_dir, "knn_neighbor_heatmap.png")
        plt.tight_layout()
        plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
        plt.close()
        return heatmap_path

    def _generate_decision_boundary(self, model, feature_names):
        """生成KNN决策边界图（仅支持二维特征+分类任务）"""
        if not model.is_classifier:
            raise ValueError("决策边界可视化仅支持分类任务")

        # 提取二维特征
        X_train_2d, feat_names = self._get_2d_features(model.X_train, feature_names)
        y_train = model.y_train

        # 构建网格（覆盖训练集特征范围）
        x_min, x_max = X_train_2d[:, 0].min() - 1, X_train_2d[:, 0].max() + 1
        y_min, y_max = X_train_2d[:, 1].min() - 1, X_train_2d[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
        # 网格样本的完整特征（补全其他维度为0，适配模型预测）
        grid_samples = np.c_[xx.ravel(), yy.ravel()]
        if grid_samples.shape[1] < model.X_train.shape[1]:
            grid_samples = np.hstack([grid_samples, np.zeros((grid_samples.shape[0], model.X_train.shape[1] - 2))])

        # 预测网格样本类别（生成决策边界）
        grid_pred = model.predict(grid_samples).reshape(xx.shape)

        # 绘制决策边界与训练样本
        fig, ax = plt.subplots(figsize=(10, 8))
        # 绘制决策边界（填充不同类别区域）
        contour = ax.contourf(xx, yy, grid_pred, alpha=0.3, cmap="viridis")
        # 绘制训练样本（按真实类别着色）
        unique_classes = model.classes_
        for cls in unique_classes:
            mask = y_train == cls
            ax.scatter(
                X_train_2d[mask, 0], X_train_2d[mask, 1],
                label=f"类别 {cls}", alpha=0.8, s=60, edgecolors="black", linewidth=0.5
            )

        # 设置图表样式
        ax.set_xlabel(feat_names[0], **self.font_params)
        ax.set_ylabel(feat_names[1], **self.font_params)
        ax.set_title(f"KNN决策边界（K={model.n_neighbors}，权重={model.weights}）", fontsize=14, fontweight="bold")
        ax.legend(**self.font_params)
        ax.grid(alpha=0.3)

        # 保存图片
        db_path = os.path.join(self.output_dir, "knn_decision_boundary.png")
        plt.tight_layout()
        plt.savefig(db_path, dpi=150, bbox_inches='tight')
        plt.close()
        return db_path

    def _generate_truth_vs_pred(self, y_test, y_pred, model):
        """生成真实值vs预测值对比图（分类/回归分别适配）"""
        y_test = np.array(y_test)
        y_pred = np.array(y_pred)
        fig, ax = plt.subplots(figsize=(9, 6))

        if model.is_classifier:
            # 分类任务：类别分布对比（条形图）
            unique_classes = np.unique(np.concatenate([y_test, y_pred]))
            true_counts = [np.sum(y_test == cls) for cls in unique_classes]
            pred_counts = [np.sum(y_pred == cls) for cls in unique_classes]

            x = np.arange(len(unique_classes))
            width = 0.35
            # 绘制真实标签条形
            bars1 = ax.bar(x - width / 2, true_counts, width, label="真实标签", color=self.colors["train"], alpha=0.8)
            # 绘制预测标签条形
            bars2 = ax.bar(x + width / 2, pred_counts, width, label="预测标签", color=self.colors["test"], alpha=0.8)

            # 添加数值标签
            for bar, count in zip(bars1, true_counts):
                ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.5,
                        str(count), ha="center", va="bottom", **self.font_params)
            for bar, count in zip(bars2, pred_counts):
                ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.5,
                        str(count), ha="center", va="bottom", **self.font_params)

            ax.set_xlabel("类别", **self.font_params)
            ax.set_ylabel("样本数量", **self.font_params)
            ax.set_title(f"KNN：真实类别vs预测类别（测试集，K={model.n_neighbors}）", fontsize=14, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels([str(cls) for cls in unique_classes], **self.font_params)

        else:
            # 回归任务：散点图（真实值vs预测值）
            ax.scatter(y_test, y_pred, alpha=0.6, color="#9B59B6", s=60, edgecolors="black", linewidth=0.5)
            # 添加理想预测线（y_pred = y_test）
            min_val = min(np.min(y_test), np.min(y_pred))
            max_val = max(np.max(y_test), np.max(y_pred))
            ax.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.8, label="理想预测线")

            ax.set_xlabel("真实值", **self.font_params)
            ax.set_ylabel("预测值", **self.font_params)
            ax.set_title(f"KNN：真实值vs预测值（回归任务，K={model.n_neighbors}）", fontsize=14, fontweight="bold")
            ax.legend(**self.font_params)

        ax.grid(alpha=0.3)
        # 保存图片
        truth_pred_path = os.path.join(self.output_dir, "knn_truth_vs_pred.png")
        plt.tight_layout()
        plt.savefig(truth_pred_path, dpi=150, bbox_inches='tight')
        plt.close()
        return truth_pred_path

    def generate_visualizations(self, model, X_train, X_test, y_train, y_test, y_pred, y_pred_proba, feature_names=None,
                                target_names=None):
        """
        生成KNN全面可视化（涵盖数据探索、特征工程、模型评估三个阶段）
        返回格式：{可视化名称: 静态文件路径}（适配前端访问）
        """
        visualizations = {}
        try:
            # ========== 第一阶段：数据探索与理解 ==========
            # 1. 数据分布图（单变量分析）
            data_dist_path = self._plot_data_distribution(y_train, y_test, target_names)
            if data_dist_path:
                visualizations["data_distribution"] = data_dist_path
            
            # 2. 特征相关性矩阵（多变量关系分析）
            corr_matrix_path = self._plot_correlation_matrix(X_train, feature_names)
            if corr_matrix_path:
                visualizations["correlation_matrix"] = corr_matrix_path

            # 3. PCA降维可视化（适用于高维数据）
            if X_train.shape[1] > 2:
                pca_path = self._plot_pca_visualization(model, X_train, y_train, X_test, y_test)
                if pca_path:
                    visualizations["pca_visualization"] = pca_path

            # ========== 第二阶段：特征工程与模型构建 ==========
            # 4. 特征重要性图（基于距离贡献）
            feat_imp_path = self._plot_feature_importance(model, X_test, y_test, feature_names)
            if feat_imp_path:
                visualizations["feature_importance"] = feat_imp_path

            # 5. K值选择曲线（KNN特有）
            k_selection_path = self._plot_k_selection(model, X_train, y_train, X_test, y_test)
            if k_selection_path:
                visualizations["k_selection"] = k_selection_path

            # 6. 近邻分布热力图（取第一个测试样本作为示例）
            if len(X_test) > 0:
                heatmap_path = self._generate_neighbor_heatmap(model, X_test[0], feature_names)
                heatmap_viz_path = f"/static/images/{os.path.basename(self.output_dir)}/{os.path.basename(heatmap_path)}"
                visualizations["neighbor_heatmap"] = heatmap_viz_path

            # 7. 决策边界图（仅分类任务+二维特征）
            if model.is_classifier and X_train.shape[1] >= 2:
                db_path = self._generate_decision_boundary(model, feature_names)
                db_viz_path = f"/static/images/{os.path.basename(self.output_dir)}/{os.path.basename(db_path)}"
                visualizations["decision_boundary"] = db_viz_path

            # ========== 第三阶段：模型评估与性能分析 ==========
            # 8. 混淆矩阵（分类任务）
            if model.is_classifier:
                confusion_path = self._plot_confusion_matrix(y_test, y_pred, target_names)
                if confusion_path:
                    visualizations["confusion_matrix"] = confusion_path
                
                # 9. ROC曲线
                roc_path = self._plot_roc_curves(y_test, y_pred_proba, target_names)
                if roc_path:
                    visualizations["roc_curve"] = roc_path
                
                # 10. 精确率-召回率曲线
                pr_path = self._plot_precision_recall_curve(y_test, y_pred_proba, target_names)
                if pr_path:
                    visualizations["precision_recall_curve"] = pr_path
                
                # 11. 校准曲线（仅二分类）
                if len(np.unique(y_test)) == 2:
                    calib_path = self._plot_calibration_curve(y_test, y_pred_proba)
                    if calib_path:
                        visualizations["calibration_curve"] = calib_path
            else:
                # 回归任务：残差图
                residual_path = self._plot_residuals(y_test, y_pred)
                if residual_path:
                    visualizations["residual_plot"] = residual_path

            # 12. 学习曲线（通用）
            learning_path = self._plot_learning_curve(model, X_train, y_train)
            if learning_path:
                visualizations["learning_curve"] = learning_path

            # 13. 真实vs预测对比可视化（通用）
            truth_pred_path = self._generate_truth_vs_pred(y_test, y_pred, model)
            truth_pred_viz_path = f"/static/images/{os.path.basename(self.output_dir)}/{os.path.basename(truth_pred_path)}"
            visualizations["truth_vs_pred"] = truth_pred_viz_path

        except Exception as e:
            print(f"KNN可视化生成失败: {str(e)}")
            return {}  # 失败时返回空字典，避免框架报错

        return visualizations

    def _plot_data_distribution(self, y_train, y_test, target_names=None):
        """绘制数据分布图（单变量分析）"""
        save_path = os.path.join(self.output_dir, 'data_distribution.png')
        if os.path.exists(save_path):
            return '/static/' + os.path.join('images', os.path.basename(self.output_dir), os.path.basename(save_path)).replace('\\', '/')
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 训练集分布
        unique_train, counts_train = np.unique(y_train, return_counts=True)
        ax1.bar(unique_train, counts_train, color='skyblue', alpha=0.7, label='训练集')
        ax1.set_title('训练集类别分布', fontsize=12, fontweight='bold')
        ax1.set_xlabel('类别', **self.font_params)
        ax1.set_ylabel('样本数量', **self.font_params)
        ax1.grid(True, alpha=0.3)
        
        # 测试集分布
        unique_test, counts_test = np.unique(y_test, return_counts=True)
        ax2.bar(unique_test, counts_test, color='lightcoral', alpha=0.7, label='测试集')
        ax2.set_title('测试集类别分布', fontsize=12, fontweight='bold')
        ax2.set_xlabel('类别', **self.font_params)
        ax2.set_ylabel('样本数量', **self.font_params)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return '/static/' + os.path.join('images', os.path.basename(self.output_dir), os.path.basename(save_path)).replace('\\', '/')

    def _plot_correlation_matrix(self, X_train, feature_names=None):
        """绘制特征相关性热力图（多变量关系分析）"""
        save_path = os.path.join(self.output_dir, 'correlation_matrix.png')
        if os.path.exists(save_path):
            return '/static/' + os.path.join('images', os.path.basename(self.output_dir), os.path.basename(save_path)).replace('\\', '/')
            
        # 计算相关性矩阵
        corr_matrix = np.corrcoef(X_train.T)
        
        # 设置特征名称
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(X_train.shape[1])]
        
        # 绘制热力图
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # 只显示下三角
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f', cbar_kws={"shrink": .8},
                   xticklabels=feature_names, yticklabels=feature_names)
        plt.title('特征相关性矩阵', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right', **self.font_params)
        plt.yticks(rotation=0, **self.font_params)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return '/static/' + os.path.join('images', os.path.basename(self.output_dir), os.path.basename(save_path)).replace('\\', '/')

    def _plot_feature_importance(self, model, X_test, y_test, feature_names=None):
        """绘制特征重要性图（基于距离贡献）"""
        save_path = os.path.join(self.output_dir, 'feature_importance.png')
        if os.path.exists(save_path):
            return '/static/' + os.path.join('images', os.path.basename(self.output_dir), os.path.basename(save_path)).replace('\\', '/')
            
        # 计算特征重要性
        feature_importance = model._calculate_feature_importance(X_test, y_test)
        
        # 设置特征名称
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(len(feature_importance))]
        
        # 排序（按重要性降序）
        sorted_idx = np.argsort(feature_importance)[::-1]
        sorted_feat_names = [feature_names[i] for i in sorted_idx]
        sorted_importances = feature_importance[sorted_idx]
        
        # 绘制柱状图
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(range(len(sorted_importances)), sorted_importances, color="#3498DB", alpha=0.8)
        # 添加数值标签
        for bar, imp in zip(bars, sorted_importances):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                    f"{imp:.3f}", ha="center", va="bottom", **self.font_params)
        
        # 设置图表样式
        ax.set_xlabel("特征", **self.font_params)
        ax.set_ylabel("重要性（基于距离贡献）", **self.font_params)
        ax.set_title("KNN特征重要性", fontsize=14, fontweight="bold")
        ax.set_xticks(range(len(sorted_importances)))
        ax.set_xticklabels(sorted_feat_names, rotation=45, ha="right", **self.font_params)
        ax.set_ylim(0, max(sorted_importances) * 1.2)
        ax.grid(axis="y", alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return '/static/' + os.path.join('images', os.path.basename(self.output_dir), os.path.basename(save_path)).replace('\\', '/')

    def _plot_confusion_matrix(self, y_test, y_pred, target_names=None):
        """绘制混淆矩阵热力图"""
        save_path = os.path.join(self.output_dir, 'confusion_matrix.png')
        if os.path.exists(save_path):
            return '/static/' + os.path.join('images', os.path.basename(self.output_dir), os.path.basename(save_path)).replace('\\', '/')
            
        # 计算混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        
        # 设置类别名称
        if target_names is None:
            target_names = [str(i) for i in range(len(np.unique(y_test)))]
        
        # 绘制热力图
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=target_names, yticklabels=target_names)
        plt.title('混淆矩阵', fontsize=14, fontweight='bold')
        plt.xlabel('预测类别', **self.font_params)
        plt.ylabel('真实类别', **self.font_params)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return '/static/' + os.path.join('images', os.path.basename(self.output_dir), os.path.basename(save_path)).replace('\\', '/')

    def _plot_roc_curves(self, y_test, y_pred_proba, target_names=None):
        """绘制ROC曲线"""
        if y_pred_proba is None:
            return None
            
        save_path = os.path.join(self.output_dir, 'roc_curve.png')
        if os.path.exists(save_path):
            return '/static/' + os.path.join('images', os.path.basename(self.output_dir), os.path.basename(save_path)).replace('\\', '/')
            
        plt.figure(figsize=(8, 6))
        
        # 获取类别
        classes = np.unique(y_test)
        n_classes = len(classes)
        
        if n_classes == 2:
            # 二分类：直接绘制ROC曲线
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC曲线 (AUC = {roc_auc:.2f})')
        else:
            # 多分类：One-vs-Rest
            for i, class_label in enumerate(classes):
                # 将多分类转换为二分类
                y_binary = (y_test == class_label).astype(int)
                fpr, tpr, _ = roc_curve(y_binary, y_pred_proba[:, i])
                roc_auc = auc(fpr, tpr)
                class_name = target_names[i] if target_names else f'类别 {class_label}'
                plt.plot(fpr, tpr, lw=2, 
                        label=f'{class_name} (AUC = {roc_auc:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='随机分类器')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假正率 (FPR)', **self.font_params)
        plt.ylabel('真正率 (TPR)', **self.font_params)
        plt.title('ROC曲线', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", **self.font_params)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return '/static/' + os.path.join('images', os.path.basename(self.output_dir), os.path.basename(save_path)).replace('\\', '/')

    def _plot_precision_recall_curve(self, y_test, y_pred_proba, target_names=None):
        """绘制精确率-召回率曲线"""
        if y_pred_proba is None:
            return None
            
        save_path = os.path.join(self.output_dir, 'precision_recall_curve.png')
        if os.path.exists(save_path):
            return '/static/' + os.path.join('images', os.path.basename(self.output_dir), os.path.basename(save_path)).replace('\\', '/')
            
        plt.figure(figsize=(8, 6))
        
        # 获取类别
        classes = np.unique(y_test)
        n_classes = len(classes)
        
        if n_classes == 2:
            # 二分类
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba[:, 1])
            plt.plot(recall, precision, color='darkorange', lw=2, label='PR曲线')
        else:
            # 多分类：One-vs-Rest
            for i, class_label in enumerate(classes):
                y_binary = (y_test == class_label).astype(int)
                precision, recall, _ = precision_recall_curve(y_binary, y_pred_proba[:, i])
                class_name = target_names[i] if target_names else f'类别 {class_label}'
                plt.plot(recall, precision, lw=2, label=f'{class_name}')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('召回率', **self.font_params)
        plt.ylabel('精确率', **self.font_params)
        plt.title('精确率-召回率曲线', fontsize=14, fontweight='bold')
        plt.legend(loc="lower left", **self.font_params)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return '/static/' + os.path.join('images', os.path.basename(self.output_dir), os.path.basename(save_path)).replace('\\', '/')

    def _plot_learning_curve(self, model, X_train, y_train):
        """绘制学习曲线"""
        save_path = os.path.join(self.output_dir, 'learning_curve.png')
        if os.path.exists(save_path):
            return '/static/' + os.path.join('images', os.path.basename(self.output_dir), os.path.basename(save_path)).replace('\\', '/')
            
        # 定义学习曲线函数
        def custom_learning_curve(estimator, X, y, train_sizes, cv=5):
            train_scores = []
            val_scores = []
            
            for train_size in train_sizes:
                train_scores_fold = []
                val_scores_fold = []
                
                # 简单的交叉验证
                indices = np.random.permutation(len(X))
                fold_size = len(X) // cv
                
                for i in range(cv):
                    start_idx = i * fold_size
                    end_idx = (i + 1) * fold_size if i < cv - 1 else len(X)
                    val_indices = indices[start_idx:end_idx]
                    train_indices = np.concatenate([indices[:start_idx], indices[end_idx:]])
                    
                    # 限制训练集大小
                    if len(train_indices) > train_size:
                        train_indices = train_indices[:train_size]
                    
                    X_train_fold = X[train_indices]
                    y_train_fold = y[train_indices]
                    X_val_fold = X[val_indices]
                    y_val_fold = y[val_indices]
                    
                    # 训练模型（只复制超参数，不复制训练数据）
                    model_copy = type(estimator)(
                        learning_rate=estimator.learning_rate,
                        n_neighbors=getattr(estimator, 'n_neighbors', 5),
                        weights=getattr(estimator, 'weights', 'uniform')
                    )
                    model_copy.train(X_train_fold, y_train_fold)
                    
                    # 评估
                    train_pred = model_copy.predict(X_train_fold)
                    val_pred = model_copy.predict(X_val_fold)
                    
                    if model.is_classifier:
                        train_score = np.mean(train_pred == y_train_fold)
                        val_score = np.mean(val_pred == y_val_fold)
                    else:
                        train_score = 1 - np.mean((y_train_fold - train_pred) ** 2) / np.var(y_train_fold)
                        val_score = 1 - np.mean((y_val_fold - val_pred) ** 2) / np.var(y_val_fold)
                    
                    train_scores_fold.append(train_score)
                    val_scores_fold.append(val_score)
                
                train_scores.append(np.mean(train_scores_fold))
                val_scores.append(np.mean(val_scores_fold))
            
            return np.array(train_scores), np.array(val_scores)
        
        # 计算学习曲线
        train_sizes = np.linspace(0.1, 1.0, 10) * len(X_train)
        train_sizes = train_sizes.astype(int)
        train_sizes = np.unique(train_sizes)
        
        train_scores, val_scores = custom_learning_curve(model, X_train, y_train, train_sizes)
        
        # 绘制学习曲线
        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_scores, 'o-', color='blue', label='训练集')
        plt.plot(train_sizes, val_scores, 'o-', color='red', label='验证集')
        plt.xlabel('训练样本数', **self.font_params)
        plt.ylabel('准确率' if model.is_classifier else 'R²分数', **self.font_params)
        plt.title('学习曲线', fontsize=14, fontweight='bold')
        plt.legend(**self.font_params)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return '/static/' + os.path.join('images', os.path.basename(self.output_dir), os.path.basename(save_path)).replace('\\', '/')

    def _plot_calibration_curve(self, y_test, y_pred_proba):
        """绘制校准曲线（仅分类任务）"""
        if y_pred_proba is None or len(np.unique(y_test)) != 2:
            return None
            
        save_path = os.path.join(self.output_dir, 'calibration_curve.png')
        if os.path.exists(save_path):
            return '/static/' + os.path.join('images', os.path.basename(self.output_dir), os.path.basename(save_path)).replace('\\', '/')
            
        from sklearn.calibration import calibration_curve
        
        # 计算校准曲线
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_test, y_pred_proba[:, 1], n_bins=10)
        
        # 绘制校准曲线
        plt.figure(figsize=(8, 6))
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", 
                label="KNN", color='blue')
        plt.plot([0, 1], [0, 1], "k:", label="完美校准")
        plt.xlabel('平均预测概率', **self.font_params)
        plt.ylabel('正例比例', **self.font_params)
        plt.title('校准曲线', fontsize=14, fontweight='bold')
        plt.legend(**self.font_params)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return '/static/' + os.path.join('images', os.path.basename(self.output_dir), os.path.basename(save_path)).replace('\\', '/')

    def _plot_residuals(self, y_test, y_pred):
        """绘制残差图（仅回归任务）"""
        save_path = os.path.join(self.output_dir, 'residual_plot.png')
        if os.path.exists(save_path):
            return '/static/' + os.path.join('images', os.path.basename(self.output_dir), os.path.basename(save_path)).replace('\\', '/')
            
        # 计算残差
        residuals = y_test - y_pred
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 残差 vs 预测值
        ax1.scatter(y_pred, residuals, alpha=0.6, color='blue')
        ax1.axhline(y=0, color='red', linestyle='--')
        ax1.set_xlabel('预测值', **self.font_params)
        ax1.set_ylabel('残差', **self.font_params)
        ax1.set_title('残差 vs 预测值', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 残差直方图
        ax2.hist(residuals, bins=20, alpha=0.7, color='green', edgecolor='black')
        ax2.set_xlabel('残差', **self.font_params)
        ax2.set_ylabel('频次', **self.font_params)
        ax2.set_title('残差分布', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return '/static/' + os.path.join('images', os.path.basename(self.output_dir), os.path.basename(save_path)).replace('\\', '/')

    def _plot_k_selection(self, model, X_train, y_train, X_test, y_test):
        """绘制K值选择曲线（展示不同K值对性能的影响）"""
        save_path = os.path.join(self.output_dir, 'k_selection.png')
        if os.path.exists(save_path):
            return '/static/' + os.path.join('images', os.path.basename(self.output_dir), os.path.basename(save_path)).replace('\\', '/')
            
        # 测试不同的K值
        k_values = range(1, min(21, len(X_train) // 2))  # 限制K值范围
        train_scores = []
        test_scores = []
        
        for k in k_values:
            # 创建临时模型
            temp_model = type(model)(n_neighbors=k, weights=model.weights)
            temp_model.train(X_train, y_train)
            
            # 计算训练集和测试集性能
            train_pred = temp_model.predict(X_train)
            test_pred = temp_model.predict(X_test)
            
            if model.is_classifier:
                train_score = np.mean(train_pred == y_train)
                test_score = np.mean(test_pred == y_test)
            else:
                train_score = 1 - np.mean((y_train - train_pred) ** 2) / np.var(y_train)
                test_score = 1 - np.mean((y_test - test_pred) ** 2) / np.var(y_test)
            
            train_scores.append(train_score)
            test_scores.append(test_score)
        
        # 绘制K值选择曲线
        plt.figure(figsize=(10, 6))
        plt.plot(k_values, train_scores, 'o-', color='blue', label='训练集')
        plt.plot(k_values, test_scores, 'o-', color='red', label='测试集')
        plt.xlabel('K值', **self.font_params)
        plt.ylabel('准确率' if model.is_classifier else 'R²分数', **self.font_params)
        plt.title('K值选择曲线', fontsize=14, fontweight='bold')
        plt.legend(**self.font_params)
        plt.grid(True, alpha=0.3)
        
        # 标注最优K值
        best_k_idx = np.argmax(test_scores)
        best_k = k_values[best_k_idx]
        best_score = test_scores[best_k_idx]
        plt.annotate(f'最优K={best_k}\n性能={best_score:.3f}', 
                    xy=(best_k, best_score), xytext=(best_k+2, best_score-0.05),
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                    arrowprops=dict(arrowstyle='->', color='black'))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return '/static/' + os.path.join('images', os.path.basename(self.output_dir), os.path.basename(save_path)).replace('\\', '/')

    def _plot_pca_visualization(self, model, X_train, y_train, X_test, y_test):
        """绘制PCA降维可视化（适用于高维数据）"""
        save_path = os.path.join(self.output_dir, 'pca_visualization.png')
        if os.path.exists(save_path):
            return '/static/' + os.path.join('images', os.path.basename(self.output_dir), os.path.basename(save_path)).replace('\\', '/')
            
        # 合并训练集和测试集进行PCA
        X_combined = np.vstack([X_train, X_test])
        y_combined = np.concatenate([y_train, y_test])
        
        # PCA降维到2D
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_combined)
        
        # 分离训练集和测试集的PCA结果
        X_train_pca = X_pca[:len(X_train)]
        X_test_pca = X_pca[len(X_train):]
        
        # 绘制PCA可视化
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 左图：训练集PCA
        unique_classes = np.unique(y_train)
        for cls in unique_classes:
            mask = y_train == cls
            ax1.scatter(X_train_pca[mask, 0], X_train_pca[mask, 1], 
                       label=f'类别 {cls}', alpha=0.7, s=60)
        ax1.set_title('训练集PCA降维可视化', fontsize=12, fontweight='bold')
        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} 方差)', **self.font_params)
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} 方差)', **self.font_params)
        ax1.legend(**self.font_params)
        ax1.grid(True, alpha=0.3)
        
        # 右图：测试集PCA
        for cls in unique_classes:
            mask = y_test == cls
            ax2.scatter(X_test_pca[mask, 0], X_test_pca[mask, 1], 
                       label=f'类别 {cls}', alpha=0.7, s=60)
        ax2.set_title('测试集PCA降维可视化', fontsize=12, fontweight='bold')
        ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} 方差)', **self.font_params)
        ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} 方差)', **self.font_params)
        ax2.legend(**self.font_params)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return '/static/' + os.path.join('images', os.path.basename(self.output_dir), os.path.basename(save_path)).replace('\\', '/')