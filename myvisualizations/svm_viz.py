import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.model_selection import learning_curve
import seaborn as sns

class CustomVisualization:
    def __init__(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir  # 可视化保存目录
        self._set_fonts()
        
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
        """获取二维特征（优先用前两维，无特征名时用默认名）"""
        if X.shape[1] < 2:
            raise ValueError("SVM可视化需要至少2个特征（当前仅支持二维特征可视化）")
        # 取前两维特征用于绘图
        X_2d = X[:, :2]
        # 处理特征名（无则用默认名）
        if feature_names and len(feature_names) >= 2:
            feat_names = feature_names[:2]
        else:
            feat_names = ["Feature 1", "Feature 2"]
        return X_2d, feat_names

    def _plot_decision_boundary(self, ax, model, X_2d, y, feat_names):
        """绘制SVM决策边界、支持向量和类别分布"""
        # 1. 绘制样本点（按真实标签着色）
        unique_y = np.unique(y)
        colors = ['#FF6B6B', '#4ECDC4']  # 两类样本的颜色
        for i, label in enumerate(unique_y):
            mask = y == label
            ax.scatter(
                X_2d[mask, 0], X_2d[mask, 1],
                c=colors[i], label=f"Class {label}",
                alpha=0.7, edgecolors='black', s=60
            )

        # 2. 绘制决策边界（w1x1 + w2x2 + b = 0）
        x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
        y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                             np.linspace(y_min, y_max, 200))
        # 计算网格点的预测分数
        grid_X = np.c_[xx.ravel(), yy.ravel()]
        # 补全特征维度（若原始特征>2维，其他维度用0填充）
        if grid_X.shape[1] < model.w.shape[0]:
            grid_X = np.hstack([grid_X, np.zeros((grid_X.shape[0], model.w.shape[0]-2))])
        scores = np.dot(grid_X, model.w) + model.b
        Z = scores.reshape(xx.shape)

        # 绘制决策边界（Z=0）和间隔线（Z=±1）
        ax.contour(xx, yy, Z, levels=[-1, 0, 1], colors=['#95A5A6', '#E74C3C', '#95A5A6'],
                   linestyles=['--', '-', '--'], linewidths=[1.5, 2.5, 1.5])
        # 绘制类别区域（填充色）
        ax.contourf(xx, yy, Z, levels=[-np.inf, 0, np.inf], colors=colors, alpha=0.1)

        # 3. 标记支持向量（满足|w·x + b| ≤ 1的样本）
        support_mask = np.abs(np.dot(X_2d, model.w[:2]) + model.b) <= 1.05  # 允许微小误差
        if np.sum(support_mask) > 0:
            ax.scatter(
                X_2d[support_mask, 0], X_2d[support_mask, 1],
                s=150, edgecolors='#F39C12', facecolors='none',
                linewidths=2, label="Support Vectors"
            )

        # 4. 设置图表样式
        ax.set_xlabel(feat_names[0], fontsize=12)
        ax.set_ylabel(feat_names[1], fontsize=12)
        ax.set_title("SVM Decision Boundary & Support Vectors", fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    def generate_visualizations(self, model, X_train, X_test, y_train, y_test, y_pred, y_pred_proba, feature_names=None, target_names=None):
        """
        生成SVM全面可视化（涵盖数据探索、特征工程、模型评估三个阶段）
        参数：kwargs含feature_names（特征名列表，默认feat_0/feat_1...）
        返回：dict{图表名称: 前端访问路径}
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

            # ========== 第二阶段：特征工程与模型构建 ==========
            # 3. 特征重要性图（基于权重绝对值）
            feat_imp_path = self._plot_feature_importance(model, feature_names)
            if feat_imp_path:
                visualizations["feature_importance"] = feat_imp_path

            # 4. 决策边界与支持向量图（SVM特有）
            try:
                X_train_2d, feat_names = self._get_2d_features(X_train, feature_names)
                fig, ax = plt.subplots(figsize=(10, 7))
                self._plot_decision_boundary(ax, model, X_train_2d, y_train, feat_names)
                # 保存图片
                db_path = os.path.join(self.output_dir, "decision_boundary.png")
                plt.tight_layout()
                plt.savefig(db_path, dpi=150, bbox_inches='tight')
                plt.close()
                # 记录路径（适配前端访问格式）
                visualizations["decision_boundary"] = f"/static/images/{os.path.basename(self.output_dir)}/decision_boundary.png"
            except Exception as e:
                print(f"生成决策边界图表失败：{str(e)}")

            # 5. 支持向量分析图（SVM特有）
            support_vec_path = self._plot_support_vectors(model, X_train, y_train, feature_names)
            if support_vec_path:
                visualizations["support_vectors"] = support_vec_path

            # ========== 第三阶段：模型评估与性能分析 ==========
            # 判断任务类型
            is_classifier = getattr(model, 'is_classifier', None)
            
            if is_classifier:
                # 分类任务特有可视化
                # 6. 混淆矩阵
                confusion_path = self._plot_confusion_matrix(y_test, y_pred, target_names)
                if confusion_path:
                    visualizations["confusion_matrix"] = confusion_path
                    
                # 7. ROC曲线
                roc_path = self._plot_roc_curves(y_test, y_pred_proba, target_names)
                if roc_path:
                    visualizations["roc_curve"] = roc_path
                    
                # 8. 精确率-召回率曲线
                pr_path = self._plot_precision_recall_curve(y_test, y_pred_proba, target_names)
                if pr_path:
                    visualizations["precision_recall_curve"] = pr_path
                    
                # 9. 校准曲线（仅二分类）
                if len(np.unique(y_test)) == 2:
                    calib_path = self._plot_calibration_curve(y_test, y_pred_proba)
                    if calib_path:
                        visualizations["calibration_curve"] = calib_path
            else:
                # 回归任务特有可视化
                # 6. 残差图
                try:
                    residuals_path = self._plot_residuals(y_test, y_pred)
                    if residuals_path:
                        visualizations["residuals"] = residuals_path
                except Exception as e:
                    print(f"生成残差图失败：{str(e)}")

            # 10. 学习曲线（通用）
            learning_path = self._plot_learning_curve(model, X_train, y_train)
            if learning_path:
                visualizations["learning_curve"] = learning_path

            # 11. 真实vs预测对比可视化（通用）
            truth_pred_path = self._plot_truth_vs_pred(y_test, y_pred, model)
            if truth_pred_path:
                visualizations["truth_vs_pred"] = truth_pred_path

        except Exception as e:
            print(f"SVM可视化生成失败: {str(e)}")
            return {}  # 失败时返回空字典

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

    def _plot_feature_importance(self, model, feature_names=None):
        """绘制特征重要性图（基于权重绝对值）"""
        save_path = os.path.join(self.output_dir, 'feature_importance.png')
        if os.path.exists(save_path):
            return '/static/' + os.path.join('images', os.path.basename(self.output_dir), os.path.basename(save_path)).replace('\\', '/')
            
        # 获取特征重要性
        feature_importance = model.feature_importances_
        
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
        ax.set_ylabel("重要性（基于权重绝对值）", **self.font_params)
        ax.set_title("SVM特征重要性", fontsize=14, fontweight="bold")
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
                    
                    # 训练模型 - 只传入构造函数参数
                    model_copy = type(estimator)(
                        learning_rate=estimator.learning_rate,
                        C=estimator.C,
                        max_iter=estimator.max_iter,
                        tol=estimator.tol,
                        epsilon=estimator.epsilon
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
                label="SVM", color='blue')
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

    def _plot_truth_vs_pred(self, y_test, y_pred, model):
        """绘制真实值vs预测值对比图（分类/回归分别适配）"""
        save_path = os.path.join(self.output_dir, 'truth_vs_pred.png')
        if os.path.exists(save_path):
            return '/static/' + os.path.join('images', os.path.basename(self.output_dir), os.path.basename(save_path)).replace('\\', '/')
            
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
            bars1 = ax.bar(x - width / 2, true_counts, width, label="真实标签", color="#2ECC71", alpha=0.8)
            bars2 = ax.bar(x + width / 2, pred_counts, width, label="预测标签", color="#E74C3C", alpha=0.8)

            # 添加数值标签
            for bar, count in zip(bars1, true_counts):
                ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.5,
                        str(count), ha="center", va="bottom", **self.font_params)
            for bar, count in zip(bars2, pred_counts):
                ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.5,
                        str(count), ha="center", va="bottom", **self.font_params)

            ax.set_xlabel("类别", **self.font_params)
            ax.set_ylabel("样本数量", **self.font_params)
            ax.set_title("SVM：真实类别vs预测类别（测试集）", fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels([str(cls) for cls in unique_classes], **self.font_params)

        else:
            # 回归任务：散点图（真实值vs预测值）
            ax.scatter(y_test, y_pred, alpha=0.6, color="#9B59B6", s=60, edgecolors="black", linewidth=0.5)
            # 添加对角线（理想预测线：y_pred = y_test）
            min_val = min(np.min(y_test), np.min(y_pred))
            max_val = max(np.max(y_test), np.max(y_pred))
            ax.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.8, label="理想预测线")

            ax.set_xlabel("真实值", **self.font_params)
            ax.set_ylabel("预测值", **self.font_params)
            ax.set_title("SVM：真实值vs预测值（回归任务-测试集）", fontsize=14, fontweight='bold')
            ax.legend(**self.font_params)

        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return '/static/' + os.path.join('images', os.path.basename(self.output_dir), os.path.basename(save_path)).replace('\\', '/')
    
    def _plot_residuals(self, y_test, y_pred):
        """绘制残差图（回归任务）"""
        try:
            save_path = os.path.join(self.output_dir, 'residuals.png')
            if os.path.exists(save_path):
                return '/static/' + os.path.join('images', os.path.basename(self.output_dir), os.path.basename(save_path)).replace('\\', '/')
            
            residuals = np.array(y_test) - np.array(y_pred)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            # 残差vs预测值
            ax1.scatter(y_pred, residuals, alpha=0.5, color='#3498db')
            ax1.axhline(y=0, color='r', linestyle='--', linewidth=2)
            ax1.set_xlabel('预测值', **self.font_params)
            ax1.set_ylabel('残差', **self.font_params)
            ax1.set_title('残差 vs 预测值', fontsize=12, fontweight='bold')
            ax1.grid(alpha=0.3)
            
            # 残差分布直方图
            ax2.hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='#2ecc71')
            ax2.set_xlabel('残差', **self.font_params)
            ax2.set_ylabel('频数', **self.font_params)
            ax2.set_title('残差分布', fontsize=12, fontweight='bold')
            ax2.grid(alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            return '/static/' + os.path.join('images', os.path.basename(self.output_dir), os.path.basename(save_path)).replace('\\', '/')
        except Exception as e:
            print(f"生成残差图失败: {str(e)}")
            return None

    def _plot_support_vectors(self, model, X_train, y_train, feature_names=None):
        """绘制支持向量分析图（SVM特有）"""
        save_path = os.path.join(self.output_dir, 'support_vectors.png')
        if os.path.exists(save_path):
            return '/static/' + os.path.join('images', os.path.basename(self.output_dir), os.path.basename(save_path)).replace('\\', '/')
            
        if not hasattr(model, 'support_vectors_') or model.support_vectors_ is None:
            return None
            
        # 获取二维特征
        X_2d, feat_names = self._get_2d_features(X_train, feature_names)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 左图：支持向量分布
        unique_y = np.unique(y_train)
        colors = ['#FF6B6B', '#4ECDC4']
        
        for i, label in enumerate(unique_y):
            mask = y_train == label
            ax1.scatter(X_2d[mask, 0], X_2d[mask, 1], c=colors[i], 
                       label=f"类别 {label}", alpha=0.6, s=50)
        
        # 标记支持向量
        if model.support_vector_indices_ is not None:
            sv_indices = model.support_vector_indices_
            ax1.scatter(X_2d[sv_indices, 0], X_2d[sv_indices, 1], 
                      s=150, edgecolors='red', facecolors='none', 
                      linewidths=2, label="支持向量")
        
        ax1.set_xlabel(feat_names[0], **self.font_params)
        ax1.set_ylabel(feat_names[1], **self.font_params)
        ax1.set_title("支持向量分布", fontsize=14, fontweight='bold')
        ax1.legend(**self.font_params)
        ax1.grid(True, alpha=0.3)
        
        # 右图：权重向量可视化
        if model.w is not None:
            weights = np.abs(model.w[:2])  # 前两个特征的权重
            bars = ax2.bar(range(len(weights)), weights, color='skyblue', alpha=0.7)
            
            # 添加数值标签
            for bar, weight in zip(bars, weights):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                        f"{weight:.3f}", ha="center", va="bottom", **self.font_params)
            
            ax2.set_xlabel("特征", **self.font_params)
            ax2.set_ylabel("权重绝对值", **self.font_params)
            ax2.set_title("权重向量分析", fontsize=14, fontweight='bold')
            ax2.set_xticks(range(len(weights)))
            ax2.set_xticklabels(feat_names, **self.font_params)
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return '/static/' + os.path.join('images', os.path.basename(self.output_dir), os.path.basename(save_path)).replace('\\', '/')