import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
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

    def _plot_tree_structure(self, tree, feature_names, ax, x=0.5, y=1.0, level=0, width=0.5, depth_step=0.1):
        """递归绘制决策树结构（基于matplotlib图形）"""
        if tree["leaf"]:
            # 绘制叶子节点（矩形）
            bbox = FancyBboxPatch(
                (x - width / 2, y - 0.03), width, 0.06,
                boxstyle="round,pad=0.02",
                facecolor="#E8F4FD", edgecolor="#2E86AB"
            )
            ax.add_patch(bbox)
            # 叶子节点文本（分类=类别，回归=均值）
            if isinstance(tree["value"], (int, np.integer)):
                text = f"Class: {int(tree['value'])}"
            else:
                text = f"Mean: {tree['value']:.2f}"
            ax.text(x, y, text, ha="center", va="center", **self.font_params)
            return x  # 返回叶子节点x坐标（用于连接父节点）

        # 绘制内部节点（矩形）
        bbox = FancyBboxPatch(
            (x - width / 2, y - 0.03), width, 0.06,
            boxstyle="round,pad=0.02",
            facecolor="#F7DC6F", edgecolor="#F39C12"
        )
        ax.add_patch(bbox)
        # 内部节点文本（特征+阈值）
        feat_name = feature_names[tree["feature_idx"]] if feature_names else f"Feature {tree['feature_idx']}"
        text = f"{feat_name}\n≤ {tree['threshold']:.2f}"
        ax.text(x, y, text, ha="center", va="center", **self.font_params)

        # 计算左右子树位置（宽度减半，y坐标下移）
        child_width = width / 2
        child_y = y - depth_step
        # 递归绘制左子树
        left_x = self._plot_tree_structure(tree["left"], feature_names, ax, x - child_width / 2, child_y, level + 1,
                                           child_width, depth_step)
        # 递归绘制右子树
        right_x = self._plot_tree_structure(tree["right"], feature_names, ax, x + child_width / 2, child_y, level + 1,
                                            child_width, depth_step)
        # 绘制父节点到左子树的连线
        ax.plot([x, left_x], [y - 0.03, child_y + 0.03], color="#34495E", linewidth=1)
        # 绘制父节点到右子树的连线
        ax.plot([x, right_x], [y - 0.03, child_y + 0.03], color="#34495E", linewidth=1)
        return x

    def _generate_tree_plot(self, model, feature_names):
        """生成决策树结构可视化图"""
        if model.tree is None:
            raise ValueError("决策树未训练，无法生成可视化")

        # 设置画布（根据树深度调整高度）
        max_depth = self._get_tree_depth(model.tree)
        fig_height = max(6, max_depth * 0.8 + 2)
        fig, ax = plt.subplots(figsize=(12, fig_height))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")  # 隐藏坐标轴

        # 绘制树结构
        self._plot_tree_structure(model.tree, feature_names, ax)
        # 设置标题
        task_type = "分类" if model.is_classifier else "回归"
        ax.text(0.5, 0.95, f"决策树结构（{task_type}任务）", ha="center", va="top", fontsize=14, fontweight="bold",
                **self.font_params)

        # 保存图片
        tree_path = os.path.join(self.output_dir, "decision_tree_structure.png")
        plt.tight_layout()
        plt.savefig(tree_path, dpi=150, bbox_inches='tight')
        plt.close()
        return tree_path

    def _get_tree_depth(self, tree):
        """计算决策树深度（递归）"""
        if tree["leaf"]:
            return 1
        left_depth = self._get_tree_depth(tree["left"])
        right_depth = self._get_tree_depth(tree["right"])
        return max(left_depth, right_depth) + 1

    def _generate_feature_importance_plot(self, model, feature_names):
        """生成特征重要性柱状图"""
        n_features = len(model.feature_importances_)
        # 处理特征名（无则用默认名）
        if feature_names is None or len(feature_names) != n_features:
            feature_names = [f"Feature {i}" for i in range(n_features)]

        # 排序（按重要性降序）
        sorted_idx = np.argsort(model.feature_importances_)[::-1]
        sorted_feat_names = [feature_names[i] for i in sorted_idx]
        sorted_importances = model.feature_importances_[sorted_idx]

        # 绘制柱状图
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(range(n_features), sorted_importances, color="#3498DB", alpha=0.8)
        # 添加数值标签
        for bar, imp in zip(bars, sorted_importances):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                    f"{imp:.3f}", ha="center", va="bottom", **self.font_params)

        # 设置图表样式
        ax.set_xlabel("特征", **self.font_params)
        ax.set_ylabel("重要性（归一化）", **self.font_params)
        ax.set_title("决策树特征重要性", fontsize=14, fontweight="bold")
        ax.set_xticks(range(n_features))
        ax.set_xticklabels(sorted_feat_names, rotation=45, ha="right", **self.font_params)
        ax.set_ylim(0, max(sorted_importances) * 1.2)
        ax.grid(axis="y", alpha=0.3)

        # 保存图片
        feat_imp_path = os.path.join(self.output_dir, "decision_tree_feature_importance.png")
        plt.tight_layout()
        plt.savefig(feat_imp_path, dpi=150, bbox_inches='tight')
        plt.close()
        return feat_imp_path

    def _generate_truth_vs_pred_plot(self, y_test, y_pred, model):
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
            ax.set_title("决策树：真实类别vs预测类别（测试集）", fontsize=14, fontweight="bold")
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
            ax.set_title("决策树：真实值vs预测值（回归任务-测试集）", fontsize=14, fontweight="bold")
            ax.legend(**self.font_params)

        ax.grid(alpha=0.3)
        # 保存图片
        truth_pred_path = os.path.join(self.output_dir, "decision_tree_truth_vs_pred.png")
        plt.tight_layout()
        plt.savefig(truth_pred_path, dpi=150, bbox_inches='tight')
        plt.close()
        return truth_pred_path

    def generate_visualizations(self, model, X_train, X_test, y_train, y_test, y_pred, y_pred_proba, feature_names=None,
                                target_names=None):
        """
        生成决策树全面可视化（涵盖数据探索、特征工程、模型评估三个阶段）
        返回格式：{可视化名称: 静态文件路径}
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
            # 3. 特征重要性图
            feat_imp_path = self._generate_feature_importance_plot(model, feature_names)
            feat_imp_viz_path = f"/static/images/{os.path.basename(self.output_dir)}/{os.path.basename(feat_imp_path)}"
            visualizations["feature_importance"] = feat_imp_viz_path

            # 4. 决策树结构可视化（仅当树不太深时）
            if self._get_tree_depth(model.tree) <= 5:  # 限制深度避免过于复杂
                tree_path = self._generate_tree_plot(model, feature_names)
                tree_viz_path = f"/static/images/{os.path.basename(self.output_dir)}/{os.path.basename(tree_path)}"
                visualizations["tree_structure"] = tree_viz_path

            # ========== 第三阶段：模型评估与性能分析 ==========
            # 5. 混淆矩阵（分类任务）
            if model.is_classifier:
                confusion_path = self._plot_confusion_matrix(y_test, y_pred, target_names)
                if confusion_path:
                    visualizations["confusion_matrix"] = confusion_path
                
                # 6. ROC曲线
                roc_path = self._plot_roc_curves(y_test, y_pred_proba, target_names)
                if roc_path:
                    visualizations["roc_curve"] = roc_path
                
                # 7. 精确率-召回率曲线
                pr_path = self._plot_precision_recall_curve(y_test, y_pred_proba, target_names)
                if pr_path:
                    visualizations["precision_recall_curve"] = pr_path
                
                # 8. 校准曲线（仅二分类）
                if len(np.unique(y_test)) == 2:
                    calib_path = self._plot_calibration_curve(y_test, y_pred_proba)
                    if calib_path:
                        visualizations["calibration_curve"] = calib_path
            else:
                # 回归任务：残差图
                residual_path = self._plot_residuals(y_test, y_pred)
                if residual_path:
                    visualizations["residual_plot"] = residual_path

            # 9. 学习曲线（通用）
            learning_path = self._plot_learning_curve(model, X_train, y_train)
            if learning_path:
                visualizations["learning_curve"] = learning_path

            # 10. 真实vs预测对比可视化（通用）
            truth_pred_path = self._generate_truth_vs_pred_plot(y_test, y_pred, model)
            truth_pred_viz_path = f"/static/images/{os.path.basename(self.output_dir)}/{os.path.basename(truth_pred_path)}"
            visualizations["truth_vs_pred"] = truth_pred_viz_path

        except Exception as e:
            print(f"决策树可视化生成失败: {str(e)}")
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
                    
                    # 训练模型
                    model_copy = type(estimator)(
                        learning_rate=estimator.learning_rate,
                        max_depth=estimator.max_depth,
                        min_samples_split=estimator.min_samples_split,
                        min_samples_leaf=estimator.min_samples_leaf
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
                label="决策树", color='blue')
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