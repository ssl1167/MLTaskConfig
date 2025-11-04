import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

class CustomVisualization:
    def __init__(self, output_dir: str):
        """Initialize: Create visualization output directory"""
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        # Set fonts for visualization
        self._set_fonts()
    
    def _set_fonts(self):
        """Set fonts for visualization"""
        try:
            plt.rcParams['axes.unicode_minus'] = False
            # Prefer Chinese-capable fonts; fallback to default if unavailable
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'Arial Unicode MS', 'sans-serif']
        except Exception as e:
            print(f"Error setting fonts: {str(e)}")

    # no deletion: we keep existing images if already generated

    def _calculate_feature_importance(self, model, X, y, feature_names):
        """Calculate Random Forest feature importance (based on impurity reduction).
        Uses training data (X, y) approximation to traverse each tree and measure impurity reduction.
        """
        X = np.array(X)
        y = np.array(y)
        n_features = X.shape[1]
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(n_features)]
        
        # 初始化特征重要性（每棵树的重要性累加）
        feature_importance = defaultdict(float)
        total_trees = len(model.estimators)

        # 遍历每棵树，计算其特征重要性（树内节点feature_idx为局部索引，需要映射回原索引）
        for tree, feature_indices in model.estimators:
            if not isinstance(tree.root, dict):  # 跳过单节点树（无分裂）
                continue
            
            # 递归计算树的特征重要性
            def _traverse_node(node, X_sub, y_sub, depth=0):
                if not isinstance(node, dict):  # 叶子节点，终止递归
                    return
                
                # 计算当前节点的不纯度减少量（重要性贡献）
                parent_impurity = tree._calculate_impurity(y_sub)
                (X_left, y_left), (X_right, y_right) = tree._split_data(
                    X_sub, y_sub, node['feature_idx'], node['threshold']
                )
                weight_left = len(y_left) / len(y_sub)
                weight_right = len(y_right) / len(y_sub)
                child_impurity = (weight_left * tree._calculate_impurity(y_left) 
                                 + weight_right * tree._calculate_impurity(y_right))
                impurity_reduction = parent_impurity - child_impurity

                # 映射到原始特征索引并累加重要性
                orig_feature_idx = feature_indices[node['feature_idx']]
                feature_importance[orig_feature_idx] += impurity_reduction

                # 递归遍历左右子树
                _traverse_node(node['left'], X_left, y_left, depth + 1)
                _traverse_node(node['right'], X_right, y_right, depth + 1)

            # 对当前树的根节点开始遍历（使用树训练时的特征子集）
            X_tree_subset = X[:, feature_indices]
            _traverse_node(tree.root, X_tree_subset, y)


        # 归一化：重要性总和为1，按原始特征名排序
        total_importance = sum(feature_importance.values())
        if total_importance == 0:
            return {name: 0.0 for name in feature_names}
        
        # 映射到特征名并排序
        importance_dict = {
            feature_names[idx]: (imp / total_importance) 
            for idx, imp in feature_importance.items()
        }
        # 补充未被任何树使用的特征（重要性为0）
        for name in feature_names:
            if name not in importance_dict:
                importance_dict[name] = 0.0
        
        # 按重要性降序排序
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

    def _plot_feature_importance(self, model, X_train, y_train, feature_names):
        """Plot feature importance bar chart"""
        try:
            # Idempotent save: if already exists, reuse
            save_path = os.path.join(self.output_dir, 'feature_importance.png')
            if os.path.exists(save_path):
                return '/static/' + os.path.join(
                    'images', os.path.basename(self.output_dir), os.path.basename(save_path)
                ).replace('\\', '/')
            # 计算特征重要性
            importance = self._calculate_feature_importance(model, X_train, y_train, feature_names)
            features = list(importance.keys())
            importances = list(importance.values())

            # 绘制前10个重要特征（避免图表过宽）
            top_n = min(10, len(features))
            top_features = features[:top_n]
            top_importances = importances[:top_n]

            # 生成图表
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.barh(range(top_n), top_importances, align='center')
            ax.set_yticks(range(top_n))
            ax.set_yticklabels(top_features)
            ax.set_xlabel('Feature Importance (Normalized)')
            ax.set_title(f'Random Forest Top {top_n} Features\n(Importance Based on Impurity Reduction)')
            ax.grid(axis='x', alpha=0.3)

            # 在柱子上添加数值标签
            for i, bar in enumerate(bars):
                width = bar.get_width()
                ax.text(width + 0.005, bar.get_y() + bar.get_height()/2, 
                        f'{width:.3f}', ha='left', va='center')

            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()

            # 返回相对路径（适配前端）
            return '/static/' + os.path.join(
                'images', os.path.basename(self.output_dir), os.path.basename(save_path)
            ).replace('\\', '/')
        except Exception as e:
            print(f"Failed to plot feature importance: {str(e)}")
            return None

    def _plot_prediction_confidence(self, model, X_test, y_pred_proba):
        """Plot prediction confidence distribution (classification tasks only)"""
        if not model.is_classifier or y_pred_proba is None:
            return None
        
        try:
            save_path = os.path.join(self.output_dir, 'prediction_confidence.png')
            if os.path.exists(save_path):
                return '/static/' + os.path.join(
                    'images', os.path.basename(self.output_dir), os.path.basename(save_path)
                ).replace('\\', '/')
            # 计算每个样本的最大置信度（预测类的概率）
            max_confidence = np.max(y_pred_proba, axis=1)

            # 生成图表
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(max_confidence, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax.axvline(np.mean(max_confidence), color='red', linestyle='--', 
                       label=f'Mean Confidence: {np.mean(max_confidence):.3f}')
            ax.set_xlabel('Prediction Confidence (Max Probability)')
            ax.set_ylabel('Sample Count')
            ax.set_title('Random Forest Classification Prediction Confidence Distribution')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)

            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()

            return '/static/' + os.path.join(
                'images', os.path.basename(self.output_dir), os.path.basename(save_path)
            ).replace('\\', '/')
        except Exception as e:
            print(f"Failed to plot confidence distribution: {str(e)}")
            return None

    def _plot_truth_vs_pred(self, model, y_test, y_pred):
        """Plot truth vs prediction (distribution comparison for classification, scatter for regression)"""
        try:
            y_test = np.array(y_test)
            y_pred = np.array(y_pred)
            save_path = os.path.join(self.output_dir, 'truth_vs_pred.png')
            if os.path.exists(save_path):
                return '/static/' + os.path.join(
                    'images', os.path.basename(self.output_dir), os.path.basename(save_path)
                ).replace('\\', '/')

            fig, ax = plt.subplots(figsize=(8, 6))
            if model.is_classifier:
                # 分类任务：绘制真实/预测类别分布对比
                unique_classes = np.unique(np.concatenate([y_test, y_pred]))
                true_counts = np.array([np.sum(y_test == c) for c in unique_classes])
                pred_counts = np.array([np.sum(y_pred == c) for c in unique_classes])

                # 计算准确率（用于标题）
                accuracy = np.mean(y_test == y_pred)

                # 绘制双柱状图
                x = np.arange(len(unique_classes))
                width = 0.35
                bars1 = ax.bar(x - width/2, true_counts, width, label='True Class', color='lightcoral')
                bars2 = ax.bar(x + width/2, pred_counts, width, label='Predicted Class', color='lightgreen')

                # Add value labels
                for bars in [bars1, bars2]:
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                                f'{int(height)}', ha='center', va='bottom')

                ax.set_xlabel('Class')
                ax.set_ylabel('Sample Count')
                ax.set_title(f'True vs Predicted Class Distribution\n(Accuracy: {accuracy:.3f})')
                ax.set_xticks(x)
                ax.set_xticklabels([str(c) for c in unique_classes])
                ax.legend()
            else:
                # 回归任务：绘制真实值vs预测值散点图（含对角线参考线）
                # 计算R²（用于标题）
                ss_res = np.sum((y_test - y_pred) ** 2)
                ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0

                # 绘制散点图
                ax.scatter(y_test, y_pred, alpha=0.6, color='cornflowerblue', s=50)
                # 添加对角线（完美预测线）
                min_val = min(np.min(y_test), np.min(y_pred))
                max_val = max(np.max(y_test), np.max(y_pred))
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect Prediction Line')

                ax.set_xlabel('True Values')
                ax.set_ylabel('Predicted Values')
                ax.set_title(f'True vs Predicted Values Scatter Plot\n(R²: {r2:.3f})')
                ax.legend()
                ax.grid(alpha=0.3)

            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()

            return '/static/' + os.path.join(
                'images', os.path.basename(self.output_dir), os.path.basename(save_path)
            ).replace('\\', '/')
        except Exception as e:
            print(f"Failed to plot truth vs prediction: {str(e)}")
            return None

    def _plot_roc_curves(self, model, y_test, y_pred_proba):
        """Plot ROC curve: binary or One-vs-Rest using voted probabilities.
        Column order follows model.classes_ when available."""
        if y_pred_proba is None:
            return None
        try:
            save_path = os.path.join(self.output_dir, 'roc_curve.png')
            if os.path.exists(save_path):
                return '/static/' + os.path.join(
                    'images', os.path.basename(self.output_dir), os.path.basename(save_path)
                ).replace('\\', '/')
            y_test = np.array(y_test)
            proba = np.array(y_pred_proba)
            n_classes = proba.shape[1] if proba.ndim == 2 else 1
            classes_order = getattr(model, 'classes_', None)

            def _binary_roc(y_true_bin, scores):
                # Sort by descending score; compute cumulative TPR/FPR
                order = np.argsort(-scores)
                y_sorted = y_true_bin[order]
                P = np.sum(y_sorted == 1)
                N = np.sum(y_sorted == 0)
                if P == 0 or N == 0:
                    return None, None
                tp = np.cumsum(y_sorted == 1)
                fp = np.cumsum(y_sorted == 0)
                tpr = tp / P
                fpr = fp / N
                # prepend (0,0), append (1,1)
                tpr = np.concatenate([[0.0], tpr, [1.0]])
                fpr = np.concatenate([[0.0], fpr, [1.0]])
                return fpr, tpr

            fig, ax = plt.subplots(figsize=(7, 6))
            if n_classes == 2:
                # Determine positive class column using classes_ if available
                if classes_order is not None and len(classes_order) == 2:
                    pos_class = classes_order[1]
                    y_bin = (y_test == pos_class).astype(int)
                    col_idx = 1
                else:
                    # Fallback: assume column 1 corresponds to the larger label
                    unique_labels = np.unique(y_test)
                    pos_class = unique_labels[-1]
                    y_bin = (y_test == pos_class).astype(int)
                    col_idx = 1
                fpr, tpr = _binary_roc(y_bin, proba[:, col_idx])
                if fpr is None:
                    return None
                ax.plot(fpr, tpr, label='ROC (positive=1)')
            else:
                # One-vs-Rest using model.classes_ order to align with columns
                if classes_order is None:
                    classes_order = np.unique(y_test)
                for i, lab in enumerate(classes_order):
                    if i >= proba.shape[1]:
                        continue
                    y_bin = (y_test == lab).astype(int)
                    fpr, tpr = _binary_roc(y_bin, proba[:, i])
                    if fpr is None:
                        continue
                    ax.plot(fpr, tpr, label=f'Class {lab} vs Rest')
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curve')
            ax.legend()
            ax.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            return '/static/' + os.path.join(
                'images', os.path.basename(self.output_dir), os.path.basename(save_path)
            ).replace('\\', '/')
        except Exception as e:
            print(f"Failed to plot ROC: {str(e)}")
            return None

    def _plot_data_distribution(self, y_train, y_test, target_names):
        """绘制数据分布图"""
        try:
            save_path = os.path.join(self.output_dir, 'data_distribution.png')
            if os.path.exists(save_path):
                return '/static/' + os.path.join('images', os.path.basename(self.output_dir), os.path.basename(save_path)).replace('\\', '/')
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            unique_train, counts_train = np.unique(y_train, return_counts=True)
            ax1.bar(unique_train, counts_train, color='skyblue', alpha=0.7)
            ax1.set_title('训练集分布', fontsize=12, fontweight='bold')
            ax1.set_xlabel('类别/值')
            ax1.set_ylabel('样本数')
            ax1.grid(alpha=0.3)
            
            unique_test, counts_test = np.unique(y_test, return_counts=True)
            ax2.bar(unique_test, counts_test, color='lightcoral', alpha=0.7)
            ax2.set_title('测试集分布', fontsize=12, fontweight='bold')
            ax2.set_xlabel('类别/值')
            ax2.set_ylabel('样本数')
            ax2.grid(alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            return '/static/' + os.path.join('images', os.path.basename(self.output_dir), os.path.basename(save_path)).replace('\\', '/')
        except Exception as e:
            print(f"生成数据分布图失败: {str(e)}")
            return None
    
    def _plot_correlation_matrix(self, X_train, feature_names):
        """绘制特征相关性矩阵"""
        try:
            save_path = os.path.join(self.output_dir, 'correlation_matrix.png')
            if os.path.exists(save_path):
                return '/static/' + os.path.join('images', os.path.basename(self.output_dir), os.path.basename(save_path)).replace('\\', '/')
            
            import seaborn as sns
            import pandas as pd
            df = pd.DataFrame(X_train, columns=feature_names if feature_names else [f'F{i}' for i in range(X_train.shape[1])])
            corr = df.corr()
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, square=True, ax=ax)
            ax.set_title('特征相关性矩阵', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            return '/static/' + os.path.join('images', os.path.basename(self.output_dir), os.path.basename(save_path)).replace('\\', '/')
        except Exception as e:
            print(f"生成相关性矩阵失败: {str(e)}")
            return None
    
    def _plot_confusion_matrix(self, y_test, y_pred, target_names):
        """绘制混淆矩阵（分类）"""
        try:
            save_path = os.path.join(self.output_dir, 'confusion_matrix.png')
            if os.path.exists(save_path):
                return '/static/' + os.path.join('images', os.path.basename(self.output_dir), os.path.basename(save_path)).replace('\\', '/')
            
            import seaborn as sns
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_test, y_pred)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names, ax=ax)
            ax.set_title('混淆矩阵', fontsize=14, fontweight='bold')
            ax.set_xlabel('预测类别')
            ax.set_ylabel('真实类别')
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            return '/static/' + os.path.join('images', os.path.basename(self.output_dir), os.path.basename(save_path)).replace('\\', '/')
        except Exception as e:
            print(f"生成混淆矩阵失败: {str(e)}")
            return None
    
    def _plot_residuals(self, y_test, y_pred):
        """绘制残差图（回归）"""
        try:
            save_path = os.path.join(self.output_dir, 'residuals.png')
            if os.path.exists(save_path):
                return '/static/' + os.path.join('images', os.path.basename(self.output_dir), os.path.basename(save_path)).replace('\\', '/')
            
            residuals = y_test - y_pred
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            # 残差vs预测值
            ax1.scatter(y_pred, residuals, alpha=0.5)
            ax1.axhline(y=0, color='r', linestyle='--')
            ax1.set_xlabel('预测值')
            ax1.set_ylabel('残差')
            ax1.set_title('残差 vs 预测值', fontsize=12, fontweight='bold')
            ax1.grid(alpha=0.3)
            
            # 残差分布直方图
            ax2.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
            ax2.set_xlabel('残差')
            ax2.set_ylabel('频数')
            ax2.set_title('残差分布', fontsize=12, fontweight='bold')
            ax2.grid(alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            return '/static/' + os.path.join('images', os.path.basename(self.output_dir), os.path.basename(save_path)).replace('\\', '/')
        except Exception as e:
            print(f"生成残差图失败: {str(e)}")
            return None
    
    def _plot_precision_recall_curve(self, y_test, y_pred_proba, target_names):
        """绘制精确率-召回率曲线（分类）"""
        try:
            save_path = os.path.join(self.output_dir, 'precision_recall_curve.png')
            if os.path.exists(save_path):
                return '/static/' + os.path.join('images', os.path.basename(self.output_dir), os.path.basename(save_path)).replace('\\', '/')
            
            from sklearn.metrics import precision_recall_curve, average_precision_score
            from sklearn.preprocessing import label_binarize
            
            y_test_arr = np.array(y_test)
            proba = np.array(y_pred_proba)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            
            if proba.shape[1] == 2:
                # 二分类
                precision, recall, _ = precision_recall_curve(y_test_arr, proba[:, 1])
                ap = average_precision_score(y_test_arr, proba[:, 1])
                ax.plot(recall, precision, label=f'AP={ap:.3f}')
            else:
                # 多分类
                classes = np.unique(y_test_arr)
                y_test_bin = label_binarize(y_test_arr, classes=classes)
                for i, cls in enumerate(classes):
                    precision, recall, _ = precision_recall_curve(y_test_bin[:, i], proba[:, i])
                    ap = average_precision_score(y_test_bin[:, i], proba[:, i])
                    ax.plot(recall, precision, label=f'Class {cls} (AP={ap:.3f})')
            
            ax.set_xlabel('召回率')
            ax.set_ylabel('精确率')
            ax.set_title('精确率-召回率曲线', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            return '/static/' + os.path.join('images', os.path.basename(self.output_dir), os.path.basename(save_path)).replace('\\', '/')
        except Exception as e:
            print(f"生成PR曲线失败: {str(e)}")
            return None

    def generate_visualizations(self, model, X_train, X_test, y_train, y_test, y_pred, y_pred_proba, 
                               feature_names=None, target_names=None):
        """Generate all visualizations and return path dictionary"""
        visualizations = {}

        # 1. 数据分布图
        try:
            data_dist_path = self._plot_data_distribution(y_train, y_test, target_names)
            if data_dist_path:
                visualizations['data_distribution'] = data_dist_path
        except Exception as e:
            print(f"数据分布图生成失败: {str(e)}")

        # 2. 特征相关性矩阵
        try:
            corr_path = self._plot_correlation_matrix(X_train, feature_names)
            if corr_path:
                visualizations['correlation_matrix'] = corr_path
        except Exception as e:
            print(f"相关性矩阵生成失败: {str(e)}")

        # 3. 特征重要性图
        try:
            feat_importance_path = self._plot_feature_importance(model, X_train, y_train, feature_names)
            if feat_importance_path:
                visualizations['feature_importance'] = feat_importance_path
        except Exception as e:
            print(f"特征重要性图生成失败: {str(e)}")

        # 4. 真实vs预测图
        try:
            truth_pred_path = self._plot_truth_vs_pred(model, y_test, y_pred)
            if truth_pred_path:
                visualizations['truth_vs_pred'] = truth_pred_path
        except Exception as e:
            print(f"真实vs预测图生成失败: {str(e)}")

        # 分类任务特有可视化
        if model.is_classifier:
            # 5. 混淆矩阵
            try:
                cm_path = self._plot_confusion_matrix(y_test, y_pred, target_names)
                if cm_path:
                    visualizations['confusion_matrix'] = cm_path
            except Exception as e:
                print(f"混淆矩阵生成失败: {str(e)}")
            
            # 6. 预测置信度图
            try:
                confidence_path = self._plot_prediction_confidence(model, X_test, y_pred_proba)
                if confidence_path:
                    visualizations['prediction_confidence'] = confidence_path
            except Exception as e:
                print(f"置信度图生成失败: {str(e)}")
            
            # 7. ROC曲线
            try:
                roc_path = self._plot_roc_curves(model, y_test, y_pred_proba)
                if roc_path:
                    visualizations['roc_curve'] = roc_path
            except Exception as e:
                print(f"ROC曲线生成失败: {str(e)}")
            
            # 8. 精确率-召回率曲线
            try:
                pr_path = self._plot_precision_recall_curve(y_test, y_pred_proba, target_names)
                if pr_path:
                    visualizations['precision_recall_curve'] = pr_path
            except Exception as e:
                print(f"PR曲线生成失败: {str(e)}")
        else:
            # 回归任务特有可视化
            # 5. 残差图
            try:
                residual_path = self._plot_residuals(y_test, y_pred)
                if residual_path:
                    visualizations['residuals'] = residual_path
            except Exception as e:
                print(f"残差图生成失败: {str(e)}")

        return visualizations