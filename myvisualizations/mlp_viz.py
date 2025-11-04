import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.model_selection import learning_curve
import seaborn as sns

class CustomVisualization:
    """MLP专属可视化模块，适配框架动态加载，支持分类/回归差异化展示"""
    def __init__(self, output_dir: str):
        """初始化：创建输出目录，配置中文字体（避免乱码）"""
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
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

    def _get_viz_paths(self, filename):
        """生成图表保存路径与前端访问路径（符合框架/static规范）"""
        save_path = os.path.join(self.output_dir, filename)
        task_id = os.path.basename(self.output_dir)
        web_path = f"/static/images/{task_id}/{filename}"
        return save_path, web_path

    def _save_figure(self, fig, filename):
        """保存图表并返回前端路径，确保高分辨率与布局完整"""
        save_path, web_path = self._get_viz_paths(filename)
        plt.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return web_path

    # --------------------------
    # 1. 任务结果对比（分类/回归差异化）
    # --------------------------
    def _plot_task_result(self, model, y_true, y_pred):
        """分类→类别分布+混淆矩阵；回归→真实vs预测散点+R²"""
        fig = plt.figure(figsize=(10, 6))
        if model.is_classifier:
            # 分类任务：双子图（类别分布+混淆矩阵）
            ax1 = plt.subplot(1, 2, 1)
            # 子图1：类别分布对比
            classes = model.classes_
            true_counts = np.bincount(y_true.astype(int), minlength=len(classes))
            pred_counts = np.bincount(y_pred.astype(int), minlength=len(classes))
            x = np.arange(len(classes))
            width = 0.35
            ax1.bar(x - width/2, true_counts, width, label='真实标签', color='#2E86AB', alpha=0.8)
            ax1.bar(x + width/2, pred_counts, width, label='预测标签', color='#A23B72', alpha=0.8)
            ax1.set_xlabel('类别', fontsize=12)
            ax1.set_ylabel('样本数量', fontsize=12)
            ax1.set_title('类别分布对比', fontsize=13, fontweight='bold')
            ax1.set_xticks(x)
            ax1.set_xticklabels([f'类别{int(c)}' for c in classes])
            ax1.legend()

            # 子图2：混淆矩阵
            ax2 = plt.subplot(1, 2, 2)
            n_classes = len(classes)
            confusion = np.zeros((n_classes, n_classes), dtype=int)
            for true, pred in zip(y_true.astype(int), y_pred.astype(int)):
                confusion[true, pred] += 1
            # 热力图展示混淆矩阵
            im = ax2.imshow(confusion, cmap='YlOrRd', aspect='auto')
            # 添加数值标签
            for i in range(n_classes):
                for j in range(n_classes):
                    ax2.text(j, i, str(confusion[i, j]), ha='center', va='center',
                             color='black' if confusion[i, j] < confusion.max()/2 else 'white')
            ax2.set_xlabel('预测类别', fontsize=12)
            ax2.set_ylabel('真实类别', fontsize=12)
            ax2.set_title('混淆矩阵', fontsize=13, fontweight='bold')
            ax2.set_xticks(range(n_classes))
            ax2.set_yticks(range(n_classes))
            ax2.set_xticklabels([f'类别{int(c)}' for c in classes])
            ax2.set_yticklabels([f'类别{int(c)}' for c in classes])
            plt.colorbar(im, ax=ax2, label='样本数量')
        else:
            # 回归任务：真实vs预测散点+R²
            ax = plt.gca()
            # 绘制散点
            ax.scatter(y_true, y_pred, alpha=0.6, color='#2E86AB', s=50)
            # 理想预测线（y=x）
            min_val = min(min(y_true), min(y_pred))
            max_val = max(max(y_true), max(y_pred))
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='理想预测线（y=x）')
            # 计算并标注R²
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0
            ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                    verticalalignment='top', fontsize=11)
            ax.set_xlabel('真实值', fontsize=12)
            ax.set_ylabel('预测值', fontsize=12)
            ax.set_title('回归任务：真实值vs预测值', fontsize=13, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)

        return self._save_figure(fig, 'task_result.png')

    # --------------------------
    # 2. 训练损失曲线（分析收敛过程）
    # --------------------------
    def _plot_training_loss(self, model, X_train, y_train):
        """重新计算训练损失并绘制曲线，观察收敛速度与过拟合风险"""
        # 取部分训练集加速计算（避免样本过多耗时）
        X_train_sample = X_train[:200] if len(X_train) > 200 else X_train
        y_train_sample = y_train[:200] if len(y_train) > 200 else y_train
        X = np.asarray(X_train_sample, dtype=np.float64)
        y_formatted = model._format_labels(y_train_sample)

        # 重新初始化参数（模拟训练过程，仅记录损失）
        model._init_params(X.shape[1])
        loss_history = []
        prev_loss = float('inf')

        for iter in range(model.max_iter):
            # 前向传播计算损失
            activations, _ = model._forward_propagation(X)
            y_pred = activations[-1]
            if model.is_classifier:
                current_loss = model._cross_entropy_loss(y_formatted, y_pred)
            else:
                current_loss = model._mse_loss(y_formatted, y_pred)
            loss_history.append(current_loss)

            # 收敛判断（与训练逻辑一致）
            if abs(prev_loss - current_loss) < model.tol:
                break
            prev_loss = current_loss

            # 计算梯度并更新参数（模拟训练）
            grads_W, grads_b = model._backward_propagation(X, y_formatted, activations, [])
            for i in range(len(model.weights)):
                model.weights[i] -= model.learning_rate * grads_W[i]
                model.biases[i] -= model.learning_rate * grads_b[i]

        # 检查是否有损失历史记录
        if not loss_history:
            print("警告：训练损失历史为空，跳过损失曲线生成")
            return None
            
        # 绘制损失曲线
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(range(1, len(loss_history)+1), loss_history, color='#E74C3C', linewidth=2, marker='o', markersize=3)
        ax.set_xlabel('迭代次数', fontsize=12)
        ax.set_ylabel('损失值', fontsize=12)
        ax.set_title('MLP训练损失曲线', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        # 标注最终损失
        final_loss = loss_history[-1]
        ax.text(0.95, 0.05, f'最终损失：{final_loss:.6f}', transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                horizontalalignment='right', fontsize=11)

        return self._save_figure(fig, 'training_loss.png')

    # --------------------------
    # 3. 隐藏层神经元激活热力图（观察内部响应）
    # --------------------------
    def _plot_neuron_activation(self, model, X_test, feature_names):
        """展示最后一层隐藏层的神经元激活状态，分析特征对模型的影响"""
        if not model.hidden_layers:
            return None  # 无隐藏层时跳过

        # 取前50个测试样本（避免图表过大）
        X_sample = X_test[:50] if len(X_test) > 50 else X_test
        X = np.asarray(X_sample, dtype=np.float64)
        # 前向传播获取隐藏层激活值
        activations, _ = model._forward_propagation(X)
        hidden_activation = activations[len(model.hidden_layers)]  # 最后一层隐藏层输出

        # 绘制热力图
        fig, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(hidden_activation.T, cmap='viridis', aspect='auto')
        # 图表美化
        ax.set_xlabel('测试样本（前50个）', fontsize=12)
        ax.set_ylabel('隐藏层神经元', fontsize=12)
        ax.set_title(f'MLP隐藏层神经元激活热力图（{model.hidden_layers[-1]}个神经元）', fontsize=13, fontweight='bold')
        # 设置坐标轴标签（间隔显示样本索引，避免拥挤）
        sample_ticks = np.arange(0, len(X_sample), 5)
        ax.set_xticks(sample_ticks)
        ax.set_xticklabels([f'样本{i}' for i in sample_ticks])
        ax.set_yticks(range(model.hidden_layers[-1]))
        ax.set_yticklabels([f'神经元{j}' for j in range(model.hidden_layers[-1])])
        # 颜色条标注激活值范围
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('ReLU激活值（0=未激活，>0=激活）', fontsize=10)

        return self._save_figure(fig, 'neuron_activation.png')

    # --------------------------
    # 4. 权重分布直方图（分析参数多样性）
    # --------------------------
    def _plot_weight_distribution(self, model):
        """展示各层权重的分布情况，判断参数初始化与训练效果"""
        n_layers = len(model.weights)
        fig, axes = plt.subplots(1, n_layers, figsize=(15, 5))
        if n_layers == 1:
            axes = [axes]  # 确保axes为列表格式

        for i, (ax, W) in enumerate(zip(axes, model.weights)):
            # 展平权重矩阵（便于统计分布）
            weights_flat = W.flatten()
            # 绘制直方图
            ax.hist(weights_flat, bins=30, alpha=0.7, color='#8E44AD', edgecolor='black')
            # 计算并标注统计信息（均值、标准差）
            mean_w = np.mean(weights_flat)
            std_w = np.std(weights_flat)
            ax.axvline(mean_w, color='red', linestyle='--', linewidth=2, label=f'均值：{mean_w:.4f}')
            ax.text(0.05, 0.95, f'标准差：{std_w:.4f}', transform=ax.transAxes,
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
                    verticalalignment='top', fontsize=10)
            # 标注层名称
            layer_name = f'输出层' if i == n_layers-1 else f'隐藏层{i+1}'
            ax.set_xlabel('权重值', fontsize=11)
            ax.set_ylabel('频次', fontsize=11)
            ax.set_title(f'{layer_name}权重分布', fontsize=12, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        return self._save_figure(fig, 'weight_distribution.png')

    # --------------------------
    # 可视化入口（框架调用接口）
    # --------------------------
    def generate_visualizations(self, model, X_train, X_test, y_train, y_test, y_pred, y_pred_proba, **kwargs):
        """
        生成MLP全面可视化（涵盖数据探索、特征工程、模型评估三个阶段）
        参数：kwargs含feature_names（特征名列表，默认feat_0/feat_1...）
        返回：dict{图表名称: 前端访问路径}
        """
        results = {}
        # 提取基础参数
        feature_names = kwargs.get('feature_names', [f'特征{i}' for i in range(X_test.shape[1])])
        target_names = kwargs.get('target_names', None)
        y_true = np.asarray(y_test)
        y_pred = np.asarray(y_pred)
        y_pred_proba = np.asarray(y_pred_proba) if y_pred_proba is not None else None

        try:
            # ========== 第一阶段：数据探索与理解 ==========
            # 1. 数据分布图（单变量分析）
            data_dist_path = self._plot_data_distribution(y_train, y_test, target_names)
            if data_dist_path:
                results["data_distribution"] = data_dist_path
            
            # 2. 特征相关性矩阵（多变量关系分析）
            corr_matrix_path = self._plot_correlation_matrix(X_train, feature_names)
            if corr_matrix_path:
                results["correlation_matrix"] = corr_matrix_path

            # ========== 第二阶段：特征工程与模型构建 ==========
            # 3. 特征重要性图（基于第一层权重）
            feat_imp_path = self._plot_feature_importance(model, feature_names)
            if feat_imp_path:
                results["feature_importance"] = feat_imp_path

            # 4. 网络架构图（MLP特有）
            network_path = self._plot_network_architecture(model)
            if network_path:
                results["network_architecture"] = network_path

            # 5. 神经元激活热力图（MLP特有）
            try:
                activation_path = self._plot_neuron_activation(model, X_test, feature_names)
                if activation_path:
                    results['neuron_activation'] = activation_path
            except Exception as e:
                print(f"生成神经元激活图表失败：{str(e)}")

            # 6. 权重分布直方图（MLP特有）
            try:
                results['weight_distribution'] = self._plot_weight_distribution(model)
            except Exception as e:
                print(f"生成权重分布图表失败：{str(e)}")

            # ========== 第三阶段：模型评估与性能分析 ==========
            # 7. 混淆矩阵（分类任务）
            if model.is_classifier:
                confusion_path = self._plot_confusion_matrix(y_test, y_pred, target_names)
                if confusion_path:
                    results["confusion_matrix"] = confusion_path
                
                # 8. ROC曲线
                roc_path = self._plot_roc_curves(y_test, y_pred_proba, target_names)
                if roc_path:
                    results["roc_curve"] = roc_path
                
                # 9. 精确率-召回率曲线
                pr_path = self._plot_precision_recall_curve(y_test, y_pred_proba, target_names)
                if pr_path:
                    results["precision_recall_curve"] = pr_path
                
                # 10. 校准曲线（仅二分类）
                if len(np.unique(y_test)) == 2:
                    calib_path = self._plot_calibration_curve(y_test, y_pred_proba)
                    if calib_path:
                        results["calibration_curve"] = calib_path
            else:
                # 回归任务：残差图
                residual_path = self._plot_residuals(y_test, y_pred)
                if residual_path:
                    results["residual_plot"] = residual_path

            # 11. 学习曲线（通用）- MLP跳过以加快速度
            # learning_path = self._plot_learning_curve(model, X_train, y_train)
            # if learning_path:
            #     results["learning_curve"] = learning_path

            # 12. 真实vs预测对比可视化（通用）
            truth_pred_path = self._plot_truth_vs_pred(y_test, y_pred, model)
            if truth_pred_path:
                results["truth_vs_pred"] = truth_pred_path

            # 13. 训练损失曲线（MLP特有）
            try:
                training_loss_path = self._plot_training_loss(model, X_train, y_train)
                if training_loss_path:
                    results['training_loss'] = training_loss_path
            except Exception as e:
                print(f"生成训练损失图表失败：{str(e)}")

        except Exception as e:
            print(f"MLP可视化生成失败: {str(e)}")
            return {}  # 失败时返回空字典，避免框架报错

        return results

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
        """绘制特征重要性图（基于第一层权重）"""
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
        ax.set_ylabel("重要性（基于第一层权重）", **self.font_params)
        ax.set_title("MLP特征重要性", fontsize=14, fontweight="bold")
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
                    from mymodels.mlp import CustomModel as MLPModel
                    model_copy = MLPModel(
                        learning_rate=estimator.learning_rate,
                        hidden_layers=estimator.hidden_layers,
                        max_iter=estimator.max_iter,
                        tol=estimator.tol
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
                label="MLP", color='blue')
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

    def _plot_network_architecture(self, model):
        """绘制网络架构图"""
        save_path = os.path.join(self.output_dir, 'network_architecture.png')
        if os.path.exists(save_path):
            return '/static/' + os.path.join('images', os.path.basename(self.output_dir), os.path.basename(save_path)).replace('\\', '/')
            
        # 创建网络架构图
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 计算层数和位置
        n_layers = len(model.weights) + 1  # +1 for input layer
        layer_sizes = [model.weights[0].shape[0]] + model.hidden_layers + [model.weights[-1].shape[1]]
        
        # 绘制网络结构
        y_positions = np.linspace(0, 1, max(layer_sizes))
        
        for layer_idx in range(n_layers):
            x_pos = layer_idx / (n_layers - 1)
            layer_size = layer_sizes[layer_idx]
            
            # 绘制神经元
            for neuron_idx in range(layer_size):
                y_pos = y_positions[neuron_idx] if layer_size > 1 else 0.5
                circle = plt.Circle((x_pos, y_pos), 0.05, color='lightblue', ec='black')
                ax.add_patch(circle)
                
                # 添加层标签
                if neuron_idx == layer_size // 2:
                    layer_name = f'输入层\n({layer_size})' if layer_idx == 0 else \
                               f'隐藏层{layer_idx}\n({layer_size})' if layer_idx < n_layers - 1 else \
                               f'输出层\n({layer_size})'
                    ax.text(x_pos, -0.1, layer_name, ha='center', va='top', fontsize=10)
        
        # 绘制连接线（简化版，只显示部分连接）
        for layer_idx in range(n_layers - 1):
            current_size = layer_sizes[layer_idx]
            next_size = layer_sizes[layer_idx + 1]
            
            # 只绘制部分连接线，避免过于密集
            step_current = max(1, current_size // 10)
            step_next = max(1, next_size // 10)
            
            for i in range(0, current_size, step_current):
                for j in range(0, next_size, step_next):
                    x1 = layer_idx / (n_layers - 1)
                    y1 = y_positions[i] if current_size > 1 else 0.5
                    x2 = (layer_idx + 1) / (n_layers - 1)
                    y2 = y_positions[j] if next_size > 1 else 0.5
                    ax.plot([x1, x2], [y1, y2], 'gray', alpha=0.3, linewidth=0.5)
        
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.2, 1.2)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title('MLP网络架构', fontsize=16, fontweight='bold')
        
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
            ax.set_title("MLP：真实类别vs预测类别（测试集）", fontsize=14, fontweight='bold')
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
            ax.set_title("MLP：真实值vs预测值（回归任务-测试集）", fontsize=14, fontweight='bold')
            ax.legend(**self.font_params)

        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return '/static/' + os.path.join('images', os.path.basename(self.output_dir), os.path.basename(save_path)).replace('\\', '/')