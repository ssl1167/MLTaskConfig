import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.model_selection import learning_curve
import seaborn as sns

class CustomVisualization:
    """GBDT专属可视化模块，适配框架动态加载逻辑，支持中文显示"""
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
            self.font_params = {"fontsize": 10, "fontfamily": "SimHei"}
        except Exception as e:
            print(f"字体设置失败: {str(e)}")
            self.font_params = {"fontsize": 10}

    def _get_viz_paths(self, filename):
        """生成图表保存路径与前端访问路径（符合框架/static路径规范）"""
        save_path = os.path.join(self.output_dir, filename)
        # 提取task_id（输出目录的basename），构建前端可访问路径
        task_id = os.path.basename(self.output_dir)
        web_path = f"/static/images/{task_id}/{filename}"
        return save_path, web_path

    def _save_figure(self, fig, filename):
        """保存图表并返回前端访问路径，确保高分辨率与布局完整"""
        save_path, web_path = self._get_viz_paths(filename)
        plt.tight_layout()  # 自动调整布局，避免标签截断
        fig.savefig(save_path, dpi=150, bbox_inches='tight')  # 150DPI保证清晰度
        plt.close(fig)  # 关闭图表释放内存
        return web_path

    # --------------------------
    # 1. 真实vs预测类别分布（条形图）
    # --------------------------
    def _plot_class_distribution(self, y_true, y_pred, classes):
        """对比真实标签与预测标签的类别数量，快速判断分类准确性"""
        fig, ax = plt.subplots(figsize=(8, 5))
        # 统计各类别样本数
        true_counts = np.bincount(y_true.astype(int), minlength=len(classes))
        pred_counts = np.bincount(y_pred.astype(int), minlength=len(classes))
        # 设置x轴位置
        x = np.arange(len(classes))
        width = 0.35  # 条形图宽度

        # 绘制双条形图
        bars1 = ax.bar(x - width/2, true_counts, width, label='真实标签', color='#2E86AB', alpha=0.8)
        bars2 = ax.bar(x + width/2, pred_counts, width, label='预测标签', color='#A23B72', alpha=0.8)

        # 图表美化与标注
        ax.set_xlabel('类别', fontsize=12)
        ax.set_ylabel('样本数量', fontsize=12)
        ax.set_title('GBDT：真实vs预测类别分布', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'类别{int(cls)}' for cls in classes], fontsize=10)
        ax.legend(fontsize=10)
        # 为条形图添加数值标签
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5, f'{int(height)}',
                    ha='center', va='bottom', fontsize=9)
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5, f'{int(height)}',
                    ha='center', va='bottom', fontsize=9)

        return self._save_figure(fig, 'class_distribution.png')

    # --------------------------
    # 2. 预测概率分布（直方图）
    # --------------------------
    def _plot_probability_distribution(self, y_true, y_pred_proba):
        """按真实类别分组展示类别1的预测概率分布，判断概率校准效果"""
        fig, ax = plt.subplots(figsize=(8, 5))
        proba_1 = y_pred_proba[:, 1]  # 提取类别1的概率

        # 按真实类别绘制直方图
        for cls in [0, 1]:
            mask = (y_true == cls)
            ax.hist(proba_1[mask], bins=15, alpha=0.6,
                    label=f'真实类别{int(cls)}',
                    color='#F18F01' if cls == 0 else '#C73E1D')

        # 添加分类阈值线（0.5）
        ax.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='分类阈值(0.5)')

        # 图表美化
        ax.set_xlabel('类别1预测概率', fontsize=12)
        ax.set_ylabel('样本数量', fontsize=12)
        ax.set_title('GBDT：预测概率分布（按真实类别分组）', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        return self._save_figure(fig, 'probability_distribution.png')

    # --------------------------
    # 3. 特征重要性（基于分裂次数）
    # --------------------------
    def _plot_feature_importance(self, model, feature_names):
        """统计特征在所有基学习器中的分裂次数，衡量特征重要性"""
        fig, ax = plt.subplots(figsize=(10, 6))
        n_features = len(feature_names)
        feature_importance = np.zeros(n_features)  # 初始化特征重要性

        # 递归统计每棵树的特征分裂次数
        def count_split(node):
            if node.value is not None:
                return  # 叶子节点无分裂
            # 累加当前特征的分裂次数
            feature_idx = node.tree['feature_idx']
            feature_importance[feature_idx] += 1
            # 递归统计左右子树
            count_split(node.tree['left'])
            count_split(node.tree['right'])

        # 遍历所有基学习器
        for tree in model.trees:
            count_split(tree)

        # 按重要性从大到小排序
        sorted_idx = np.argsort(feature_importance)[::-1]
        sorted_importance = feature_importance[sorted_idx]
        sorted_feats = [feature_names[i] for i in sorted_idx]

        # 绘制水平条形图
        colors = plt.cm.YlOrRd(np.linspace(0.4, 0.9, len(sorted_importance)))
        bars = ax.barh(sorted_feats, sorted_importance, color=colors)

        # 图表美化
        ax.set_xlabel('特征分裂次数（重要性指标）', fontsize=12)
        ax.set_ylabel('特征名称', fontsize=12)
        ax.set_title('GBDT：特征重要性排名（基于分裂次数）', fontsize=14, fontweight='bold')
        # 为条形图添加数值标签
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.1, bar.get_y() + bar.get_height()/2.,
                    f'{int(width)}', ha='left', va='center', fontsize=9)

        return self._save_figure(fig, 'feature_importance.png')

    # --------------------------
    # 4. 基学习器迭代效果（准确率曲线）
    # --------------------------
    def _plot_iteration_performance(self, model, X_train, X_test, y_train, y_test):
        """展示随基学习器数量增加，训练/测试准确率的变化（分析收敛与过拟合）"""
        fig, ax = plt.subplots(figsize=(10, 5))
        # 控制曲线点数（最多20个点，避免过于密集）
        step = max(1, model.n_estimators // 20)
        n_trees_list = list(range(step, model.n_estimators + 1, step))
        train_acc_list = []
        test_acc_list = []

        # 计算每增加step棵树的准确率
        for n_trees in n_trees_list:
            # 训练集准确率
            f_train = np.zeros(len(y_train))
            for tree in model.trees[:n_trees]:
                f_train += model.learning_rate * tree.predict(X_train)
            train_proba = model._sigmoid(f_train)
            train_acc = np.mean((train_proba >= 0.5).astype(int) == y_train)
            train_acc_list.append(train_acc)

            # 测试集准确率
            f_test = np.zeros(len(y_test))
            for tree in model.trees[:n_trees]:
                f_test += model.learning_rate * tree.predict(X_test)
            test_proba = model._sigmoid(f_test)
            test_acc = np.mean((test_proba >= 0.5).astype(int) == y_test)
            test_acc_list.append(test_acc)

        # 绘制迭代曲线
        ax.plot(n_trees_list, train_acc_list, marker='o', markersize=4,
                label='训练准确率', color='#2E86AB', linewidth=2)
        ax.plot(n_trees_list, test_acc_list, marker='s', markersize=4,
                label='测试准确率', color='#A23B72', linewidth=2)

        # 图表美化
        ax.set_xlabel('基学习器数量（决策树棵数）', fontsize=12)
        ax.set_ylabel('准确率', fontsize=12)
        ax.set_title('GBDT：迭代过程准确率变化', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        # 标注最优测试准确率
        best_test_idx = np.argmax(test_acc_list)
        best_test_acc = test_acc_list[best_test_idx]
        best_n_trees = n_trees_list[best_test_idx]
        ax.annotate(f'最优测试准确率：{best_test_acc:.4f}\n树数量：{best_n_trees}',
                    xy=(best_n_trees, best_test_acc), xytext=(best_n_trees+5, best_test_acc-0.05),
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                    arrowprops=dict(arrowstyle='->', color='black'))

        return self._save_figure(fig, 'iteration_performance.png')

    # --------------------------
    # 可视化入口（框架调用接口）
    # --------------------------
    def generate_visualizations(self, model, X_train, X_test, y_train, y_test, y_pred, y_pred_proba, **kwargs):
        """
        生成梯度提升全面可视化（涵盖数据探索、特征工程、模型评估三个阶段）
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
            # 3. 特征重要性图
            feat_imp_path = self._plot_feature_importance(model, feature_names)
            if feat_imp_path:
                results["feature_importance"] = feat_imp_path

            # 4. 迭代效果曲线图（GBDT特有）
            iteration_path = self._plot_iteration_performance(model, X_train, X_test, y_train, y_test)
            if iteration_path:
                results["iteration_performance"] = iteration_path

            # ========== 第三阶段：模型评估与性能分析 ==========
            # 5. 混淆矩阵（分类任务）
            if model.is_classifier:
                confusion_path = self._plot_confusion_matrix(y_test, y_pred, target_names)
                if confusion_path:
                    results["confusion_matrix"] = confusion_path
                
                # 6. ROC曲线
                roc_path = self._plot_roc_curves(y_test, y_pred_proba, target_names)
                if roc_path:
                    results["roc_curve"] = roc_path
                
                # 7. 精确率-召回率曲线
                pr_path = self._plot_precision_recall_curve(y_test, y_pred_proba, target_names)
                if pr_path:
                    results["precision_recall_curve"] = pr_path
                
                # 8. 校准曲线（仅二分类）
                if len(np.unique(y_test)) == 2:
                    calib_path = self._plot_calibration_curve(y_test, y_pred_proba)
                    if calib_path:
                        results["calibration_curve"] = calib_path
            else:
                # 回归任务：残差图
                residual_path = self._plot_residuals(y_test, y_pred)
                if residual_path:
                    results["residual_plot"] = residual_path

            # 9. 学习曲线（通用）
            learning_path = self._plot_learning_curve(model, X_train, y_train)
            if learning_path:
                results["learning_curve"] = learning_path

            # 10. 真实vs预测对比可视化（通用）
            truth_pred_path = self._plot_truth_vs_pred(y_test, y_pred, model)
            if truth_pred_path:
                results["truth_vs_pred"] = truth_pred_path

        except Exception as e:
            print(f"梯度提升可视化生成失败: {str(e)}")
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
        ax1.set_title('训练集类别分布', fontsize=12, fontweight='bold', **self.font_params)
        ax1.set_xlabel('类别', **self.font_params)
        ax1.set_ylabel('样本数量', **self.font_params)
        ax1.grid(True, alpha=0.3)
        
        # 测试集分布
        unique_test, counts_test = np.unique(y_test, return_counts=True)
        ax2.bar(unique_test, counts_test, color='lightcoral', alpha=0.7, label='测试集')
        ax2.set_title('测试集类别分布', fontsize=12, fontweight='bold', **self.font_params)
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
        plt.title('特征相关性矩阵', fontsize=14, fontweight='bold', **self.font_params)
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
        plt.title('混淆矩阵', fontsize=14, fontweight='bold', **self.font_params)
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
        plt.title('ROC曲线', fontsize=14, fontweight='bold', **self.font_params)
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
        plt.title('精确率-召回率曲线', fontsize=14, fontweight='bold', **self.font_params)
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
                    model_copy = type(estimator)(**estimator.__dict__)
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
        plt.title('学习曲线', fontsize=14, fontweight='bold', **self.font_params)
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
                label="梯度提升", color='blue')
        plt.plot([0, 1], [0, 1], "k:", label="完美校准")
        plt.xlabel('平均预测概率', **self.font_params)
        plt.ylabel('正例比例', **self.font_params)
        plt.title('校准曲线', fontsize=14, fontweight='bold', **self.font_params)
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
        ax1.set_title('残差 vs 预测值', fontsize=12, fontweight='bold', **self.font_params)
        ax1.grid(True, alpha=0.3)
        
        # 残差直方图
        ax2.hist(residuals, bins=20, alpha=0.7, color='green', edgecolor='black')
        ax2.set_xlabel('残差', **self.font_params)
        ax2.set_ylabel('频次', **self.font_params)
        ax2.set_title('残差分布', fontsize=12, fontweight='bold', **self.font_params)
        ax2.grid(True, alpha=0.3)
        
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
            ax.set_title("梯度提升：真实类别vs预测类别（测试集）", fontsize=14, fontweight='bold', **self.font_params)
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
            ax.set_title("梯度提升：真实值vs预测值（回归任务-测试集）", fontsize=14, fontweight='bold', **self.font_params)
            ax.legend(**self.font_params)

        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return '/static/' + os.path.join('images', os.path.basename(self.output_dir), os.path.basename(save_path)).replace('\\', '/')