import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.model_selection import learning_curve
import seaborn as sns


class CustomVisualization:
    """朴素贝叶斯专属可视化模块，适配框架动态加载逻辑，支持中文显示与特征类型差异化展示"""

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
        bars1 = ax.bar(x - width / 2, true_counts, width, label='真实标签', color='#3498db', alpha=0.8)
        bars2 = ax.bar(x + width / 2, pred_counts, width, label='预测标签', color='#e74c3c', alpha=0.8)

        # 图表美化与标注
        ax.set_xlabel('类别', fontsize=12)
        ax.set_ylabel('样本数量', fontsize=12)
        ax.set_title('朴素贝叶斯：真实vs预测类别分布', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'类别{int(cls)}' for cls in classes], fontsize=10)
        ax.legend(fontsize=10)
        # 为条形图添加数值标签
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.5, f'{int(height)}',
                    ha='center', va='bottom', fontsize=9)
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.5, f'{int(height)}',
                    ha='center', va='bottom', fontsize=9)

        return self._save_figure(fig, 'class_distribution.png')

    # --------------------------
    # 2. 后验概率分布（直方图）
    # --------------------------
    def _plot_posterior_distribution(self, y_true, y_pred_proba, classes):
        """按真实类别分组展示后验概率分布，判断概率校准效果（如真实类别1的概率应集中在高值区）"""
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.Set3(np.linspace(0, 1, len(classes)))  # 差异化颜色

        for cls_idx, cls in enumerate(classes):
            # 提取真实类别为cls的样本，对应类别cls的后验概率
            mask = (y_true == cls)
            probs = y_pred_proba[mask, cls_idx]
            if len(probs) == 0:
                continue  # 跳过无样本的类别

            # 绘制直方图
            ax.hist(probs, bins=15, alpha=0.6,
                    label=f'真实类别{int(cls)} → 预测类别{int(cls)}概率',
                    color=colors[cls_idx])

        # 添加概率阈值线（0.5，二分类专用）
        if len(classes) == 2:
            ax.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='分类阈值(0.5)')

        # 图表美化
        ax.set_xlabel('后验概率', fontsize=12)
        ax.set_ylabel('样本数量', fontsize=12)
        ax.set_title('朴素贝叶斯：后验概率分布（按真实类别分组）', fontsize=14, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        return self._save_figure(fig, 'posterior_distribution.png')

    # --------------------------
    # 3. 特征条件概率（连续→高斯曲线，离散→热力图）
    # --------------------------
    def _plot_gaussian_conditional(self, model, feature_names):
        """连续特征：绘制各特征在不同类别下的高斯条件概率曲线"""
        # 选择前2个特征（避免图表过于复杂）
        n_features = min(2, len(feature_names))
        fig, axes = plt.subplots(n_features, 1, figsize=(10, 6 * n_features))
        if n_features == 1:
            axes = [axes]  # 确保axes为列表格式

        for feat_idx, ax in enumerate(axes):
            feat_name = feature_names[feat_idx] if len(feature_names) > feat_idx else f'特征{feat_idx}'
            # 生成特征值范围（覆盖所有类别均值±3倍标准差，确保曲线完整）
            all_vals = []
            for cls in model.classes_:
                mean = model.gaussian_params[cls]['mean'][feat_idx]
                std = model.gaussian_params[cls]['std'][feat_idx]
                all_vals.extend([mean - 3 * std, mean + 3 * std])
            x = np.linspace(min(all_vals), max(all_vals), 1000)  # 生成1000个采样点

            # 绘制每个类别的高斯曲线
            colors = plt.cm.Set2(np.linspace(0, 1, len(model.classes_)))
            for cls_idx, cls in enumerate(model.classes_):
                mean = model.gaussian_params[cls]['mean'][feat_idx]
                std = model.gaussian_params[cls]['std'][feat_idx]
                # 计算条件概率密度
                pdf = model._gaussian_pdf(x, mean, std)
                # 绘制曲线并标注均值、标准差
                ax.plot(x, pdf, label=f'类别{int(cls)}（均值={mean:.2f}, 标准差={std:.2f}）',
                        color=colors[cls_idx], linewidth=2)

            # 图表美化
            ax.set_xlabel(feat_name, fontsize=12)
            ax.set_ylabel('条件概率密度 P(X|Y)', fontsize=12)
            ax.set_title(f'朴素贝叶斯：{feat_name}的类别条件概率分布（高斯）', fontsize=13, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

        return self._save_figure(fig, 'gaussian_conditional.png')

    def _plot_poly_conditional(self, model, feature_names):
        """离散特征：绘制各特征在不同类别下的条件概率热力图"""
        # 选择前2个特征（避免图表过于复杂）
        n_features = min(2, len(feature_names))
        fig, axes = plt.subplots(1, n_features, figsize=(12, 5))
        if n_features == 1:
            axes = [axes]  # 确保axes为列表格式

        for feat_idx, ax in enumerate(axes):
            feat_name = feature_names[feat_idx] if len(feature_names) > feat_idx else f'特征{feat_idx}'
            # 提取当前特征的所有可能值（所有类别中出现过的值）
            all_vals = set()
            for cls in model.classes_:
                all_vals.update(model.poly_params[cls][feat_idx].keys())
            all_vals = sorted(list(all_vals))  # 排序便于展示

            # 构建条件概率矩阵（行：类别，列：特征值）
            prob_matrix = np.zeros((len(model.classes_), len(all_vals)))
            for cls_idx, cls in enumerate(model.classes_):
                for val_idx, val in enumerate(all_vals):
                    # 填充条件概率（未出现的值概率为0）
                    prob_matrix[cls_idx, val_idx] = model.poly_params[cls][feat_idx].get(val, 0.0)

            # 绘制热力图
            im = ax.imshow(prob_matrix, cmap='YlOrRd', aspect='auto')
            # 设置坐标轴标签
            ax.set_xticks(np.arange(len(all_vals)))
            ax.set_xticklabels([f'值{int(v)}' for v in all_vals], fontsize=10, rotation=45)
            ax.set_yticks(np.arange(len(model.classes_)))
            ax.set_yticklabels([f'类别{int(c)}' for c in model.classes_], fontsize=10)

            # 为热力图添加数值标签（保留2位小数）
            for i in range(len(model.classes_)):
                for j in range(len(all_vals)):
                    text = ax.text(j, i, f'{prob_matrix[i, j]:.2f}',
                                   ha='center', va='center',
                                   color='black' if prob_matrix[i, j] < 0.5 else 'white',
                                   fontsize=9)

            # 图表美化
            ax.set_xlabel(f'{feat_name}值', fontsize=12)
            ax.set_ylabel('类别', fontsize=12)
            ax.set_title(f'朴素贝叶斯：{feat_name}的类别条件概率（多项式）', fontsize=13, fontweight='bold')
            # 添加颜色条（标注概率范围）
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('条件概率 P(X|Y)', fontsize=10)

        return self._save_figure(fig, 'polynomial_conditional.png')

    # --------------------------
    # 4. 二分类决策边界（基于前2个特征）
    # --------------------------
    def _plot_decision_boundary(self, model, X_test, y_test, classes, feature_names=None):
        """二分类场景：绘制基于前2个特征的决策边界，直观理解分类逻辑"""
        if len(classes) != 2:
            return None  # 仅支持二分类
        if X_test.shape[1] < 2:
            return None  # 至少需要2个特征

        # 提取前2个特征（简化可视化）
        X = X_test[:, :2]
        # 生成网格点（覆盖特征值范围，确保决策边界完整）
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

        # 补全其他特征（使用测试集对应特征的均值，避免特征数量不匹配）
        grid = np.c_[xx.ravel(), yy.ravel()]  # 形状：(200*200, 2)
        if X_test.shape[1] > 2:
            mean_other_feats = np.mean(X_test[:, 2:], axis=0)  # 其他特征的均值
            grid = np.hstack([grid, np.tile(mean_other_feats, (grid.shape[0], 1))])

        # 预测网格点的类别概率（类别1的概率）
        y_grid_proba = model.predict_proba(grid)[:, 1].reshape(xx.shape)  # 形状：(200, 200)

        # 绘制决策边界
        fig, ax = plt.subplots(figsize=(10, 8))
        # 绘制概率等高线（颜色深浅表示类别1概率）
        contour = ax.contourf(xx, yy, y_grid_proba, alpha=0.3, cmap='RdBu', levels=np.linspace(0, 1, 11))
        # 绘制决策边界线（概率=0.5的位置）
        ax.contour(xx, yy, y_grid_proba, levels=[0.5], colors='black', linewidths=2, linestyles='-')
        # 绘制测试集样本点（颜色表示真实类别）
        scatter = ax.scatter(X[:, 0], X[:, 1], c=y_test, cmap='RdBu', edgecolors='black', s=60, alpha=0.8)

        # 图表美化
        feat1_name = '特征1' if feature_names is None or len(feature_names) < 1 else feature_names[0]
        feat2_name = '特征2' if feature_names is None or len(feature_names) < 2 else feature_names[1]
        ax.set_xlabel(feat1_name, fontsize=12)
        ax.set_ylabel(feat2_name, fontsize=12)
        ax.set_title('朴素贝叶斯：二分类决策边界（前2个特征）', fontsize=14, fontweight='bold')

        # 添加颜色条（左侧：概率，右侧：真实类别）
        cbar1 = plt.colorbar(contour, ax=ax, shrink=0.8)
        cbar1.set_label('类别1后验概率', fontsize=10)
        cbar2 = plt.colorbar(scatter, ax=ax, shrink=0.8, pad=0.05)
        cbar2.set_label('真实类别', fontsize=10)
        cbar2.set_ticks(classes)
        cbar2.set_ticklabels([f'类别{int(c)}' for c in classes])

        return self._save_figure(fig, 'decision_boundary.png')

    # --------------------------
    # 可视化入口（框架调用接口）
    # --------------------------
    def generate_visualizations(self, model, X_train, X_test, y_train, y_test, y_pred, y_pred_proba, **kwargs):
        """
        生成朴素贝叶斯全面可视化（涵盖数据探索、特征工程、模型评估三个阶段）
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
        classes = model.classes_

        try:
            # ========== 第一阶段：数据探索与理解 ==========
            # 1. 数据分布图（单变量分析）
            try:
                data_dist_path = self._plot_data_distribution(y_train, y_test, target_names)
                if data_dist_path:
                    results["data_distribution"] = data_dist_path
            except Exception as e:
                print(f"生成数据分布图失败：{str(e)}")
                import traceback
                traceback.print_exc()
            
            # 2. 特征相关性矩阵（多变量关系分析）
            try:
                corr_matrix_path = self._plot_correlation_matrix(X_train, feature_names)
                if corr_matrix_path:
                    results["correlation_matrix"] = corr_matrix_path
            except Exception as e:
                print(f"生成相关性矩阵失败：{str(e)}")

            # ========== 第二阶段：特征工程与模型构建 ==========
            # 3. 特征重要性图（基于条件概率差异）
            try:
                feat_imp_path = self._plot_feature_importance(model, feature_names)
                if feat_imp_path:
                    results["feature_importance"] = feat_imp_path
            except Exception as e:
                print(f"生成特征重要性图失败：{str(e)}")

            # 4. 特征条件概率图（朴素贝叶斯特有）
            try:
                if model.feature_type == 'continuous':
                    results['feature_conditional'] = self._plot_gaussian_conditional(model, feature_names)
                else:
                    results['feature_conditional'] = self._plot_poly_conditional(model, feature_names)
            except Exception as e:
                print(f"生成特征条件概率图表失败：{str(e)}")

            # 5. 二分类决策边界图（仅二分类+至少2个特征）
            if len(classes) == 2 and X_test.shape[1] >= 2:
                try:
                    decision_path = self._plot_decision_boundary(model, X_test, y_true, classes, feature_names)
                    if decision_path:
                        results['decision_boundary'] = decision_path
                except Exception as e:
                    print(f"生成决策边界图表失败：{str(e)}")

            # ========== 第三阶段：模型评估与性能分析 ==========
            # 6. 混淆矩阵（分类任务）
            try:
                confusion_path = self._plot_confusion_matrix(y_test, y_pred, target_names)
                if confusion_path:
                    results["confusion_matrix"] = confusion_path
            except Exception as e:
                print(f"生成混淆矩阵失败：{str(e)}")
                
            # 7. ROC曲线
            try:
                roc_path = self._plot_roc_curves(y_test, y_pred_proba, target_names)
                if roc_path:
                    results["roc_curve"] = roc_path
            except Exception as e:
                print(f"生成ROC曲线失败：{str(e)}")
                
            # 8. 精确率-召回率曲线
            try:
                pr_path = self._plot_precision_recall_curve(y_test, y_pred_proba, target_names)
                if pr_path:
                    results["precision_recall_curve"] = pr_path
            except Exception as e:
                print(f"生成精确率-召回率曲线失败：{str(e)}")
                
            # 9. 校准曲线（仅二分类）
            if len(np.unique(y_test)) == 2:
                try:
                    calib_path = self._plot_calibration_curve(y_test, y_pred_proba)
                    if calib_path:
                        results["calibration_curve"] = calib_path
                except Exception as e:
                    print(f"生成校准曲线失败：{str(e)}")

            # 10. 学习曲线（通用）
            try:
                learning_path = self._plot_learning_curve(model, X_train, y_train)
                if learning_path:
                    results["learning_curve"] = learning_path
            except Exception as e:
                print(f"生成学习曲线失败：{str(e)}")

            # 11. 真实vs预测对比可视化（通用）
            try:
                truth_pred_path = self._plot_truth_vs_pred(y_test, y_pred, model)
                if truth_pred_path:
                    results["truth_vs_pred"] = truth_pred_path
            except Exception as e:
                print(f"生成真实vs预测图失败：{str(e)}")

            # 12. 后验概率分布图（朴素贝叶斯特有）
            try:
                results['posterior_distribution'] = self._plot_posterior_distribution(y_true, y_pred_proba, classes)
            except Exception as e:
                print(f"生成后验概率图表失败：{str(e)}")

        except Exception as e:
            print(f"朴素贝叶斯可视化生成失败: {str(e)}")
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
        """绘制特征重要性图（基于条件概率差异）"""
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
        ax.set_ylabel("重要性（基于条件概率差异）", **self.font_params)
        ax.set_title("朴素贝叶斯特征重要性", fontsize=14, fontweight="bold")
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
                        alpha=estimator.alpha
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
                label="朴素贝叶斯", color='blue')
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
            ax.set_title("朴素贝叶斯：真实类别vs预测类别（测试集）", fontsize=14, fontweight='bold')
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
            ax.set_title("朴素贝叶斯：真实值vs预测值（回归任务-测试集）", fontsize=14, fontweight='bold')
            ax.legend(**self.font_params)

        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return '/static/' + os.path.join('images', os.path.basename(self.output_dir), os.path.basename(save_path)).replace('\\', '/')