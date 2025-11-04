# myvisualizations/baseline_viz.py
import os
import matplotlib.pyplot as plt
import numpy as np

class CustomVisualization:
    def __init__(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir

    def generate_visualizations(self, model, X_train, X_test, y_train, y_test, y_pred, y_pred_proba, feature_names=None, target_names=None):
        """
        生成并保存几张基础图：
        - 真实 vs 预测（散点或条形，分类时按类计数对比）
        返回 dict: {'plot_name': 'static/images/...png', ...}
        """
        results = {}
        try:
            y_test = np.array(y_test)
            y_pred = np.array(y_pred)
            is_class = getattr(model, 'is_classifier', None)
            fname = os.path.join(self.output_dir, 'truth_vs_pred.png')

            plt.figure(figsize=(8, 5))
            if is_class is None:
                # 无类型信息：画散点
                plt.scatter(range(len(y_test)), y_test, label='True', alpha=0.6)
                plt.scatter(range(len(y_pred)), y_pred, label='Pred', alpha=0.6)
                plt.legend()
                plt.title('True vs Pred')
            elif is_class:
                # 分类：绘制真实/预测的类分布对比条形图
                unique = np.unique(np.concatenate([y_test, y_pred]))
                counts_true = [np.sum(y_test == u) for u in unique]
                counts_pred = [np.sum(y_pred == u) for u in unique]
                x = range(len(unique))
                width = 0.35
                plt.bar([xi - width/2 for xi in x], counts_true, width=width, label='True')
                plt.bar([xi + width/2 for xi in x], counts_pred, width=width, label='Pred')
                plt.xticks(x, [str(u) for u in unique])
                plt.legend()
                plt.title('Class distribution: True vs Pred')
            else:
                # 回归：画真实 vs 预测 散点
                plt.scatter(y_test, y_pred, alpha=0.6)
                plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', alpha=0.5)
                plt.xlabel('True')
                plt.ylabel('Pred')
                plt.title('True vs Pred (regression)')

            plt.tight_layout()
            plt.savefig(fname, dpi=150, bbox_inches='tight')
            plt.close()
            results['truth_vs_pred'] = '/static/' + os.path.join('images', os.path.basename(self.output_dir), os.path.basename(fname)).replace('\\', '/')
        except Exception as e:
            # 若失败，返回空 dict
            print("viz error:", e)
            return {}
        return results
