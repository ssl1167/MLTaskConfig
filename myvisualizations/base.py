# myvisualizations/base.py
import os

class BaseVisualization:
    """
    最小基类：负责创建目录并提供一个 generate_visualizations 接口
    子类需要实现 generate_visualizations 并返回 dict {name: path}
    """
    def __init__(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir

    def generate_visualizations(self, model, X_train, X_test, y_train, y_test, y_pred, y_pred_proba, **kwargs):
        # 子类重写
        return {}
