# app.py
import os
import sys
import traceback
import uuid
import logging
import json
from datetime import datetime
from io import BytesIO
from importlib import import_module
from flask import Flask, render_template, jsonify, request, session
from flask_wtf.csrf import CSRFProtect
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

# -------------------------
# 日志
# -------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ml_app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# -------------------------
# 路径与自定义模块导入准备
# -------------------------
# 保证可以导入 mymodels 和 myvisualizations
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# 导入数据加载器
from data_loader import data_loader

# matplotlib 设置 - 增强中文字体支持
try:
    # 添加更多通用中文字体选项，提高兼容性
    plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'Heiti TC', 'Arial Unicode MS', 'Microsoft YaHei', 'SimSun', 'NSimSun', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    print("Matplotlib字体配置已设置")
except Exception as e:
    print(f"设置Matplotlib字体时出错: {str(e)}")
plt.ioff()

# Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('ML_APP_SECRET_KEY', 'ml_task_config_secret_key_2024_secure')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['TEMPLATES_AUTO_RELOAD'] = True

os.makedirs('static/images', exist_ok=True)
os.makedirs('tasks', exist_ok=True)

# -------------------------
# 配置（前端下拉数据保持不变或更新 id 与 mymodels 模块名同步）
# -------------------------
DATASETS = [
    # 分类任务 - 按难度排序
    {'id': 'wholesale_customers', 'name': '批发客户数据集 (适合聚类)', 'type': 'classification'},
    {'id': 'iris', 'name': '鸢尾花数据集 (简单)', 'type': 'classification'},
    {'id': 'titanic', 'name': '泰坦尼克号数据集 (简单-中等)', 'type': 'classification'},
    {'id': 'wine', 'name': '葡萄酒数据集 (中等)', 'type': 'classification'},
    {'id': 'breast_cancer', 'name': '乳腺癌数据集 (中等)', 'type': 'classification'},
    {'id': 'adult', 'name': '收入预测数据集 (中等-复杂)', 'type': 'classification'},
    {'id': 'credit_card', 'name': '信用卡欺诈检测 (复杂)', 'type': 'classification'},
    
    # 回归任务 - 按难度排序
    {'id': 'diabetes', 'name': '糖尿病进展预测 (简单)', 'type': 'regression'},
    {'id': 'concrete', 'name': '混凝土强度预测 (简单)', 'type': 'regression'},
    {'id': 'bike_sharing', 'name': '共享单车需求预测 (中等)', 'type': 'regression'},
    {'id': 'california_housing', 'name': '加州房价预测 (中等-复杂)', 'type': 'regression'}
]

SPLIT_RATIOS = [
    {'id': '0.2', 'name': '20%测试集'},
    {'id': '0.3', 'name': '30%测试集'},
    {'id': '0.4', 'name': '40%测试集'}
]

SPLITTERS = [
    {'id': 'train_test_split', 'name': '简单分割'},
    {'id': 'kfold', 'name': 'K折交叉验证'},
    {'id': 'stratified_kfold', 'name': '分层K折交叉验证'}
]

LEARNING_RATES = [
    {'id': 'low', 'name': '低学习率 (0.001)'},
    {'id': 'medium', 'name': '中等学习率 (0.01)'},
    {'id': 'high', 'name': '高学习率 (0.1)'}
]

# 注意：models 的 id 应与 mymodels 里的模块名一致
MODELS = [
    {'id': 'random_forest', 'name': '随机森林', 'supports': ['classification', 'regression']},
    {'id': 'decision_tree', 'name': '决策树', 'supports': ['classification', 'regression']},
    {'id': 'gradient_boosting', 'name': '梯度提升', 'supports': ['classification', 'regression']},
    {'id': 'knn', 'name': 'K最近邻', 'supports': ['classification', 'regression']},
    {'id': 'logistic_regression', 'name': '逻辑回归', 'supports': ['classification']},
    {'id': 'mlp', 'name': '多层感知机', 'supports': ['classification', 'regression']},
    {'id': 'naive_bayes', 'name': '朴素贝叶斯', 'supports': ['classification']},
    {'id': 'svm', 'name': '支持向量机', 'supports': ['classification', 'regression']},
    {'id': 'kmeans', 'name': 'K-Means聚类', 'supports': ['classification', 'regression']}
]

METRICS = {
    'classification': [
        {'id': 'accuracy', 'name': '准确率'},
        {'id': 'precision', 'name': '精确率'},
        {'id': 'recall', 'name': '召回率'},
        {'id': 'f1_score', 'name': 'F1分数'},
        {'id': 'roc_auc', 'name': 'ROC AUC'}
    ],
    'regression': [
        {'id': 'r2', 'name': 'R²分数'},
        {'id': 'mse', 'name': '均方误差'},
        {'id': 'rmse', 'name': '均方根误差'},
        {'id': 'mae', 'name': '平均绝对误差'}
    ]
}

# -------------------------
# 辅助：动态加载自定义模型/可视化
# 约定：
# - mymodels/<model_id>.py 中定义类 CustomModel，必须实现 train(X, y)、predict(X)；
#   对于分类可选实现 predict_proba(X)。
# - myvisualizations/<model_id>_viz.py 中定义类 CustomVisualization，必须实现 generate_visualizations(...)
# -------------------------
def load_custom_model(model_id, **kwargs):
    try:
        module = import_module(f"mymodels.{model_id}")
        if hasattr(module, 'CustomModel'):
            return module.CustomModel(**kwargs)
        else:
            raise ImportError(f"模块 mymodels.{model_id} 未定义 CustomModel 类")
    except Exception as e:
        raise ImportError(f"加载自定义模型失败: {e}")

def load_custom_visualization(model_id, output_dir):
    try:
        module = import_module(f"myvisualizations.{model_id}_viz")
        if hasattr(module, 'CustomVisualization'):
            return module.CustomVisualization(output_dir)
    except Exception as e:
        logger.warning(f"加载可视化模块失败 {model_id}_viz: {str(e)}")
        # 退回到基类
        try:
            base = import_module("myvisualizations.base")
            if hasattr(base, 'BaseVisualization'):
                return base.BaseVisualization(output_dir)
        except Exception as e:
            logger.warning(f"加载基础可视化类失败: {str(e)}")
            # 最后退回到一个内联最小实现
            class InlineBaseViz:
                def __init__(self, output_dir):
                    os.makedirs(output_dir, exist_ok=True)
                    self.output_dir = output_dir
                def generate_visualizations(self, *args, **kwargs):
                    return {}
            return InlineBaseViz(output_dir)
    # fallback if module exists but no class
    return None

# -------------------------
# KMeans 辅助：对聚类标签进行对齐（将簇ID映射到真实标签，按多数投票）
# -------------------------
def align_cluster_labels_majority(y_true, y_pred):
    try:
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        true_labels = np.unique(y_true)
        clusters = np.unique(y_pred)

        # 统计每个簇内真实标签的数量
        mapping = {}
        used_true = set()
        counts = {}
        for c in clusters:
            mask = (y_pred == c)
            vals, cnts = np.unique(y_true[mask], return_counts=True)
            if len(cnts) == 0:
                mapping[c] = None
                continue
            # 记录计数用于后续冲突处理
            max_idx = int(np.argmax(cnts))
            counts[c] = (int(vals[max_idx]), int(cnts[max_idx]))

        # 按簇内最大计数从大到小，贪心分配到未占用的真实标签
        for c, (label, cnt) in sorted(counts.items(), key=lambda kv: kv[1][1], reverse=True):
            if label not in used_true:
                mapping[c] = label
                used_true.add(label)
            else:
                # 找次优的标签
                mask = (y_pred == c)
                vals, cnts = np.unique(y_true[mask], return_counts=True)
                order = np.argsort(cnts)[::-1]
                assigned = None
                for idx in order:
                    cand = int(vals[idx])
                    if cand not in used_true:
                        assigned = cand
                        break
                mapping[c] = assigned if assigned is not None else label

        # 生成对齐后的预测
        y_aligned = np.array([mapping.get(int(c), int(c)) for c in y_pred])
        return y_aligned
    except Exception:
        # 任何异常回退原预测
        return y_pred

# -------------------------
# 页面与 API
# -------------------------
@app.route('/')
def index():
    return render_template(
        'index.html',
        datasets=DATASETS,
        split_ratios=SPLIT_RATIOS,
        splitters=SPLITTERS,
        learning_rates=LEARNING_RATES,
        models=MODELS,
        metrics=METRICS
    )

@app.route('/get_compatible_options', methods=['POST'])
def get_compatible_options():
    try:
        dataset_id = request.json.get('dataset_id')
        if not dataset_id:
            return jsonify({'success': False, 'error': '未提供数据集ID'})
        dataset = next((d for d in DATASETS if d['id'] == dataset_id), None)
        if not dataset:
            return jsonify({'success': False, 'error': '未知数据集'})
        dataset_type = dataset['type']
        compatible_models = [m for m in MODELS if dataset_type in m['supports']]
        compatible_metrics = METRICS.get(dataset_type, [])
        return jsonify({'success': True, 'models': compatible_models, 'metrics': compatible_metrics})
    except Exception as e:
        logger.error(f"获取兼容选项失败: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

# -------------------------
# 运行任务主逻辑（使用自定义模型）
# -------------------------
@app.route('/run_task', methods=['POST'])
def run_task():
    task_id = str(uuid.uuid4())
    start_time = datetime.now()
    logs = [f"任务ID: {task_id}", f"任务开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}"]
    try:
        dataset_id = request.form.get('dataset')
        split_ratio = float(request.form.get('split_ratio', 0.2))
        splitter = request.form.get('splitter', 'train_test_split')
        learning_rate = request.form.get('learning_rate', 'medium')
        model_id = request.form.get('model')
        metrics = request.form.getlist('metrics')

        if not all([dataset_id, model_id]):
            raise ValueError("数据集和模型为必填项")

        def get_name_safely(items, item_id, default="未知"):
            for item in items:
                if item['id'] == item_id:
                    return item['name']
            return f"{default}({item_id})"

        logs.append(f"选择的数据集: {get_name_safely(DATASETS, dataset_id)}")
        logs.append(f"选择的算法: {get_name_safely(MODELS, model_id)}")
        all_metrics = METRICS['classification'] + METRICS['regression']
        metric_names = [get_name_safely(all_metrics, metric) for metric in metrics]
        logs.append(f"选择的指标: {metric_names}")
        logs.append(f"选择的学习率: {get_name_safely(LEARNING_RATES, learning_rate)}")

        dataset_info = next((d for d in DATASETS if d['id'] == dataset_id), None)
        if not dataset_info:
            raise ValueError(f"未知的数据集ID: {dataset_id}")
        dataset_type = dataset_info['type']

        task_info = {
            'task_id': task_id,
            'model': get_name_safely(MODELS, model_id),
            'dataset': get_name_safely(DATASETS, dataset_id),
            'dataset_type': dataset_type,
            'metrics': metric_names,
            'learning_rate': get_name_safely(LEARNING_RATES, learning_rate),
            'split_ratio': split_ratio,
            'splitter': next(s['name'] for s in SPLITTERS if s['id'] == splitter)
        }

        # 加载数据集或原始数据（根据分割策略）
        logs.append("开始加载数据集...")
        try:
            if splitter == 'train_test_split':
                X_train, X_test, y_train, y_test, feature_names, target_names, _, preprocessor = data_loader.load_dataset(
                    dataset_id,
                    return_type='array',
                    preprocess=True,
                    test_size=split_ratio
                )
                logs.append(f"数据集加载完成，训练集: {X_train.shape[0]} 个样本, 测试集: {X_test.shape[0]} 个样本, 特征数: {X_train.shape[1]}")
            else:
                # 交叉验证：加载原始数据，后续每折各自预处理
                X_raw, y_raw, feature_names, target_names = data_loader._load_raw_data(dataset_id)
                logs.append(f"数据集加载完成，总样本: {X_raw.shape[0]} 个, 特征数: {X_raw.shape[1]}")
        except Exception as e:
            logs.append(f"数据集加载失败: {str(e)}")
            raise

        # 学习率值转换（只是传递给模型的参考值）
        lr_value = 0.001
        if learning_rate == 'medium':
            lr_value = 0.01
        elif learning_rate == 'high':
            lr_value = 0.1

        # 动态加载自定义模型
        logs.append("初始化自定义模型...")
        try:
            # 自动为KMeans设置簇数（分类任务：等于训练集真实类别数；回归任务：默认3）
            extra_kwargs = {}
            if model_id == 'kmeans':
                if dataset_type == 'classification':
                    # 依据训练集类别数设置k（避免信息泄露到测试集）
                    k_val = len(np.unique(y_train)) if 'y_train' in locals() else 3
                else:
                    k_val = 3
                extra_kwargs['n_clusters'] = int(max(2, k_val))
            model = load_custom_model(model_id, learning_rate=lr_value, **extra_kwargs)
            logs.append(f"成功加载模型模块: mymodels.{model_id}")
        except Exception as e:
            logs.append(f"加载自定义模型失败: {str(e)}")
            raise

        # 训练/评估
        visualization_paths = {}
        if splitter == 'train_test_split':
            logs.append("开始模型训练...")
            try:
                model.train(X_train, y_train)
                logs.append("模型训练完成")
            except Exception as e:
                logs.append(f"自定义模型训练失败: {str(e)}")
                raise

            logs.append("进行模型预测...")
            try:
                y_pred = model.predict(X_test)
                logs.append("模型预测完成")
            except Exception as e:
                logs.append(f"自定义模型预测失败: {str(e)}")
                raise
            
            # 尝试获取概率预测（仅分类任务）
            y_pred_proba = None
            if dataset_type == 'classification' and hasattr(model, 'predict_proba'):
                try:
                    y_pred_proba = model.predict_proba(X_test)
                except (NotImplementedError, AttributeError, Exception) as e:
                    logs.append(f"predict_proba不可用: {str(e)}")
                    y_pred_proba = None

            logs.append("开始模型评估...")
            # 若为KMeans聚类+分类数据，先对齐簇标签再评估
            if model_id == 'kmeans' and dataset_type == 'classification':
                y_eval = align_cluster_labels_majority(y_test, y_pred)
            else:
                y_eval = y_pred
            results = evaluate_model(y_test, y_eval, y_pred_proba, metrics, dataset_type)
            logs.append("模型评估完成")
            for metric, value in results.items():
                logs.append(f"{metric}: {value:.6f}")

            logs.append("开始生成可视化...")
            output_dir = f'static/images/{task_id}'
            viz = load_custom_visualization(model_id, output_dir=output_dir)
            try:
                visualization_paths = viz.generate_visualizations(model, X_train, X_test, y_train, y_test, y_pred, y_pred_proba, feature_names=feature_names, target_names=target_names)
                visualization_paths = normalize_visualization_paths(visualization_paths)
                logs.append(f"成功生成 {len(visualization_paths)} 个可视化")
            except Exception as e:
                logs.append(f"生成可视化失败: {str(e)}")
                visualization_paths = {}
        else:
            # 交叉验证
            n_splits = 5
            if splitter == 'kfold':
                kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
                split_gen = kf.split(X_raw)
            else:
                # 确保目标变量是分类类型且非连续
                if len(np.unique(y_raw)) > 10:
                    logger.warning(f"目标变量类别数过多({len(np.unique(y_raw))})，可能不适合使用StratifiedKFold")
                skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                split_gen = skf.split(X_raw, y_raw)

            # 累加指标
            metric_sums = {}
            fold_idx = 0
            last_fold_data = None

            for train_idx, test_idx in split_gen:
                fold_idx += 1
                logs.append(f"开始第 {fold_idx}/{n_splits} 折...")

                X_tr_raw, X_te_raw = X_raw[train_idx], X_raw[test_idx]
                y_tr, y_te = y_raw[train_idx], y_raw[test_idx]

                # 每折单独预处理（仅用训练集拟合）
                X_tr, X_te, _ = data_loader.preprocess_data(X_tr_raw, X_te_raw, y_tr, dataset_type)

                # 初始化并训练模型
                try:
                    extra_kwargs = {}
                    if model_id == 'kmeans':
                        if dataset_type == 'classification':
                            k_val = len(np.unique(y_tr))
                        else:
                            k_val = 3
                        extra_kwargs['n_clusters'] = int(max(2, k_val))
                    model = load_custom_model(model_id, learning_rate=lr_value, **extra_kwargs)
                    model.train(X_tr, y_tr)
                except Exception as e:
                    logs.append(f"第 {fold_idx} 折训练失败: {str(e)}")
                    raise

                # 预测
                try:
                    y_pred = model.predict(X_te)
                except Exception as e:
                    logs.append(f"第 {fold_idx} 折预测失败: {str(e)}")
                    raise
                
                # 尝试获取概率预测（仅分类任务）
                y_pred_proba = None
                if dataset_type == 'classification' and hasattr(model, 'predict_proba'):
                    try:
                        y_pred_proba = model.predict_proba(X_te)
                    except (NotImplementedError, AttributeError, Exception):
                        y_pred_proba = None

                # 评估
                if model_id == 'kmeans' and dataset_type == 'classification':
                    y_eval = align_cluster_labels_majority(y_te, y_pred)
                else:
                    y_eval = y_pred
                fold_results = evaluate_model(y_te, y_eval, y_pred_proba, metrics, dataset_type)
                for k, v in fold_results.items():
                    metric_sums[k] = metric_sums.get(k, 0.0) + float(v)
                logs.append("第 {} 折结果: {}".format(fold_idx, {k: float(v) for k, v in fold_results.items()}))

                # 保存最后一折用于可视化
                last_fold_data = (model, X_tr, X_te, y_tr, y_te, y_pred, y_pred_proba)

            # 平均指标
            results = {k: (metric_sums[k] / n_splits) for k in metric_sums}
            logs.append("交叉验证平均结果: " + str(results))

            # 可视化（最后一折）
            if last_fold_data is not None:
                model, X_tr, X_te, y_tr, y_te, y_pred, y_pred_proba = last_fold_data
                logs.append("开始生成可视化(最后一折)...")
                output_dir = f'static/images/{task_id}'
                viz = load_custom_visualization(model_id, output_dir=output_dir)
                try:
                    visualization_paths = viz.generate_visualizations(model, X_tr, X_te, y_tr, y_te, y_pred, y_pred_proba, feature_names=feature_names, target_names=target_names)
                    visualization_paths = normalize_visualization_paths(visualization_paths)
                    logs.append(f"成功生成 {len(visualization_paths)} 个可视化")
                except Exception as e:
                    logs.append(f"生成可视化失败: {str(e)}")
                    visualization_paths = {}

        # 保存结果并返回
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        task_info['duration'] = f"{duration:.2f}秒"
        task_info['end_time'] = end_time.strftime('%Y-%m-%d %H:%M:%S')

        save_task_results(task_id, task_info, logs, results, visualization_paths)

        return jsonify({
            'success': True,
            'task_id': task_id,
            'taskInfo': task_info,
            'logs': logs,
            'results': results,
            'visualizations': visualization_paths
        })

    except Exception as e:
        error_msg = f"任务执行失败: {str(e)}"
        logs.append(error_msg)
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        save_task_results(task_id, {'task_id': task_id, 'status': 'failed'}, logs, {}, {})
        return jsonify({'success': False, 'task_id': task_id, 'logs': logs, 'error': str(e)})

# -------------------------
# 自实现评估函数（不依赖 sklearn）
# - 分类: accuracy, precision (weighted), recall (weighted), f1 (weighted), roc_auc (binary or OVR approx)
# - 回归: r2, mse, rmse, mae
# -------------------------
def accuracy_score_custom(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return float((y_true == y_pred).sum()) / len(y_true) if len(y_true) else 0.0

def precision_recall_f1_weighted(y_true, y_pred):
    # 计算每个类的 TP, FP, FN
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    labels, counts = np.unique(y_true, return_counts=True)
    precisions = []
    recalls = []
    f1s = []
    supports = []
    for lab, sup in zip(labels, counts):
        tp = int(((y_pred == lab) & (y_true == lab)).sum())
        fp = int(((y_pred == lab) & (y_true != lab)).sum())
        fn = int(((y_pred != lab) & (y_true == lab)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        supports.append(sup)
    total = np.sum(supports) if len(supports) else 1
    weighted_precision = float(np.sum(np.array(precisions) * np.array(supports)) / total)
    weighted_recall = float(np.sum(np.array(recalls) * np.array(supports)) / total)
    weighted_f1 = float(np.sum(np.array(f1s) * np.array(supports)) / total)
    return weighted_precision, weighted_recall, weighted_f1

def roc_auc_binary_from_proba(y_true, y_score):
    # y_score: probability of positive class
    # Use Mann-Whitney U / numeric ranking to compute AUC
    y_true = np.array(y_true)
    y_score = np.array(y_score)
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.0
    # compute pairwise comparisons
    count = 0.0
    for p in pos:
        count += np.sum(p > neg) + 0.5 * np.sum(p == neg)
    auc = count / (len(pos) * len(neg))
    return float(auc)

def normalize_visualization_paths(paths):
    """规范化可视化路径，确保它们以正确的格式返回给前端"""
    normalized = {}
    if isinstance(paths, dict):
        for k, v in paths.items():
            if isinstance(v, str):
                if not v.startswith('/static/'):
                    if v.startswith('static/'):
                        normalized[k] = '/' + v
                    else:
                        normalized[k] = '/static/' + v.lstrip('/')
                else:
                    normalized[k] = v
        return normalized
    return {}

def evaluate_model(y_test, y_pred, y_pred_proba, metrics, task_type):
    results = {}
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)

    if task_type == 'classification':
        if 'accuracy' in metrics:
            results['accuracy'] = accuracy_score_custom(y_test, y_pred)
        if any(m in metrics for m in ['precision', 'recall', 'f1_score']):
            p, r, f1 = precision_recall_f1_weighted(y_test, y_pred)
            if 'precision' in metrics:
                results['precision'] = p
            if 'recall' in metrics:
                results['recall'] = r
            if 'f1_score' in metrics:
                results['f1_score'] = f1
        if 'roc_auc' in metrics:
            if y_pred_proba is not None:
                y_pred_proba = np.array(y_pred_proba)
                # 二分类
                if y_pred_proba.ndim == 2 and y_pred_proba.shape[1] == 2:
                    # 假设正类为列索引 1
                    auc = roc_auc_binary_from_proba(y_test, y_pred_proba[:, 1])
                    results['roc_auc'] = auc
                else:
                    # 多类情况：对每个类做 OVR，然后平均（简单实现）
                    try:
                        labels = np.unique(y_test)
                        aucs = []
                        for i, lab in enumerate(labels):
                            # scores for class i: column i
                            if y_pred_proba.shape[1] == len(labels):
                                scores = y_pred_proba[:, i]
                                binary_true = (y_test == lab).astype(int)
                                aucs.append(roc_auc_binary_from_proba(binary_true, scores))
                        results['roc_auc'] = float(np.mean(aucs)) if aucs else 0.0
                    except Exception:
                        results['roc_auc'] = 0.0
            else:
                results['roc_auc'] = 0.0

    else:
        # 回归指标
        y_test_f = y_test.astype(float)
        y_pred_f = y_pred.astype(float)
        if 'r2' in metrics:
            ss_res = np.sum((y_test_f - y_pred_f) ** 2)
            ss_tot = np.sum((y_test_f - np.mean(y_test_f)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0
            results['r2'] = float(r2)
        if 'mse' in metrics:
            results['mse'] = float(np.mean((y_test_f - y_pred_f) ** 2))
        if 'rmse' in metrics:
            results['rmse'] = float(np.sqrt(np.mean((y_test_f - y_pred_f) ** 2)))
        if 'mae' in metrics:
            results['mae'] = float(np.mean(np.abs(y_test_f - y_pred_f)))

    return results

# -------------------------
# 可视化生成与保存
# -------------------------
def save_task_results(task_id, task_info, logs, results, visualizations):
    try:
        task_dir = os.path.join('tasks', task_id)
        os.makedirs(task_dir, exist_ok=True)
        with open(os.path.join(task_dir, 'task_info.json'), 'w', encoding='utf-8') as f:
            json.dump(task_info, f, ensure_ascii=False, indent=2)
        with open(os.path.join(task_dir, 'logs.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(logs))
        with open(os.path.join(task_dir, 'results.json'), 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        if isinstance(visualizations, dict):
            with open(os.path.join(task_dir, 'visualizations.json'), 'w', encoding='utf-8') as f:
                json.dump(visualizations, f, ensure_ascii=False, indent=2)
        logger.info(f"任务 {task_id} 结果已保存")
    except Exception as e:
        logger.error(f"保存任务结果失败: {str(e)}")

@app.route('/results/<task_id>')
def show_results(task_id):
    try:
        task_dir = os.path.join('tasks', task_id)
        with open(os.path.join(task_dir, 'task_info.json'), 'r', encoding='utf-8') as f:
            task_info = json.load(f)
        with open(os.path.join(task_dir, 'logs.txt'), 'r', encoding='utf-8') as f:
            logs = f.readlines()
        with open(os.path.join(task_dir, 'results.json'), 'r', encoding='utf-8') as f:
            results = json.load(f)
        visualizations = {}
        # 优先读取保存的可视化映射，避免目录历史图片重复展示
        try:
            with open(os.path.join(task_dir, 'visualizations.json'), 'r', encoding='utf-8') as f:
                saved_viz = json.load(f)
                if isinstance(saved_viz, dict):
                    visualizations.update(saved_viz)
        except Exception:
            # 回退到目录扫描（仅白名单文件，避免历史或第三方生成的重复图片）
            whitelist = {'feature_importance', 'truth_vs_pred', 'prediction_confidence', 'roc_curve'}
            viz_dir = os.path.join('static', 'images', task_id)
            if os.path.exists(viz_dir):
                for filename in os.listdir(viz_dir):
                    name_no_ext = os.path.splitext(filename)[0]
                    if name_no_ext in whitelist and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        visualizations[name_no_ext] = '/static/' + os.path.join('images', task_id, filename).replace('\\', '/')
        return render_template('results.html', task_id=task_id, task_info=task_info, logs=logs, results=results, visualizations=visualizations)
    except Exception as e:
        logger.error(f"加载任务结果失败: {str(e)}")
        return render_template('error.html', message=f"无法加载任务 {task_id} 的结果: {str(e)}")

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    logger.error(f"服务器内部错误: {str(e)}\n{traceback.format_exc()}")
    return render_template('500.html'), 500

@app.errorhandler(403)
def forbidden(e):
    return render_template('403.html'), 403

if __name__ == '__main__':
    debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(
        debug=debug_mode,
        host=os.getenv('FLASK_HOST', '127.0.0.1'),
        port=int(os.getenv('FLASK_PORT', 5001))
    )
