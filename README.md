## MLTaskConfig

一个基于 Flask 的机器学习教学/实践项目：支持多数据集选择、训练/验证/可视化一体化，包含多种经典算法的纯 Numpy/手写实现与可视化模块。

### 功能特性
- 数据集在线加载与预处理（含标准化、缺失处理、类别编码）
- 多算法可选，一键训练/预测/评估
- 自动生成评估可视化图（分类/回归/聚类）
- 任务结果持久化（logs、results、visualizations）
- UI 端交互（`templates/index.html`）

---

## 环境要求

- Python 3.9+
- 操作系统：Windows/Linux/macOS

安装依赖：
```bash
pip install -r requirements.txt
```

注意：
- 若网络受限，个别数据集可能需手动下载（代码会提示路径与来源）。
- 如遇中文字体问题，可在 `app.py` 中调整 `matplotlib` 字体配置。

---

## 启动项目

```bash
# 进入项目目录
cd MLTaskConfig

# 安装依赖
pip install -r requirements.txt

# 启动服务
python app.py
# 默认 http://127.0.0.1:5001
```

浏览器打开后：
1) 选择数据集、划分方式、算法与指标
2) 点击“开始任务”
3) 查看训练日志、指标与可视化结果

---

## 已支持数据集

- 批发客户（`wholesale_customers`，适合聚类）
- 鸢尾花（`iris`）
- 泰坦尼克号（`titanic`）
- 葡萄酒（`wine`）
- 乳腺癌（`breast_cancer`）
- 成人收入（`adult`）
- 信用卡欺诈（`credit_card`，需手动下载到 data 目录）
- 糖尿病（`diabetes`）
- 混凝土强度（`concrete`）
- 共享单车（`bike_sharing`）
- 加州房价（`california_housing`）

数据加载与预处理逻辑见 `data_loader.py`。

---

## 已实现算法（mymodels）

- `baseline`：基线（分类取众数，回归取均值）
- `decision_tree`：决策树（CART，分类/回归，预剪枝）
- `random_forest`：随机森林（Bagging+特征子采样）
- `gradient_boosting`：GBDT（手写，分类/回归）
- `knn`：K近邻（均值/投票，支持距离加权）
- `logistic_regression`：逻辑回归（梯度下降，二分类）
- `mlp`：多层感知机（ReLU+He/Xavier 初始化）
- `naive_bayes`：朴素贝叶斯（高斯/多项式自适配）
- `svm`：支持向量机（线性核，分类/回归）
- `kmeans`：K-Means 聚类（k-means++/random 初始化）

接口规范（统一）：
- `train(X, y)`：训练模型（聚类忽略 y）
- `predict(X)`：预测
- `predict_proba(X)`（可选）：分类概率或“拟概率”（K-Means 为相似度归一化）

---

## 可视化（myvisualizations）

通用（因模型而异）：
- 数据分布图、特征相关性热力图
- 混淆矩阵、ROC、PR、校准曲线（分类）
- 残差图、真实值 vs 预测值（回归）
- 学习曲线

模型特有示例：
- 决策树：树结构图、特征重要性
- 逻辑回归：特征权重、决策边界
- SVM：决策边界、支持向量
- K-Means：PCA 二维聚类散点、簇大小、列联表（真实标签 vs 簇）

图片输出到 `static/images/{task_id}/...`，并记录在 `tasks/{task_id}/visualizations.json`。

---

## 评估指标

- 分类：accuracy、precision（加权）、recall（加权）、f1（加权）、roc_auc（自实现）
- 回归：R²、MSE、RMSE、MAE

说明：
- 聚类（K-Means）用于分类数据时，会自动将簇数设为训练集类别数，并在评估前用“簇内多数投票”对齐簇标签与真实标签，以获得可比较的分类指标。若需 NMI/ARI/轮廓系数等无监督指标，可扩展 `app.py:evaluate_model`。

---

## 目录结构

- `app.py`：Flask 后端、任务流控制、模型/可视化动态加载、评估与结果落盘
- `data_loader.py`：数据集加载、预处理（缺失/编码/缩放）、缓存/下载
- `mymodels/`：各算法实现
- `myvisualizations/`：各算法可视化
- `templates/`：前端页面（`index.html`、`results.html` 等）
- `static/`：样式、图片输出目录
- `tasks/{task_id}/`：任务日志、指标、可视化路径映射

---

## 常见问题

- 字体乱码：修改 `app.py` 中 `matplotlib` 字体列表（如添加本机可用中文字体）。
- 信用卡欺诈数据集无法下载：该数据需登录 Kaggle，请按日志提示将 `creditcard.csv` 放到 `data/creditcard.csv`。
- 评估与聚类标签不一致：本项目已在评估前对齐 K-Means 簇标签，若自定义流程，注意做同样对齐或使用无监督评估指标。

---

## 扩展与二次开发

- 新增算法：在 `mymodels/{model_id}.py` 实现 `CustomModel` 并在 `app.py:MODELS` 注册
- 新增可视化：在 `myvisualizations/{model_id}_viz.py` 实现 `CustomVisualization`
- 新增数据集：在 `data_loader.py` 中添加 `_load_xxx` 并注册到 `dataset_info`、`dataset_urls` 和路由

---

## License
For educational purposes. Check datasets’ original licenses before redistribution.
