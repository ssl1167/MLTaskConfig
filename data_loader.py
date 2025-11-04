import numpy as np
import pandas as pd
import os
import logging
import requests
from sklearn import datasets
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import hashlib
from datetime import datetime
import zipfile
from io import BytesIO

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataLoader:
    """数据集加载器类，用于加载和预处理各种机器学习数据集"""
    
    def __init__(self, data_dir='data', random_state=42):
        """初始化数据加载器
        
        Args:
            data_dir (str): 数据集存储路径
            random_state (int): 随机种子，确保结果可复现
        """
        self.data_dir = os.path.abspath(data_dir)
        self.random_state = random_state
        np.random.seed(random_state)
        
        # 确保数据目录存在
        os.makedirs(self.data_dir, exist_ok=True)
        
        # 数据集元信息：名称、类型(分类/回归)、来源
        self.dataset_info = {
            # 分类任务 - 简单
            'watermelon': {'type': 'classification', 'description': '西瓜数据集，用于演示 (简单)'},
            'iris': {'type': 'classification', 'description': '鸢尾花数据集，3类别分类 (简单)'},
            'titanic': {'type': 'classification', 'description': '泰坦尼克号生存预测，二分类 (简单-中等)'},
            
            # 分类任务 - 中等
            'wine': {'type': 'classification', 'description': '葡萄酒数据集，3类别分类 (中等)'},
            'breast_cancer': {'type': 'classification', 'description': '乳腺癌数据集，二分类 (中等)'},
            'adult': {'type': 'classification', 'description': '收入预测，二分类，大样本 (中等-复杂)'},
            
            # 分类任务 - 复杂
            'credit_card': {'type': 'classification', 'description': '信用卡欺诈检测，极度不平衡 (复杂)'},
            # 聚类友好（含可用监督标签 Channel）
            'wholesale_customers': {'type': 'classification', 'description': 'UCI 批发客户数据集 (适合聚类)'},
            
            # 回归任务 - 简单
            'diabetes': {'type': 'regression', 'description': '糖尿病进展预测 (简单)'},
            'concrete': {'type': 'regression', 'description': '混凝土强度预测，物理数据 (简单)'},
            
            # 回归任务 - 中等
            'bike_sharing': {'type': 'regression', 'description': '共享单车需求预测，时间序列 (中等)'},
            'california_housing': {'type': 'regression', 'description': '加州房价预测，大样本 (中等-复杂)'}
        }
        
        # 预处理参数配置
        self.preprocessing_config = {
            'scaler': 'standard',  # 'standard' 或 'minmax'
            'handle_missing': 'mean',  # 'mean', 'median' 或 'drop'
            'encode_categorical': True  # 是否编码分类特征
        }
        
        # 数据集下载链接
        self.dataset_urls = {
            'titanic': 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv',
            'credit_card': 'https://www.kaggle.com/mlg-ulb/creditcardfraud/download',
            'watermelon': 'https://raw.githubusercontent.com/terminusle/ML-Datasets/master/watermelon3_0_En.csv',
            'concrete': 'https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls',
            'bike_sharing': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00560/SeoulBikeData.csv',
            'adult': 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
            , 'wholesale_customers': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00292/Wholesale%20customers%20data.csv'
        }

    def list_datasets(self):
        """列出所有支持的数据集"""
        return {k: v['description'] for k, v in self.dataset_info.items()}
    
    def load_dataset(self, dataset_id, return_type='array', preprocess=True, test_size=0.2):
        """根据数据集ID加载相应的数据集
        
        Args:
            dataset_id (str): 数据集的标识符
            return_type (str): 返回数据类型，'array' 为numpy数组，'dataframe' 为pandas数据框
            preprocess (bool): 是否进行预处理
            test_size (float): 测试集比例，0到1之间
            
        Returns:
            tuple: 包含以下元素的元组
                - X_train: 训练特征
                - X_test: 测试特征
                - y_train: 训练目标
                - y_test: 测试目标
                - feature_names: 特征名称
                - target_names: 目标变量名称
                - dataset_type: 数据集类型（classification/regression）
        """
        if dataset_id not in self.dataset_info:
            raise ValueError(f"未知的数据集ID: {dataset_id}。支持的数据集: {list(self.dataset_info.keys())}")
        
        # 加载原始数据
        try:
            X, y, feature_names, target_names = self._load_raw_data(dataset_id)
            dataset_type = self.dataset_info[dataset_id]['type']
            logger.info(f"成功加载 {dataset_id} 数据集，样本数: {X.shape[0]}, 特征数: {X.shape[1]}")
        except Exception as e:
            logger.error(f"加载 {dataset_id} 数据集失败: {str(e)}")
            raise
        
        # 分割训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state,
            stratify=y if dataset_type == 'classification' else None
        )
        
        # 数据预处理
        preprocessor = None
        if preprocess:
            X_train, X_test, preprocessor = self.preprocess_data(
                X_train, X_test, y_train, dataset_type
            )
        
        # 转换返回类型
        if return_type == 'dataframe' and isinstance(X_train, np.ndarray):
            X_train = pd.DataFrame(X_train, columns=feature_names)
            X_test = pd.DataFrame(X_test, columns=feature_names)
        
        return (X_train, X_test, y_train, y_test, feature_names, 
                target_names, dataset_type, preprocessor)
    
    def _load_raw_data(self, dataset_id):
        """加载原始数据（未预处理）"""
        load_methods = {
            'watermelon': self._load_watermelon,
            'iris': self._load_iris,
            'wine': self._load_wine,
            'breast_cancer': self._load_breast_cancer,
            'diabetes': self._load_diabetes,
            'california_housing': self._load_california_housing,
            'credit_card': self._load_credit_card,
            'titanic': self._load_titanic,
            'concrete': self._load_concrete,
            'bike_sharing': self._load_bike_sharing,
            'adult': self._load_adult,
            'wholesale_customers': self._load_wholesale_customers
        }
        
        return load_methods[dataset_id]()
    
    def _get_cache_path(self, dataset_id):
        """获取数据集缓存路径"""
        return os.path.join(self.data_dir, f"{dataset_id}.csv")
    
    def _download_file(self, url, dataset_id):
        """下载文件并缓存到本地"""
        cache_path = self._get_cache_path(dataset_id)
        
        # 检查缓存是否存在
        if os.path.exists(cache_path):
            logger.info(f"使用缓存的 {dataset_id} 数据集: {cache_path}")
            return cache_path
        
        # 特殊处理需要登录的Kaggle数据集
        if 'kaggle.com' in url:
            logger.warning(f"{dataset_id} 数据集需要从Kaggle下载，请手动下载并保存到 {cache_path}")
            logger.warning(f"下载地址: {url}")
            raise RuntimeError(f"需要手动下载 {dataset_id} 数据集")
        
        # 下载文件
        logger.info(f"下载 {dataset_id} 数据集: {url}")
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()  # 检查HTTP错误
            
            # 保存到本地
            with open(cache_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"数据集已保存到: {cache_path}")
            return cache_path
        except Exception as e:
            logger.error(f"下载 {dataset_id} 数据集失败: {str(e)}")
            if os.path.exists(cache_path):
                os.remove(cache_path)  # 清理不完整文件
            raise
    
    def _load_iris(self):
        """加载鸢尾花数据集"""
        iris = datasets.load_iris()
        X = iris.data
        y = iris.target
        feature_names = iris.feature_names
        target_names = list(iris.target_names)
        
        return X, y, feature_names, target_names
    
    def _load_wine(self):
        """加载葡萄酒数据集"""
        wine = datasets.load_wine()
        X = wine.data
        y = wine.target
        feature_names = wine.feature_names
        target_names = [f'类别{i}' for i in range(len(wine.target_names))]
        
        return X, y, feature_names, target_names
    
    def _load_breast_cancer(self):
        """加载乳腺癌数据集"""
        breast_cancer = datasets.load_breast_cancer()
        X = breast_cancer.data
        y = breast_cancer.target
        feature_names = breast_cancer.feature_names
        target_names = ['恶性', '良性']
        
        return X, y, feature_names, target_names
    
    def _load_diabetes(self):
        """加载糖尿病数据集"""
        diabetes = datasets.load_diabetes()
        X = diabetes.data
        y = diabetes.target
        feature_names = diabetes.feature_names
        target_names = ['病情指数']
        
        return X, y, feature_names, target_names
    
    def _load_california_housing(self):
        """加载加州房价数据集"""
        from sklearn.datasets import fetch_california_housing
        housing = fetch_california_housing()
        X = housing.data
        y = housing.target
        feature_names = housing.feature_names
        target_names = ['房价中位数']
        
        return X, y, feature_names, target_names
    
    def _load_credit_card(self):
        """加载信用卡欺诈检测数据集"""
        cache_path = self._get_cache_path('creditcard')
        
        # 检查是否已缓存
        if not os.path.exists(cache_path):
            # 提示用户手动下载，因为Kaggle需要登录
            raise RuntimeError(
                "信用卡欺诈数据集需要从Kaggle下载。\n"
                "请访问: https://www.kaggle.com/mlg-ulb/creditcardfraud\n"
                f"下载后将creditcard.csv文件保存到: {cache_path}"
            )
        
        # 加载缓存的CSV文件
        try:
            df = pd.read_csv(cache_path)
            # 特征列 V1-V28, Amount, Time
            feature_names = df.columns.drop('Class').tolist()
            X = df[feature_names].values
            y = df['Class'].values
            target_names = ['正常', '欺诈']
            
            return X, y, feature_names, target_names
        except Exception as e:
            logger.error(f"加载信用卡数据集失败: {str(e)}")
            raise
    
    def _load_titanic(self):
        """加载泰坦尼克号数据集"""
        # 下载或使用缓存
        cache_path = self._download_file(self.dataset_urls['titanic'], 'titanic')
        
        # 加载并处理数据
        try:
            df = pd.read_csv(cache_path)
            
            # 选择相关特征
            features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
            target = 'Survived'
            
            # 处理缺失值
            df = df[features + [target]].dropna(subset=['Embarked'])
            
            # 特征名称映射为中文
            feature_name_mapping = {
                'Pclass': '客舱等级',
                'Sex': '性别',
                'Age': '年龄',
                'SibSp': '兄弟姐妹数量',
                'Parch': '父母子女数量',
                'Fare': '票价',
                'Embarked': '登船港口'
            }
            feature_names = [feature_name_mapping[f] for f in features]
            
            X = df[features].values
            y = df[target].values
            target_names = ['遇难', '幸存']
            
            return X, y, feature_names, target_names
        except Exception as e:
            logger.error(f"加载泰坦尼克号数据集失败: {str(e)}")
            raise
    
    def _load_watermelon(self):
        """加载西瓜数据集"""
        # 下载或使用缓存
        cache_path = self._download_file(self.dataset_urls['watermelon'], 'watermelon')
        
        try:
            df = pd.read_csv(cache_path)
            # 选择特征和目标
            features = df.columns[1:-1].tolist()
            X = df[features].values
            y = df['label'].values
            
            # 转换目标变量为0和1
            y = np.where(y == 'good', 1, 0)
            
            target_names = ['坏瓜', '好瓜']
            return X, y, features, target_names
        except Exception as e:
            logger.error(f"加载西瓜数据集失败: {str(e)}")
            raise
    
    def _load_digits(self):
        """加载手写数字识别数据集（sklearn内置）"""
        try:
            from sklearn.datasets import load_digits
            digits = load_digits()
            X = digits.data  # 8x8图像展平为64维向量
            y = digits.target  # 0-9的数字标签
            feature_names = [f'pixel_{i}' for i in range(64)]
            target_names = [str(i) for i in range(10)]
            
            logger.info(f"成功加载 digits 数据集，样本数: {X.shape[0]}, 特征数: {X.shape[1]}")
            return X, y, feature_names, target_names
        except Exception as e:
            logger.error(f"加载手写数字数据集失败: {str(e)}")
            raise
    
    def _load_concrete(self):
        """加载混凝土强度数据集（UCI）"""
        # 先尝试查找.xls文件，再查找.csv文件
        xls_path = os.path.join(self.data_dir, 'concrete.xls')
        csv_path = self._get_cache_path('concrete')
        
        df = None
        
        # 优先尝试读取XLS文件
        if os.path.exists(xls_path):
            try:
                df = pd.read_excel(xls_path)
                logger.info(f"从Excel文件读取混凝土数据集: {xls_path}")
            except Exception as e:
                logger.warning(f"读取XLS文件失败: {str(e)}")
        
        # 如果XLS读取失败，尝试CSV
        if df is None and os.path.exists(csv_path):
            # 尝试多种编码读取CSV
            encodings = ['utf-8', 'gbk', 'gb2312', 'latin1', 'iso-8859-1']
            for enc in encodings:
                try:
                    df = pd.read_csv(csv_path, encoding=enc)
                    logger.info(f"使用编码 {enc} 成功读取混凝土数据集")
                    break
                except (UnicodeDecodeError, Exception):
                    continue
        
        # 如果都不存在，尝试下载
        if df is None:
            logger.warning(
                f"混凝土强度数据集未找到，尝试自动下载...\n"
                f"下载地址: {self.dataset_urls['concrete']}"
            )
            try:
                import urllib.request
                urllib.request.urlretrieve(self.dataset_urls['concrete'], xls_path)
                df = pd.read_excel(xls_path)
                logger.info(f"成功下载混凝土数据集到: {xls_path}")
            except Exception as e:
                logger.error(f"自动下载失败: {str(e)}")
                raise FileNotFoundError(
                    f"请手动下载数据集\n"
                    f"下载地址: {self.dataset_urls['concrete']}\n"
                    f"保存到: {xls_path} 或 {csv_path}"
                )
        
        if df is None:
            raise ValueError("无法读取混凝土数据集")
        
        try:
            # 混凝土数据集标准列名（英文）
            expected_columns = [
                'Cement (component 1)(kg in a m^3 mixture)',
                'Blast Furnace Slag (component 2)(kg in a m^3 mixture)',
                'Fly Ash (component 3)(kg in a m^3 mixture)',
                'Water  (component 4)(kg in a m^3 mixture)',
                'Superplasticizer (component 5)(kg in a m^3 mixture)',
                'Coarse Aggregate  (component 6)(kg in a m^3 mixture)',
                'Fine Aggregate (component 7)(kg in a m^3 mixture)',
                'Age (day)',
                'Concrete compressive strength(MPa, megapascals) '
            ]
            
            # 中文列名
            feature_names = ['水泥', '高炉矿渣', '粉煤灰', '水', '超塑化剂', '粗骨料', '细骨料', '龄期']
            target_name = '抗压强度'
            
            # 检查是否需要重命名列
            if len(df.columns) == 9:
                # 使用位置索引获取数据，避免列名问题
                X = df.iloc[:, :8].values
                y = df.iloc[:, 8].values
                target_names = [target_name]
                
                logger.info(f"成功加载 concrete 数据集，样本数: {X.shape[0]}, 特征数: {X.shape[1]}")
                return X, y, feature_names, target_names
            else:
                raise ValueError(f"数据集列数不正确，期望9列，实际{len(df.columns)}列")
                
        except Exception as e:
            logger.error(f"处理混凝土数据集失败: {str(e)}")
            raise
    
    def _load_bike_sharing(self):
        """加载共享单车需求预测数据集（UCI）"""
        cache_path = self._get_cache_path('bike_sharing')
        
        # 检查缓存
        if not os.path.exists(cache_path):
            logger.warning(
                f"共享单车数据集未找到，请手动下载\n"
                f"下载地址: {self.dataset_urls['bike_sharing']}\n"
                f"保存到: {cache_path}\n"
                f"或者我会尝试自动下载..."
            )
            try:
                # 尝试下载
                import urllib.request
                urllib.request.urlretrieve(self.dataset_urls['bike_sharing'], cache_path)
                logger.info(f"成功下载共享单车数据集到: {cache_path}")
            except Exception as e:
                logger.error(f"自动下载失败: {str(e)}")
                raise FileNotFoundError(f"请手动下载数据集到: {cache_path}")
        
        try:
            # 尝试多种编码（韩文数据集）
            encodings = ['cp949', 'euc-kr', 'utf-8', 'latin1']
            df = None
            for enc in encodings:
                try:
                    df = pd.read_csv(cache_path, encoding=enc)
                    logger.info(f"使用编码 {enc} 成功读取共享单车数据集")
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                raise ValueError("无法用任何编码读取数据集")
            
            # 选择相关特征（排除日期列和目标列）
            # 列结构：Date(0), Rented Bike Count(1-目标), Hour(2), ..., Functioning Day(13)
            # 目标：第2列（索引1）"Rented Bike Count"
            # 特征：第3列到最后（索引2-13）
            target_col = df.columns[1]  # Rented Bike Count - 租赁数量
            feature_cols = df.columns[2:].tolist()  # Hour到Functioning Day的所有列
            
            # 提取特征和目标
            X_df = df[feature_cols].copy()
            y_series = df[target_col]
            
            # 处理分类特征（转换Yes/No等字符串为数值）
            for col in X_df.columns:
                if X_df[col].dtype == 'object' or X_df[col].dtype.name == 'category':
                    # 字符串类型，需要编码
                    unique_vals = X_df[col].unique()
                    logger.info(f"编码列 {col}，唯一值: {unique_vals}")
                    
                    # 创建映射字典
                    value_map = {val: i for i, val in enumerate(sorted(unique_vals, key=str))}
                    X_df[col] = X_df[col].map(value_map)
            
            # 转换为数值数组
            X = X_df.values.astype(float)
            y = pd.to_numeric(y_series, errors='coerce').values  # 将目标变量转换为数值
            
            # 重命名为中文（根据实际列数调整）
            feature_names = ['小时', '温度', '湿度', '风速', '能见度', '露点温度', '太阳辐射', 
                           '降雨量', '降雪量', '季节', '假期', '是否工作日']
            target_names = ['租赁数量']
            
            logger.info(f"成功加载 bike_sharing 数据集，样本数: {X.shape[0]}, 特征数: {X.shape[1]}")
            return X, y, feature_names[:X.shape[1]], target_names
        except Exception as e:
            logger.error(f"加载共享单车数据集失败: {str(e)}")
            raise
    
    def _load_adult(self):
        """加载成人收入预测数据集（UCI）"""
        cache_path = self._get_cache_path('adult')
        
        # 检查缓存
        if not os.path.exists(cache_path):
            logger.warning(
                f"成人收入数据集未找到，请手动下载\n"
                f"下载地址: {self.dataset_urls['adult']}\n"
                f"保存为CSV格式到: {cache_path}\n"
                f"或者我会尝试自动下载..."
            )
            try:
                # 尝试下载
                import urllib.request
                urllib.request.urlretrieve(self.dataset_urls['adult'], cache_path)
                logger.info(f"成功下载成人收入数据集到: {cache_path}")
            except Exception as e:
                logger.error(f"自动下载失败: {str(e)}")
                raise FileNotFoundError(f"请手动下载数据集到: {cache_path}")
        
        try:
            # Adult数据集没有列名，需要手动指定
            column_names = ['年龄', '工作类型', '权重', '教育程度', '受教育年限', '婚姻状况',
                          '职业', '家庭关系', '种族', '性别', '资本收益', '资本损失',
                          '每周工作小时', '国籍', '收入']
            
            df = pd.read_csv(cache_path, names=column_names, skipinitialspace=True)
            
            # 去除缺失值（标记为'?'）
            df = df.replace(' ?', np.nan).dropna()
            
            # 选择特征（排除权重列）
            feature_names = [col for col in column_names if col not in ['权重', '收入']]
            X = df[feature_names].values
            
            # 目标变量：>50K为1，<=50K为0
            y = (df['收入'].str.strip() == '>50K').astype(int).values
            target_names = ['<=50K', '>50K']
            
            logger.info(f"成功加载 adult 数据集，样本数: {X.shape[0]}, 特征数: {X.shape[1]}")
            return X, y, feature_names, target_names
        except Exception as e:
            logger.error(f"加载成人收入数据集失败: {str(e)}")
            raise
    
    def set_preprocessing_config(self, **kwargs):
        """设置预处理配置
        
        Args:
            scaler (str): 缩放方法，'standard' 或 'minmax'
            handle_missing (str): 缺失值处理方法，'mean', 'median' 或 'drop'
            encode_categorical (bool): 是否编码分类特征
        """
        for key, value in kwargs.items():
            if key in self.preprocessing_config:
                self.preprocessing_config[key] = value
            else:
                logger.warning(f"未知的预处理参数: {key}")
    
    def preprocess_data(self, X_train, X_test, y_train=None, task_type='classification'):
        """数据预处理函数，支持训练集和测试集的正确处理
        
        Args:
            X_train (numpy.ndarray): 训练特征数据
            X_test (numpy.ndarray): 测试特征数据
            y_train (numpy.ndarray): 训练目标变量，用于某些预处理步骤
            task_type (str): 任务类型，'classification' 或 'regression'
        
        Returns:
            tuple: 预处理后的训练特征、测试特征和预处理对象
        """
        # 确保输入是numpy数组（保持原有接口），允许object与数值混合
        if not isinstance(X_train, np.ndarray):
            X_train = np.array(X_train)
        if not isinstance(X_test, np.ndarray):
            X_test = np.array(X_test)
        
        # 存储预处理对象，便于后续复用
        preprocessor = {
            'scaler': None,
            'label_encoders': {},
            'missing_values': {}
        }
        
        # 1. 处理缺失值（对数值与非数值分别处理）
        X_train, X_test = self._handle_missing_values(X_train, X_test, preprocessor)
        
        # 2. 编码分类特征（仅对分类特征）
        if self.preprocessing_config['encode_categorical']:
            X_train, X_test = self._encode_categorical_features(
                X_train, X_test, preprocessor, task_type
            )
        
        # 3. 特征缩放
        X_train, X_test = self._scale_features(X_train, X_test, preprocessor)
        
        return X_train, X_test, preprocessor
    
    def _handle_missing_values(self, X_train, X_test, preprocessor):
        """处理缺失值"""
        method = self.preprocessing_config['handle_missing']
        
        # 对每一列处理缺失值
        for col in range(X_train.shape[1]):
            train_col = X_train[:, col]
            test_col = X_test[:, col]
            
            # 检查是否有缺失值
            has_missing_train = pd.isna(train_col).any()
            has_missing_test = pd.isna(test_col).any()
            
            if not has_missing_train and not has_missing_test:
                continue
                
            # 记录缺失值处理方式
            if method == 'drop':
                # 这种方法可能会减少样本数量，谨慎使用
                if has_missing_train:
                    non_missing_idx = ~pd.isna(train_col)
                    X_train = X_train[non_missing_idx]
                    # 如果提供了y_train，也需要相应调整
                    # 注意：这里简化处理，实际应用中需要更复杂的逻辑
                if has_missing_test:
                    non_missing_idx = ~pd.isna(test_col)
                    X_test = X_test[non_missing_idx]
                    
            else:
                # 计算填充值（使用训练集的统计量）
                is_numeric = np.issubdtype(np.array(train_col).dtype, np.number)
                if is_numeric:
                    if method == 'mean':
                        fill_value = np.nanmean(train_col.astype(float))
                    elif method == 'median':
                        fill_value = np.nanmedian(train_col.astype(float))
                    else:
                        raise ValueError(f"不支持的缺失值处理方法: {method}")
                else:
                    # 非数值列：使用众数（mode）填充；若不存在众数则用占位字符串
                    try:
                        non_missing_vals = pd.Series(train_col).dropna()
                        fill_value = non_missing_vals.mode().iloc[0] if not non_missing_vals.empty else 'missing'
                    except Exception:
                        fill_value = 'missing'
                
                preprocessor['missing_values'][col] = {
                    'method': method,
                    'value': fill_value
                }
                
                # 填充缺失值
                if has_missing_train:
                    mask_train = pd.isna(train_col)
                    X_train[mask_train, col] = fill_value
                if has_missing_test:
                    mask_test = pd.isna(test_col)
                    X_test[mask_test, col] = fill_value
        
        return X_train, X_test
    
    def _encode_categorical_features(self, X_train, X_test, preprocessor, task_type):
        """编码分类特征"""
        # 判断哪些列需要编码：非数值列 或 数值但唯一值很少（类别型）
        n_samples = X_train.shape[0]
        num_cols = X_train.shape[1]
        for col in range(num_cols):
            train_col = X_train[:, col]
            is_numeric = np.issubdtype(np.array(train_col).dtype, np.number)

            # 统一缺失判断
            not_nan_train = ~pd.isna(train_col)

            unique_count = len(pd.unique(train_col[not_nan_train]))
            is_categorical = (not is_numeric) or (unique_count < min(10, max(2, n_samples // 20)))

            if not is_categorical:
                # 确保数值列中可能的字符串被安全转换
                continue

            # 创建并训练编码器（将值转为字符串以兼容混合类型）
            le = LabelEncoder()
            train_values = pd.Series(train_col[not_nan_train]).astype(str).values
            le.fit(train_values)
            preprocessor['label_encoders'][col] = le

            # 转换训练集
            encoded_train = np.full(shape=train_col.shape, fill_value=np.nan, dtype=float)
            encoded_train[not_nan_train] = le.transform(train_values).astype(float)
            X_train[:, col] = encoded_train

            # 转换测试集，未知类别置为 -1
            test_col = X_test[:, col]
            not_nan_test = ~pd.isna(test_col)
            test_values = pd.Series(test_col[not_nan_test]).astype(str).values
            class_to_index = {cls: idx for idx, cls in enumerate(le.classes_)}
            mapped = np.array([class_to_index.get(v, -1) for v in test_values], dtype=float)
            encoded_test = np.full(shape=test_col.shape, fill_value=np.nan, dtype=float)
            encoded_test[not_nan_test] = mapped
            X_test[:, col] = encoded_test
        
        # 确保数据类型正确
        return X_train.astype(float), X_test.astype(float)
    
    def _scale_features(self, X_train, X_test, preprocessor):
        """特征缩放"""
        scaler_type = self.preprocessing_config['scaler']
        
        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'minmax':
            scaler = MinMaxScaler()
        else:
            logger.info(f"不使用特征缩放，方法: {scaler_type}")
            return X_train, X_test
        
        # 使用训练集拟合缩放器
        X_train_scaled = scaler.fit_transform(X_train)
        # 使用相同的缩放器转换测试集
        X_test_scaled = scaler.transform(X_test)
        
        # 保存缩放器
        preprocessor['scaler'] = scaler
        
        return X_train_scaled, X_test_scaled

    def _load_wholesale_customers(self):
        """加载UCI批发客户数据集（含 Channel 标签，适合聚类）"""
        cache_path = self._get_cache_path('wholesale_customers')
        # 下载或使用缓存
        if not os.path.exists(cache_path):
            try:
                import urllib.request
                urllib.request.urlretrieve(self.dataset_urls['wholesale_customers'], cache_path)
                logger.info(f"成功下载 wholesale_customers 数据集到: {cache_path}")
            except Exception as e:
                logger.error(f"下载批发客户数据集失败: {str(e)}")
                raise FileNotFoundError(f"请手动下载并保存到: {cache_path}\n来源: {self.dataset_urls['wholesale_customers']}")

        try:
            # 文件包含列: Channel, Region, Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen
            df = pd.read_csv(cache_path)
            # 兼容不同标题情况（有些版本含空格或不同大小写）
            cols = {c.lower().replace(' ', '_'): c for c in df.columns}
            def col(name):
                return cols.get(name, name)

            feature_cols = [
                col('fresh'), col('milk'), col('grocery'), col('frozen'),
                col('detergents_paper'), col('delicassen')
            ]
            X = df[feature_cols].values.astype(float)

            # 使用 Channel 作为监督标签(1/2) → 映射到 0/1，便于与聚类结果比较
            ch_col = col('channel')
            if ch_col in df.columns:
                y_raw = df[ch_col].values
                # 有的文件Channel从1开始
                uniq = np.unique(y_raw)
                if set(uniq) == {1, 2}:
                    y = (y_raw.astype(int) - 1).astype(int)
                else:
                    # 兜底: 排序后映射到 0..k-1
                    mapping = {val: i for i, val in enumerate(sorted(uniq))}
                    y = np.array([mapping[v] for v in y_raw], dtype=int)
            else:
                # 若无Channel列，则用Region作为标签占位
                rg_col = col('region')
                uniq = np.unique(df[rg_col].values)
                mapping = {val: i for i, val in enumerate(sorted(uniq))}
                y = np.array([mapping[v] for v in df[rg_col].values], dtype=int)

            feature_names = ['新鲜', '牛奶', '杂货', '冷冻', '纸品清洁', '熟食']
            target_names = [str(i) for i in np.unique(y)]
            logger.info(f"成功加载 wholesale_customers 数据集，样本数: {X.shape[0]}, 特征数: {X.shape[1]}")
            return X, y, feature_names[:X.shape[1]], target_names
        except Exception as e:
            logger.error(f"处理批发客户数据集失败: {str(e)}")
            raise

# 创建全局数据加载器实例，方便其他模块导入使用
data_loader = DataLoader()
