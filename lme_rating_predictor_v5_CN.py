"""
线性混合模型 (LME) Rating预测器 - 最终版 (v5.4)

核心改进:
1. Database效应：考虑所有训练database的平均效应
2. Subject ID：新被试自动使用773+编号，避免与训练数据冲突
3. Bash交互：提供完整的命令行交互界面
4. Rating统计量：硬编码原始训练数据统计量（mean=5.2503, std=2.3568）

使用场景:
- 有被试的特征历史数据（5-50个试次）
- 无需rating历史数据
- 命令行友好的交互式预测
- 自动反向标准化到原始rating尺度

预测流程说明:
1. 特征标准化: 原始特征 → Z-score特征 (使用被试内或全局统计量)
2. 模型预测: Z-score特征 → Rating Z-score (LME模型)
3. 反向标准化: Rating Z-score → 原始Rating (使用硬编码的原始统计量)

训练数据Rating统计量（MATLAB）:
- mean(all_bigdata_rating) = 5.2503  ← 原始rating均值
- std(all_bigdata_rating) = 2.3568   ← 原始rating标准差
- 注意: JSON中的rating统计量是Z-score尺度的，不用于反向标准化

作者: Yun Zhuang
日期: 2025-11
版本: v5.4
如果使用本工具发表论文，请引用：
Zhuang Y., Zhang L.B., Wang X.Q., Geng X.Y., & Hu L., (in preparation) From Normative Features to Multidimensional Estimation of Pain: A Large-Scale Study of Laser-Evoked Brain Responses.
"""

import json
import numpy as np
import pandas as pd
from scipy import stats
import warnings
import sys
import os
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from pathlib import Path

warnings.filterwarnings('ignore')


@dataclass
class SubjectProfile:
    """
    被试档案：存储特征标准化参数
    """
    subject_id: Union[int, str]
    
    # 历史特征数据
    historical_features: Dict[str, np.ndarray]
    
    # 特征标准化参数
    feature_means: Dict[str, float] = None
    feature_stds: Dict[str, float] = None
    
    def __post_init__(self):
        """计算特征标准化参数"""
        self.feature_means = {}
        self.feature_stds = {}
        
        for feat_name, feat_values in self.historical_features.items():
            self.feature_means[feat_name] = np.mean(feat_values)
            feat_std = np.std(feat_values, ddof=0)
            self.feature_stds[feat_name] = feat_std if feat_std > 0 else 1.0
    
    def standardize_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """将原始特征标准化为Z-score"""
        standardized = {}
        
        for feat_name, feat_value in features.items():
            if feat_name not in self.feature_means:
                raise ValueError(f"特征 {feat_name} 不在历史数据中")
            
            mean = self.feature_means[feat_name]
            std = self.feature_stds[feat_name]
            standardized[feat_name] = (feat_value - mean) / std
        
        return standardized
    
    def get_info(self) -> str:
        """返回被试信息摘要"""
        n_features = len(self.feature_means)
        if n_features > 0:
            feat_name = list(self.historical_features.keys())[0]
            n_trials = len(self.historical_features[feat_name])
        else:
            n_trials = 0
        
        return (f"Subject {self.subject_id}\n"
                f"  历史试次数: {n_trials}\n"
                f"  特征数: {n_features}")


class LMERatingPredictor:
    """
    线性混合模型Rating预测器 - 最终版 v5.1
    
    特点:
    - 自动处理所有训练database
    - 新被试自动分配773+编号
    - 完整的命令行交互界面
    - Rating统计量硬编码（mean=5.2503, std=2.3568）
    - 自动反向标准化到原始rating尺度
    
    重要说明:
    - JSON文件中的rating统计量是Z-score尺度的（已标准化）
    - 硬编码的统计量是原始rating尺度的（用于反向标准化）
    - 预测流程: 特征Z-score → 模型预测rating Z-score → 反向标准化到原始rating
    """
    
    def __init__(self, params_file: str, silent: bool = False):
        """
        初始化预测器
        
        参数:
            params_file: MATLAB导出的模型参数JSON文件路径
            silent: 是否静默模式（不打印初始化信息）
        """
        if not silent:
            print("="*80)
            print("LME Rating预测器 - 最终版 v5.1")
            print("="*80)
        
        # 加载模型参数
        with open(params_file, 'r', encoding='utf-8') as f:
            self.params = json.load(f)
        
        # 提取关键参数
        self.fixed_effects = self.params['fixed_effects']
        self.random_effects = self.params['random_effects']
        self.feature_names = self.params['feature_names'][0]
        
        # 重新排序特征名称为指定顺序
        desired_order = ['N1_amp', 'N2_amp', 'P2_amp', 'N1_lat', 'N2_lat', 'P2_lat', 
                        'ERP_mag', 'Alpha_mag', 'Beta_mag', 'Gamma_mag']
        # 验证所有特征都存在
        if set(self.feature_names) == set(desired_order):
            self.feature_names = desired_order
        else:
            if not silent:
                print("⚠️  警告: 特征名称与预期不匹配，保持原顺序")
        
        self.data_stats = self.params.get('data_stats', {})
        self.model_info = self.params['model_info']
        
        # 提取训练数据信息
        self.training_info = self._extract_training_info()
        
        # 提取全局标准化参数
        self._extract_global_stats(silent)
        
        # 提取固定效应系数
        self._extract_fixed_coefficients(silent)
        
        # 存储被试档案
        self.subject_profiles: Dict[str, SubjectProfile] = {}
        
        # 新被试计数器（从773开始）
        self.next_new_subject_id = 773
        
        if not silent:
            print("\n✓ 预测器初始化完成")
            print(f"  训练Database数: {self.training_info['num_databases']}")
            print(f"  训练Subject数: {self.training_info['num_subjects']}")
            print(f"  新被试ID起始: {self.next_new_subject_id}")
            print("="*80)
    
    def _extract_training_info(self) -> Dict:
        """提取训练数据信息"""
        info = {
            'num_databases': self.model_info.get('num_groups_database', 10),
            'num_subjects': self.model_info.get('num_groups_subject', 772),
            'num_observations': self.model_info.get('num_observations', 0)
        }
        
        # 从grouping_info提取详细信息（如果有）
        if 'grouping_info' in self.params:
            grouping = self.params['grouping_info']
            info['database_ids'] = grouping.get('unique_databases', [])
            info['subject_ids'] = grouping.get('unique_subjects', [])
        
        return info
    
    def _extract_global_stats(self, silent: bool = False):
        """提取全局标准化参数"""
        # 特征的全局参数
        self.global_feature_means = {}
        self.global_feature_stds = {}
        
        for feat in self.feature_names:
            if feat in self.data_stats:
                self.global_feature_means[feat] = self.data_stats[feat].get('mean', 0.0)
                self.global_feature_stds[feat] = self.data_stats[feat].get('std', 1.0)
            else:
                if not silent:
                    print(f"⚠️  特征 {feat} 缺少全局统计量，使用默认值 0±1")
                self.global_feature_means[feat] = 0.0
                self.global_feature_stds[feat] = 1.0
        
        # Rating的反向标准化参数（硬编码原始rating统计量）
        # 训练数据原始rating统计: mean(all_bigdata_rating) = 5.2503, std = 2.3568
        # 注意：JSON文件中的rating统计量是Z-score尺度的（已标准化），不能用于反向标准化
        
        # 始终使用原始训练数据的rating统计量（用于反向标准化）
        self.global_rating_mean = 5.2503  # 原始rating均值
        self.global_rating_std = 2.3568   # 原始rating标准差
        
        if not silent:
            print(f"\n✓ Rating反向标准化参数（硬编码）:")
            print(f"  原始rating均值: {self.global_rating_mean:.4f}")
            print(f"  原始rating标准差: {self.global_rating_std:.4f}")
            print(f"  数据来源: MATLAB all_bigdata_rating")
            
            # 如果JSON中有rating统计量，显示但不使用（因为是Z-score尺度的）
            if 'rating' in self.data_stats:
                json_mean = self.data_stats['rating'].get('mean', 0.0)
                json_std = self.data_stats['rating'].get('std', 1.0)
                print(f"  注意: JSON中rating统计量为Z-score尺度（mean={json_mean:.4f}, std={json_std:.4f}）")
                print(f"        已忽略，使用原始统计量进行反向标准化")
    
    def set_rating_scale(self, mean: float, std: float):
        """
        设置rating量表的均值和标准差
        
        参数:
            mean: rating的均值（例如：4.5 对于0-10量表）
            std: rating的标准差（例如：2.0）
        """
        self.global_rating_mean = mean
        self.global_rating_std = std
        
        print(f"\n✓ 已设置rating量表:")
        print(f"  均值: {mean:.2f}")
        print(f"  标准差: {std:.2f}")
    
    def _extract_fixed_coefficients(self, silent: bool = False):
        """提取固定效应系数"""
        # 截距
        intercept_key = [k for k in self.fixed_effects.keys() if 'Intercept' in k]
        if intercept_key:
            self.intercept = self.fixed_effects[intercept_key[0]]['estimate']
            self.intercept_se = self.fixed_effects[intercept_key[0]]['se']
        else:
            self.intercept = 0.0
            self.intercept_se = 0.0
        
        # 各特征的系数
        self.coefficients = {}
        self.coefficients_se = {}
        
        for feat in self.feature_names:
            if feat in self.fixed_effects:
                self.coefficients[feat] = self.fixed_effects[feat]['estimate']
                self.coefficients_se[feat] = self.fixed_effects[feat]['se']
            else:
                # 尝试匹配
                found = False
                for key in self.fixed_effects.keys():
                    if feat.lower() in key.lower().replace('_', ''):
                        self.coefficients[feat] = self.fixed_effects[key]['estimate']
                        self.coefficients_se[feat] = self.fixed_effects[key]['se']
                        found = True
                        break
                
                if not found:
                    if not silent:
                        print(f"⚠️  警告: 未找到特征 {feat} 的系数，设为0")
                    self.coefficients[feat] = 0.0
                    self.coefficients_se[feat] = 0.0
        
        if not silent:
            # 打印系数摘要
            print(f"\n✓ 模型系数 (前5个最重要的特征):")
            
            # 按系数绝对值排序
            coef_sorted = sorted(
                [(feat, abs(self.coefficients[feat])) for feat in self.feature_names],
                key=lambda x: x[1],
                reverse=True
            )
            
            for i, (feat, abs_coef) in enumerate(coef_sorted[:5], 1):
                coef = self.coefficients[feat]
                pval = self.fixed_effects.get(feat, {}).get('pValue', np.nan)
                sig = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else ""))
                print(f"  {i}. {feat:<15s}: {coef:7.3f} {sig}")
    
    def register_subject(self,
                        subject_id: Optional[Union[int, str]] = None,
                        historical_features: pd.DataFrame = None,
                        feature_cols: Optional[List[str]] = None) -> SubjectProfile:
        """
        注册被试
        
        参数:
            subject_id: 被试ID（可选，如果不提供则自动分配773+编号）
            historical_features: 包含特征的DataFrame
            feature_cols: 特征列名列表（默认使用模型的所有特征）
        
        返回:
            SubjectProfile对象
        """
        # 如果未提供subject_id，自动分配新编号
        if subject_id is None:
            subject_id = self.next_new_subject_id
            self.next_new_subject_id += 1
            print(f"\n✓ 自动分配被试ID: {subject_id}")
        
        if feature_cols is None:
            feature_cols = self.feature_names
        
        # 检查特征列
        missing_cols = [col for col in feature_cols if col not in historical_features.columns]
        if missing_cols:
            raise ValueError(f"历史数据中缺少特征列: {missing_cols}")
        
        # 提取特征数据
        hist_features = {
            feat: historical_features[feat].values 
            for feat in feature_cols
        }
        
        # 创建被试档案
        profile = SubjectProfile(
            subject_id=subject_id,
            historical_features=hist_features
        )
        
        # 存储档案
        self.subject_profiles[str(subject_id)] = profile
        
        print(f"\n✓ 已注册被试: {subject_id}")
        print(f"  {profile.get_info()}")
        
        return profile
    
    def predict(self,
               subject_id: Union[int, str],
               features: Dict[str, float],
               confidence_level: float = 0.95) -> Dict:
        """
        预测rating
        
        参数:
            subject_id: 被试ID
            features: 原始特征值字典（10个特征）
            confidence_level: 置信水平（默认0.95）
        
        返回:
            包含预测结果的字典
        """
        subject_key = str(subject_id)
        
        # 检查被试是否已注册
        if subject_key in self.subject_profiles:
            # 已注册：使用被试的特征标准化
            profile = self.subject_profiles[subject_key]
            features_zscore = profile.standardize_features(features)
            is_registered = True
        else:
            # 未注册：使用全局特征标准化
            features_zscore = {}
            for feat_name, feat_value in features.items():
                if feat_name not in self.global_feature_means:
                    raise ValueError(f"特征 {feat_name} 没有全局统计参数")
                
                mean = self.global_feature_means[feat_name]
                std = self.global_feature_stds[feat_name]
                features_zscore[feat_name] = (feat_value - mean) / std
            
            is_registered = False
        
        # 在Z-score尺度上预测
        zscore_result = self._predict_zscore(features_zscore, confidence_level)
        
        # 转换到原始rating尺度
        rating_original = (zscore_result['rating_mean'] * self.global_rating_std + 
                          self.global_rating_mean)
        rating_se_original = zscore_result['rating_se'] * self.global_rating_std
        ci_lower_original = (zscore_result['ci_lower'] * self.global_rating_std + 
                            self.global_rating_mean)
        ci_upper_original = (zscore_result['ci_upper'] * self.global_rating_std + 
                            self.global_rating_mean)
        
        # 构建返回结果
        result = {
            # Z-score尺度（相对值）
            'rating_zscore': zscore_result['rating_mean'],
            'rating_se_zscore': zscore_result['rating_se'],
            'ci_lower_zscore': zscore_result['ci_lower'],
            'ci_upper_zscore': zscore_result['ci_upper'],
            
            # 原始尺度（估算的绝对值）
            'rating_original': rating_original,
            'rating_se_original': rating_se_original,
            'ci_lower_original': ci_lower_original,
            'ci_upper_original': ci_upper_original,
            
            # 元信息
            'confidence_level': confidence_level,
            'subject_id': subject_id,
            'subject_registered': is_registered
        }
        
        return result
    
    def _predict_zscore(self, 
                       features_zscore: Dict[str, float],
                       confidence_level: float = 0.95) -> Dict:
        """在Z-score尺度上预测（内部方法）"""
        # 转换为数组
        X = self._dict_to_array(features_zscore)
        
        # 计算固定效应预测
        y_pred = self.intercept
        for i, feat in enumerate(self.feature_names):
            y_pred += self.coefficients[feat] * X[i]
        
        # 计算标准误差
        var_pred = self.intercept_se ** 2
        for i, feat in enumerate(self.feature_names):
            var_pred += (self.coefficients_se[feat] * X[i]) ** 2
        
        # 加上残差方差
        var_residual = self.random_effects.get('residual_variance', 0)
        var_pred += var_residual
        
        se_pred = np.sqrt(var_pred)
        
        # 计算置信区间
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        ci_lower = y_pred - z_score * se_pred
        ci_upper = y_pred + z_score * se_pred
        
        return {
            'rating_mean': float(y_pred),
            'rating_se': float(se_pred),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper)
        }
    
    def _dict_to_array(self, features: Dict[str, float]) -> np.ndarray:
        """将特征字典转为数组"""
        X = np.zeros(len(self.feature_names))
        
        for i, feat in enumerate(self.feature_names):
            if feat in features:
                X[i] = features[feat]
            else:
                raise ValueError(f"缺少必需的特征: {feat}")
        
        return X
    
    def batch_predict(self,
                     subject_id: Union[int, str],
                     features_df: pd.DataFrame,
                     confidence_level: float = 0.95) -> pd.DataFrame:
        """批量预测"""
        # 检查必需的列
        missing_cols = [col for col in self.feature_names if col not in features_df.columns]
        if missing_cols:
            raise ValueError(f"输入DataFrame缺少以下列: {missing_cols}")
        
        results = []
        
        print(f"\n批量预测: Subject {subject_id}")
        print(f"样本数: {len(features_df)}")
        
        for idx, row in features_df.iterrows():
            features = {feat: row[feat] for feat in self.feature_names}
            
            try:
                result = self.predict(subject_id, features, confidence_level)
                result['trial_index'] = idx
                
                # 添加输入特征
                for feat in self.feature_names:
                    result[f'{feat}_input'] = features[feat]
                
                results.append(result)
            
            except Exception as e:
                print(f"  ⚠️  Trial {idx} 预测失败: {e}")
        
        results_df = pd.DataFrame(results)
        
        print(f"✓ 完成: {len(results)}/{len(features_df)} 个试次")
        print(f"  平均Z-score: {results_df['rating_zscore'].mean():.3f}")
        print(f"  平均原始rating: {results_df['rating_original'].mean():.2f}")
        
        return results_df
    
    def print_prediction_summary(self, result: Dict):
        """打印预测结果的友好摘要"""
        print(f"\n" + "="*60)
        print(f"预测结果 - Subject {result['subject_id']}")
        print("="*60)
        
        # Z-score结果
        print(f"\nZ-score尺度（相对值）:")
        print(f"  预测值: {result['rating_zscore']:6.3f} ± {result['rating_se_zscore']:.3f}")
        print(f"  {result['confidence_level']*100:.0f}% CI: "
              f"[{result['ci_lower_zscore']:6.3f}, {result['ci_upper_zscore']:6.3f}]")
        
        # 转换为百分位
        zscore = result['rating_zscore']
        percentile = stats.norm.cdf(zscore) * 100
        print(f"  百分位: 该疼痛水平超过了 {percentile:.1f}% 的试次")
        
        # 原始尺度结果
        print(f"\n原始rating尺度（估算值）:")
        print(f"  预测值: {result['rating_original']:6.2f} ± {result['rating_se_original']:.2f}")
        print(f"  {result['confidence_level']*100:.0f}% CI: "
              f"[{result['ci_lower_original']:6.2f}, {result['ci_upper_original']:6.2f}]")
        
        # 提示
        if not result['subject_registered']:
            print(f"\n⚠️  被试未注册，使用全局标准化")
            print(f"  建议先注册被试以提高准确性")
        
        print("="*60)
    
    def list_registered_subjects(self):
        """列出所有已注册的被试"""
        n_subjects = len(self.subject_profiles)
        print(f"\n已注册被试数: {n_subjects}")
        
        if n_subjects > 0:
            print(f"\n{'Subject ID':<15s} {'试次数':>10s} {'特征数':>10s}")
            print("-" * 40)
            
            for subject_id, profile in self.subject_profiles.items():
                feat_name = list(profile.historical_features.keys())[0]
                n_trials = len(profile.historical_features[feat_name])
                n_features = len(profile.feature_means)
                
                print(f"{str(subject_id):<15s} {n_trials:10d} {n_features:10d}")
    
    def save_subject_profile(self, subject_id: Union[int, str], output_file: str):
        """保存被试档案到CSV"""
        subject_key = str(subject_id)
        
        if subject_key not in self.subject_profiles:
            raise ValueError(f"被试 {subject_id} 未注册")
        
        profile = self.subject_profiles[subject_key]
        
        # 构建DataFrame
        data = {
            'subject_id': [subject_id] * len(self.feature_names),
            'feature': self.feature_names,
            'mean': [profile.feature_means[f] for f in self.feature_names],
            'std': [profile.feature_stds[f] for f in self.feature_names]
        }
        
        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False)
        
        print(f"\n✓ 被试档案已保存: {output_file}")
    
    def load_subject_profile(self, subject_id: Union[int, str], input_file: str):
        """从CSV加载被试档案"""
        df = pd.read_csv(input_file)
        
        # 重构历史特征（生成符合均值和标准差的随机数据）
        n_samples = 20  # 生成20个样本点
        hist_features = {}
        
        for _, row in df.iterrows():
            feat = row['feature']
            mean = row['mean']
            std = row['std']
            
            # 生成符合统计量的样本
            hist_features[feat] = np.random.normal(mean, std, n_samples)
        
        # 创建档案
        profile = SubjectProfile(
            subject_id=subject_id,
            historical_features=hist_features
        )
        
        # 直接设置统计参数（更准确）
        profile.feature_means = {row['feature']: row['mean'] for _, row in df.iterrows()}
        profile.feature_stds = {row['feature']: row['std'] for _, row in df.iterrows()}
        
        # 存储档案
        self.subject_profiles[str(subject_id)] = profile
        
        print(f"\n✓ 已从文件加载被试档案: {subject_id}")
        print(f"  文件: {input_file}")


def interactive_mode():
    """命令行交互模式"""
    print("\n" + "="*80)
    print(" "*20 + "LME Rating预测器 - 交互模式")
    print("="*80)
    
    # 1. 加载模型
    params_file = input("\n模型参数文件路径 (默认: lme_model_params.json): ").strip()
    if not params_file:
        params_file = "lme_model_params.json"
    
    if not os.path.exists(params_file):
        print(f"\n❌ 文件不存在: {params_file}")
        return
    
    try:
        predictor = LMERatingPredictor(params_file)
    except Exception as e:
        print(f"\n❌ 加载失败: {e}")
        return
    
    # 2. 显示rating量表信息（已硬编码，无需手动设置）
    print("\n" + "-"*80)
    print("Rating量表信息")
    print("-"*80)
    print(f"已使用训练数据的rating统计量（硬编码）:")
    print(f"  均值: {predictor.global_rating_mean:.4f}")
    print(f"  标准差: {predictor.global_rating_std:.4f}")
    print(f"  数据来源: MATLAB all_bigdata_rating")
    print("\n如需修改，可在Python脚本中调用:")
    print("  predictor.set_rating_scale(mean=?, std=?)")
    
    # 主循环
    while True:
        print("\n" + "="*80)
        print("主菜单")
        print("="*80)
        print("1. 注册新被试（单个）")
        print("2. 预测单个试次")
        print("3. 批量注册新被试")
        print("4. 批量预测多名被试的多个试次")
        print("5. 退出")
        
        choice = input("\n请选择 (1-5): ").strip()
        
        if choice == '1':
            # 注册新被试（单个）
            print("\n" + "-"*80)
            print("注册新被试")
            print("-"*80)
            
            subject_id = input("被试ID (留空自动分配773+编号): ").strip()
            if not subject_id:
                subject_id = None
            else:
                try:
                    subject_id = int(subject_id)
                except ValueError:
                    pass  # 保持字符串
            
            csv_file = input("特征历史数据CSV文件路径: ").strip()
            
            if not os.path.exists(csv_file):
                print(f"❌ 文件不存在: {csv_file}")
                continue
            
            try:
                historical_data = pd.read_csv(csv_file)
                print(f"\n✓ 读取到 {len(historical_data)} 条历史数据")
                print(f"  列名: {list(historical_data.columns)}")
                
                profile = predictor.register_subject(subject_id, historical_data)
                
            except Exception as e:
                print(f"\n❌ 注册失败: {e}")
        
        elif choice == '2':
            # 预测单个试次
            print("\n" + "-"*80)
            print("预测单个试次")
            print("-"*80)
            
            # 先显示已注册的被试
            if len(predictor.subject_profiles) == 0:
                print("⚠️  当前没有已注册的被试，请先注册（菜单1）")
                continue
            
            print(f"\n当前已注册被试: {list(predictor.subject_profiles.keys())}")
            subject_id = input("被试ID: ").strip()
            
            # 检查被试是否存在
            if subject_id not in predictor.subject_profiles:
                print(f"⚠️  被试 {subject_id} 未注册")
                use_global = input("是否使用全局标准化预测? (y/n): ").strip().lower()
                if use_global != 'y':
                    continue
            
            print("\n请输入各特征值:")
            features = {}
            
            for feat in predictor.feature_names:
                while True:
                    value = input(f"  {feat}: ").strip()
                    
                    if value.lower() == 'q':
                        break
                    
                    try:
                        features[feat] = float(value)
                        break
                    except ValueError:
                        print(f"    ⚠️  无效输入，请输入数字")
            
            if len(features) == len(predictor.feature_names):
                try:
                    result = predictor.predict(subject_id, features)
                    predictor.print_prediction_summary(result)
                    
                    # 保存选项
                    save = input("\n保存结果到CSV？(y/n): ").strip().lower()
                    if save == 'y':
                        output_file = input("输出文件名 (默认: prediction_result.csv): ").strip()
                        if not output_file:
                            output_file = "prediction_result.csv"
                        
                        result_df = pd.DataFrame([{**features, **result}])
                        result_df.to_csv(output_file, index=False)
                        print(f"✓ 已保存: {output_file}")
                
                except Exception as e:
                    print(f"\n❌ 预测错误: {e}")
        
        elif choice == '3':
            # 批量注册新被试
            print("\n" + "-"*80)
            print("批量注册新被试")
            print("-"*80)
            print("\n说明: 请提供包含被试ID和特征历史数据的文件夹或文件列表")
            
            # 方式1: 从文件夹批量注册
            mode = input("\n注册方式 (1=从文件夹, 2=手动输入文件列表): ").strip()
            
            if mode == '1':
                folder = input("特征数据文件夹路径: ").strip()
                
                if not os.path.exists(folder):
                    print(f"❌ 文件夹不存在: {folder}")
                    continue
                
                # 查找所有CSV文件
                import glob
                csv_files = glob.glob(os.path.join(folder, "*.csv"))
                
                if not csv_files:
                    print(f"❌ 文件夹中没有CSV文件")
                    continue
                
                print(f"\n找到 {len(csv_files)} 个CSV文件:")
                for f in csv_files[:5]:  # 显示前5个
                    print(f"  - {os.path.basename(f)}")
                if len(csv_files) > 5:
                    print(f"  ... 还有 {len(csv_files)-5} 个文件")
                
                confirm = input("\n开始批量注册? (y/n): ").strip().lower()
                if confirm != 'y':
                    continue
                
                success_count = 0
                for csv_file in csv_files:
                    try:
                        # 从文件名提取被试ID（假设格式: subject_XXX_*.csv）
                        filename = os.path.basename(csv_file)
                        # 尝试多种命名格式
                        if 'subject_' in filename:
                            subject_id = filename.split('subject_')[1].split('_')[0]
                        else:
                            # 使用自动分配
                            subject_id = None
                        
                        historical_data = pd.read_csv(csv_file)
                        profile = predictor.register_subject(subject_id, historical_data)
                        success_count += 1
                        print(f"✓ 已注册: {profile.subject_id} (来自 {filename})")
                    
                    except Exception as e:
                        print(f"✗ 失败: {filename} - {e}")
                
                print(f"\n批量注册完成: {success_count}/{len(csv_files)} 成功")
            
            elif mode == '2':
                # 手动输入文件列表
                print("\n请输入文件路径（每行一个，输入空行结束）:")
                csv_files = []
                while True:
                    line = input("  文件路径: ").strip()
                    if not line:
                        break
                    csv_files.append(line)
                
                if not csv_files:
                    print("❌ 没有输入任何文件")
                    continue
                
                success_count = 0
                for csv_file in csv_files:
                    if not os.path.exists(csv_file):
                        print(f"✗ 文件不存在: {csv_file}")
                        continue
                    
                    # 询问被试ID
                    subject_id = input(f"  {os.path.basename(csv_file)} 的被试ID (留空自动分配): ").strip()
                    if not subject_id:
                        subject_id = None
                    
                    try:
                        historical_data = pd.read_csv(csv_file)
                        profile = predictor.register_subject(subject_id, historical_data)
                        success_count += 1
                        print(f"✓ 已注册: {profile.subject_id}")
                    
                    except Exception as e:
                        print(f"✗ 失败: {e}")
                
                print(f"\n批量注册完成: {success_count}/{len(csv_files)} 成功")
            
            # 显示所有已注册被试
            print("\n当前已注册被试:")
            predictor.list_registered_subjects()
        
        elif choice == '4':
            # 批量预测多名被试的多个试次
            print("\n" + "-"*80)
            print("批量预测多名被试的多个试次")
            print("-"*80)
            
            if len(predictor.subject_profiles) == 0:
                print("⚠️  当前没有已注册的被试，请先注册（菜单1或3）")
                continue
            
            print(f"\n当前已注册被试: {list(predictor.subject_profiles.keys())}")
            
            # 方式1: 单个被试的批量预测
            mode = input("\n预测方式 (1=单个被试批量预测, 2=多个被试分别预测): ").strip()
            
            if mode == '1':
                # 单个被试的批量预测
                subject_id = input("被试ID: ").strip()
                
                if subject_id not in predictor.subject_profiles:
                    print(f"⚠️  被试 {subject_id} 未注册")
                    continue
                
                input_csv = input("输入CSV文件路径（包含特征列）: ").strip()
                
                if not os.path.exists(input_csv):
                    print(f"❌ 文件不存在: {input_csv}")
                    continue
                
                output_csv = input("输出CSV文件路径 (默认: predictions_{subject_id}.csv): ").strip()
                if not output_csv:
                    output_csv = f"predictions_{subject_id}.csv"
                
                try:
                    features_df = pd.read_csv(input_csv)
                    print(f"\n✓ 读取到 {len(features_df)} 条数据")
                    
                    results_df = predictor.batch_predict(subject_id, features_df)
                    
                    results_df.to_csv(output_csv, index=False)
                    print(f"\n✓ 批量预测完成！")
                    print(f"  输出文件: {output_csv}")
                
                except Exception as e:
                    print(f"\n❌ 错误: {e}")
            
            elif mode == '2':
                # 多个被试分别预测
                print("\n请输入要预测的被试ID（逗号分隔，或输入'all'预测所有）:")
                subject_input = input("被试ID: ").strip()
                
                if subject_input.lower() == 'all':
                    subject_ids = list(predictor.subject_profiles.keys())
                else:
                    subject_ids = [s.strip() for s in subject_input.split(',')]
                
                # 询问数据文件方式
                data_mode = input("\n数据文件方式 (1=统一文件, 2=分别指定): ").strip()
                
                if data_mode == '1':
                    # 所有被试用同一个特征文件
                    input_csv = input("输入CSV文件路径（包含特征列）: ").strip()
                    
                    if not os.path.exists(input_csv):
                        print(f"❌ 文件不存在: {input_csv}")
                        continue
                    
                    features_df = pd.read_csv(input_csv)
                    print(f"\n✓ 读取到 {len(features_df)} 条数据")
                    
                    output_folder = input("输出文件夹 (默认: ./predictions/): ").strip()
                    if not output_folder:
                        output_folder = "./predictions/"
                    
                    os.makedirs(output_folder, exist_ok=True)
                    
                    success_count = 0
                    for subject_id in subject_ids:
                        if subject_id not in predictor.subject_profiles:
                            print(f"⚠️  跳过未注册被试: {subject_id}")
                            continue
                        
                        try:
                            output_csv = os.path.join(output_folder, f"predictions_{subject_id}.csv")
                            results_df = predictor.batch_predict(subject_id, features_df)
                            results_df.to_csv(output_csv, index=False)
                            success_count += 1
                            print(f"✓ 完成: {subject_id} → {output_csv}")
                        
                        except Exception as e:
                            print(f"✗ 失败: {subject_id} - {e}")
                    
                    print(f"\n批量预测完成: {success_count}/{len(subject_ids)} 成功")
                
                elif data_mode == '2':
                    # 每个被试单独指定文件
                    success_count = 0
                    for subject_id in subject_ids:
                        if subject_id not in predictor.subject_profiles:
                            print(f"\n⚠️  跳过未注册被试: {subject_id}")
                            continue
                        
                        input_csv = input(f"\n{subject_id} 的输入CSV文件: ").strip()
                        
                        if not os.path.exists(input_csv):
                            print(f"✗ 文件不存在: {input_csv}")
                            continue
                        
                        output_csv = input(f"输出文件 (默认: predictions_{subject_id}.csv): ").strip()
                        if not output_csv:
                            output_csv = f"predictions_{subject_id}.csv"
                        
                        try:
                            features_df = pd.read_csv(input_csv)
                            results_df = predictor.batch_predict(subject_id, features_df)
                            results_df.to_csv(output_csv, index=False)
                            success_count += 1
                            print(f"✓ 完成: {subject_id}")
                        
                        except Exception as e:
                            print(f"✗ 失败: {subject_id} - {e}")
                    
                    print(f"\n批量预测完成: {success_count}/{len(subject_ids)} 成功")
        
        elif choice == '5':
            print("\n再见！")
            break
        
        else:
            print("\n无效选择，请重试")


def demo():
    """演示预测器的基本使用"""
    print("\n" + "="*80)
    print(" "*25 + "使用演示")
    print("="*80)
    
    # 1. 加载预测器（rating统计量已硬编码）
    print("\n步骤1: 加载预测器")
    print("-" * 80)
    predictor = LMERatingPredictor('lme_model_params.json')
    
    # 2. 准备特征历史数据
    print("\n步骤2: 准备被试的特征历史数据")
    print("-" * 80)
    
    np.random.seed(42)
    feature_history = pd.DataFrame({
        'N1_amp': np.random.normal(-9, 1.5, 20),
        'N2_amp': np.random.normal(-17, 3, 20),
        'P2_amp': np.random.normal(13.5, 2, 20),
        'N1_lat': np.random.normal(152, 10, 20),
        'N2_lat': np.random.normal(200, 15, 20),
        'P2_lat': np.random.normal(390, 20, 20),
        'ERP_mag': np.random.normal(31, 4, 20),
        'Alpha_mag': np.random.normal(79, 8, 20),
        'Beta_mag': np.random.normal(1.7, 0.3, 20),
        'Gamma_mag': np.random.normal(2.6, 0.5, 20)
    })
    
    print(f"特征历史数据: {len(feature_history)} 个试次")
    
    # 3. 注册被试（自动分配ID）
    print("\n步骤3: 注册新被试（自动分配ID）")
    print("-" * 80)
    profile = predictor.register_subject(subject_id=None, historical_features=feature_history)
    assigned_id = profile.subject_id
    
    # 4. 预测单个试次
    print("\n步骤4: 预测新试次")
    print("-" * 80)
    
    new_features = {
        'N1_amp': -9.1,
        'N2_amp': -17.5,
        'P2_amp': 13.8,
        'N1_lat': 152,
        'N2_lat': 200,
        'P2_lat': 392,
        'ERP_mag': 31.2,
        'Alpha_mag': 79,
        'Beta_mag': 1.7,
        'Gamma_mag': 2.7
    }
    
    result = predictor.predict(subject_id=assigned_id, features=new_features)
    
    # 打印结果
    predictor.print_prediction_summary(result)
    
    print("\n" + "="*80)
    print("✓ 演示完成")
    print("="*80)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == 'demo':
            demo()
        elif sys.argv[1] == 'interactive' or sys.argv[1] == '-i':
            interactive_mode()
        else:
            print("使用方法:")
            print("  python lme_rating_predictor_v5_final.py demo         # 运行演示")
            print("  python lme_rating_predictor_v5_final.py interactive # 交互模式")
            print("  python lme_rating_predictor_v5_final.py -i          # 交互模式（简写）")
    else:
        # 默认进入交互模式
        interactive_mode()
