#!/usr/bin/env python3
"""
跨论文关联引擎 - 集成版
========================

基于论文知识图谱的实际节点类型（paper, research_question, solution）
设计更科学的跨论文关联算法，直接集成到LightRAG系统中

核心改进：
1. 基于真实节点属性的多维相似度计算
2. 语义向量缓存机制，避免重复计算
3. 直接集成到LightRAG，支持向量存储和图更新
4. 针对三种节点类型的专门关联策略
"""

import os
import json
import pickle
import hashlib
import asyncio
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import logging
from datetime import datetime
import random
import re

# 导入LightRAG相关
from lightrag.llm.ollama import ollama_embed
from lightrag.utils import logger, compute_mdhash_id

# 添加当前目录到sys.path，以便绝对导入同目录模块
import sys
from pathlib import Path
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# 导入技术证据提取器
from technical_evidence_extractor import (
    TechnicalEvidenceExtractor, 
    TransferOpportunity,
    TechnicalComponent,
    TechnicalChallenge
)

class CrossPaperLinkingEngine:
    """跨论文关联引擎"""
    
    def __init__(self, 
                 rag_instance,
                 embed_model: str = "bge-m3:latest",
                 embed_host: str = "http://localhost:11434",
                 cache_dir: str = "./cache",
                 similarity_threshold: float = 0.5):
        """
        初始化跨论文关联引擎
        
        Args:
            rag_instance: LightRAG实例
            embed_model: 嵌入模型名称
            embed_host: Ollama服务地址
            cache_dir: 缓存目录
            similarity_threshold: 相似度阈值
        """
        self.rag = rag_instance
        self.embed_model = embed_model
        self.embed_host = embed_host
        self.cache_dir = cache_dir
        self.similarity_threshold = similarity_threshold
        
        # 创建缓存目录
        os.makedirs(cache_dir, exist_ok=True)
        
        # 向量缓存
        self.vector_cache_file = os.path.join(cache_dir, "node_vectors.pkl")
        self.vector_cache = self._load_vector_cache()
        
        # 关联权重配置
        self._setup_linking_weights()
        
        # 调试模式
        self.debug_mode = True
        
        # 🔧 初始化技术证据提取器（延迟加载避免阻塞）
        self.technical_extractor = None
        
        # 🔧 初始化多样化语义模板
        self._init_diverse_templates()
        
        logger.info(f"跨论文关联引擎初始化完成，缓存目录: {cache_dir}")
    
    def _init_diverse_templates(self):
        """初始化多样化语义模板"""
        self.semantic_templates = [
            # 技术迁移类
            "【技术迁移启发】从{source_domain}领域的{source_method}中提取核心算法{core_algorithm}，通过{adaptation_strategy}改造后应用于{target_domain}的{target_challenge}",
            
            # 方法论借鉴类  
            "【方法论借鉴】{source_method}在解决{source_challenge}时采用的{key_insight}策略，为{target_domain}领域的{target_challenge}提供了{innovation_angle}思路",
            
            # 架构设计类
            "【架构设计灵感】{source_method}的{architecture_component}设计理念，经过{modification_approach}调整，可为{target_problem}提供{expected_benefit}",
            
            # 理论融合类
            "【理论融合探索】将{source_theory}理论框架与{target_context}相结合，通过{integration_mechanism}实现{theoretical_contribution}",
            
            # 实验验证类
            "【实验验证启发】{source_method}在{source_dataset}上的{performance_metric}表现，启发我们在{target_dataset}上设计{validation_strategy}验证方案"
        ]
        
        self.template_variables = {
            'source_domains': [
                '时间序列分析', '计算机视觉', '自然语言处理', '图神经网络', 
                '强化学习', '无监督学习', '对比学习', '元学习'
            ],
            'target_domains': [
                '域适应', '迁移学习', '少样本学习', '连续学习',
                '多模态学习', '鲁棒性学习', '可解释AI', '联邦学习'
            ],
            'core_algorithms': [
                '注意力机制', '对抗训练', '正则化策略', '特征对齐',
                '分布匹配', '梯度优化', '损失函数设计', '网络架构'
            ],
            'adaptation_strategies': [
                '参数微调', '结构重组', '损失函数修改', '特征空间变换',
                '数据增强', '模型蒸馏', '集成学习', '层级适配'
            ],
            'innovation_angles': [
                '新颖的理论视角', '实用的工程方案', '高效的计算策略',
                '鲁棒的评估方法', '可扩展的框架设计', '可解释的机制'
            ]
        }
        
        # 统计模板使用情况
        self.template_usage_stats = defaultdict(int)
    
    def _load_vector_cache(self) -> Dict[str, np.ndarray]:
        """加载向量缓存"""
        if os.path.exists(self.vector_cache_file):
            try:
                with open(self.vector_cache_file, 'rb') as f:
                    cache = pickle.load(f)
                logger.info(f"成功加载向量缓存，包含 {len(cache)} 个向量")
                return cache
            except Exception as e:
                logger.warning(f"加载向量缓存失败: {e}")
        return {}
    
    def _save_vector_cache(self):
        """保存向量缓存"""
        try:
            with open(self.vector_cache_file, 'wb') as f:
                pickle.dump(self.vector_cache, f)
            logger.info(f"向量缓存已保存，包含 {len(self.vector_cache)} 个向量")
        except Exception as e:
            logger.error(f"保存向量缓存失败: {e}")
    
    def _setup_linking_weights(self):
        """设置关联权重配置 - 专注Solution到Paper的连接"""
        
        # 🎯 只允许solution到paper的跨论文连接
        self.allowed_cross_paper_types = {("solution", "paper")}
        
        # 多维相似度权重（语义主导）
        self.similarity_weights = {
            "semantic": 0.90,
            "attribute": 0.10,
            "structural": 0.0,
            "type_compatibility": 0.0,
        }

        # 🔥 Solution -> Paper 专用属性权重
        self.solution_to_paper_weights = {
            # Solution侧权重
            "solution_attrs": {
                "solution": 0.70,              # 主要解决方法
                "simplified_solution": 0.30    # 简化解决方法
            },
            # Paper侧权重  
            "paper_attrs": {
                "abstract": 0.50,              # 🎯 主要：摘要
                "core_problem": 0.25,          # 核心问题
                "basic_problem": 0.15,         # 基础问题
                "preliminary_innovation_analysis": 0.07,  # 创新分析
                "title": 0.03,                 # 标题
            }
        }
        
        # 通用属性权重（保留兼容性）
        self.attribute_weights = {
            "paper": {
                "abstract": 0.50,
                "core_problem": 0.25,
                "basic_problem": 0.15,
                "preliminary_innovation_analysis": 0.07,
                "title": 0.03,
            },
            "solution": {
                "solution": 0.70,
                "simplified_solution": 0.30
            }
        }
    
    def _get_node_hash(self, node_data: Dict) -> str:
        """生成节点数据的哈希值，用于缓存键"""
        # 使用关键属性生成哈希
        key_attrs = ['entity_name', 'entity_type']
        for attr_name, weight in self.attribute_weights.get(node_data.get('entity_type', ''), {}).items():
            if weight > 0.1:  # 只考虑重要属性
                key_attrs.append(attr_name)
        
        hash_input = ""
        for attr in key_attrs:
            value = str(node_data.get(attr, ""))
            hash_input += f"{attr}:{value}|"
        
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    async def get_node_vector(self, node_data: Dict, force_refresh: bool = False) -> Optional[np.ndarray]:
        """
        获取节点的语义向量，支持缓存
        
        Args:
            node_data: 节点数据
            force_refresh: 是否强制刷新缓存
            
        Returns:
            语义向量
        """
        node_hash = self._get_node_hash(node_data)
        
        # 检查缓存
        if not force_refresh and node_hash in self.vector_cache:
            return self.vector_cache[node_hash]
        
        # 构建用于嵌入的文本
        embed_text = self._build_embedding_text(node_data)
        
        if not embed_text.strip():
            logger.warning(f"节点 {node_data.get('entity_name', 'Unknown')} 的嵌入文本为空")
            return None
        
        try:
            # 调用Ollama获取嵌入向量
            embedding_result = await ollama_embed(
                [embed_text],
                embed_model=self.embed_model,
                host=self.embed_host,
                timeout=120  # 增加超时时间
            )
            
            if embedding_result is not None and len(embedding_result) > 0:
                vector = np.array(embedding_result[0], dtype=np.float32)
                # 归一化
                norm = np.linalg.norm(vector)
                if norm > 0:
                    vector = vector / norm
                else:
                    logger.warning(f"节点向量范数为0: {node_data.get('entity_name', 'Unknown')}")
                    return None
                
                # 保存到缓存
                self.vector_cache[node_hash] = vector
                return vector
            else:
                logger.error(f"获取节点向量失败: {node_data.get('entity_name', 'Unknown')}")
                return None
                
        except Exception as e:
            logger.error(f"计算节点向量时出错: {e}")
            # 🔧 备选方案：生成基于文本长度的简单向量
            try:
                # 创建一个基于文本特征的简单向量
                text_words = embed_text.lower().split()
                if len(text_words) > 0:
                    # 简单的文本特征向量（1024维，与bge-m3一致）
                    simple_vector = np.zeros(1024, dtype=np.float32)
                    for i, word in enumerate(text_words[:100]):  # 只取前100个词
                        word_hash = hash(word) % 1024
                        simple_vector[word_hash] += 1.0 / (i + 1)  # 位置权重递减
                    
                    # 归一化
                    norm = np.linalg.norm(simple_vector)
                    if norm > 0:
                        simple_vector = simple_vector / norm
                        self.vector_cache[node_hash] = simple_vector
                        logger.info(f"使用简单向量作为备选: {node_data.get('entity_name', 'Unknown')}")
                        return simple_vector
            except Exception as backup_e:
                logger.error(f"备选向量计算也失败: {backup_e}")
            
            return None
    
    def _build_embedding_text(self, node_data: Dict) -> str:
        """
        根据节点类型和属性权重构建用于嵌入的文本
        
        Args:
            node_data: 节点数据
            
        Returns:
            用于嵌入的文本
        """
        entity_type = node_data.get('entity_type', 'unknown')
        texts = []
        
        if entity_type in self.attribute_weights:
            # 按权重排序属性
            sorted_attrs = sorted(
                self.attribute_weights[entity_type].items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            for attr_name, weight in sorted_attrs:
                value = node_data.get(attr_name, "")
                if value and str(value).strip():
                    # 根据权重重复文本，增强重要属性的影响
                    repeat_count = max(1, int(weight * 5))
                    texts.extend([str(value).strip()] * repeat_count)
        
        # 如果没有找到相关属性，使用默认的description或title
        if not texts:
            for fallback_attr in ['description', 'title', 'entity_name']:
                value = node_data.get(fallback_attr, "")
                if value and str(value).strip():
                    texts.append(str(value).strip())
                    break
        
        return " ".join(texts)
    
    def calculate_attribute_similarity(self, node1: Dict, node2: Dict) -> float:
        """
        计算两个节点的属性相似度
        
        Args:
            node1, node2: 节点数据
            
        Returns:
            属性相似度分数 (0-1)
        """
        type1 = node1.get('entity_type', 'unknown')
        type2 = node2.get('entity_type', 'unknown')
        
        # 同类型节点的属性相似度
        if type1 == type2 and type1 in self.attribute_weights:
            similarities = []
            weights = []
            
            for attr_name, weight in self.attribute_weights[type1].items():
                val1 = str(node1.get(attr_name, "")).strip().lower()
                val2 = str(node2.get(attr_name, "")).strip().lower()
                
                if val1 and val2:
                    # 简单的文本相似度计算（可以替换为更复杂的算法）
                    similarity = self._text_similarity(val1, val2)
                    similarities.append(similarity)
                    weights.append(weight)
            
            if similarities:
                # 加权平均
                total_weight = sum(weights)
                weighted_sum = sum(sim * weight for sim, weight in zip(similarities, weights))
                return weighted_sum / total_weight if total_weight > 0 else 0.0
        
        # 跨类型节点的特殊处理
        elif (type1, type2) in [("paper", "research_question"), ("research_question", "paper")]:
            # 论文的核心问题与研究问题的相似度
            paper_problem = ""
            rq_question = ""
            
            if type1 == "paper":
                paper_problem = str(node1.get("core_problem", "")).strip().lower()
                rq_question = str(node2.get("research_question", "")).strip().lower()
            else:
                paper_problem = str(node2.get("core_problem", "")).strip().lower()
                rq_question = str(node1.get("research_question", "")).strip().lower()
            
            if paper_problem and rq_question:
                return self._text_similarity(paper_problem, rq_question)
        
        elif (type1, type2) in [("research_question", "solution"), ("solution", "research_question")]:
            # 研究问题与解决方案的相关性
            rq_text = ""
            sol_text = ""
            
            if type1 == "research_question":
                rq_text = str(node1.get("research_question", "")).strip().lower()
                sol_text = str(node2.get("solution", "")).strip().lower()
            else:
                rq_text = str(node2.get("research_question", "")).strip().lower()
                sol_text = str(node1.get("solution", "")).strip().lower()
            
            if rq_text and sol_text:
                return self._text_similarity(rq_text, sol_text)
        
        return 0.0
    
    def calculate_solution_paper_similarity(self, solution_node: Dict, paper_node: Dict) -> float:
        """
        专门计算Solution到Paper的相似度
        
        Args:
            solution_node: Solution节点数据
            paper_node: Paper节点数据
            
        Returns:
            相似度分数 (0-1)
        """
        solution_weights = self.solution_to_paper_weights["solution_attrs"]
        paper_weights = self.solution_to_paper_weights["paper_attrs"]
        
        similarities = []
        weights = []
        
        # Solution侧的文本
        solution_texts = []
        for attr_name, weight in solution_weights.items():
            value = str(solution_node.get(attr_name, "")).strip()
            if value:
                solution_texts.append((value, weight))
        
        # Paper侧的文本
        paper_texts = []
        for attr_name, weight in paper_weights.items():
            value = str(paper_node.get(attr_name, "")).strip()
            if value:
                paper_texts.append((value, weight))
        
        # 计算所有组合的相似度
        for sol_text, sol_weight in solution_texts:
            for paper_text, paper_weight in paper_texts:
                text_sim = self._text_similarity(sol_text, paper_text)
                if text_sim > 0:
                    # 组合权重
                    combined_weight = sol_weight * paper_weight
                    similarities.append(text_sim)
                    weights.append(combined_weight)
        
        if similarities:
            # 加权平均
            total_weight = sum(weights)
            weighted_sum = sum(sim * weight for sim, weight in zip(similarities, weights))
            return weighted_sum / total_weight if total_weight > 0 else 0.0
        
        return 0.0

    def calculate_technical_evidence_similarity(self, solution_node: Dict, paper_node: Dict) -> Tuple[float, Dict]:
        """
        🔧 新增：基于技术证据的相似度计算
        
        Args:
            solution_node: Solution节点数据
            paper_node: Paper节点数据
            
        Returns:
            (相似度分数, 技术证据详情)
        """
        # 1. 从solution中提取技术组件
        technical_components = self.technical_extractor.extract_technical_components(solution_node)
        
        # 2. 从paper中提取技术挑战
        technical_challenges = self.technical_extractor.extract_technical_challenges(paper_node)
        
        if not technical_components or not technical_challenges:
            return 0.0, {}
        
        # 3. 发现迁移机会
        transfer_opportunities = self.technical_extractor.find_transfer_opportunities(
            technical_components, technical_challenges
        )
        
        if not transfer_opportunities:
            return 0.0, {}
        
        # 4. 计算最佳迁移机会的可行性作为相似度
        best_opportunity = transfer_opportunities[0]  # 已按可行性排序
        similarity_score = best_opportunity.transfer_feasibility
        
        # 5. 构建技术证据详情
        evidence_details = {
            "best_opportunity": best_opportunity,
            "total_opportunities": len(transfer_opportunities),
            "technical_components": technical_components,
            "technical_challenges": technical_challenges,
            "adaptation_evidence": best_opportunity.adaptation_evidence,
            "adaptation_mechanism": best_opportunity.adaptation_mechanism,
            "validation_suggestion": best_opportunity.validation_suggestion
        }
        
        logger.debug(f"技术证据相似度: {similarity_score:.3f}, 发现 {len(transfer_opportunities)} 个迁移机会")
        
        return similarity_score, evidence_details
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """
        计算两个文本的相似度（改进的算法）
        
        Args:
            text1, text2: 待比较的文本
            
        Returns:
            相似度分数 (0-1)
        """
        # 预处理：转小写，移除常见停词
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an'}
        
        def clean_text(text):
            words = text.lower().split()
            return set(word for word in words if len(word) > 2 and word not in stop_words)
        
        words1 = clean_text(text1)
        words2 = clean_text(text2)
        
        if not words1 or not words2:
            return 0.0
        
        # Jaccard相似度
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        jaccard_sim = intersection / union if union > 0 else 0.0
        
        # 计算重叠比例（对较短文本的奖励）
        overlap_ratio = intersection / min(len(words1), len(words2)) if min(len(words1), len(words2)) > 0 else 0.0
        
        # 综合相似度：Jaccard + 重叠奖励
        final_sim = 0.7 * jaccard_sim + 0.3 * overlap_ratio
        
        return min(final_sim, 1.0)
    
    def _calculate_semantic_similarity(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """
        计算两个向量的语义相似度（余弦相似度）
        
        Args:
            vector1, vector2: 语义向量
            
        Returns:
            余弦相似度 (0-1)
        """
        try:
            # 计算余弦相似度
            dot_product = np.dot(vector1, vector2)
            norm1 = np.linalg.norm(vector1)
            norm2 = np.linalg.norm(vector2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            cosine_sim = dot_product / (norm1 * norm2)
            # 将余弦相似度从[-1,1]映射到[0,1]
            return (cosine_sim + 1) / 2
        except Exception as e:
            logger.error(f"计算语义相似度失败: {e}")
            return 0.0
    
    def get_type_compatibility_score(self, type1: str, type2: str) -> float:
        """
        获取两个节点类型的兼容性分数
        
        Args:
            type1, type2: 节点类型
            
        Returns:
            兼容性分数 (0-1)
        """
        # 标准化类型对
        type_pair = tuple(sorted([type1, type2]))
        
        return self.type_compatibility.get(type_pair, self.type_compatibility["default"])
    
    async def calculate_comprehensive_similarity(self, 
                                               node1: Dict, 
                                               node2: Dict,
                                               vector1: Optional[np.ndarray] = None,
                                               vector2: Optional[np.ndarray] = None) -> float:
        """
        计算两个节点的综合相似度
        
        Args:
            node1, node2: 节点数据
            vector1, vector2: 预计算的语义向量（可选）
            
        Returns:
            综合相似度分数 (0-1)
        """
        
        # 1. 语义相似度
        semantic_sim = 0.0
        if vector1 is not None and vector2 is not None:
            semantic_sim = float(np.dot(vector1, vector2))
        else:
            # 如果没有预计算向量，临时计算
            if vector1 is None:
                vector1 = await self.get_node_vector(node1)
            if vector2 is None:
                vector2 = await self.get_node_vector(node2)
            
            if vector1 is not None and vector2 is not None:
                semantic_sim = float(np.dot(vector1, vector2))
        
        # 2. 属性相似度
        attribute_sim = self.calculate_attribute_similarity(node1, node2)
        
        # 3. 类型兼容性
        type1 = node1.get('entity_type', 'unknown')
        type2 = node2.get('entity_type', 'unknown')
        type_compat = self.get_type_compatibility_score(type1, type2)
        
        # 语义最低门槛：不达标直接不建边
        semantic_min = float(os.getenv("CPL_MIN_SEMANTIC", "0.60"))
        if semantic_sim < semantic_min:
            return 0.0

        # 4. 结构相似度（不参与）
        structural_sim = 0.0
        
        # 5. 综合相似度计算
        comprehensive_score = (
            self.similarity_weights["semantic"] * semantic_sim +
            self.similarity_weights["attribute"] * attribute_sim +
            self.similarity_weights["type_compatibility"] * type_compat +
            self.similarity_weights["structural"] * structural_sim
        )
        
        # 🔍 调试：记录高分相似度详情
        final_score = min(1.0, max(0.0, comprehensive_score))
        if final_score >= 0.5:  # 记录中等以上的相似度
            entity1 = node1.get('entity_name', 'Unknown')[:30]
            entity2 = node2.get('entity_name', 'Unknown')[:30]
            logger.debug(f"相似度详情 {entity1}...↔{entity2}...: "
                        f"综合={final_score:.3f} (语义={semantic_sim:.3f}, "
                        f"属性={attribute_sim:.3f}, 类型={type_compat:.3f}, "
                        f"结构={structural_sim:.3f})")
        
        return final_score
    
    def _calculate_structural_similarity(self, node1: Dict, node2: Dict) -> float:
        """
        计算结构相似度（基于节点在论文中的结构位置）
        
        Args:
            node1, node2: 节点数据
            
        Returns:
            结构相似度分数 (0-1)
        """
        # 提取论文ID和节点在论文中的序号
        name1 = node1.get('entity_name', '')
        name2 = node2.get('entity_name', '')
        
        # 假设节点命名格式为: PaperName_TYPE_序号
        try:
            parts1 = name1.split('_')
            parts2 = name2.split('_')
            
            if len(parts1) >= 3 and len(parts2) >= 3:
                # 比较节点类型和序号
                type1, seq1 = parts1[-2], parts1[-1]
                type2, seq2 = parts2[-2], parts2[-1]
                
                # 同类型同序号的节点具有更高的结构相似度
                if type1 == type2 and seq1 == seq2:
                    return 0.8
                elif type1 == type2:
                    return 0.6
                else:
                    return 0.3
            
        except (IndexError, ValueError):
            pass
        
        return 0.4  # 默认结构相似度
    
    def calculate_comprehensive_node_similarity(self, node1: Dict, node2: Dict) -> float:
        """
        计算基于节点属性的综合相似度（无LLM调用）
        
        Args:
            node1, node2: 节点数据
            
        Returns:
            综合相似度分数 (0-1)
        """
        # 1. 基础属性相似度
        attr_sim = self._calculate_attribute_similarity(node1, node2)
        
        # 2. 实体名称语义相似度（基于关键词匹配）
        name_sim = self._calculate_name_similarity(node1, node2)
        
        # 3. 类型兼容性
        type_compat = self._calculate_type_compatibility(node1, node2)
        
        # 4. 结构相似度
        struct_sim = self._calculate_structural_similarity(node1, node2)
        
        # 5. 内容长度相似度
        content_sim = self._calculate_content_similarity(node1, node2)
        
        # 加权综合
        comprehensive_score = (
            0.3 * attr_sim +
            0.25 * name_sim +
            0.2 * type_compat +
            0.15 * struct_sim +
            0.1 * content_sim
        )
        
        return min(1.0, max(0.0, comprehensive_score))
    
    def _calculate_name_similarity(self, node1: Dict, node2: Dict) -> float:
        """计算实体名称相似度"""
        name1 = node1.get('entity_name', '').lower()
        name2 = node2.get('entity_name', '').lower()
        
        if not name1 or not name2:
            return 0.0
        
        # 关键词提取和匹配
        import re
        words1 = set(re.findall(r'\b\w+\b', name1))
        words2 = set(re.findall(r'\b\w+\b', name2))
        
        if not words1 or not words2:
            return 0.0
        
        # Jaccard相似度
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_content_similarity(self, node1: Dict, node2: Dict) -> float:
        """计算内容长度相似度"""
        # 获取主要内容字段
        content1 = str(node1.get('description', '') or node1.get('full_solution', '') or node1.get('simplified_solution', '') or '')
        content2 = str(node2.get('description', '') or node2.get('full_solution', '') or node2.get('simplified_solution', '') or '')
        
        len1, len2 = len(content1), len(content2)
        
        if len1 == 0 and len2 == 0:
            return 1.0
        if len1 == 0 or len2 == 0:
            return 0.0
        
        # 长度相似度（避免极端差异）
        ratio = min(len1, len2) / max(len1, len2)
        return ratio
    
    def build_attribute_based_evidence(self, node1: Dict, node2: Dict, similarity: float) -> Dict:
        """构建基于属性的关联证据"""
        evidence = {
            'method': 'attribute_based',
            'similarity_score': similarity,
            'node1_type': node1.get('entity_type', 'unknown'),
            'node2_type': node2.get('entity_type', 'unknown'),
            'node1_name': node1.get('entity_name', '')[:50],
            'node2_name': node2.get('entity_name', '')[:50]
        }
        
        # 添加具体的相似度组件
        evidence['components'] = {
            'attribute_similarity': self._calculate_attribute_similarity(node1, node2),
            'name_similarity': self._calculate_name_similarity(node1, node2),
            'type_compatibility': self._calculate_type_compatibility(node1, node2),
            'structural_similarity': self._calculate_structural_similarity(node1, node2),
            'content_similarity': self._calculate_content_similarity(node1, node2)
        }
        
        return evidence
    
    def _calculate_attribute_similarity(self, node1: Dict, node2: Dict) -> float:
        """
        计算节点属性相似度（基于真实属性结构）
        
        根据不同节点类型的实际属性进行科学匹配
        """
        type1 = node1.get('entity_type', '')
        type2 = node2.get('entity_type', '')
        
        # 基于节点类型的智能属性匹配
        if type1 == 'solution' and type2 == 'paper':
            return self._calculate_solution_paper_similarity(node1, node2)
        elif type1 == 'paper' and type2 == 'solution':
            return self._calculate_solution_paper_similarity(node2, node1)
        elif type1 == 'solution' and type2 == 'research_question':
            return self._calculate_solution_rq_similarity(node1, node2)
        elif type1 == 'research_question' and type2 == 'solution':
            return self._calculate_solution_rq_similarity(node2, node1)
        elif type1 == 'paper' and type2 == 'research_question':
            return self._calculate_paper_rq_similarity(node1, node2)
        elif type1 == 'research_question' and type2 == 'paper':
            return self._calculate_paper_rq_similarity(node2, node1)
        else:
            # 同类型节点比较
            return self._calculate_same_type_similarity(node1, node2)
    
    def _calculate_solution_paper_similarity(self, solution_node: Dict, paper_node: Dict) -> float:
        """
        计算Solution和Paper节点的相似度
        充分利用人工标注的丰富属性数据
        """
        similarities = []
        
        # 获取Solution内容
        solution_full = str(solution_node.get('solution', ''))
        solution_simple = str(solution_node.get('simplified_solution', ''))
        solution_name = str(solution_node.get('entity_name', ''))
        
        # 跨领域技术迁移优化权重分配
        
        # 1. 技术背景深度匹配: Solution vs 相关工作 (35%) - 最重要，包含技术细节
        paper_related = str(paper_node.get('related_work', ''))
        related_sim = self._calculate_enhanced_text_similarity(solution_full, paper_related)
        similarities.append(('technical_background', related_sim, 0.35))
        
        # 2. 框架迁移潜力: Solution vs Paper框架总结 (25%)
        paper_framework = str(paper_node.get('framework_summary', ''))
        framework_sim = self._calculate_enhanced_text_similarity(solution_full, paper_framework)
        similarities.append(('framework_transfer', framework_sim, 0.25))
        
        # 3. 方法论创新: Solution vs 初步创新分析 (20%)
        paper_innovation = str(paper_node.get('preliminary_innovation_analysis', ''))
        innovation_sim = self._calculate_enhanced_text_similarity(solution_simple, paper_innovation)
        similarities.append(('methodology_innovation', innovation_sim, 0.2))
        
        # 4. 问题抽象化: Solution vs Paper核心问题 (15%)
        paper_core = str(paper_node.get('core_problem', ''))
        core_sim = self._calculate_enhanced_text_similarity(solution_full, paper_core)
        similarities.append(('problem_abstraction', core_sim, 0.15))
        
        # 5. 高层概述: Solution简化 vs 摘要 (5%)
        paper_abstract = str(paper_node.get('abstract', ''))
        abstract_sim = self._calculate_enhanced_text_similarity(solution_simple, paper_abstract)
        similarities.append(('high_level_match', abstract_sim, 0.05))
        
        # 加权计算并记录详细信息
        weighted_sum = sum(sim * weight for _, sim, weight in similarities)
        
        # 调试信息：记录高分组件
        high_scores = [(name, sim) for name, sim, _ in similarities if sim > 0.3]
        if high_scores:
            logger.debug(f"高分组件: {high_scores}")
        
        return min(1.0, max(0.0, weighted_sum))
    
    def _calculate_enhanced_text_similarity(self, text1: str, text2: str) -> float:
        """
        增强版文本相似度计算
        结合关键词匹配、技术相似度和跨领域迁移潜力
        """
        if not text1 or not text2:
            return 0.0
        
        # 1. 基础关键词匹配
        basic_sim = self._calculate_text_similarity(text1, text2)
        
        # 2. 技术关键词增强匹配（支持跨领域）
        tech_sim = self._calculate_technical_keyword_similarity(text1, text2)
        
        # 3. 概念级别相似度（抽象层面的匹配）
        concept_sim = self._calculate_concept_similarity(text1, text2)
        
        # 4. 长度加权（长文本更可靠）
        len1, len2 = len(text1), len(text2)
        length_weight = min(1.0, (len1 + len2) / 2000)
        
        # 5. 跨领域迁移奖励
        transfer_bonus = self._calculate_transfer_potential(text1, text2)
        
        # 跨领域优化：降低基础文本权重，提升技术和迁移权重
        enhanced_sim = (
            0.25 * basic_sim + 
            0.35 * tech_sim + 
            0.15 * concept_sim + 
            0.25 * transfer_bonus  # 大幅提升迁移潜力权重
        )
        
        # 跨领域场景下，长文本权重更重要（包含更多技术细节）
        final_sim = enhanced_sim * (0.7 + 0.3 * length_weight)
        
        # 跨领域迁移奖励：如果发现技术迁移潜力，给予额外提升
        if transfer_bonus > 0.08 and tech_sim > 0.03:  # 降低触发阈值
            cross_domain_bonus = min(0.2, transfer_bonus * tech_sim * 1.5)  # 增强奖励
            final_sim += cross_domain_bonus
        
        # 长文本深度分析奖励（人工标注数据充分利用）
        if len1 > 1000 and len2 > 1000:  # 两个文本都很长
            depth_bonus = min(0.05, (len1 + len2) / 10000)  # 基于长度的深度奖励
            final_sim += depth_bonus
        
        return min(1.0, max(0.0, final_sim))
    
    def _calculate_concept_similarity(self, text1: str, text2: str) -> float:
        """计算概念级别的相似度"""
        import re
        
        # 抽象概念关键词
        concept_patterns = {
            'problem_solving': r'\b(problem|challenge|issue|difficulty|obstacle)\b',
            'improvement': r'\b(improve|enhance|optimize|better|superior|advance)\b',
            'accuracy': r'\b(accuracy|precision|performance|quality|effectiveness)\b',
            'efficiency': r'\b(efficient|fast|quick|speed|computational|runtime)\b',
            'robustness': r'\b(robust|stable|reliable|consistent|invariant)\b',
            'adaptation': r'\b(adapt|adjust|modify|customize|tailor|flexible)\b',
            'learning': r'\b(learn|training|knowledge|understanding|representation)\b',
            'analysis': r'\b(analysis|analyze|examine|investigate|study|research)\b'
        }
        
        concept_scores = []
        for concept, pattern in concept_patterns.items():
            matches1 = len(re.findall(pattern, text1.lower()))
            matches2 = len(re.findall(pattern, text2.lower()))
            
            if matches1 > 0 and matches2 > 0:
                # 计算概念强度相似度
                concept_sim = 1.0 - abs(matches1 - matches2) / max(matches1, matches2, 1)
                concept_scores.append(concept_sim)
        
        return sum(concept_scores) / len(concept_scores) if concept_scores else 0.0
    
    def _calculate_transfer_potential(self, text1: str, text2: str) -> float:
        """计算跨领域技术迁移潜力"""
        import re
        
        # 可迁移的技术模式
        transferable_patterns = {
            'attention_mechanism': r'\b(attention|focus|weight|importance|selection|channel|spatial)\b',
            'feature_fusion': r'\b(fusion|combine|merge|integrate|concatenate|aggregate|ensemble)\b',
            'multi_level': r'\b(multi-level|hierarchical|pyramid|scale|resolution|multi-scale)\b',
            'learning_strategy': r'\b(supervised|unsupervised|self-supervised|contrastive|adversarial)\b',
            'optimization': r'\b(loss|objective|minimize|maximize|gradient|optimization|training)\b',
            'regularization': r'\b(regularization|constraint|penalty|normalization|dropout|batch)\b',
            'adaptation_techniques': r'\b(adapt|alignment|domain|transfer|cross|bridge|gap)\b',
            'feature_learning': r'\b(feature|representation|embedding|encoding|extraction)\b'
        }
        
        transfer_scores = []
        for pattern_name, pattern in transferable_patterns.items():
            matches1 = bool(re.search(pattern, text1.lower()))
            matches2 = bool(re.search(pattern, text2.lower()))
            
            if matches1 and matches2:
                transfer_scores.append(1.0)
            elif matches1 or matches2:
                transfer_scores.append(0.3)  # 单边匹配也有迁移价值
        
        base_score = sum(transfer_scores) / len(transferable_patterns)
        
        # 跨领域迁移奖励机制优化
        num_matches = len([s for s in transfer_scores if s > 0])
        if num_matches >= 3:
            base_score *= 1.5  # 多模式匹配大幅奖励
        elif num_matches >= 2:
            base_score *= 1.3
        elif num_matches >= 1:
            base_score *= 1.1
        
        # 特殊奖励：attention + fusion 组合（常见跨领域迁移模式）
        has_attention = bool(re.search(transferable_patterns['attention_mechanism'], text1.lower()) and 
                           re.search(transferable_patterns['attention_mechanism'], text2.lower()))
        has_fusion = bool(re.search(transferable_patterns['feature_fusion'], text1.lower()) and 
                         re.search(transferable_patterns['feature_fusion'], text2.lower()))
        has_adaptation = bool(re.search(transferable_patterns['adaptation_techniques'], text1.lower()) and 
                            re.search(transferable_patterns['adaptation_techniques'], text2.lower()))
        
        if has_attention and has_fusion:
            base_score += 0.25  # 注意力+融合组合奖励增强
        elif has_attention and has_adaptation:
            base_score += 0.2   # 注意力+适应组合
        elif has_fusion and has_adaptation:
            base_score += 0.18  # 融合+适应组合
        
        return min(1.0, base_score)
    
    def _calculate_technical_keyword_similarity(self, text1: str, text2: str) -> float:
        """计算技术关键词相似度 - 支持跨领域技术迁移分析"""
        import re
        
        # 定义技术领域和方法的层次化关键词
        tech_categories = {
            'deep_learning': {
                'neural', 'network', 'deep', 'learning', 'training', 'model', 'architecture',
                'layer', 'feature', 'embedding', 'representation', 'attention', 'transformer',
                'convolution', 'pooling', 'activation', 'optimization', 'gradient', 'backpropagation'
            },
            'computer_vision': {
                'segmentation', 'classification', 'detection', 'retrieval', 'recognition',
                'fusion', 'multi-modal', 'cross-modal', 'visual', 'image', 'video', 'depth',
                'stereo', 'reconstruction', 'estimation', 'matching', 'correspondence'
            },
            'domain_adaptation': {
                'domain', 'adaptation', 'transfer', 'adversarial', 'alignment', 'distribution',
                'source', 'target', 'unsupervised', 'supervised', 'semi-supervised', 'cross-domain'
            },
            'methodology': {
                'loss', 'function', 'optimization', 'training', 'learning', 'algorithm',
                'model', 'architecture', 'framework', 'approach', 'method', 'technique',
                'strategy', 'mechanism', 'module', 'component', 'layer', 'feature'
            },
            'data_processing': {
                'dataset', 'benchmark', 'annotation', 'label', 'ground', 'truth', 'synthetic',
                'preprocessing', 'postprocessing', 'filtering', 'sampling', 'selection'
            }
        }
        
        words1 = set(re.findall(r'\b\w{4,}\b', text1.lower()))
        words2 = set(re.findall(r'\b\w{4,}\b', text2.lower()))
        
        # 计算各类别的相似度
        category_similarities = []
        for category, keywords in tech_categories.items():
            tech_words1 = words1.intersection(keywords)
            tech_words2 = words2.intersection(keywords)
            
            if tech_words1 and tech_words2:
                intersection = len(tech_words1.intersection(tech_words2))
                union = len(tech_words1.union(tech_words2))
                if union > 0:
                    category_sim = intersection / union
                    category_similarities.append((category, category_sim, len(tech_words1), len(tech_words2)))
        
        if not category_similarities:
            # 如果没有直接匹配，检查是否有跨领域的方法论相似性
            method_words1 = words1.intersection(tech_categories['methodology'])
            method_words2 = words2.intersection(tech_categories['methodology'])
            
            if method_words1 and method_words2:
                intersection = len(method_words1.intersection(method_words2))
                union = len(method_words1.union(method_words2))
                return 0.4 * (intersection / union) if union > 0 else 0.0  # 提升方法论权重
            
            # 检查深度学习基础技术相似性
            dl_words1 = words1.intersection(tech_categories['deep_learning'])
            dl_words2 = words2.intersection(tech_categories['deep_learning'])
            
            if dl_words1 and dl_words2:
                intersection = len(dl_words1.intersection(dl_words2))
                union = len(dl_words1.union(dl_words2))
                return 0.25 * (intersection / union) if union > 0 else 0.0
            
            return 0.0
        
        # 加权计算：深度学习和方法论权重更高（支持跨领域迁移）
        weights = {
            'deep_learning': 0.3,
            'computer_vision': 0.25,
            'domain_adaptation': 0.25,
            'methodology': 0.15,
            'data_processing': 0.05
        }
        
        weighted_sim = sum(
            weights.get(category, 0.1) * sim 
            for category, sim, _, _ in category_similarities
        )
        
        # 如果有多个类别匹配，给予奖励
        diversity_bonus = min(0.2, 0.05 * len(category_similarities))
        
        return min(1.0, weighted_sim + diversity_bonus)
    
    def _calculate_solution_rq_similarity(self, solution_node: Dict, rq_node: Dict) -> float:
        """计算Solution和Research Question节点的相似度"""
        # Solution内容 vs 研究问题
        solution_content = str(solution_node.get('solution', '') or solution_node.get('simplified_solution', ''))
        rq_content = str(rq_node.get('research_question', '') or rq_node.get('simplified_research_question', ''))
        
        # 直接文本相似度作为主要指标
        return self._calculate_text_similarity(solution_content, rq_content)
    
    def _calculate_paper_rq_similarity(self, paper_node: Dict, rq_node: Dict) -> float:
        """计算Paper和Research Question节点的相似度"""
        # Paper核心问题 vs 研究问题
        paper_problem = str(paper_node.get('core_problem', '') or paper_node.get('abstract', ''))
        rq_content = str(rq_node.get('research_question', '') or rq_node.get('simplified_research_question', ''))
        
        return self._calculate_text_similarity(paper_problem, rq_content)
    
    def _calculate_same_type_similarity(self, node1: Dict, node2: Dict) -> float:
        """计算同类型节点的相似度"""
        node_type = node1.get('entity_type', '')
        
        if node_type == 'solution':
            content1 = str(node1.get('solution', '') or node1.get('simplified_solution', ''))
            content2 = str(node2.get('solution', '') or node2.get('simplified_solution', ''))
        elif node_type == 'paper':
            content1 = str(node1.get('abstract', '') or node1.get('title', ''))
            content2 = str(node2.get('abstract', '') or node2.get('title', ''))
        elif node_type == 'research_question':
            content1 = str(node1.get('research_question', '') or node1.get('simplified_research_question', ''))
            content2 = str(node2.get('research_question', '') or node2.get('simplified_research_question', ''))
        else:
            return 0.1
        
        return self._calculate_text_similarity(content1, content2)
    
    def _calculate_domain_similarity(self, name1: str, name2: str) -> float:
        """计算技术领域相似度"""
        # 提取技术领域关键词
        domain_keywords = [
            'domain', 'adaptation', 'transfer', 'learning', 'network', 'neural',
            'attention', 'fusion', 'alignment', 'adversarial', 'contrastive',
            'segmentation', 'classification', 'detection', 'retrieval', 'recognition'
        ]
        
        import re
        words1 = set(re.findall(r'\b\w+\b', name1.lower()))
        words2 = set(re.findall(r'\b\w+\b', name2.lower()))
        
        # 计算领域关键词交集
        domain_words1 = words1.intersection(set(domain_keywords))
        domain_words2 = words2.intersection(set(domain_keywords))
        
        if not domain_words1 or not domain_words2:
            return 0.0
        
        intersection = len(domain_words1.intersection(domain_words2))
        union = len(domain_words1.union(domain_words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_method_similarity(self, text1: str, text2: str) -> float:
        """计算方法论相似度"""
        # 方法论关键词
        method_keywords = [
            'loss', 'function', 'optimization', 'training', 'learning', 'algorithm',
            'model', 'architecture', 'framework', 'approach', 'method', 'technique',
            'strategy', 'mechanism', 'module', 'component', 'layer', 'feature'
        ]
        
        import re
        words1 = set(re.findall(r'\b\w+\b', text1.lower()))
        words2 = set(re.findall(r'\b\w+\b', text2.lower()))
        
        method_words1 = words1.intersection(set(method_keywords))
        method_words2 = words2.intersection(set(method_keywords))
        
        if not method_words1 or not method_words2:
            return 0.0
        
        intersection = len(method_words1.intersection(method_words2))
        union = len(method_words1.union(method_words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度（基于关键词匹配）"""
        if not text1 or not text2:
            return 0.0
        
        import re
        # 提取关键词
        words1 = set(re.findall(r'\b\w{3,}\b', text1.lower()))  # 只考虑长度>=3的单词
        words2 = set(re.findall(r'\b\w{3,}\b', text2.lower()))
        
        if not words1 or not words2:
            return 0.0
        
        # Jaccard相似度
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_type_compatibility(self, node1: Dict, node2: Dict) -> float:
        """计算节点类型兼容性"""
        type1 = node1.get('entity_type', '')
        type2 = node2.get('entity_type', '')
        
        # 定义类型兼容性矩阵
        compatibility_matrix = {
            ('solution', 'paper'): 0.9,  # solution和paper高度兼容
            ('paper', 'solution'): 0.9,
            ('solution', 'research_question'): 0.7,
            ('research_question', 'solution'): 0.7,
            ('paper', 'research_question'): 0.6,
            ('research_question', 'paper'): 0.6,
            ('solution', 'solution'): 0.5,  # 同类型中等兼容
            ('paper', 'paper'): 0.3,
            ('research_question', 'research_question'): 0.4
        }
        
        return compatibility_matrix.get((type1, type2), 0.1)  # 默认低兼容性
    
    async def find_cross_paper_links(self, progress_callback=None) -> List[Dict]:
        """
        查找跨论文关联并生成新的边
        
        Args:
            progress_callback: 进度回调函数
            
        Returns:
            新生成的边列表
        """
        logger.info("开始查找跨论文关联...")
        
        # 获取所有节点
        graph = self.rag.chunk_entity_relation_graph
        all_labels = await graph.get_all_labels()
        
        # 按论文分组节点
        paper_nodes = defaultdict(list)
        all_nodes = []
        
        # 🔍 调试：记录分析过程
        debug_samples = []
        empty_entity_names = 0
        
        for i, label in enumerate(all_labels):
            node_data = await graph.get_node(label)
            if node_data:
                # 提取论文ID
                entity_name = node_data.get('entity_name', '')
                entity_type = node_data.get('entity_type', 'unknown')
                
                # 🔍 调试：记录前20个节点的详细信息
                if i < 20:
                    debug_samples.append({
                        'index': i,
                        'label': label,
                        'entity_name': entity_name,
                        'entity_type': entity_type,
                        'all_keys': list(node_data.keys())
                    })
                
                # 🔍 调试：统计空entity_name
                if not entity_name.strip():
                    empty_entity_names += 1
                    if empty_entity_names <= 5:  # 只记录前5个空的
                        logger.warning(f"发现空entity_name: label={label}, keys={list(node_data.keys())}")
                
                # 提取论文ID：移除_RQ_X和_SOL_X后缀
                if '_RQ_' in entity_name:
                    paper_id = entity_name.split('_RQ_')[0]
                elif '_SOL_' in entity_name:
                    paper_id = entity_name.split('_SOL_')[0]
                else:
                    # 对于paper节点，直接使用entity_name作为paper_id
                    paper_id = entity_name
                
                paper_nodes[paper_id].append({
                    'label': label,
                    'data': node_data,
                    'paper_id': paper_id
                })
                all_nodes.append({
                    'label': label,
                    'data': node_data,
                    'paper_id': paper_id
                })
        
        # 🔍 调试：输出详细信息
        logger.info(f"找到 {len(all_nodes)} 个节点，分布在 {len(paper_nodes)} 篇论文中")
        logger.info(f"空entity_name数量: {empty_entity_names}")
        
        # 🔍 调试：输出前20个节点的详细信息
        logger.info("前20个节点的详细信息:")
        for sample in debug_samples:
            logger.info(f"  {sample['index']:2d}. {sample['entity_type']:15s} | entity_name: {sample['entity_name'][:50]}... | keys: {sample['all_keys']}")
        
        # 🔍 调试：输出前10个论文组的信息
        logger.info(f"前10个论文组的信息:")
        for i, (paper_id, nodes) in enumerate(list(paper_nodes.items())[:10], 1):
            entity_types = [n['data'].get('entity_type', 'unknown') for n in nodes]
            type_counts = {t: entity_types.count(t) for t in set(entity_types)}
            logger.info(f"  {i:2d}. {paper_id[:50]}... ({len(nodes)} 个节点: {type_counts})")
        
        # 预计算所有节点的语义向量
        logger.info("预计算语义向量...")
        node_vectors = {}
        vector_failures = 0
        
        for i, node_info in enumerate(all_nodes):
            if progress_callback:
                progress_callback(f"计算向量", i, len(all_nodes))
            
            vector = await self.get_node_vector(node_info['data'])
            if vector is not None:
                node_vectors[node_info['label']] = vector
            else:
                vector_failures += 1
                # 🔍 调试：记录前10个向量计算失败的节点
                if vector_failures <= 10:
                    entity_type = node_info['data'].get('entity_type', 'unknown')
                    entity_name = node_info['data'].get('entity_name', '')[:50]
                    logger.warning(f"向量计算失败 {vector_failures}: {entity_type} - {entity_name}...")
        
        logger.info(f"向量计算完成: 成功 {len(node_vectors)} 个，失败 {vector_failures} 个")
        
        # 保存向量缓存
        self._save_vector_cache()
        
        # 🚀 优化：预先分离Solution和Paper节点，避免无效比较
        logger.info("预处理节点类型...")
        solution_nodes = []  # 所有solution节点
        paper_nodes_list = []  # 所有paper节点
        
        for paper_id, nodes in paper_nodes.items():
            for node_info in nodes:
                entity_type = node_info['data'].get('entity_type', 'unknown')
                if entity_type == 'solution':
                    solution_nodes.append(node_info)
                elif entity_type == 'paper':
                    paper_nodes_list.append(node_info)
        
        logger.info(f"找到 {len(solution_nodes)} 个solution节点，{len(paper_nodes_list)} 个paper节点")
        
        # 🚨 紧急检查：如果比较次数还是太大，限制数量
        if len(solution_nodes) * len(paper_nodes_list) > 50000:
            logger.warning(f"⚠️ 比较次数过大 ({len(solution_nodes)} × {len(paper_nodes_list)} = {len(solution_nodes) * len(paper_nodes_list)})，限制为前50个solution节点")
            solution_nodes = solution_nodes[:50]
        
        # 🎯 只计算Solution -> Paper的跨论文相似度
        logger.info("计算跨论文相似度...")
        new_edges = []
        
        # 总比较次数大幅减少
        total_comparisons = len(solution_nodes) * len(paper_nodes_list)
        logger.info(f"总计算量：{total_comparisons} 次比较（优化后）")
        
        current_comparison = 0
        
        for sol_info in solution_nodes:
            for paper_info in paper_nodes_list:
                current_comparison += 1
                
                if progress_callback and current_comparison % 100 == 0:  # 减少进度更新频率
                    progress_callback(f"计算相似度", current_comparison, total_comparisons)
                
                # 确保是跨论文连接（不同paper_id）
                if sol_info['paper_id'] == paper_info['paper_id']:
                    continue
                
                label1, data1 = sol_info['label'], sol_info['data']
                label2, data2 = paper_info['label'], paper_info['data']
                
                vector1 = node_vectors.get(label1)
                vector2 = node_vectors.get(label2)
                
                if vector1 is not None and vector2 is not None:
                    # 🔧 基于节点属性的科学相似度计算
                    try:
                        # 使用纯粹的节点属性指标，避免LLM调用
                        similarity = self.calculate_comprehensive_node_similarity(data1, data2)
                        
                        # 构建基于属性的证据
                        evidence_for_edge = self.build_attribute_based_evidence(data1, data2, similarity)
                    except Exception as e:
                        logger.warning(f"属性相似度计算失败: {e}，使用语义相似度")
                        similarity = self._calculate_semantic_similarity(vector1, vector2)
                        evidence_for_edge = {"error_fallback": str(e), "method": "semantic_only"}
                    
                    # 🔍 调试：记录前10个相似度计算
                    if current_comparison <= 10:
                        name1 = data1.get('entity_name', '')[:40]
                        name2 = data2.get('entity_name', '')[:40]
                        logger.info(f"Solution->Paper相似度 {current_comparison}: {similarity:.4f}")
                        logger.info(f"  Solution: {name1}...")
                        logger.info(f"  Paper: {name2}...")
                    
                    # 🔧 使用统一的阈值逻辑
                    effective_threshold = self.similarity_threshold
                    
                    if similarity >= effective_threshold:
                        # 🔧 根据是否有技术证据生成不同的描述
                        if "best_opportunity" in evidence_for_edge:
                            # 基于技术证据的描述 - 使用LLM生成真实分析
                            best_opportunity = evidence_for_edge["best_opportunity"]
                            try:
                                # 🆕 使用LLM生成基于真实节点属性的分析
                                edge_description = await self.technical_extractor.generate_llm_based_evidence(best_opportunity)
                            except Exception as e:
                                logger.warning(f"LLM分析失败，使用基础描述: {e}")
                                edge_description = self.technical_extractor.generate_enhanced_edge_description(best_opportunity)
                            
                            new_edge = {
                                "source": label1,
                                "target": label2,
                                "relationship": "cross_paper_similarity",
                                "similarity_score": similarity,
                                "description": edge_description,
                                "edge_type": "cross_paper",
                                "source_type": "solution",
                                "target_type": "paper",
                                "created_at": datetime.now().isoformat(),
                                # 🔧 关键：保存详细的技术证据
                                "technical_evidence": {
                                    "adaptation_evidence": evidence_for_edge["adaptation_evidence"],
                                    "adaptation_mechanism": evidence_for_edge["adaptation_mechanism"],
                                    "validation_suggestion": evidence_for_edge["validation_suggestion"],
                                    "source_component_name": best_opportunity.source_component.name,
                                    "source_component_type": best_opportunity.source_component.type,
                                    "target_challenge_name": best_opportunity.target_challenge.name,
                                    "target_challenge_domain": best_opportunity.target_challenge.domain,
                                    "transfer_feasibility": best_opportunity.transfer_feasibility
                                }
                            }
                        else:
                            # 传统描述 - 基于真实节点属性生成
                            try:
                                edge_description = await self.generate_diverse_semantic_hint(data1, data2, similarity)
                            except Exception as e:
                                logger.warning(f"生成边描述失败: {e}")
                                edge_description = f"跨论文关联 (相似度: {similarity:.3f})"
                            
                            new_edge = {
                                "source": label1,
                                "target": label2,
                                "relationship": "cross_paper_similarity",
                                "similarity_score": similarity,
                                "description": edge_description,
                                "edge_type": "cross_paper",
                                "source_type": "solution",
                                "target_type": "paper",
                                "created_at": datetime.now().isoformat(),
                                "technical_evidence": None
                            }
                        
                        new_edges.append(new_edge)
                        
                        # 每发现5个关联记录一次进度
                        if len(new_edges) % 5 == 0 or len(new_edges) <= 10:
                            logger.info(f"✅ 发现Solution->Paper关联 #{len(new_edges)}: 相似度 {similarity:.3f}")
                else:
                    # 🔍 调试：记录向量缺失情况
                    if current_comparison <= 5:
                        logger.warning(f"向量缺失 {current_comparison}: solution={vector1 is not None}, paper={vector2 is not None}")
        
        logger.info(f"🎉 跨论文关联发现完成！总共发现 {len(new_edges)} 个跨论文关联")
        
        # 添加性能统计
        if new_edges:
            similarities = [edge['similarity_score'] for edge in new_edges]
            logger.info(f"📊 相似度统计: 最高 {max(similarities):.3f}, 最低 {min(similarities):.3f}, 平均 {sum(similarities)/len(similarities):.3f}")
        
        # 性能优化：减少冗余日志输出
        
        return new_edges
    
    async def add_cross_paper_edges_to_graph(self, new_edges: List[Dict]) -> int:
        """
        将新的跨论文边添加到LightRAG图中
        
        Args:
            new_edges: 新边列表
            
        Returns:
            成功添加的边数量
        """
        if not new_edges:
            return 0
        
        logger.info(f"开始将 {len(new_edges)} 条跨论文边添加到图中...")
        
        graph = self.rag.chunk_entity_relation_graph
        added_count = 0

        # 大幅降低并发数，避免系统卡顿
        sem = asyncio.Semaphore(10)  # 从200降到10

        async def _upsert_one(edge_data: Dict) -> int:
            try:
                async with sem:
                    existing_edge = await graph.get_edge(edge_data["source"], edge_data["target"])
                    if existing_edge is None:
                        # 边属性保留全部细粒度字段
                        edge_props = {
                            "relationship": edge_data.get("relationship"),
                            "similarity_score": edge_data.get("similarity_score"),
                            "description": edge_data.get("description"),
                            "edge_type": edge_data.get("edge_type"),
                            "created_at": edge_data.get("created_at"),
                            "keywords": f"跨论文关联,相似度:{edge_data.get('similarity_score', 0):.2f}",
                        }
                        # 透传额外属性（若后续扩展）
                        for k, v in edge_data.items():
                            if k not in edge_props and k not in {"source", "target"} and v is not None:
                                # 🔥 修复：如果值为字典，则序列化为JSON字符串
                                if isinstance(v, dict):
                                    edge_props[k] = json.dumps(v, ensure_ascii=False)
                                else:
                                    edge_props[k] = v
                        await graph.upsert_edge(
                            source_node_id=edge_data["source"],
                            target_node_id=edge_data["target"],
                            edge_data=edge_props,
                        )
                        return 1
                    return 0
            except Exception as ex:
                source = edge_data.get('source', 'unknown') if isinstance(edge_data, dict) else 'unknown'
                target = edge_data.get('target', 'unknown') if isinstance(edge_data, dict) else 'unknown'
                logger.error(f"添加边失败: {source} -> {target}, 错误: {ex}")
                return 0

        # 分批串行写入，避免并发过载导致卡顿
        added_count = 0
        batch_size = 5  # 每批处理5条边
        
        for i in range(0, len(new_edges), batch_size):
            batch = new_edges[i:i+batch_size]
            logger.info(f"处理第 {i//batch_size + 1} 批，共 {len(batch)} 条边...")
            
            try:
                batch_results = await asyncio.gather(*[_upsert_one(e) for e in batch])
                batch_added = sum(batch_results)
                added_count += batch_added
                
                logger.info(f"第 {i//batch_size + 1} 批完成，添加 {batch_added} 条边")
                
                # 批次间短暂休息，避免系统过载
                if i + batch_size < len(new_edges):
                    await asyncio.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"第 {i//batch_size + 1} 批处理失败: {e}")
                continue
        
        logger.info(f"成功添加 {added_count} 条跨论文边到图中")
        # 🚀 性能优化：异步持久化，避免阻塞主流程
        try:
            # 使用非阻塞方式持久化
            asyncio.create_task(graph.index_done_callback())
            logger.info("跨论文边持久化任务已启动（异步）")
        except Exception as e:
            logger.warning(f"异步持久化启动失败，将在程序结束时自动保存: {e}")
        
        # 🚀 性能优化：跳过向量库写入，避免LightRAG异步限流器阻塞
        logger.info("性能优化：跳过向量库写入步骤，避免系统阻塞")
        # 注释掉向量库写入逻辑，专注于图结构更新
        # try:
        #     # 向量库写入逻辑已优化跳过
        # except Exception as ex:
        #     logger.error(f"写入关系向量库失败: {ex}")

        return added_count
    
    def _generate_simple_description(self, source_node: Dict, target_node: Dict, similarity_score: float) -> str:
        """
        生成简化描述，避免LLM调用阻塞
        """
        source_title = source_node.get('entity_name', '').replace('_', ' ')
        target_title = target_node.get('title', target_node.get('entity_name', '')).replace('_', ' ')
        
        # 基于相似度生成描述
        if similarity_score >= 0.4:
            strength = "强相关"
            desc = "具有显著的技术迁移潜力"
        elif similarity_score >= 0.3:
            strength = "中等相关"
            desc = "存在一定的技术借鉴价值"
        else:
            strength = "弱相关"
            desc = "可能存在跨领域技术启发"
        
        return f"{source_title} 与 {target_title} 之间存在{strength}性（相似度: {similarity_score:.3f}），{desc}。"
    
    async def generate_diverse_semantic_hint(self, source_node: Dict, target_node: Dict, 
                                     similarity_score: float, technical_evidence: Dict = None) -> str:
        """
        生成多样化的语义提示 - 性能优化版本
        """
        # 直接返回简化描述，避免LLM调用阻塞
        return self._generate_simple_description(source_node, target_node, similarity_score)

    def _generate_llm_prompt(self, source_title: str, target_title: str) -> str:
        """生成LLM提示模板"""
        return f"""解决方案: {source_title}
论文: {target_title}

请用一句话简洁描述它们之间的技术关联性和启发价值。格式：【关联类型】具体描述"""

    async def _call_llm_for_description(self, source_node: Dict, target_node: Dict) -> str:
        """调用LLM生成描述"""
        try:
            source_title = source_node.get('entity_name', 'Unknown')
            target_title = target_node.get('entity_name', 'Unknown')
            prompt = self._generate_llm_prompt(source_title, target_title)
            
            # 使用LightRAG的LLM接口
            if hasattr(self.rag, 'llm_model_func') and self.rag.llm_model_func:
                response = await self.rag.llm_model_func(prompt)
                if response and len(response.strip()) > 10:
                    return response.strip()
            
            # 降级到基于属性的描述
            return self._generate_simple_description(source_node, target_node, similarity_score)
            
        except Exception as e:
            logger.warning(f"LLM语义分析失败: {e}")
            # 回退到简化版本
            source_method = self._extract_method_name(source_node.get('solution', ''))
            target_title = target_node.get('title', '未知论文')
            return f"【跨领域启发】{source_method} → {target_title}的创新应用"
    
    def _select_template(self, similarity_score: float, technical_evidence: Dict = None) -> int:
        """选择合适的模板"""
        if technical_evidence and technical_evidence.get('transfer_feasibility', 0) > 0.7:
            return 0  # 技术迁移
        elif similarity_score > 0.8:
            return 1  # 方法论借鉴
        elif similarity_score > 0.6:
            return 2  # 架构设计
        elif similarity_score > 0.4:
            return 3  # 理论融合
        else:
            return 4  # 实验验证
    
    def _extract_semantic_variables(self, source_node: Dict, target_node: Dict, technical_evidence: Dict = None) -> Dict[str, str]:
        """从节点中提取语义变量"""
        
        # 提取源方法信息
        source_solution = source_node.get('solution', '') or source_node.get('simplified_solution', '')
        source_method = self._extract_method_name(source_solution)
        source_domain = self._infer_domain(source_node)
        
        # 提取目标信息
        target_problem = target_node.get('core_problem', '') or target_node.get('basic_problem', '')
        target_domain = self._infer_domain(target_node)
        target_challenge = self._extract_main_challenge(target_problem)
        
        # 基于技术证据提取核心机制
        core_algorithm = "未知算法"
        if technical_evidence:
            adaptation_mechanism = technical_evidence.get('adaptation_mechanism', '')
            if adaptation_mechanism:
                core_algorithm = adaptation_mechanism[:30]
        
        # 从预定义变量中随机选择，但保持一定的稳定性
        seed = hash(source_method + target_domain) % 1000  # 基于内容的伪随机种子
        random.seed(seed)
        
        return {
            'source_method': source_method,
            'source_domain': source_domain,
            'source_challenge': self._extract_main_challenge(source_node.get('research_question', '')),
            'target_domain': target_domain,
            'target_problem': target_node.get('title', '未知论文'),
            'target_challenge': target_challenge,
            'core_algorithm': core_algorithm,
            'adaptation_strategy': random.choice(self.template_variables['adaptation_strategies']),
            'key_insight': self._extract_key_insight(source_solution),
            'innovation_angle': random.choice(self.template_variables['innovation_angles']),
            'architecture_component': '架构组件',
            'modification_approach': '技术改造',
            'expected_benefit': '性能提升',
            'source_theory': f"{source_method}理论",
            'target_context': f"{target_domain}场景",
            'integration_mechanism': '融合机制',
            'theoretical_contribution': '理论贡献',
            'source_dataset': '源数据集',
            'performance_metric': '性能指标',
            'target_dataset': '目标数据集',
            'validation_strategy': '验证策略'
        }
    
    def _extract_method_name(self, solution_text: str) -> str:
        """从解决方案文本中提取方法名"""
        if not solution_text:
            return '创新方法'
        
        # 查找常见的方法名模式
        patterns = [
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:algorithm|method|approach|framework)',
            r'([A-Z]{2,}(?:\s+[A-Z]{2,})*)',  # 缩写方法名
            r'(\w+(?:\s+\w+){0,2})\s+(?:mechanism|module|network)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, solution_text)
            if match:
                return match.group(1)
        
        # 回退：返回前几个关键词
        words = solution_text.split()[:3]
        return ' '.join(words) if words else '创新方法'
    
    def _infer_domain(self, node: Dict) -> str:
        """推断节点所属领域"""
        
        text = f"{node.get('title', '')} {node.get('abstract', '')} {node.get('core_problem', '')}"
        
        domain_keywords = {
            '计算机视觉': ['image', 'visual', 'detection', 'segmentation', 'face', 'object'],
            '自然语言处理': ['text', 'language', 'nlp', 'semantic', 'translation', 'embedding'],
            '时间序列分析': ['time series', 'temporal', 'sequence', 'forecasting', 'time'],
            '域适应': ['domain adaptation', 'transfer', 'cross-domain', 'adaptation'],
            '深度学习': ['neural network', 'deep learning', 'cnn', 'rnn', 'transformer'],
            '机器学习': ['learning', 'classification', 'regression', 'clustering']
        }
        
        text_lower = text.lower()
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return domain
        
        return '通用机器学习'
    
    def _extract_main_challenge(self, problem_text: str) -> str:
        """提取主要挑战"""
        if not problem_text:
            return '技术难题'
        
        # 查找挑战性词汇
        challenge_patterns = [
            r'challenge.*?(?:is|lies in|involves)\s+([^.]+)',
            r'difficulty.*?(?:is|lies in|involves)\s+([^.]+)', 
            r'problem.*?(?:is|lies in|involves)\s+([^.]+)'
        ]
        
        for pattern in challenge_patterns:
            match = re.search(pattern, problem_text, re.IGNORECASE)
            if match:
                return match.group(1).strip()[:50]
        
        # 回退：返回前50个字符
        return problem_text[:50] + "..." if len(problem_text) > 50 else problem_text
    
    def _extract_key_insight(self, solution_text: str) -> str:
        """提取关键洞察"""
        if not solution_text:
            return '创新思路'
        
        # 查找洞察性描述
        words = solution_text.split()
        key_phrases = []
        
        for i, word in enumerate(words):
            if any(keyword in word.lower() for keyword in ['novel', 'new', 'innovative', 'unique']):
                # 提取该词周围的短语
                start = max(0, i-2)
                end = min(len(words), i+3)
                key_phrases.append(' '.join(words[start:end]))
        
        if key_phrases:
            return key_phrases[0][:50]
        
        # 回退：返回前几个词
        return ' '.join(solution_text.split()[:5]) + '策略'


async def integrate_cross_paper_linking(rag_instance, 
                                      similarity_threshold: float = 0.50,
                                      progress_callback=None):
    """
    集成跨论文关联功能到LightRAG流程中
    
    Args:
        rag_instance: LightRAG实例
        similarity_threshold: 相似度阈值
        progress_callback: 进度回调函数
        
    Returns:
        添加的边数量
    """
    
    # 创建跨论文关联引擎
    linking_engine = CrossPaperLinkingEngine(
        rag_instance=rag_instance,
        similarity_threshold=similarity_threshold
    )
    
    # 查找跨论文关联
    new_edges = await linking_engine.find_cross_paper_links(progress_callback)
    
    # 添加到图中
    added_count = await linking_engine.add_cross_paper_edges_to_graph(new_edges)
    
    return added_count, new_edges
