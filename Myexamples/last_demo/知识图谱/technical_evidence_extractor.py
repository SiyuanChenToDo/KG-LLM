#!/usr/bin/env python3
"""
技术证据提取器 - 知识图谱深度集成版
=====================================

充分利用细粒度知识图谱的高质量人工标注信息，从三种节点类型中提取
具体的技术证据，用于建立真正有意义的跨论文连接。

核心改进：
1. 🎯 深度集成node_attributes_api.py，充分利用所有节点属性
2. 📊 基于paper节点的丰富元数据（authors, conference, year等）
3. 🔍 利用research_question的详细问题描述和简化版本
4. ⚙️ 从solution的完整和简化描述中提取技术组件
5. 🧠 智能的跨论文技术迁移机会发现
6. 📝 生成高质量的技术改造证据和验证建议
"""

import re
import json
import asyncio
import os
import sys
from typing import Dict, List, Tuple, Optional, Set, Union
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import logging
from pathlib import Path

# 导入节点属性API
sys.path.append(str(Path(__file__).parent.parent))
from node_attributes_api import (
    get_solution_attributes_by_name, 
    get_paper_attributes_by_name, 
    get_research_question_attributes_by_name,
    get_solution_attributes,
    find_paper_by_title,
    get_paper_research_questions,
    get_research_question_solutions,
    print_node_attributes
)

logger = logging.getLogger(__name__)

@dataclass
class TechnicalComponent:
    """技术组件 - 增强版"""
    name: str                    # 组件名称 
    type: str                   # 组件类型：loss_function, module, algorithm, constraint, etc.
    description: str            # 详细描述
    source_context: str         # 源上下文
    parameters: List[str]       # 参数列表
    mathematical_form: str      # 数学表达（如果有）
    
    # 🆕 新增属性：基于知识图谱的丰富信息
    source_paper_info: Dict = field(default_factory=dict)  # 来源论文信息
    source_solution_info: Dict = field(default_factory=dict)  # 来源解决方案信息
    technical_domain: str = ""   # 技术领域
    complexity_level: str = ""   # 复杂度级别：low, medium, high
    implementation_details: List[str] = field(default_factory=list)  # 实现细节
    related_concepts: List[str] = field(default_factory=list)  # 相关概念
    performance_metrics: List[str] = field(default_factory=list)  # 性能指标
    
@dataclass 
class TechnicalChallenge:
    """技术挑战 - 增强版"""
    name: str                   # 挑战名称
    domain: str                 # 领域
    description: str            # 挑战描述
    constraints: List[str]      # 约束条件
    desired_properties: List[str] # 期望的解决方案特性
    
    # 🆕 新增属性：基于知识图谱的丰富信息
    source_paper_info: Dict = field(default_factory=dict)  # 来源论文信息
    source_rq_info: Dict = field(default_factory=dict)     # 来源研究问题信息
    challenge_category: str = ""  # 挑战类别
    difficulty_level: str = ""    # 难度级别
    research_context: str = ""    # 研究背景
    existing_approaches: List[str] = field(default_factory=list)  # 现有方法
    evaluation_criteria: List[str] = field(default_factory=list)  # 评估标准

@dataclass
class TransferOpportunity:
    """迁移机会 - 增强版"""
    source_component: TechnicalComponent
    target_challenge: TechnicalChallenge
    adaptation_evidence: str    # 改造证据
    adaptation_mechanism: str   # 改造机制
    validation_suggestion: str  # 验证建议
    transfer_feasibility: float # 迁移可行性 (0-1)
    
    # 🆕 新增属性：基于知识图谱的深度分析
    cross_paper_context: Dict = field(default_factory=dict)  # 跨论文上下文
    technical_gap_analysis: str = ""  # 技术差距分析
    innovation_potential: float = 0.0  # 创新潜力 (0-1)
    implementation_complexity: str = ""  # 实现复杂度
    expected_performance_gain: str = ""  # 预期性能提升
    risk_assessment: str = ""  # 风险评估
    related_work_analysis: str = ""  # 相关工作分析

class TechnicalEvidenceExtractor:
    """技术证据提取器 - 知识图谱深度集成版"""
    
    def __init__(self, kg_file_path: Optional[str] = None):
        """初始化技术证据提取器
        
        Args:
            kg_file_path: 知识图谱文件路径，如果为None则使用API自动检测
        """
        self.kg_file_path = kg_file_path
        
        # 🔧 设置技术模式库（保留原有功能）
        self.extraction_stats = {
            "components_extracted": 0,
            "challenges_extracted": 0,
            "opportunities_found": 0
        }
        # 添加缓存机制避免重复提取
        self._component_cache = {}
        self._challenge_cache = {}
        
        # 初始化所有模式
        self._setup_technical_patterns()
        self._setup_component_types()
        
        logger.info("技术证据提取器初始化完成 - 知识图谱深度集成版")
    
    def _setup_enhanced_extractors(self):
        """🆕 设置基于知识图谱的增强提取器"""
        # 论文元数据分析权重
        self.paper_metadata_weights = {
            'title': 0.25,
            'abstract': 0.40, 
            'authors': 0.15,
            'conference': 0.10,
            'year': 0.05,
            'core_problem': 0.05
        }
        
        
        # 性能指标关键词
        self.performance_keywords = [
            'accuracy', 'precision', 'recall', 'f1', 'auc', 'map', 'bleu', 'rouge',
            'perplexity', 'loss', 'error', 'mse', 'mae', 'iou', 'dice', 'psnr'
        ]
    
    def _setup_cross_paper_analysis(self):
        """🆕 设置跨论文分析功能"""
        # 创新潜力评估因子
        self.innovation_factors = {
            'novelty': 0.30,      # 新颖性
            'impact': 0.25,       # 影响力
            'feasibility': 0.20,  # 可行性
            'generality': 0.15,   # 通用性
            'efficiency': 0.10    # 效率
        }
        
        # 风险评估类别
        self.risk_categories = {
            'technical': ['implementation', 'scalability', 'compatibility'],
            'performance': ['degradation', 'instability', 'overfitting'],
            'resource': ['computational', 'memory', 'time'],
            'domain': ['transferability', 'generalization', 'adaptation']
        }
    
    def _setup_evidence_templates(self):
        """🆕 设置证据生成模板"""
        self.evidence_templates = {
            'technical_transfer': [
                "从{source_paper}的{source_method}中提取的{component_type}技术，通过{adaptation}改造后，可以解决{target_paper}中的{target_challenge}问题。",
                "基于{source_conference}发表的{source_method}，其{core_technique}机制为{target_domain}领域的{specific_challenge}提供了新的解决思路。",
                "将{source_authors}提出的{technical_component}与{target_context}相结合，预期能够在{performance_metric}上获得{expected_gain}的提升。"
            ],
            'methodological_insight': [
                "从{source_method}的{key_insight}中获得启发，{target_method}可以采用类似的{strategy}来处理{target_challenge}。",
                "{source_paper}在{source_domain}中的成功经验表明，{approach}对于{target_scenario}具有良好的适用性。"
            ],
            'architectural_adaptation': [
                "借鉴{source_architecture}的{design_principle}，为{target_application}设计{adapted_architecture}，预期能够{benefit}。",
                "通过对{source_model}的{modification_type}改造，使其适应{target_constraints}，同时保持{preserved_properties}。"
            ]
        }
    
    def _setup_technical_patterns(self):
        """设置技术模式识别"""
        # 🔧 知识图谱文件路径（多个可能的位置）
        self.kg_file_paths = [
            '/root/autodl-tmp/LightRAG/data/final_custom_kg_papers.json',
            '/root/autodl-tmp/LightRAG/data/test_custom_kg_papers.json',
            'data/final_custom_kg_papers.json',
            'data/custom_kg_papers.json', 
            'data/test_custom_kg_papers.json',
            'data/custom_kg_fixed.json',
            'data/custom_kg.json',
            '../data/final_custom_kg_papers.json',
            '../data/custom_kg_papers.json', 
            '../data/test_custom_kg_papers.json',
            '../data/custom_kg_fixed.json',
            '../data/custom_kg.json',
            '../../data/final_custom_kg_papers.json',
            '../../data/custom_kg_papers.json', 
            '../../data/test_custom_kg_papers.json',
            '../../data/custom_kg_fixed.json',
            '../../data/custom_kg.json'
        ]
        # 损失函数模式
        self.loss_patterns = [
            r'(\w+(?:\s+\w+)*)\s*loss',
            r'L[_\s]*(\w+)',
            r'(\w+(?:\s+\w+)*)\s*损失',
            r'(\w+(?:\s+\w+)*)\s*constraint',
            r'(\w+(?:\s+\w+)*)\s*regularization'
        ]
        
        # 算法/模块模式  
        self.algorithm_patterns = [
            r'(\w+(?:\s+\w+)*)\s*algorithm',
            r'(\w+(?:\s+\w+)*)\s*module',
            r'(\w+(?:\s+\w+)*)\s*mechanism',
            r'(\w+(?:\s+\w+)*)\s*framework',
            r'(\w+(?:\s+\w+)*)\s*approach',
            r'(\w+(?:\s+\w+)*)\s*method'
        ]
        
        # 数学表达模式
        self.math_patterns = [
            r'([A-Z]_\w+)\s*=\s*([^.]+)',
            r'(\w+)\s*\(\s*([^)]+)\s*\)',
            r'argmin|argmax|min|max|sum|∑|∏'
        ]
        
        # 技术挑战模式
        self.challenge_patterns = [
            r'challenge[s]?\s*(?:of|in|for)?\s*([^.]+)',
            r'problem[s]?\s*(?:of|in|for)?\s*([^.]+)',
            r'issue[s]?\s*(?:of|in|for)?\s*([^.]+)',
            r'limitation[s]?\s*(?:of|in|for)?\s*([^.]+)',
            r'difficulty\s*(?:of|in|for)?\s*([^.]+)'
        ]
    
    def _setup_component_types(self):
        """设置技术组件类型"""
        self.component_keywords = {
            'loss_function': ['loss', 'objective', 'cost', 'regularization', 'penalty'],
            'attention_mechanism': ['attention', 'self-attention', 'cross-attention', 'multi-head'],
            'contrastive_learning': ['contrastive', 'contrast', 'positive', 'negative', 'pairs'],
            'gradient_method': ['gradient', 'backprop', 'optimization', 'descent', 'update'],
            'normalization': ['normalization', 'batch norm', 'layer norm', 'instance norm'],
            'activation': ['relu', 'sigmoid', 'tanh', 'activation', 'nonlinearity'],
            'regularization': ['dropout', 'weight decay', 'regularization', 'constraint'],
            'embedding': ['embedding', 'representation', 'encoding', 'feature'],
            'domain_adaptation': ['adaptation', 'transfer', 'domain', 'shift', 'alignment']
        }
    
    def _setup_domain_keywords(self):
        """设置领域关键词"""
        self.domain_keywords = {
            'computer_vision': ['image', 'video', 'visual', 'CNN', 'convolution', 'detection'],
            'nlp': ['text', 'language', 'word', 'sentence', 'semantic', 'transformer'],
            'multimodal': ['multimodal', 'cross-modal', 'vision-language', 'text-video'],
            'time_series': ['temporal', 'time', 'sequence', 'sequential', 'time-series'],
            'graph': ['graph', 'node', 'edge', 'network', 'relation'],
            'retrieval': ['retrieval', 'search', 'ranking', 'similarity', 'matching'],
            'continual_learning': ['continual', 'incremental', 'lifelong', 'catastrophic'],
            'domain_adaptation': ['domain', 'adaptation', 'transfer', 'source', 'target']
        }

    def _get_related_paper_info(self, entity_name: str) -> Optional[Dict]:
        """获取与节点相关的论文信息"""
        try:
            # 从entity_name推导论文ID
            if '_SOL_' in entity_name:
                paper_id = entity_name.split('_SOL_')[0]
            elif '_RQ_' in entity_name:
                paper_id = entity_name.split('_RQ_')[0]
            else:
                paper_id = entity_name
            
            # 尝试通过API获取论文信息
            papers = find_paper_by_title(paper_id)
            if papers:
                return papers[0]
            
            # 如果找不到，尝试直接获取paper属性
            return get_paper_attributes_by_name('') or {}
            
        except Exception as e:
            logger.warning(f"获取论文信息失败: {e}")
            return None
    
    def _assess_complexity(self, text: str) -> str:
        """简化的复杂度评估"""
        return 'medium'  # 统一返回中等复杂度
    
    def _extract_performance_metrics(self, text: str) -> List[str]:
        """提取性能指标"""
        metrics = []
        text_lower = text.lower()
        
        for metric in self.performance_keywords:
            if metric in text_lower:
                metrics.append(metric)
        
        return list(set(metrics))  # 去重
    
    def _extract_implementation_details(self, text: str) -> List[str]:
        """提取实现细节"""
        details = []
        
        # 查找实现相关的模式
        implementation_patterns = [
            r'implement[ed]?\s+([^.]+)',
            r'use[sd]?\s+([^.]+)',
            r'apply\s+([^.]+)',
            r'employ[ed]?\s+([^.]+)',
            r'adopt[ed]?\s+([^.]+)'
        ]
        
        for pattern in implementation_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                detail = match.group(1)
                if detail and isinstance(detail, str):
                    detail = detail.strip()
                    if len(detail) > 5 and len(detail) < 100:
                        details.append(detail)
        
        return details[:5]  # 限制数量
    
    def _extract_related_concepts(self, text: str) -> List[str]:
        """提取相关概念"""
        concepts = []
        
        # 查找相关概念的模式
        concept_patterns = [
            r'based\s+on\s+([^.]+)',
            r'inspired\s+by\s+([^.]+)',
            r'similar\s+to\s+([^.]+)',
            r'related\s+to\s+([^.]+)',
            r'extends?\s+([^.]+)'
        ]
        
        for pattern in concept_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                concept = match.group(1)
                if concept and isinstance(concept, str):
                    concept = concept.strip()
                    if len(concept) > 3 and len(concept) < 50:
                        concepts.append(concept)
        
        return concepts[:5]  # 限制数量
    
    def _enrich_component_context(self, component: TechnicalComponent, solution_data: Dict, paper_info: Optional[Dict]):
        """丰富组件上下文信息"""
        component.source_context = solution_data.get('entity_name', '')
        component.source_solution_info = solution_data
        component.source_paper_info = paper_info or {}
        
        # 设置技术领域
        component.technical_domain = 'general'
        
    
    def extract_technical_components(self, solution_data: Dict) -> List[TechnicalComponent]:
        """
        从solution节点中提取技术组件
        
        Args:
            solution_data: solution节点数据
            
        Returns:
            List[TechnicalComponent]: 技术组件列表
        """
        try:
            # 🔧 使用节点属性API获取真实solution属性
            entity_name = solution_data.get('entity_name', '')
            
            # 🔧 节点存在性验证
            if not entity_name or not entity_name.strip():
                logger.warning(f"solution节点缺少entity_name")
                return []
            
            # 🔧 缓存机制：先检查缓存
            cache_key = f"components_{entity_name}"
            if cache_key in self._component_cache:
                logger.debug(f"使用缓存的技术组件 for {entity_name}")
                return self._component_cache[cache_key]
            
            solution_attrs = get_solution_attributes_by_name(entity_name)
            
            if solution_attrs:
                # 基于真实属性提取技术组件
                full_solution = solution_attrs.get('full_solution', '')
                simplified_solution = solution_attrs.get('simplified_solution', '')
                components = self._extract_real_technical_components(full_solution, simplified_solution, entity_name, solution_attrs)
                
                # 缓存结果
                self._component_cache[cache_key] = components
                logger.info(f"提取到 {len(components)} 个技术组件 from {entity_name}")
                
                return components
            else:
                logger.warning(f"无法获取solution属性: {entity_name}")
                # 缓存空结果，避免重复查询
                self._component_cache[cache_key] = []
                return []
                
        except Exception as e:
            logger.error(f"技术组件提取失败: {e}")
            return []

    def _extract_real_technical_components(self, full_solution: str, simplified_solution: str, 
                                         entity_name: str, solution_data: Dict) -> List[TechnicalComponent]:
        """从真实solution描述中提取技术组件"""
        components = []
        
        # 分析simplified_solution中的关键技术点
        if simplified_solution:
            tech_points = [point.strip() for point in simplified_solution.split(';') if point.strip()]
            
            for point in tech_points:
                # 提取技术组件名称和类型
                component_name, component_type = self._parse_technical_point(point)
                
                if component_name:
                    # 从full_solution中找到相关描述
                    detailed_desc = self._find_detailed_description(component_name, full_solution)
                    
                    component = TechnicalComponent(
                        name=component_name,
                        type=component_type,
                        description=detailed_desc or point,
                        source_context=f"来自 {entity_name}",
                        parameters=self._extract_parameters_from_description(detailed_desc or point),
                        mathematical_form=self._extract_mathematical_form(detailed_desc or point),
                        source_paper_info=self._get_paper_info_from_solution_data(solution_data),
                        technical_domain='general',
                        complexity_level='medium'
                    )
                    components.append(component)
        
        return components

    def _parse_technical_point(self, tech_point: str) -> Tuple[str, str]:
        """解析技术点，提取组件名称和类型"""
        tech_point = tech_point.lower().strip()
        
        # 技术类型映射
        type_mappings = {
            'loss': 'loss_function',
            'divergence': 'loss_function', 
            'distance': 'loss_function',
            'pool': 'data_structure',
            'representation': 'feature_representation',
            'embedding': 'feature_representation',
            'network': 'neural_architecture',
            'module': 'neural_module',
            'mechanism': 'algorithm',
            'strategy': 'algorithm',
            'framework': 'system_architecture',
            'selector': 'component',
            'encoder': 'neural_module',
            'normalization': 'preprocessing'
        }
        
        # 识别技术类型
        component_type = 'algorithm'  # 默认类型
        for keyword, mapped_type in type_mappings.items():
            if keyword in tech_point:
                component_type = mapped_type
                break
        
        # 提取组件名称（保留原始描述的关键部分）
        component_name = tech_point.replace(';', '').strip()
        
        return component_name, component_type

    def _find_detailed_description(self, component_name: str, full_solution: str) -> str:
        """从完整solution中找到组件的详细描述"""
        # 查找包含组件名称的句子段落
        sentences = full_solution.split('.')
        relevant_sentences = []
        
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in component_name.lower().split()):
                relevant_sentences.append(sentence.strip())
        
        return '. '.join(relevant_sentences[:2])  # 返回前两个相关句子

    def _get_paper_info_from_solution_data(self, solution_data: Dict) -> Dict:
        """从solution数据中获取论文信息"""
        source_id = solution_data.get('source_id', '')
        if source_id:
            try:
                # 通过source_id查找对应的paper信息
                paper_attrs = get_paper_attributes_by_name(f"paper_{source_id}")
                if paper_attrs:
                    return {
                        'title': paper_attrs.get('title', ''),
                        'authors': paper_attrs.get('authors', ''),
                        'conference': paper_attrs.get('conference', ''),
                        'year': paper_attrs.get('year', ''),
                        'abstract': paper_attrs.get('abstract', '')
                    }
            except:
                pass
        return {}

    def _extract_loss_functions(self, text: str) -> List[TechnicalComponent]:
        """提取损失函数组件"""
        components = []
        
        for pattern in self.loss_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                name = match.group(1)
                if name and isinstance(name, str):
                    name = name.strip()
                    if len(name) > 2:  # 过滤过短的匹配
                        # 提取周围的上下文
                        start = max(0, match.start() - 50)
                        end = min(len(text), match.end() + 100)
                        context = text[start:end].strip()
                        
                        # 查找数学表达
                        math_form = self._extract_math_near_text(text, match.start(), match.end())
                        
                        component = TechnicalComponent(
                            name=name,
                            type='loss_function',
                            description=context,
                            source_context='',
                            parameters=self._extract_parameters(context),
                            mathematical_form=math_form
                        )
                        components.append(component)
        
        return components
    
    def _extract_algorithms(self, text: str) -> List[TechnicalComponent]:
        """提取算法/模块组件"""
        components = []
        
        for pattern in self.algorithm_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                name = match.group(1)
                if name and isinstance(name, str):
                    name = name.strip()
                    if len(name) > 3:  # 过滤过短的匹配
                        # 判断组件类型
                        component_type = self._classify_component_type(name, text)
                        
                        # 提取周围的上下文
                        start = max(0, match.start() - 30)
                        end = min(len(text), match.end() + 80)
                        context = text[start:end].strip()
                        
                        component = TechnicalComponent(
                            name=name,
                            type=component_type,
                            description=context,
                            source_context='',
                            parameters=self._extract_parameters(context),
                            mathematical_form=''
                        )
                        components.append(component)
        
        return components
    
    def _extract_constraints(self, text: str) -> List[TechnicalComponent]:
        """提取约束条件"""
        components = []
        
        # 查找约束模式
        constraint_patterns = [
            r'constraint[s]?\s*(?:that|:)?\s*([^.]+)',
            r'subject\s*to\s*([^.]+)',
            r'such\s*that\s*([^.]+)',
            r'ensure[s]?\s*(?:that)?\s*([^.]+)',
            r'guarantee[s]?\s*(?:that)?\s*([^.]+)'
        ]
        
        for pattern in constraint_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                constraint_desc = match.group(1)
                if constraint_desc and isinstance(constraint_desc, str):
                    constraint_desc = constraint_desc.strip()
                    if len(constraint_desc) > 10:
                        component = TechnicalComponent(
                            name=f"Constraint: {constraint_desc[:50]}...",
                            type='constraint',
                            description=constraint_desc,
                            source_context='',
                            parameters=[],
                            mathematical_form=''
                        )
                        components.append(component)
        
        return components
    
    def _classify_component_type(self, name: str, context: str) -> str:
        """根据名称和上下文分类组件类型"""
        name_lower = name.lower()
        context_lower = context.lower()
        
        for comp_type, keywords in self.component_keywords.items():
            for keyword in keywords:
                if keyword in name_lower or keyword in context_lower:
                    return comp_type
        
        return 'algorithm'  # 默认类型
    
    def _extract_parameters(self, text: str) -> List[str]:
        """从文本中提取参数"""
        parameters = []
        
        # 查找参数模式
        param_patterns = [
            r'parameter[s]?\s*([^.]+)',
            r'with\s+([a-zA-Z_]\w*)\s*=',
            r'where\s+([a-zA-Z_]\w*)\s+(?:is|represents)',
            r'([a-zA-Z_]\w*)\s*∈\s*[^,.]+'
        ]
        
        for pattern in param_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                param = match.group(1)
                if param and isinstance(param, str):
                    param = param.strip()
                    if len(param) <= 20:  # 过滤过长的匹配
                        parameters.append(param)
        
        return list(set(parameters))  # 去重
    
    def _extract_math_near_text(self, text: str, start: int, end: int) -> str:
        """提取指定位置附近的数学表达"""
        # 在匹配位置前后200字符内查找数学表达
        search_start = max(0, start - 200)
        search_end = min(len(text), end + 200)
        search_text = text[search_start:search_end]
        
        for pattern in self.math_patterns:
            match = re.search(pattern, search_text)
            if match:
                return match.group(0)
        
        return ''

    def extract_technical_challenges(self, paper_data: Dict, rq_data: Optional[Dict] = None) -> List[TechnicalChallenge]:
        """从paper/RQ节点中提取技术挑战 - 知识图谱增强版
        
        Args:
            paper_data: paper节点数据（包含完整的知识图谱属性）
            rq_data: research_question节点数据（可选，用于更精确的挑战提取）
            
        Returns:
            技术挑战列表（包含丰富的上下文信息）
        """
        challenges = []
        
        # 获取相关文本 - 充分利用知识图谱属性
        entity_type = paper_data.get('entity_type', '')
        
        if entity_type == 'paper':
            text_sources = {
                'core_problem': paper_data.get('core_problem', ''),
                'basic_problem': paper_data.get('basic_problem', ''),
                'abstract': paper_data.get('abstract', ''),
                'title': paper_data.get('title', ''),
                'preliminary_innovation_analysis': paper_data.get('preliminary_innovation_analysis', ''),
                'authors': ', '.join(paper_data.get('authors', [])) if isinstance(paper_data.get('authors'), list) else str(paper_data.get('authors', '')),
                'conference': paper_data.get('conference', ''),
                'year': str(paper_data.get('year', ''))
            }
        elif rq_data:
            text_sources = {
                'research_question': rq_data.get('research_question', ''),
                'simplified_research_question': rq_data.get('simplified_research_question', ''),
                'paper_context': paper_data.get('abstract', ''),
                'paper_title': paper_data.get('title', '')
            }
        else:
            text_sources = {}
        
        # 合并所有文本
        full_text = ' '.join(filter(None, text_sources.values()))
        
        if not full_text.strip():
            logger.warning(f"节点 {paper_data.get('entity_name', 'Unknown')} 缺少文本内容")
            return challenges
        
        # 识别领域和背景
        domain = self._identify_domain(full_text)
        research_context = self._extract_research_context(paper_data, rq_data)
        
        # 1. 提取明确提到的挑战（增强版）
        explicit_challenges = self._extract_explicit_challenges_enhanced(text_sources, domain, paper_data, rq_data)
        challenges.extend(explicit_challenges)
        
        # 2. 从问题描述中推断隐含挑战（增强版）
        implicit_challenges = self._infer_implicit_challenges_enhanced(text_sources, domain, paper_data, rq_data)
        challenges.extend(implicit_challenges)
        
        # 为每个挑战补充丰富的上下文信息
        for challenge in challenges:
            self._enrich_challenge_context(challenge, paper_data, rq_data)
        
        # 更新统计信息
        self.extraction_stats["challenges_extracted"] += len(challenges)
        
        logger.debug(f"从{entity_type}提取到 {len(challenges)} 个增强技术挑战")
        return challenges
    
    def _identify_domain(self, text: str) -> str:
        """简化的领域识别"""
        return 'general'  # 统一返回通用领域
    
    def _identify_domain_enhanced(self, text: str, paper_data: Dict) -> str:
        """增强的领域识别 - 结合论文元数据"""
        # 基础文本领域识别
        text_domain = self._identify_domain(text)
        
        # 基于会议/期刊的领域推断
        conference = paper_data.get('conference', '').upper()
        venue_domain_mapping = {
            'CVPR': 'computer_vision', 'ICCV': 'computer_vision', 'ECCV': 'computer_vision',
            'ACL': 'nlp', 'EMNLP': 'nlp', 'NAACL': 'nlp', 'COLING': 'nlp',
            'ICML': 'general_ml', 'NIPS': 'general_ml', 'ICLR': 'general_ml',
            'SIGIR': 'retrieval', 'WWW': 'retrieval', 'CIKM': 'retrieval'
        }
        
        for venue_key, domain in venue_domain_mapping.items():
            if venue_key in conference:
                return domain
        
        return text_domain
    
    def _extract_research_context(self, paper_data: Dict, rq_data: Optional[Dict]) -> str:
        """🆕 提取研究背景上下文"""
        context_parts = []
        
        # 论文背景
        if paper_data:
            authors = paper_data.get('authors', [])
            if isinstance(authors, list) and authors:
                context_parts.append(f"作者团队: {', '.join(authors[:3])}")
            
            conference = paper_data.get('conference', '')
            year = paper_data.get('year', '')
            if conference and year:
                context_parts.append(f"发表于: {conference} {year}")
        
        # 研究问题背景
        if rq_data:
            rq_text = rq_data.get('research_question', '')
            if rq_text:
                context_parts.append(f"核心研究问题: {rq_text[:100]}...")
        
        return ' | '.join(context_parts)
    
    def _extract_explicit_challenges_enhanced(self, text_sources: Dict[str, str], domain: str, 
                                            paper_data: Dict, rq_data: Optional[Dict]) -> List[TechnicalChallenge]:
        """🆕 增强版明确挑战提取"""
        challenges = []
        
        for source_type, text in text_sources.items():
            if not text:
                continue
                
            for pattern in self.challenge_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    challenge_desc = match.group(1)
                    if challenge_desc and isinstance(challenge_desc, str):
                        challenge_desc = challenge_desc.strip()
                        if len(challenge_desc) > 10:
                            # 提取约束条件
                            constraints = self._extract_constraints_from_challenge(challenge_desc)
                            
                            # 提取期望特性
                            desired_props = self._extract_desired_properties(challenge_desc)
                            
                            # 评估挑战难度
                            difficulty = self._assess_challenge_difficulty(challenge_desc, paper_data)
                            
                            challenge = TechnicalChallenge(
                                name=f"{domain} Challenge: {challenge_desc[:30]}...",
                                domain=domain,
                                description=challenge_desc,
                                constraints=constraints,
                                desired_properties=desired_props,
                                source_paper_info=paper_data,
                                source_rq_info=rq_data or {},
                                challenge_category=self._categorize_challenge(challenge_desc),
                                difficulty_level=difficulty,
                                research_context=self._extract_research_context(paper_data, rq_data)
                        )
                        challenges.append(challenge)
        
        return challenges
    
    def _infer_implicit_challenges_enhanced(self, text_sources: Dict[str, str], domain: str,
                                          paper_data: Dict, rq_data: Optional[Dict]) -> List[TechnicalChallenge]:
        """🆕 增强版隐含挑战推断"""
        challenges = []
        
        # 基于领域的常见挑战模式（扩展版）
        domain_challenge_patterns = {
            'domain_adaptation': [
                'domain shift', 'distribution mismatch', 'transferability', 
                'domain gap', 'source-target alignment', 'cross-domain generalization'
            ],
            'multimodal': [
                'modality gap', 'alignment', 'fusion', 'correspondence',
                'cross-modal understanding', 'semantic consistency'
            ],
            'computer_vision': [
                'occlusion', 'illumination variation', 'scale variation',
                'viewpoint changes', 'background clutter', 'object detection accuracy'
            ],
            'nlp': [
                'semantic ambiguity', 'context understanding', 'long-range dependencies',
                'out-of-vocabulary words', 'syntactic complexity'
            ]
        }
        
        full_text = ' '.join(filter(None, text_sources.values())).lower()
        
        if domain in domain_challenge_patterns:
            for challenge_keyword in domain_challenge_patterns[domain]:
                if challenge_keyword.lower() in full_text:
                    challenge = TechnicalChallenge(
                        name=f"{domain}: {challenge_keyword}",
                        domain=domain,
                        description=f"Implicit challenge related to {challenge_keyword} in {domain}",
                        constraints=[],
                        desired_properties=[],
                        source_paper_info=paper_data,
                        source_rq_info=rq_data or {},
                        challenge_category='implicit',
                        difficulty_level='medium',
                        research_context=self._extract_research_context(paper_data, rq_data)
                    )
                    challenges.append(challenge)
        
        return challenges
    
    def _assess_challenge_difficulty(self, challenge_desc: str, paper_data: Dict) -> str:
        """🆕 评估挑战难度"""
        difficulty_indicators = {
            'high': ['novel', 'unprecedented', 'unsolved', 'fundamental', 'complex', 'challenging'],
            'medium': ['difficult', 'non-trivial', 'significant', 'important', 'substantial'],
            'low': ['simple', 'straightforward', 'basic', 'minor', 'incremental']
        }
        
        challenge_lower = challenge_desc.lower()
        
        for level, indicators in difficulty_indicators.items():
            for indicator in indicators:
                if indicator in challenge_lower:
                    return level
        
        # 基于发表会议推断难度
        conference = paper_data.get('conference', '').upper()
        top_venues = ['NIPS', 'ICML', 'ICLR', 'CVPR', 'ICCV', 'ECCV', 'ACL', 'EMNLP']
        if any(venue in conference for venue in top_venues):
            return 'high'
        
        return 'medium'
    
    def _categorize_challenge(self, challenge_desc: str) -> str:
        """🆕 挑战分类"""
        categories = {
            'data': ['data', 'dataset', 'annotation', 'labeling', 'collection'],
            'model': ['model', 'architecture', 'network', 'representation'],
            'optimization': ['training', 'optimization', 'convergence', 'gradient'],
            'evaluation': ['evaluation', 'metric', 'benchmark', 'assessment'],
            'scalability': ['scalability', 'efficiency', 'speed', 'memory'],
            'generalization': ['generalization', 'robustness', 'transfer', 'adaptation']
        }
        
        challenge_lower = challenge_desc.lower()
        
        for category, keywords in categories.items():
            if any(keyword in challenge_lower for keyword in keywords):
                return category
        
        return 'general'
    
    def _enrich_challenge_context(self, challenge: TechnicalChallenge, paper_data: Dict, rq_data: Optional[Dict]):
        """🆕 丰富挑战上下文信息"""
        # 设置来源信息
        challenge.source_paper_info = paper_data
        challenge.source_rq_info = rq_data or {}
        
        # 提取现有方法
        if paper_data:
            abstract = paper_data.get('abstract', '')
            challenge.existing_approaches = self._extract_existing_approaches(abstract)
            
            # 设置评估标准
            challenge.evaluation_criteria = self._extract_evaluation_criteria(abstract)
            
            # 设置研究动机
            challenge.research_motivation = self._extract_research_motivation(paper_data)
    
    def _extract_existing_approaches(self, text: str) -> List[str]:
        """🆕 提取现有方法"""
        approaches = []
        approach_patterns = [
            r'existing\s+(?:methods?|approaches?)\s+([^.]+)',
            r'previous\s+(?:work|studies?)\s+([^.]+)',
            r'traditional\s+(?:methods?|approaches?)\s+([^.]+)',
            r'current\s+(?:methods?|approaches?)\s+([^.]+)'
        ]
        
        for pattern in approach_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                approach = match.group(1)
                if approach and isinstance(approach, str):
                    approach = approach.strip()
                    if len(approach) > 10 and len(approach) < 100:
                        approaches.append(approach)
        
        return approaches[:3]  # 限制数量
    
    def _extract_evaluation_criteria(self, text: str) -> List[str]:
        """🆕 提取评估标准"""
        criteria = []
        criteria_patterns = [
            r'evaluat(?:e|ed|ion)\s+(?:using|with|on)\s+([^.]+)',
            r'measur(?:e|ed)\s+(?:using|with|by)\s+([^.]+)',
            r'assess(?:ed)?\s+(?:using|with|by)\s+([^.]+)'
        ]
        
        for pattern in criteria_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                criterion = match.group(1)
                if criterion and isinstance(criterion, str):
                    criterion = criterion.strip()
                    if len(criterion) > 5 and len(criterion) < 50:
                        criteria.append(criterion)
        
        return criteria[:3]  # 限制数量
    
    def _extract_research_motivation(self, paper_data: Dict) -> str:
        """🆕 提取研究动机"""
        motivation_sources = [
            paper_data.get('core_problem', ''),
            paper_data.get('basic_problem', ''),
            paper_data.get('preliminary_innovation_analysis', '')
        ]
        
        motivation_text = ' '.join(filter(None, motivation_sources))
        
        # 查找动机相关的句子
        motivation_patterns = [
            r'motivat(?:ed|ion)\s+by\s+([^.]+)',
            r'inspired\s+by\s+([^.]+)',
            r'aim(?:s|ed)?\s+to\s+([^.]+)',
            r'goal\s+(?:is|was)\s+to\s+([^.]+)'
        ]
        
        for pattern in motivation_patterns:
            match = re.search(pattern, motivation_text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return motivation_text[:200] + '...' if len(motivation_text) > 200 else motivation_text
    
    def _extract_explicit_challenges(self, text: str, domain: str) -> List[TechnicalChallenge]:
        """提取明确提到的技术挑战"""
        challenges = []
        
        for pattern in self.challenge_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                challenge_desc = match.group(1)
                if challenge_desc and isinstance(challenge_desc, str):
                    challenge_desc = challenge_desc.strip()
                    if len(challenge_desc) > 10:
                        # 提取约束条件
                        constraints = self._extract_constraints_from_challenge(challenge_desc)
                        
                        # 提取期望特性
                        desired_props = self._extract_desired_properties(challenge_desc)
                        
                        challenge = TechnicalChallenge(
                            name=f"{domain} Challenge: {challenge_desc[:30]}...",
                            domain=domain,
                            description=challenge_desc,
                            constraints=constraints,
                            desired_properties=desired_props
                        )
                        challenges.append(challenge)
        
        return challenges
    
    def _infer_implicit_challenges(self, text: str, domain: str) -> List[TechnicalChallenge]:
        """从问题描述中推断隐含的技术挑战"""
        challenges = []
        
        # 基于领域的常见挑战模式
        domain_challenge_patterns = {
            'domain_adaptation': [
                'domain shift', 'distribution mismatch', 'transferability', 
                'domain gap', 'source-target alignment'
            ],
            'multimodal': [
                'modality gap', 'alignment', 'fusion', 'correspondence',
                'cross-modal understanding'
            ],
            'continual_learning': [
                'catastrophic forgetting', 'plasticity-stability', 
                'memory efficiency', 'task interference'
            ],
            'time_series': [
                'temporal dependency', 'drift', 'non-stationarity',
                'sequence modeling', 'long-term dependency'
            ]
        }
        
        if domain in domain_challenge_patterns:
            for challenge_keyword in domain_challenge_patterns[domain]:
                if challenge_keyword.lower() in text.lower():
                    challenge = TechnicalChallenge(
                        name=f"{domain}: {challenge_keyword}",
                        domain=domain,
                        description=f"Implicit challenge related to {challenge_keyword}",
                        constraints=[],
                        desired_properties=[]
                    )
                    challenges.append(challenge)
        
        return challenges
    
    def _extract_constraints_from_challenge(self, challenge_desc: str) -> List[str]:
        """从挑战描述中提取约束条件"""
        constraints = []
        
        constraint_indicators = [
            'must', 'should', 'cannot', 'limited', 'restricted',
            'without', 'only', 'require', 'need'
        ]
        
        sentences = challenge_desc.split('.')
        for sentence in sentences:
            for indicator in constraint_indicators:
                if indicator in sentence.lower():
                    constraints.append(sentence.strip())
                    break
        
        return constraints
    
    def _extract_desired_properties(self, challenge_desc: str) -> List[str]:
        """从挑战描述中提取期望的解决方案特性"""
        properties = []
        
        property_indicators = [
            'robust', 'efficient', 'accurate', 'fast', 'scalable',
            'generalizable', 'transferable', 'adaptive', 'stable'
        ]
        
        for prop in property_indicators:
            if prop in challenge_desc.lower():
                properties.append(prop)
        
        return properties

    def find_enhanced_transfer_opportunities(self, 
                                           components: List[TechnicalComponent],
                                           challenges: List[TechnicalChallenge]) -> List[TransferOpportunity]:
        """
        发现技术组件到挑战的迁移机会
        
        Args:
            components: 技术组件列表
            challenges: 技术挑战列表
            
        Returns:
            迁移机会列表
        """
        opportunities = []
        
        for component in components:
            for challenge in challenges:
                # 计算迁移可行性
                feasibility = self._calculate_transfer_feasibility(component, challenge)
                
                if feasibility > 0.3:  # 可行性阈值
                    # 生成改造证据
                    adaptation_evidence = self._generate_adaptation_evidence(component, challenge)
                    
                    # 生成改造机制
                    adaptation_mechanism = self._generate_adaptation_mechanism(component, challenge)
                    
                    # 生成验证建议
                    validation_suggestion = self._generate_validation_suggestion(component, challenge)
                    
                    opportunity = TransferOpportunity(
                        source_component=component,
                        target_challenge=challenge,
                        adaptation_evidence=adaptation_evidence,
                        adaptation_mechanism=adaptation_mechanism,
                        validation_suggestion=validation_suggestion,
                        transfer_feasibility=feasibility
                    )
                    opportunities.append(opportunity)
        
        # 按可行性排序
        opportunities.sort(key=lambda x: x.transfer_feasibility, reverse=True)
        
        logger.debug(f"发现 {len(opportunities)} 个迁移机会")
        return opportunities
    
    def _calculate_transfer_feasibility(self, 
                                      component: TechnicalComponent, 
                                      challenge: TechnicalChallenge) -> float:
        """计算迁移可行性"""
        feasibility = 0.0
        
        # 1. 类型匹配度
        type_compatibility = self._get_type_challenge_compatibility(component.type, challenge.domain)
        feasibility += 0.4 * type_compatibility
        
        # 2. 语义相似度
        semantic_sim = self._calculate_text_similarity(
            component.description, challenge.description
        )
        feasibility += 0.3 * semantic_sim
        
        # 3. 参数适配性
        param_adaptability = self._assess_parameter_adaptability(component, challenge)
        feasibility += 0.2 * param_adaptability
        
        # 4. 约束兼容性
        constraint_compatibility = self._assess_constraint_compatibility(component, challenge)
        feasibility += 0.1 * constraint_compatibility
        
        return min(1.0, feasibility)
    
    def _get_type_challenge_compatibility(self, component_type: str, challenge_domain: str) -> float:
        """获取组件类型与挑战领域的兼容性"""
        compatibility_matrix = {
            # 损失函数相关
            ('loss_function', 'domain_adaptation'): 0.9,
            ('loss_function', 'continual_learning'): 0.8,
            ('loss_function', 'multimodal'): 0.7,
            ('loss_function', 'time_series'): 0.6,
            
            # 注意力机制相关
            ('attention_mechanism', 'multimodal'): 0.9,
            ('attention_mechanism', 'nlp'): 0.9,
            ('attention_mechanism', 'computer_vision'): 0.8,
            ('attention_mechanism', 'time_series'): 0.7,
            
            # 对比学习相关
            ('contrastive_learning', 'domain_adaptation'): 0.9,
            ('contrastive_learning', 'multimodal'): 0.8,
            ('contrastive_learning', 'retrieval'): 0.9,
            
            # 其他核心组件
            ('embedding', 'multimodal'): 0.8,
            ('embedding', 'nlp'): 0.9,
            ('embedding', 'graph'): 0.8,
            ('regularization', 'domain_adaptation'): 0.7,
            ('regularization', 'continual_learning'): 0.8,
            ('gradient_method', 'optimization'): 0.9,
        }
        
        return compatibility_matrix.get((component_type, challenge_domain), 0.5)
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """计算文本语义相似度（简化版）"""
        if not text1 or not text2:
            return 0.0
        
        # 简单的词汇重叠度
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def _assess_parameter_adaptability(self, 
                                     component: TechnicalComponent, 
                                     challenge: TechnicalChallenge) -> float:
        """评估参数的适配性"""
        if not component.parameters:
            return 0.5  # 没有参数时返回中等适配性
        
        # 检查参数是否可以适配到目标领域
        adaptable_params = 0
        total_params = len(component.parameters)
        
        for param in component.parameters:
            # 简单启发式：长参数名通常更具体，适配性较低
            if isinstance(param, str) and len(param) <= 10:
                adaptable_params += 1
        
        return adaptable_params / total_params if total_params > 0 else 0.5
    
    def _assess_constraint_compatibility(self, 
                                       component: TechnicalComponent, 
                                       challenge: TechnicalChallenge) -> float:
        """评估约束兼容性"""
        if not challenge.constraints:
            return 1.0  # 没有约束时完全兼容
        
        # 检查组件是否能满足挑战的约束
        compatibility_score = 1.0
        
        for constraint in challenge.constraints:
            # 简单启发式：检查是否有冲突的关键词
            conflict_keywords = ['cannot', 'must not', 'impossible', 'limited']
            for keyword in conflict_keywords:
                if keyword in constraint.lower() and keyword in component.description.lower():
                    compatibility_score *= 0.7  # 降低兼容性
        
        return compatibility_score
    
    def _generate_adaptation_evidence(self, 
                                    component: TechnicalComponent, 
                                    challenge: TechnicalChallenge) -> str:
        """生成改造证据"""
        evidence_parts = []
        
        # 1. 核心机制说明
        evidence_parts.append(f"源方法 '{component.name}' 的核心机制是 {component.description[:100]}...")
        
        # 2. 目标挑战分析
        evidence_parts.append(f"目标挑战 '{challenge.name}' 需要解决 {challenge.description[:100]}...")
        
        # 3. 迁移理由
        if component.type == 'loss_function' and 'adaptation' in challenge.domain:
            evidence_parts.append("损失函数类型的组件通常具有良好的领域适配能力。")
        elif component.type == 'attention_mechanism' and challenge.domain in ['multimodal', 'nlp']:
            evidence_parts.append("注意力机制在多模态和自然语言处理任务中具有通用性。")
        else:
            evidence_parts.append(f"{component.type} 类型的组件可以通过适当修改应用于 {challenge.domain} 领域。")
        
        return " ".join(evidence_parts)
    
    def _generate_adaptation_mechanism(self, 
                                     component: TechnicalComponent, 
                                     challenge: TechnicalChallenge) -> str:
        """生成具体的改造机制"""
        mechanisms = []
        
        # 基于组件类型生成具体的改造建议
        if component.type == 'loss_function':
            if 'domain' in challenge.domain:
                mechanisms.append(f"将 {component.name} 扩展为领域自适应版本，添加域判别器约束")
            elif 'temporal' in challenge.description.lower():
                mechanisms.append(f"在 {component.name} 中引入时序一致性约束，使用滑动窗口机制")
            else:
                mechanisms.append(f"修改 {component.name} 的正负样本定义，适应目标任务的特殊需求")
        
        elif component.type == 'attention_mechanism':
            if 'cross-modal' in challenge.description.lower():
                mechanisms.append(f"将 {component.name} 扩展为跨模态注意力，处理不同模态间的对齐")
            else:
                mechanisms.append(f"调整 {component.name} 的注意力权重计算，增加对目标领域特征的敏感性")
        
        elif component.type == 'constraint':
            mechanisms.append(f"将约束条件 '{component.name}' 重新表述为适用于 {challenge.domain} 的形式")
        
            mechanisms.append(f"通过参数调整和结构修改，使 {component.name} 适应 {challenge.domain} 的需求")
        
        return " ".join(mechanisms) if mechanisms else f"通过领域特化改造 {component.name}"
    
    def _generate_validation_suggestion(self, 
                                      component: TechnicalComponent, 
                                      challenge: TechnicalChallenge) -> str:
        """生成验证建议"""
        suggestions = []
        
        # 1. 基准对比
        suggestions.append(f"在 {challenge.domain} 标准数据集上对比改造后的 {component.name} 与原始版本的性能")
        
        # 2. 消融实验
        if component.parameters:
            suggestions.append(f"进行消融实验，验证 {component.name} 中各参数在目标任务上的贡献")
        
        # 3. 特定指标验证
        if 'adaptation' in challenge.domain:
            suggestions.append("使用域适配特定指标（如A-distance、MMD）验证改造效果")
        elif 'temporal' in challenge.description.lower():
            suggestions.append("评估改造方法在时序一致性和长期依赖建模上的表现")
        elif 'multimodal' in challenge.domain:
            suggestions.append("在跨模态检索和对齐任务上验证改造方法的有效性")
        
        return " ".join(suggestions)

    def generate_enhanced_cross_paper_links(self, opportunities: List[TransferOpportunity]) -> List[Dict]:
        """🆕 生成增强的跨论文关联边"""
        links = []
        
        for opportunity in opportunities:
            if opportunity.transfer_feasibility > 0.5:  # 高质量阈值
                # 🔧 使用知识图谱丰富信息生成详细的边描述
                source_paper = opportunity.source_component.source_paper_info
                target_paper = opportunity.target_challenge.source_paper_info
                
                link = {
                    'source_entity': opportunity.source_component.source_context,
                    'target_entity': target_paper.get('entity_name', ''),
                    'relationship_type': 'technical_transfer',
                    'evidence': self._generate_enhanced_evidence(opportunity),
                    'mechanism': opportunity.adaptation_mechanism,
                    'validation': opportunity.validation_suggestion,
                    'feasibility_score': opportunity.transfer_feasibility,
                    'source_paper_info': {
                        'title': source_paper.get('title', ''),
                        'authors': source_paper.get('authors', []),
                        'conference': source_paper.get('conference', ''),
                        'year': source_paper.get('year', '')
                    },
                    'target_paper_info': {
                        'title': target_paper.get('title', ''),
                        'authors': target_paper.get('authors', []),
                        'conference': target_paper.get('conference', ''),
                        'year': target_paper.get('year', '')
                    },
                    'technical_details': {
                        'component_type': opportunity.source_component.type,
                        'challenge_domain': opportunity.target_challenge.domain,
                        'complexity_level': getattr(opportunity.source_component, 'complexity_level', 'medium'),
                        'difficulty_level': getattr(opportunity.target_challenge, 'difficulty_level', 'medium')
                    }
                }
                links.append(link)
        
        logger.info(f"生成 {len(links)} 个高质量跨论文关联边")
        return links
    
    def _generate_enhanced_evidence(self, opportunity: TransferOpportunity) -> str:
        """🆕 生成增强的技术证据"""
        component = opportunity.source_component
        challenge = opportunity.target_challenge
        
        # 使用模板生成证据
        template_type = self._select_evidence_template(component, challenge)
        template = self.evidence_templates[template_type][0]  # 使用第一个模板
        
        # 填充模板变量
        evidence = template.format(
            source_paper=component.source_paper_info.get('title', 'Unknown Paper')[:50],
            source_method=component.name,
            component_type=component.type,
            adaptation=f"{component.type} adaptation",
            target_paper=challenge.source_paper_info.get('title', 'Unknown Paper')[:50],
            target_challenge=challenge.name[:30],
            source_conference=component.source_paper_info.get('conference', 'Unknown'),
            core_technique=component.type.replace('_', ' '),
            target_domain=challenge.domain,
            specific_challenge=challenge.description[:50]
        )
        
        return evidence
    
    def _select_evidence_template(self, component: TechnicalComponent, challenge: TechnicalChallenge) -> str:
        """🆕 选择合适的证据模板"""
        if component.type in ['loss_function', 'algorithm', 'constraint']:
            return 'technical_transfer'
        elif 'architecture' in component.type or 'network' in component.type:
            return 'architectural_adaptation'
        else:
            return 'methodological_insight'

    def get_extraction_statistics(self) -> Dict:
        """🆕 获取提取统计信息"""
        return self.extraction_stats.copy()
    
    def reset_statistics(self):
        """🆕 重置统计信息"""
        for key in self.extraction_stats:
            self.extraction_stats[key] = 0

    def find_transfer_opportunities(self, components: List[TechnicalComponent], 
                                  challenges: List[TechnicalChallenge]) -> List[TransferOpportunity]:
        """🔄 兼容性方法：调用增强的迁移机会发现方法"""
        return self.find_enhanced_transfer_opportunities(components, challenges)

    def generate_enhanced_edge_description(self, 
                                         opportunity: TransferOpportunity) -> str:
        """生成增强的边描述，包含具体的技术证据"""
        component = opportunity.source_component
        challenge = opportunity.target_challenge
        
        description = (
            f"技术迁移机会：{component.name} ({component.type}) → {challenge.name} "
            f"(可行性: {opportunity.transfer_feasibility:.3f}). "
            f"改造机制: {opportunity.adaptation_mechanism[:100]}..."
        )
        
        return description

    async def generate_llm_based_evidence(self, opportunity: TransferOpportunity) -> str:
        """🆕 使用LLM基于真实节点属性生成技术证据"""
        component = opportunity.source_component
        challenge = opportunity.target_challenge
        
        # 构建LLM分析提示
        prompt = f"""
基于以下真实的技术信息，分析技术迁移的可行性和具体机制：

源技术组件：
- 名称: {component.name}
- 类型: {component.type}
- 描述: {component.description}
- 来源论文: {component.source_paper_info.get('title', '')}
- 技术领域: {component.technical_domain}

目标技术挑战：
- 挑战: {challenge.name}
- 领域: {challenge.domain}
- 描述: {challenge.description}
- 来源论文: {challenge.source_paper_info.get('title', '')}

请分析：
1. 这两个技术之间的具体关联性
2. 技术迁移的可行机制
3. 预期的改进效果
4. 实施的具体步骤

要求：基于真实技术内容，避免模板化描述，提供具体可操作的分析。
"""

        # 直接返回简化的技术迁移描述，不使用LLM
        return f"技术迁移：{component.name} → {challenge.name} (可行性: {opportunity.transfer_feasibility:.3f})"

    def _infer_technical_domain(self, description: str) -> str:
        """简化的技术领域推断"""
        return 'general'  # 统一返回通用领域

    def _extract_parameters_from_description(self, description: str) -> List[str]:
        """🆕 从描述中提取参数"""
        parameters = []
        
        # 查找常见参数模式
        param_patterns = [
            r'temperature\s+parameter\s+\(([^)]+)\)',
            r'parameter\s+([α-ωτλμσ]\w*)',
            r'threshold\s+([0-9.]+)',
            r'learning\s+rate\s+([0-9.e-]+)',
            r'batch\s+size\s+([0-9]+)',
            r'top-k\s+([0-9]+)',
            r'dimension\s+([0-9]+)'
        ]
        
        for pattern in param_patterns:
            matches = re.findall(pattern, description, re.IGNORECASE)
            parameters.extend(matches)
        
        return parameters[:5]  # 限制参数数量

    def _extract_mathematical_form(self, description: str) -> str:
        """🆕 提取数学形式"""
        # 查找数学表达式
        math_patterns = [
            r'KL-divergence',
            r'Wasserstein\s+distance',
            r'symmetric\s+KL',
            r'softmax\s+function',
            r'running\s+mean',
            r'cosine\s+similarity',
            r'euclidean\s+distance'
        ]
        
        for pattern in math_patterns:
            if re.search(pattern, description, re.IGNORECASE):
                return pattern.replace('\\s+', ' ')
        
        return ''
