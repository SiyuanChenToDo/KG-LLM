#!/usr/bin/env python3
"""
节点属性查询API - 实用指南
========================

这个文件展示了在各种环境中如何获取节点属性的最佳实践

使用场景：
1. 📁 直接从JSON文件查询 (最快速)
2. 🔍 在LightRAG环境中查询 (最完整)
3. 🎯 特定节点查询 (最灵活)
"""

import json
import os

# 智能路径检测 - 支持从不同目录调用
def get_kg_file_path():
    """
    获取知识图谱文件路径，按优先级尝试多个可能的位置
    """
    possible_paths = [
        "/root/autodl-tmp/LightRAG/data/final_custom_kg_papers.json",
        "/root/autodl-tmp/LightRAG/data/test_custom_kg_papers.json",
        "data/final_custom_kg_papers.json",
        "data/custom_kg_papers.json", 
        "data/test_custom_kg_papers.json",
        "data/custom_kg_fixed.json",
        "data/custom_kg.json",
        "../data/final_custom_kg_papers.json",
        "../data/custom_kg_papers.json", 
        "../data/test_custom_kg_papers.json",
        "../data/custom_kg_fixed.json",
        "../data/custom_kg.json",
        "../../data/final_custom_kg_papers.json",
        "../../data/custom_kg_papers.json", 
        "../../data/test_custom_kg_papers.json",
        "../../data/custom_kg_fixed.json",
        "../../data/custom_kg.json"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    # 如果都找不到，抛出异常并显示尝试过的路径
    raise FileNotFoundError(f"知识图谱文件未找到，尝试过的路径: {possible_paths}")

# ========================
# 方法1: 直接从JSON文件查询 (推荐)
# ========================

def get_paper_attributes(paper_index=0):
    """
    获取指定paper节点的所有属性
    
    Args:
        paper_index (int): paper节点的索引 (默认0，即第一篇论文)
        
    Returns:
        dict: paper节点的所有属性
    """
    kg_file_path = get_kg_file_path()
    with open(kg_file_path, 'r', encoding='utf-8') as f:
        kg_data = json.load(f)
    
    papers = [e for e in kg_data["entities"] if e["entity_type"] == "paper"]
    
    if paper_index < len(papers):
        return papers[paper_index]
    return None

def get_research_question_attributes(rq_index=0):
    """
    获取指定research_question节点的所有属性
    
    Args:
        rq_index (int): research_question节点的索引
        
    Returns:
        dict: research_question节点的所有属性
    """
    kg_file_path = get_kg_file_path()
    with open(kg_file_path, 'r', encoding='utf-8') as f:
        kg_data = json.load(f)
    
    rqs = [e for e in kg_data["entities"] if e["entity_type"] == "research_question"]
    
    if rq_index < len(rqs):
        return rqs[rq_index]
    return None

def get_solution_attributes(sol_index=0):
    """
    获取指定solution节点的所有属性
    
    Args:
        sol_index (int): solution节点的索引
        
    Returns:
        dict: solution节点的所有属性
    """
    kg_file_path = get_kg_file_path()
    with open(kg_file_path, 'r', encoding='utf-8') as f:
        kg_data = json.load(f)
    
    solutions = [e for e in kg_data["entities"] if e["entity_type"] == "solution"]
    
    if sol_index < len(solutions):
        return solutions[sol_index]
    return None

def find_paper_by_title(title_keyword):
    """
    根据标题关键词查找论文
    
    Args:
        title_keyword (str): 标题中的关键词
        
    Returns:
        list: 匹配的paper节点列表
    """
    kg_file_path = get_kg_file_path()
    with open(kg_file_path, 'r', encoding='utf-8') as f:
        kg_data = json.load(f)
    
    papers = [e for e in kg_data["entities"] if e["entity_type"] == "paper"]
    
    matching_papers = []
    for paper in papers:
        if title_keyword.lower() in paper.get("title", "").lower():
            matching_papers.append(paper)
    
    return matching_papers

def get_paper_research_questions(paper_entity_name):
    """
    获取指定论文的所有研究问题
    
    Args:
        paper_entity_name (str): paper节点的entity_name
        
    Returns:
        list: 该论文的所有research_question节点
    """
    kg_file_path = get_kg_file_path()
    with open(kg_file_path, 'r', encoding='utf-8') as f:
        kg_data = json.load(f)
    
    # 找到属于该论文的研究问题
    rqs = []
    for entity in kg_data["entities"]:
        if (entity["entity_type"] == "research_question" and 
            entity["entity_name"].startswith(paper_entity_name + "_RQ_")):
            rqs.append(entity)
    
    return rqs

def get_research_question_solutions(rq_entity_name):
    """
    获取指定研究问题的所有解决方案
    
    Args:
        rq_entity_name (str): research_question节点的entity_name
        
    Returns:
        list: 该研究问题的所有solution节点
    """
    kg_file_path = get_kg_file_path()
    with open(kg_file_path, 'r', encoding='utf-8') as f:
        kg_data = json.load(f)
    
    # 从research_question的entity_name推导对应的solution
    # 例如: Paper_RQ_1 -> Paper_SOL_1
    base_name = rq_entity_name.replace("_RQ_", "_SOL_")
    if base_name.endswith("_RQ_"):
        base_name = base_name[:-4] + "_SOL_"
    
    solutions = []
    for entity in kg_data["entities"]:
        if (entity["entity_type"] == "solution" and 
            entity["entity_name"].startswith(base_name)):
            solutions.append(entity)
    
    return solutions

def get_solution_attributes_by_name(entity_name):
    """
    根据实体名称获取solution节点的所有属性
    
    Args:
        entity_name (str): solution节点的entity_name
        
    Returns:
        dict: solution节点的所有属性，如果未找到则返回None
    """
    kg_file_path = get_kg_file_path()
    with open(kg_file_path, 'r', encoding='utf-8') as f:
        kg_data = json.load(f)
    
    solutions = [e for e in kg_data["entities"] if e["entity_type"] == "solution"]
    
    for solution in solutions:
        if solution.get("entity_name") == entity_name:
            return solution
    
    return None

def get_paper_attributes_by_name(entity_name):
    """
    根据实体名称获取paper节点的所有属性
    
    Args:
        entity_name (str): paper节点的entity_name
        
    Returns:
        dict: paper节点的所有属性，如果未找到则返回None
    """
    kg_file_path = get_kg_file_path()
    with open(kg_file_path, 'r', encoding='utf-8') as f:
        kg_data = json.load(f)
    
    papers = [e for e in kg_data["entities"] if e["entity_type"] == "paper"]
    
    for paper in papers:
        if paper.get("entity_name") == entity_name:
            return paper
    
    return None

def get_research_question_attributes_by_name(entity_name):
    """
    根据实体名称获取research_question节点的所有属性
    
    Args:
        entity_name (str): research_question节点的entity_name
        
    Returns:
        dict: research_question节点的所有属性，如果未找到则返回None
    """
    kg_file_path = get_kg_file_path()
    with open(kg_file_path, 'r', encoding='utf-8') as f:
        kg_data = json.load(f)
    
    rqs = [e for e in kg_data["entities"] if e["entity_type"] == "research_question"]
    
    for rq in rqs:
        if rq.get("entity_name") == entity_name:
            return rq
    
    return None

# ========================
# 方法2: 在LightRAG中查询节点属性的代码片段
# ========================

# 在lightrag_ollama_demo.py中添加这段代码来查询节点属性：
"""
async def query_node_attributes_in_lightrag(rag):
    '''在LightRAG环境中查询节点属性'''
    
    # 获取图存储实例
    graph = rag.chunk_entity_relation_graph
    
    # 获取所有节点标签
    all_labels = await graph.get_all_labels()
    print(f"总节点数: {len(all_labels)}")
    
    # 按类型分类节点
    papers = []
    research_questions = []
    solutions = []
    
    for label in all_labels:
        node_data = await graph.get_node(label)
        if node_data:
            entity_type = node_data.get("entity_type", "unknown")
            if entity_type == "paper":
                papers.append((label, node_data))
            elif entity_type == "research_question":
                research_questions.append((label, node_data))
            elif entity_type == "solution":
                solutions.append((label, node_data))
    
    # 显示第一篇论文的属性
    if papers:
        label, attributes = papers[0]
        print(f"Paper节点: {label}")
        for key, value in attributes.items():
            print(f"  {key}: {value}")
        
        # 获取该论文的关系
        edges = await graph.get_node_edges(label)
        print(f"关联关系数: {len(edges)}")
        
        for src, tgt in edges:
            edge_data = await graph.get_edge(src, tgt)
            if edge_data:
                print(f"  关系: {src} -> {tgt}")
                print(f"    属性: {edge_data}")
    
    return papers, research_questions, solutions
"""

# ========================
# 实用工具函数
# ========================

def print_node_attributes(node, node_type="Node"):
    """
    美观地打印节点属性
    
    Args:
        node (dict): 节点数据
        node_type (str): 节点类型名称
    """
    if not node:
        print("❌ 节点不存在")
        return
    
    print(f"\n📋 {node_type}节点属性:")
    print("=" * 50)
    
    # 定义各类型节点的重要属性
    important_attrs = {
        "paper": ["entity_name", "title", "authors", "year", "conference", "abstract"],
        "research_question": ["entity_name", "research_question", "simplified_research_question"],
        "solution": ["entity_name", "solution", "simplified_solution"]
    }
    
    node_type_key = node.get("entity_type", "").lower()
    if node_type_key in important_attrs:
        # 先显示重要属性
        for attr in important_attrs[node_type_key]:
            if attr in node:
                value = node[attr]
                if isinstance(value, str) and len(value) > 100:
                    print(f"🔹 {attr}: {value[:80]}...")
                else:
                    print(f"🔹 {attr}: {value}")
        
        print("\n📝 其他属性:")
        # 再显示其他属性
        for key, value in node.items():
            if key not in important_attrs[node_type_key]:
                if isinstance(value, str) and len(value) > 100:
                    print(f"   {key}: {value[:80]}...")
                else:
                    print(f"   {key}: {value}")
    else:
        # 显示所有属性
        for key, value in node.items():
            if isinstance(value, str) and len(value) > 100:
                print(f"🔹 {key}: {value[:80]}...")
            else:
                print(f"🔹 {key}: {value}")

# ========================
# 使用示例
# ========================

if __name__ == "__main__":
    print("🚀 节点属性查询API - 使用示例")
    print("=" * 60)
    
    # 示例1: 获取第一篇论文的属性
    print("\n📄 示例1: 获取第一篇论文的属性")
    paper = get_paper_attributes(0)
    print_node_attributes(paper, "Paper")
    
    # 示例2: 根据关键词查找论文
    print("\n🔍 示例2: 根据关键词查找论文")
    papers = find_paper_by_title("Attention")
    print(f"找到 {len(papers)} 篇包含'Attention'的论文:")
    for i, paper in enumerate(papers, 1):
        print(f"  {i}. {paper['title']}")
    
    # 示例3: 获取第一篇论文的研究问题
    if papers:
        print(f"\n❓ 示例3: 获取第一篇论文的研究问题")
        rqs = get_paper_research_questions(papers[0]["entity_name"])
        print(f"该论文有 {len(rqs)} 个研究问题:")
        for i, rq in enumerate(rqs, 1):
            print(f"  {i}. {rq.get('simplified_research_question', 'N/A')}")
    
    # 示例4: 获取第一个研究问题的解决方案
    if rqs:
        print(f"\n💡 示例4: 获取第一个研究问题的解决方案")
        solutions = get_research_question_solutions(rqs[0]["entity_name"])
        print(f"该研究问题有 {len(solutions)} 个解决方案:")
        for i, sol in enumerate(solutions, 1):
            print(f"  {i}. {sol.get('simplified_solution', 'N/A')}")
    
    print(f"\n✅ API使用示例完成!")
    print(f"💡 您可以在自己的代码中导入这些函数来查询节点属性")

def get_all_papers():
    """获取所有论文节点"""
    kg_file_path = get_kg_file_path()
    with open(kg_file_path, 'r', encoding='utf-8') as f:
        kg_data = json.load(f)
    return [e for e in kg_data["entities"] if e["entity_type"] == "paper"]

def get_all_research_questions():
    """获取所有研究问题节点"""
    kg_file_path = get_kg_file_path()
    with open(kg_file_path, 'r', encoding='utf-8') as f:
        kg_data = json.load(f)
    return [e for e in kg_data["entities"] if e["entity_type"] == "research_question"]

def get_all_solutions():
    """获取所有解决方案节点"""
    kg_file_path = get_kg_file_path()
    with open(kg_file_path, 'r', encoding='utf-8') as f:
        kg_data = json.load(f)
    return [e for e in kg_data["entities"] if e["entity_type"] == "solution"]
