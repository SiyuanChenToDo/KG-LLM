#!/usr/bin/env python3
"""
完整节点属性显示工具

用法:
python show_all_attributes.py

功能:
显示知识图谱中所有节点的完整属性，不截断任何内容
"""

import json
import os
from pathlib import Path

def get_kg_file_path():
    """获取知识图谱文件的正确路径"""
    # 可能的文件名（按优先级排序）
    possible_filenames = [
        "custom_kg_papers.json",       # 标准文件名
        "test_custom_kg_papers.json",  # 测试文件名
        "custom_kg_fixed.json",        # 备用文件名
        "custom_kg.json",              # 原始文件名
    ]
    
    # 可能的路径（按优先级排序）
    possible_dirs = [
        "data",                        # 从项目根目录
        "../data",                     # 从Mydemo目录
        "../../data",                  # 从Mydemo/知识图谱目录
    ]
    
    # 尝试所有组合
    tried_paths = []
    for dir_path in possible_dirs:
        for filename in possible_filenames:
            full_path = os.path.join(dir_path, filename)
            tried_paths.append(full_path)
            if os.path.exists(full_path):
                return full_path
    
    # 如果都找不到，返回默认路径并让用户知道
    raise FileNotFoundError(f"知识图谱文件未找到，尝试过的路径: {tried_paths}")

def load_kg_data():
    """加载知识图谱数据"""
    kg_file_path = get_kg_file_path()
    print(f"使用知识图谱文件: {kg_file_path}")
    
    with open(kg_file_path, 'r', encoding='utf-8') as f:
        kg_data = json.load(f)
    
    return kg_data

def show_all_papers(kg_data):
    """显示所有论文节点及其完整属性"""
    papers = [e for e in kg_data["entities"] if e["entity_type"] == "paper"]
    
    print(f"\n===== 所有论文节点 ({len(papers)}个) =====")
    for i, paper in enumerate(papers, 1):
        print(f"\n论文 {i}/{len(papers)}: {paper.get('entity_name', 'Unknown')}")
        print("-" * 80)
        for key, value in paper.items():
            print(f"{key}: {value}")

def show_paper_with_research_questions(kg_data, paper_index=0):
    """显示指定论文及其所有研究问题和解决方案的完整属性"""
    papers = [e for e in kg_data["entities"] if e["entity_type"] == "paper"]
    
    if paper_index >= len(papers):
        print(f"错误: 论文索引 {paper_index} 超出范围，只有 {len(papers)} 篇论文")
        return
    
    paper = papers[paper_index]
    print(f"\n===== 论文 {paper_index+1}/{len(papers)} =====")
    print(f"论文标题: {paper.get('title', 'Unknown')}")
    print("-" * 80)
    
    # 显示论文的所有属性
    print("【论文属性】:")
    for key, value in paper.items():
        print(f"{key}: {value}")
    
    # 获取该论文的所有研究问题
    paper_name = paper["entity_name"]
    research_questions = []
    for entity in kg_data["entities"]:
        if (entity["entity_type"] == "research_question" and 
            entity["entity_name"].startswith(paper_name + "_RQ_")):
            research_questions.append(entity)
    
    print(f"\n【研究问题】: 共 {len(research_questions)} 个")
    for i, rq in enumerate(research_questions, 1):
        print(f"\n研究问题 {i}/{len(research_questions)}:")
        print("-" * 60)
        for key, value in rq.items():
            print(f"{key}: {value}")
        
        # 获取该研究问题的解决方案
        rq_name = rq["entity_name"]
        base_name = rq_name.replace("_RQ_", "_SOL_")
        
        solutions = []
        for entity in kg_data["entities"]:
            if (entity["entity_type"] == "solution" and 
                entity["entity_name"] == base_name):
                solutions.append(entity)
        
        print(f"\n解决方案: 共 {len(solutions)} 个")
        for j, sol in enumerate(solutions, 1):
            print(f"\n解决方案 {j}/{len(solutions)}:")
            print("-" * 40)
            for key, value in sol.items():
                print(f"{key}: {value}")

def show_all_relationships(kg_data):
    """显示所有关系及其完整属性"""
    relationships = kg_data.get("relationships", [])
    
    print(f"\n===== 所有关系 ({len(relationships)}个) =====")
    for i, rel in enumerate(relationships, 1):
        print(f"\n关系 {i}/{len(relationships)}: {rel.get('source', '')} -> {rel.get('target', '')}")
        print("-" * 80)
        for key, value in rel.items():
            print(f"{key}: {value}")

def main():
    """主函数"""
    print("🔍 知识图谱节点完整属性查看工具")
    print("=" * 80)
    
    try:
        kg_data = load_kg_data()
        
        # 统计实体类型
        entity_types = {}
        for entity in kg_data.get("entities", []):
            entity_type = entity.get("entity_type", "unknown")
            entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
        
        print("\n知识图谱统计:")
        print(f"- 实体总数: {len(kg_data.get('entities', []))}")
        print(f"- 关系总数: {len(kg_data.get('relationships', []))}")
        print("- 实体类型分布:")
        for entity_type, count in entity_types.items():
            print(f"  - {entity_type}: {count}个")
        
        # 显示第一篇论文及其所有研究问题和解决方案
        show_paper_with_research_questions(kg_data, 0)
        
        # 询问是否显示更多内容
        print("\n\n是否需要显示更多内容? (输入选项编号)")
        print("1. 显示所有论文的完整属性")
        print("2. 显示所有关系的完整属性")
        print("3. 显示另一篇论文的完整属性")
        print("4. 退出")
        
        choice = input("\n请选择 (1-4): ")
        
        if choice == "1":
            show_all_papers(kg_data)
        elif choice == "2":
            show_all_relationships(kg_data)
        elif choice == "3":
            paper_index = int(input("请输入论文索引 (从0开始): "))
            show_paper_with_research_questions(kg_data, paper_index)
        else:
            print("退出程序")
    
    except Exception as e:
        print(f"错误: {e}")

if __name__ == "__main__":
    main()
