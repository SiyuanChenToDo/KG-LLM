#!/usr/bin/env python3
"""
论文数据集处理程序
================

处理人工标注的论文Excel数据，构建包含三种节点类型的知识图谱：
1. paper节点：论文主节点
2. research_question节点：研究问题节点
3. solution节点：解决方案节点

每篇论文形成一个子图（树结构）：paper -> research_questions -> solutions

作者：LightRAG团队
版本：2.0
"""

import pandas as pd
import json
import re
import os
from datetime import datetime

def clean_text(text):
    """清理文本内容"""
    if pd.isna(text) or text == "":
        return ""
    return str(text).strip()

def create_entity_name(base_name):
    """创建实体名称（移除特殊字符，保持可读性）"""
    # 保留字母、数字、中文和基本符号
    cleaned = re.sub(r'[^\w\u4e00-\u9fff\s\-_]', '', str(base_name))
    # 替换空格为下划线，但保持可读性
    cleaned = re.sub(r'\s+', '_', cleaned)
    # 移除开头结尾的下划线
    cleaned = cleaned.strip('_')
    return cleaned

def get_safe_attribute(row, column_name, default=""):
    """安全获取Excel列值"""
    try:
        value = row.get(column_name, default)
        return clean_text(value)
    except:
        return default

def process_papers_to_kg():
    """处理论文数据为知识图谱格式"""
    
    print("🚀 开始处理论文数据集...")
    
    # 读取Excel文件
    #excel_path = "data/final_data.xlsx"
    #excel_path = "/root/autodl-tmp/LightRAG/data/test_data.xlsx"
    excel_path = "/root/autodl-tmp/LightRAG/data/final_data.xlsx"
    
    if not os.path.exists(excel_path):
        print(f"❌ Excel文件不存在: {excel_path}")
        return
    
    try:
        df = pd.read_excel(excel_path)
        print(f"✅ 成功读取Excel文件: {len(df)} 篇论文")
    except Exception as e:
        print(f"❌ 读取Excel文件失败: {e}")
        return
    
    # 初始化知识图谱结构
    knowledge_graph = {
        "entities": [],
        "relationships": [],
        "chunks": [],
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "total_papers": len(df),
            "node_types": ["paper", "research_question", "solution"],
            "description": "论文研究问题解决方案知识图谱"
        }
    }
    
    print("\n📊 开始处理每篇论文...")
    
    # 遍历每篇论文
    for idx, row in df.iterrows():
        paper_id = get_safe_attribute(row, "file_id", f"paper_{idx}")
        title = get_safe_attribute(row, "title")
        
        if not title:
            print(f"⚠️ 跳过第{idx+1}行：标题为空")
            continue
            
        print(f"📄 处理论文 {idx+1}/{len(df)}: {title[:50]}...")
        
        # 创建论文实体名称
        paper_entity_name = create_entity_name(title)
        
        # 1. 创建论文节点
        paper_entity = {
            "entity_name": paper_entity_name,
            "entity_type": "paper",
            "abstract": get_safe_attribute(row, "abstract"),
            "title": title,
            "authors": get_safe_attribute(row, "authors"),
            "year": get_safe_attribute(row, "year"),
            "conference": get_safe_attribute(row, "conference"),
            "venue": get_safe_attribute(row, "venue"),
            "citationCount": get_safe_attribute(row, "citationCount"),
            "core_problem": get_safe_attribute(row, "core_problem"),
            "related_work": get_safe_attribute(row, "related_work"),
            "preliminary_innovation_analysis": get_safe_attribute(row, "preliminary_innovation_analysis"),
            "basic_problem": get_safe_attribute(row, "basic_problem"),
            "datasets": get_safe_attribute(row, "datasets"),
            "experimental_results": get_safe_attribute(row, "experimental_results"),
            "evaluation_metrics": get_safe_attribute(row, "evaluation_metrics"),
            "framework_summary": get_safe_attribute(row, "framework_summary"),
            "source_id": paper_id,
            "file_path": f"papers/{paper_id}.pdf"
        }
        
        knowledge_graph["entities"].append(paper_entity)
        
        # 2. 创建文档块
        chunk_content = f"""论文标题: {title}
        作者: {get_safe_attribute(row, 'authors')}
        年份: {get_safe_attribute(row, 'year')}
        会议: {get_safe_attribute(row, 'conference')}
        摘要: {get_safe_attribute(row, 'abstract')[:200]}..."""
        
        knowledge_graph["chunks"].append({
            "content": chunk_content,
            "source_id": paper_id,
            "file_path": f"papers/{paper_id}.pdf"
        })
        
        # 3. 处理研究问题和解决方案（最多4组）
        research_questions_created = 0
        solutions_created = 0
        
        for i in range(1, 5):  # 处理research_question_1到research_question_4
            rq_text = get_safe_attribute(row, f"research_question_{i}")
            simplified_rq = get_safe_attribute(row, f"simplified_research_question_{i}")
            solution_text = get_safe_attribute(row, f"solution_{i}")
            simplified_solution = get_safe_attribute(row, f"simplified_solution_{i}")
            
            # 只有当研究问题存在时才创建节点
            if rq_text:
                research_questions_created += 1
                
                # 创建研究问题实体名称
                rq_entity_name = f"{paper_entity_name}_RQ_{i}"
                
                # 创建研究问题节点
                rq_entity = {
                    "entity_name": rq_entity_name,
                    "entity_type": "research_question",
                    "research_question": rq_text,
                    "simplified_research_question": simplified_rq,
                    "source_id": paper_id,
                    "file_path": f"papers/{paper_id}.pdf"
                }
                
                knowledge_graph["entities"].append(rq_entity)
                
                # 创建论文->研究问题的关系
                paper_to_rq_relation = {
                    "src_id": paper_entity_name,
                    "tgt_id": rq_entity_name,
                    "description": f"论文提出了研究问题{i}",
                    "keywords": f"has_research_question_{i}",
                    "weight": 1.0,
                    "source_id": paper_id,
                    "file_path": f"papers/{paper_id}.pdf"
                }
                
                knowledge_graph["relationships"].append(paper_to_rq_relation)
                
                # 4. 处理对应的解决方案
                if solution_text:
                    solutions_created += 1
                    
                    # 创建解决方案实体名称
                    sol_entity_name = f"{paper_entity_name}_SOL_{i}"
                    
                    # 创建解决方案节点
                    solution_entity = {
                        "entity_name": sol_entity_name,
                        "entity_type": "solution",
                        "solution": solution_text,
                        "simplified_solution": simplified_solution,
                        "source_id": paper_id,
                        "file_path": f"papers/{paper_id}.pdf"
                    }
                    
                    knowledge_graph["entities"].append(solution_entity)
                    
                    # 创建研究问题->解决方案的关系
                    rq_to_sol_relation = {
                        "src_id": rq_entity_name,
                        "tgt_id": sol_entity_name,
                        "description": f"针对研究问题{i}的解决方案",
                        "keywords": f"solved_by_solution_{i}",
                        "weight": 1.0,
                        "source_id": paper_id,
                        "file_path": f"papers/{paper_id}.pdf"
                    }
                    
                    knowledge_graph["relationships"].append(rq_to_sol_relation)
        
        print(f"   ✅ 创建了 {research_questions_created} 个研究问题和 {solutions_created} 个解决方案")
    
    # 更新元数据统计
    knowledge_graph["metadata"].update({
        "total_entities": len(knowledge_graph["entities"]),
        "total_relationships": len(knowledge_graph["relationships"]),
        "total_chunks": len(knowledge_graph["chunks"]),
        "entity_types_count": {
            "paper": len([e for e in knowledge_graph["entities"] if e["entity_type"] == "paper"]),
            "research_question": len([e for e in knowledge_graph["entities"] if e["entity_type"] == "research_question"]),
            "solution": len([e for e in knowledge_graph["entities"] if e["entity_type"] == "solution"])
        }
    })
    
    return knowledge_graph

def save_knowledge_graph(kg_data, output_path="data/custom_kg_papers.json"):
    """保存知识图谱到JSON文件"""
    try:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(kg_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ 知识图谱已保存到: {output_path}")
        return True
    except Exception as e:
        print(f"❌ 保存文件失败: {e}")
        return False

def print_statistics(kg_data):
    """打印统计信息"""
    print("\n📊 知识图谱统计信息:")
    print("=" * 50)
    
    metadata = kg_data.get("metadata", {})
    
    print(f"📄 总论文数: {metadata.get('total_papers', 0)}")
    print(f"🏷️ 总实体数: {metadata.get('total_entities', 0)}")
    print(f"🔗 总关系数: {metadata.get('total_relationships', 0)}")
    print(f"📝 总文档块: {metadata.get('total_chunks', 0)}")
    
    print(f"\n📋 实体类型分布:")
    entity_counts = metadata.get('entity_types_count', {})
    for entity_type, count in entity_counts.items():
        print(f"   {entity_type}: {count} 个")
    
    print(f"\n🌳 图结构:")
    papers = entity_counts.get('paper', 0)
    rqs = entity_counts.get('research_question', 0)
    sols = entity_counts.get('solution', 0)
    
    print(f"   每篇论文平均研究问题数: {rqs/papers:.1f}" if papers > 0 else "   无法计算平均研究问题数")
    print(f"   每篇论文平均解决方案数: {sols/papers:.1f}" if papers > 0 else "   无法计算平均解决方案数")

def validate_knowledge_graph(kg_data):
    """验证知识图谱数据"""
    print("\n🔍 验证知识图谱数据...")
    
    entities = kg_data.get("entities", [])
    relationships = kg_data.get("relationships", [])
    
    # 验证实体
    entity_names = set()
    for entity in entities:
        if "entity_name" not in entity or "entity_type" not in entity:
            print(f"❌ 发现无效实体: {entity}")
            return False
        entity_names.add(entity["entity_name"])
    
    # 验证关系
    for rel in relationships:
        if "src_id" not in rel or "tgt_id" not in rel:
            print(f"❌ 发现无效关系: {rel}")
            return False
        
        if rel["src_id"] not in entity_names:
            print(f"❌ 关系源实体不存在: {rel['src_id']}")
            return False
            
        if rel["tgt_id"] not in entity_names:
            print(f"❌ 关系目标实体不存在: {rel['tgt_id']}")
            return False
    
    print("✅ 知识图谱数据验证通过!")
    return True

def main():
    """主函数"""
    print("🎯 论文数据集处理程序")
    print("=" * 60)
    
    # 处理数据
    kg_data = process_papers_to_kg()
    
    if not kg_data:
        print("❌ 数据处理失败")
        return
    
    # 验证数据
    if not validate_knowledge_graph(kg_data):
        print("❌ 数据验证失败")
        return
    
    # 打印统计信息
    print_statistics(kg_data)
    
    # 保存到文件
    output_path = "data/custom_kg_papers.json"
    if save_knowledge_graph(kg_data, output_path):
        print(f"\n🎉 处理完成!")
        print(f"📁 输出文件: {output_path}")
        print(f"💡 现在可以在lightrag_ollama_demo.py中使用这个文件构建知识图谱")
    else:
        print("❌ 保存失败")

if __name__ == "__main__":
    main()