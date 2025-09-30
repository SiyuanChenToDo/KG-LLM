# build_custom_kg.py
import json
import os
from pathlib import Path
from typing import Dict, Any

def get_kg_file_path():
    """动态获取知识图谱文件路径"""
    # 优先使用新生成的论文知识图谱
    base_dir = Path(__file__).parent.parent.parent / "data"
    
    # 候选文件列表（按优先级排序）
    candidate_files = [
        "final_custom_kg_papers.json",      # 新生成的论文知识图谱
        #"test_custom_kg_papers.json",
    ]
    
    for filename in candidate_files:
        file_path = base_dir / filename
        if file_path.exists():
            print(f"📁 使用知识图谱文件: {file_path}")
            return file_path
    
    # 如果都不存在，返回默认路径
    default_path = base_dir / "final_custom_kg_papers.json"
    print(f"⚠️ 未找到现有知识图谱文件，将使用: {default_path}")
    return default_path

def load_custom_kg() -> Dict[str, Any]:
    """
    加载自定义知识图谱数据
    
    返回格式符合 LightRAG.ainsert_custom_kg 的要求:
    {
        "entities": [...],
        "relationships": [...], 
        "chunks": [...]
    }
    """
    kg_file = get_kg_file_path()
    
    if not kg_file.exists():
        raise FileNotFoundError(f"知识图谱文件不存在: {kg_file}")
    
    try:
        with kg_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
        
        # 验证数据格式
        required_keys = ["entities", "relationships"]
        for key in required_keys:
            if key not in data:
                raise ValueError(f"知识图谱文件缺少必需的键: {key}")
        
        # 统计信息
        entities_count = len(data.get("entities", []))
        relationships_count = len(data.get("relationships", []))
        chunks_count = len(data.get("chunks", []))
        
        print(f"📊 知识图谱加载成功:")
        print(f"   - 实体数量: {entities_count}")
        print(f"   - 关系数量: {relationships_count}")
        print(f"   - 文档块数量: {chunks_count}")
        
        # 显示实体类型分布
        if entities_count > 0:
            entity_types = {}
            for entity in data["entities"]:
                entity_type = entity.get("entity_type", "unknown")
                entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
            
            print(f"   - 实体类型分布:")
            for entity_type, count in entity_types.items():
                print(f"     * {entity_type}: {count}个")
        
        return data
        
    except json.JSONDecodeError as e:
        raise ValueError(f"知识图谱文件格式错误: {e}")
    except Exception as e:
        raise RuntimeError(f"加载知识图谱文件失败: {e}")

def validate_kg_format(kg_data: Dict[str, Any]) -> bool:
    """验证知识图谱数据格式"""
    try:
        # 检查基本结构
        if not isinstance(kg_data, dict):
            print("❌ 知识图谱数据必须是字典格式")
            return False
        
        entities = kg_data.get("entities", [])
        relationships = kg_data.get("relationships", [])
        
        # 验证实体格式
        entity_names = set()
        for i, entity in enumerate(entities):
            if not isinstance(entity, dict):
                print(f"❌ 实体 {i} 不是字典格式")
                return False
            
            if "entity_name" not in entity or "entity_type" not in entity:
                print(f"❌ 实体 {i} 缺少必需字段 (entity_name, entity_type)")
                return False
            
            entity_names.add(entity["entity_name"])
        
        # 验证关系格式
        for i, rel in enumerate(relationships):
            if not isinstance(rel, dict):
                print(f"❌ 关系 {i} 不是字典格式")
                return False
            
            if "src_id" not in rel or "tgt_id" not in rel:
                print(f"❌ 关系 {i} 缺少必需字段 (src_id, tgt_id)")
                return False
            
            # 检查关系引用的实体是否存在
            if rel["src_id"] not in entity_names:
                print(f"❌ 关系 {i} 的源实体不存在: {rel['src_id']}")
                return False
            
            if rel["tgt_id"] not in entity_names:
                print(f"❌ 关系 {i} 的目标实体不存在: {rel['tgt_id']}")
                return False
        
        print("✅ 知识图谱格式验证通过")
        return True
        
    except Exception as e:
        print(f"❌ 验证过程出错: {e}")
        return False