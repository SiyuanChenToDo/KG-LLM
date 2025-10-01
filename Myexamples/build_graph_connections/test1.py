import os
import sys
import json
import time # 新增: 用于生成时间戳
import numpy as np
from itertools import product
from camel.storages import Neo4jGraph
from camel.embeddings import OpenAICompatibleEmbedding

# =================================================================================
# 1. 环境与数据库配置 (Environment and Database Configuration)
# =================================================================================
# --- 请确保您的环境变量已正确设置 ---
os.environ["OPENAI_COMPATIBILITY_API_KEY"] = "sk-c1a6b588f7d543adb0412c5bc61bdd7b"
os.environ["OPENAI_COMPATIBILITY_API_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# --- Neo4j 数据库连接信息 ---
# 注意：密码已替换为占位符，请使用您自己的真实密码。
n4j = Neo4jGraph(
    url="neo4j+s://b3980610.databases.neo4j.io",
    username="neo4j",
    password="ta_T6_9gzxTfrTiWjRuUhO7Lm6fBbQG8TwxnSqHpoqk",
)

# --- 嵌入模型初始化 ---
embedding_model = OpenAICompatibleEmbedding(
    model_type="text-embedding-v2",
    api_key=os.environ["OPENAI_COMPATIBILITY_API_KEY"],
    url=os.environ["OPENAI_COMPATIBILITY_API_BASE_URL"],
)

print("✅ 环境和数据库配置完成。")

# =================================================================================
# 2. 核心参数配置 (Core Parameter Configuration)
# =================================================================================
CONFIG = {
    "WEIGHTS": {
        "abstract": 0.4,
        "core_problem": 0.6
    },
    "SIMILARITY_THRESHOLD": 0.75,
    "NEW_RELATIONSHIP_TYPE": "POSSIBLY_RELATED",
    # --- 缓存路径和维度配置 ---
    # 文件将保存为 embedding_cache.json 和 embedding_cache.npy
    "EMBEDDING_CACHE_BASE_PATH": "embedding_cache", 
    "EMBEDDING_MODEL_DIM": 1536 # DashScope text-embedding-v2 的维度通常为 1536
}
print(f"✅ 核心参数配置完成，权重: {CONFIG['WEIGHTS']}, 阈值: {CONFIG['SIMILARITY_THRESHOLD']}")

# =================================================================================
# 3. 辅助及缓存函数 (Helper and Caching Functions)
# =================================================================================
def cosine_similarity(vec1, vec2):
    """计算两个向量之间的余弦相似度。"""
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0
    return dot_product / (norm_vec1 * norm_vec2)

def embed_text_batch(texts_to_embed):
    """分批处理文本并生成嵌入。"""
    batch_size = 25
    all_embeddings = []
    for i in range(0, len(texts_to_embed), batch_size):
        batch_texts = texts_to_embed[i:i + batch_size]
        print(f"  > 正在为 {len(batch_texts)} 个文本调用 API... (批次 {i//batch_size + 1})")
        try:
            batch_embeddings = embedding_model.embed_list(objs=batch_texts)
            all_embeddings.extend(batch_embeddings)
        except Exception as e:
            print(f"  ❌ 批次处理失败: {e}")
            # 确保即使失败，也能保持列表长度一致，使用 None 占位
            all_embeddings.extend([None] * len(batch_texts))
    return all_embeddings

def load_embedding_cache(base_path, dim):
    """从文件加载嵌入缓存 (JSON for keys, NumPy for vectors)。"""
    json_path = base_path + ".json"
    npy_path = base_path + ".npy"
    
    cache = {}
    
    if os.path.exists(json_path) and os.path.exists(npy_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # 1. Load keys (text) and indices from JSON
            text_to_index = metadata.get('text_to_index', {})
            
            # 2. Load vectors array from .npy file
            vectors_array = np.load(npy_path)
            
            # 检查文件完整性
            if vectors_array.shape[0] != len(text_to_index) or vectors_array.shape[1] != dim:
                print(f"⚠️ 缓存文件 '{npy_path}' 结构不一致 ({vectors_array.shape})，忽略缓存。")
                return {}

            # 3. Reconstruct cache (text: vector list)
            for text, index in text_to_index.items():
                # 存储为 list 以便在缓存字典中保持一致性
                cache[text] = vectors_array[index].tolist() 
            
            print(f"✅ 成功从 '{base_path}.json/.npy' 加载 {len(cache)} 条嵌入缓存。")
            return cache
        
        except Exception as e:
            print(f"❌ 加载缓存文件失败: {e}。返回空缓存。")
            return {}
    
    print(f"💡 未找到完整缓存文件 '{base_path}.json/.npy'，返回空缓存。")
    return {}

def save_embedding_cache(base_path, cache):
    """将嵌入缓存保存到文件 (JSON for keys, NumPy for vectors)。"""
    json_path = base_path + ".json"
    npy_path = base_path + ".npy"
    
    if not cache:
        print("⚠️ 缓存为空，跳过保存。")
        return

    # 1. Prepare metadata and vectors array
    text_to_index = {}
    vectors_list = []
    
    for i, (text, vector_list) in enumerate(cache.items()):
        text_to_index[text] = i
        vectors_list.append(vector_list)
        
    vectors_array = np.array(vectors_list)
    metadata = {
        'text_to_index': text_to_index,
        'count': len(cache),
        'timestamp': int(time.time()),
        'vector_dim': vectors_array.shape[1] if vectors_array.size > 0 else 0
    }
    
    try:
        # 2. Save vectors using NumPy binary format (速度更快)
        np.save(npy_path, vectors_array)

        # 3. Save metadata using JSON
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
            
        print(f"✅ 嵌入缓存已更新并保存到 '{base_path}.json/.npy'。")
        
    except Exception as e:
        print(f"❌ 写入缓存文件失败: {e}")


def get_embeddings_with_caching(texts, cache):
    """
    为文本列表获取嵌入，优先使用缓存。
    返回与输入文本顺序一致的嵌入列表（numpy arrays）和更新后的缓存。
    """
    texts_to_embed_map = {} # 使用 map 保证唯一性
    for i, text in enumerate(texts):
        if text not in cache:
            texts_to_embed_map[text] = None
            
    if texts_to_embed_map:
        print(f"  > 缓存未命中 {len(texts_to_embed_map)} 个唯一文本，准备生成新嵌入...")
        unique_texts_to_embed = list(texts_to_embed_map.keys())
        new_embeddings = embed_text_batch(unique_texts_to_embed)
        
        for text, embedding in zip(unique_texts_to_embed, new_embeddings):
            if embedding is not None:
                cache[text] = np.array(embedding).tolist() # 转为 list 以便 JSON 序列化
    else:
        print("  > 缓存命中所有文本！无需 API 调用。")

    # 从缓存中构建与原始输入顺序一致的结果列表
    final_embeddings = [np.array(cache.get(text)) if cache.get(text) is not None else None for text in texts]
    return final_embeddings, cache

print("✅ 辅助及缓存函数已定义。")

# =================================================================================
# 4. 主逻辑 (Main Logic)
# =================================================================================
def main():
    """执行节点获取、嵌入、加权比较和链接创建的整个流程。"""
    
    # --- 新增: 加载嵌入缓存 (使用新的 base path 和 dim) ---
    embedding_cache = load_embedding_cache(CONFIG["EMBEDDING_CACHE_BASE_PATH"], CONFIG["EMBEDDING_MODEL_DIM"])

    # --- 步骤 1: 获取 Paper 节点和 Solution 节点 (关键: 添加 file_id) ---
    print("\n🚀 步骤 1: 开始从 Neo4j 获取 Paper 和 Solution 节点...")
    try:
        # **更新点 1: Paper 查询中添加 p.file_id**
        paper_query = """
        MATCH (p:paper)
        WHERE p.abstract IS NOT NULL AND p.core_problem IS NOT NULL AND p.file_id IS NOT NULL
        RETURN elementId(p) AS node_id, p.abstract AS abstract, p.core_problem AS core_problem, p.file_id AS file_id
        """
        paper_nodes = n4j.query(query=paper_query)
        print(f"  - 成功获取 {len(paper_nodes)} 个 Paper 节点。")

        # **更新点 2: Solution 查询中添加 s.file_id**
        # 假设 Solution 节点也直接有 file_id 属性，与 Paper 节点一致。
        solution_query = """
        MATCH (s)
        WHERE (s:solution_1 OR s:solution_2 OR s:solution_3 OR s:solution_4) AND s.file_id IS NOT NULL
        WITH s, labels(s)[0] AS label
        WITH s, label, s.file_id AS file_id,
             CASE label
               WHEN 'solution_1' THEN s.solution_1
               WHEN 'solution_2' THEN s.solution_2
               WHEN 'solution_3' THEN s.solution_3
               WHEN 'solution_4' THEN s.solution_4
               ELSE null
             END AS text_content
        WHERE text_content IS NOT NULL
        RETURN elementId(s) AS node_id, label, text_content AS text, file_id
        """
        solution_nodes = n4j.query(query=solution_query)
        print(f"  - 成功获取 {len(solution_nodes)} 个 Solution 节点。")
        
        if not paper_nodes or not solution_nodes:
            print("⚠️ 缺少 Paper 或 Solution 节点，无法继续。请检查数据库。")
            return
            
    except Exception as e:
        print(f"❌ Neo4j 查询失败: {e}")
        return

    # --- 步骤 2: 使用缓存机制为节点属性生成或获取向量嵌入 ---
    print("\n🚀 步骤 2: 开始为节点属性生成或获取向量嵌入...")
    
    all_paper_texts = [text for p in paper_nodes for text in (p['abstract'], p['core_problem'])]
    all_solution_texts = [s['text'] for s in solution_nodes]
    
    # 一次性处理所有文本
    all_paper_embeddings, embedding_cache = get_embeddings_with_caching(all_paper_texts, embedding_cache)
    all_solution_embeddings, embedding_cache = get_embeddings_with_caching(all_solution_texts, embedding_cache)

    # 将嵌入向量分配回对应的节点
    for i, p_node in enumerate(paper_nodes):
        p_node['vectors'] = {
            "abstract": all_paper_embeddings[i*2],
            "core_problem": all_paper_embeddings[i*2 + 1]
        }
    for i, s_node in enumerate(solution_nodes):
        s_node['vector'] = all_solution_embeddings[i]
        
    print("✅ 向量嵌入已全部分配给对应节点。")

    # --- 步骤 3: 计算加权相似度并创建链接 (关键: 过滤 file_id) ---
    print("\n🚀 步骤 3: 开始计算加权相似度并创建链接...")
    
    link_creation_count = 0
    
    for paper, solution in product(paper_nodes, solution_nodes):
        
        # **关键过滤逻辑: 确保 Paper 和 Solution 来自不同的论文**
        if paper['file_id'] == solution['file_id']:
            # print(f"  - 💡 跳过 Paper({paper['file_id']}) 和 Solution({solution['file_id']})，因为它们属于同一篇论文。")
            continue 

        if paper['vectors']['abstract'] is None or paper['vectors']['core_problem'] is None or solution['vector'] is None:
            # print(f"  - ⚠️ 跳过 Paper({paper['node_id']}) 和 Solution({solution['node_id']})，因为缺少向量。")
            continue

        sim_abstract = cosine_similarity(paper['vectors']['abstract'], solution['vector'])
        sim_core_problem = cosine_similarity(paper['vectors']['core_problem'], solution['vector'])
        
        weighted_score = (sim_abstract * CONFIG['WEIGHTS']['abstract'] + 
                          sim_core_problem * CONFIG['WEIGHTS']['core_problem'])
        
        if weighted_score >= CONFIG['SIMILARITY_THRESHOLD']:
            link_creation_count += 1
            print(f"  ✨ 发现潜在**跨论文**关联 (总分: {weighted_score:.4f}):")
            print(f"     - Paper ID: {paper['file_id']} ({paper['node_id']})")
            print(f"     - Solution ID: {solution['file_id']} ({solution['node_id']} - {solution['label']})")
            
            # 使用 MERGE 语句在 Neo4j 中创建关系
            merge_query = f"""
            MATCH (p:paper), (s)
            WHERE elementId(p) = '{paper['node_id']}' AND elementId(s) = '{solution['node_id']}'
            MERGE (p)-[r:{CONFIG['NEW_RELATIONSHIP_TYPE']}]->(s)
            SET r.weightedScore = {weighted_score:.4f},
                r.abstractSimilarity = {sim_abstract:.4f},
                r.coreProblemSimilarity = {sim_core_problem:.4f},
                r.createdAt = timestamp()
            """
            try:
                n4j.query(query=merge_query)
                print(f"     ✅ 成功在图中创建 '{CONFIG['NEW_RELATIONSHIP_TYPE']}' 关系。")
            except Exception as e:
                print(f"     ❌ 创建关系失败: {e}")

    if link_creation_count == 0:
        print("\n✅ 完成计算，未发现相似度高于阈值的跨论文 Paper-Solution 对。")
    else:
        print(f"\n🎉 流程结束！总共创建了 {link_creation_count} 条新的**跨论文** '{CONFIG['NEW_RELATIONSHIP_TYPE']}' 关系。")
        
    # --- 新增: 保存更新后的缓存 ---
    save_embedding_cache(CONFIG["EMBEDDING_CACHE_BASE_PATH"], embedding_cache)

# =================================================================================
# 5. 脚本入口 (Script Entrypoint)
# =================================================================================
if __name__ == "__main__":
    print("=====================================================")
    print("=== Neo4j Paper-Solution 加权语义链接脚本启动 (已优化跨论文过滤) ===")
    print("=====================================================")
    main()
