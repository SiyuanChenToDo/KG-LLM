import asyncio
import os
import inspect
import logging
import logging.config
import random
from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc, logger, set_verbose_debug
from lightrag.kg.shared_storage import initialize_pipeline_status
from build_custom_kg import load_custom_kg
from cross_paper_linking import integrate_cross_paper_linking
from dotenv import load_dotenv
from pathlib import Path
import networkx as nx

# 导入节点属性查询API
import sys
import os
# 添加正确的路径以导入API
current_dir = os.path.dirname(os.path.abspath(__file__))  # 当前文件目录
parent_dir = os.path.dirname(current_dir)  # Mydemo目录
sys.path.insert(0, parent_dir)

try:
    from node_attributes_api import (
        get_paper_attributes, 
        get_research_question_attributes, 
        get_solution_attributes,
        find_paper_by_title,
        get_paper_research_questions,
        print_node_attributes
    )
    API_AVAILABLE = True
    print("✅ 节点属性API导入成功")
except ImportError as e:
    print(f"警告: 无法导入节点属性API: {e}")
    API_AVAILABLE = False

load_dotenv(dotenv_path=".env", override=False)

WORKING_DIR = "./dickens"

#日志
def configure_logging():
    """Configure logging for the application"""

    # Reset any existing handlers to ensure clean configuration
    for logger_name in ["uvicorn", "uvicorn.access", "uvicorn.error", "lightrag"]:
        logger_instance = logging.getLogger(logger_name)
        logger_instance.handlers = []
        logger_instance.filters = []

    # Get log directory path from environment variable or use current directory
    log_dir = os.getenv("LOG_DIR", os.getcwd())
    log_file_path = os.path.abspath(os.path.join(log_dir, "lightrag_ollama_demo.log"))

    print(f"\nLightRAG compatible demo log file: {log_file_path}\n")
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    # Get log file max size and backup count from environment variables
    log_max_bytes = int(os.getenv("LOG_MAX_BYTES", 10485760))  # Default 10MB
    log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", 5))  # Default 5 backups

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(levelname)s: %(message)s",
                },
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
            "handlers": {
                "console": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr",
                },
                "file": {
                    "formatter": "detailed",
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": log_file_path,
                    "maxBytes": log_max_bytes,
                    "backupCount": log_backup_count,
                    "encoding": "utf-8",
                },
            },
            "loggers": {
                "lightrag": {
                    "handlers": ["console", "file"],
                    "level": "INFO",
                    "propagate": False,
                },
            },
        }
    )

    # Set the logger level to INFO (减少冗余日志)
    logger.setLevel(logging.INFO)  # 🔥 改回INFO级别，减少冗余输出
    # Enable verbose debug if needed
    set_verbose_debug(os.getenv("VERBOSE_DEBUG", "false").lower() == "true")  # 🔥 关闭详细调试


if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


async def initialize():
    # 提取配置参数以便调试
    embed_model = os.getenv("EMBEDDING_MODEL", "bge-m3:latest")  # 添加:latest标
    embed_host = os.getenv("EMBEDDING_BINDING_HOST", "http://localhost:11434")
    llm_model = os.getenv("LLM_MODEL", "qwen3:14b")
    llm_host = os.getenv("LLM_BINDING_HOST", "http://localhost:11434")
    embedding_dim = int(os.getenv("EMBEDDING_DIM", "1024"))

    # 使用默认存储后端
    graph_backend = "json"  # 使用默认的NetworkX/JSON后端
    vector_backend = "json"  # 使用默认的NanoVectorDB后端

    # 打印配置信息用于调试
    print("\n===== RAG 初始化配置 =====")
    print(f"工作目录: {WORKING_DIR}")
    print(f"LLM模型: {llm_model}")
    print(f"LLM服务地址: {llm_host}")
    print(f"嵌入模型: {embed_model}")
    print(f"嵌入服务地址: {embed_host}")
    print(f"嵌入维度: {embedding_dim}")
    print(f"GRAPH_BACKEND: {graph_backend}")
    print(f"VECTOR_BACKEND: {vector_backend}")
    print(f"==========================\n")


    # 使用默认的存储后端
    graph_storage_name = "NetworkXStorage"  # NetworkX/JSON存储
    vector_storage_name = "NanoVectorDBStorage"  # 默认向量存储

    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=ollama_model_complete,
        llm_model_name=llm_model,
        llm_model_max_token_size=8192,
        llm_model_kwargs={
            "host": llm_host,
            "options": {"num_ctx": 8192},
            "timeout": int(os.getenv("TIMEOUT", "600")),  # 增加超时时间
        },
        embedding_func=EmbeddingFunc(
            embedding_dim=embedding_dim,
            max_token_size=int(os.getenv("MAX_EMBED_TOKENS", "8192")),
            func=lambda texts: ollama_embed(
                texts,
                embed_model=embed_model,
                host=embed_host,
                timeout=60,  # 添加超时设置
            ),
        ),
        # 🔥 禁用rerank以避免警告信息（如需启用，参考RERANK_配置指南.md）
        rerank_model_func=None,
        # 覆盖后端（字符串方式）
        graph_storage=graph_storage_name,
        vector_storage=vector_storage_name,
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()

    return rag


async def print_stream(stream):
    async for chunk in stream:
        print(chunk, end="", flush=True)


async def _demonstrate_lightrag_query_only(rag):
    """只演示LightRAG环境查询（文件访问失败时的后备方案）"""
    print(f"\n🔄 方法2: 在LightRAG环境中查询 (最完整)")
    print("-" * 50)
    
    try:
        graph = rag.chunk_entity_relation_graph
        all_labels = await graph.get_all_labels()
        
        # 找到第一个paper节点
        paper_nodes = []
        for label in all_labels:
            node_data = await graph.get_node(label)
            if node_data and node_data.get("entity_type") == "paper":
                paper_nodes.append((label, node_data))
                break
        
        if paper_nodes:
            label, node_data = paper_nodes[0]
            print(f"✅ 从LightRAG获取论文: {node_data.get('title', 'N/A')}")
            print(f"   节点ID: {label}")
            print(f"   实体类型: {node_data.get('entity_type', 'N/A')}")
            
            # 获取该节点的所有关系
            edges = await graph.get_node_edges(label)
            print(f"   关联关系数: {len(edges)}")
            
            # 显示前3个关系
            for i, (src, tgt) in enumerate(edges[:3], 1):
                edge_data = await graph.get_edge(src, tgt)
                if edge_data:
                    keywords = edge_data.get("keywords", "无关键词")
                    # 确定关系方向
                    if src == label:
                        print(f"     关系{i}: 该论文 → {tgt[:30]}... (关键词: {keywords})")
                    else:
                        print(f"     关系{i}: {src[:30]}... → 该论文 (关键词: {keywords})")
        
        # 统计不同类型节点的连接度
        print(f"\n📊 节点连接统计:")
        node_connections = {"paper": [], "research_question": [], "solution": []}
        
        for label in all_labels[:20]:  # 分析前20个节点
            node_data = await graph.get_node(label)
            if node_data:
                entity_type = node_data.get("entity_type", "unknown")
                if entity_type in node_connections:
                    edges = await graph.get_node_edges(label)
                    node_connections[entity_type].append(len(edges))
        
        for node_type, connections in node_connections.items():
            if connections:
                avg_connections = sum(connections) / len(connections)
                print(f"   {node_type}节点平均连接数: {avg_connections:.1f}")
    
    except Exception as e:
        print(f"❌ LightRAG查询失败: {e}")


async def perform_cross_paper_analysis(rag):
    """执行跨论文关联分析"""
    print("\n" + "="*70)
    print("🔗 跨论文关联分析")
    print("="*70)
    
    try:
        # 进度回调函数
        def progress_callback(stage, current, total):
            percentage = (current / total * 100) if total > 0 else 0
            print(f"\r🔄 {stage}: {current}/{total} ({percentage:.1f}%)", end="", flush=True)
        
        # 执行跨论文关联分析
        print("🚀 开始跨论文关联分析...")
        
        # 🔧 使用合理阈值确保高质量跨论文连接
        threshold = 0.5
        print(f"📊 使用阈值: {threshold} ({threshold*100:.0f}%) - 确保高质量Solution→Paper连接")
        added_count, new_edges = await integrate_cross_paper_linking(
            rag_instance=rag,
            similarity_threshold=threshold,  # 🔧 使用合理阈值
            progress_callback=progress_callback
        )
        
        print(f"\n✅ 跨论文关联分析完成！")
        print(f"📊 发现 {len(new_edges)} 个跨论文关联")
        print(f"📈 成功添加 {added_count} 条新边到知识图谱")
        
        if new_edges:
            print(f"\n🔍 跨论文关联示例（前5个）:")
            for i, edge in enumerate(new_edges[:5], 1):
                print(f"   {i}. {edge['source'][:30]}... ↔ {edge['target'][:30]}...")
                print(f"      相似度: {edge['similarity_score']:.3f}")
                print(f"      描述: {edge['description']}")
                print()
        
        # 从GraphML读取图统计
        try:
            graphml_path = Path(WORKING_DIR) / "graph_chunk_entity_relation.graphml"
            G = nx.read_graphml(str(graphml_path))
            total_nodes = G.number_of_nodes()
            total_edges = G.number_of_edges()
            cross_edges = sum(
                1 for _, _, d in G.edges(data=True)
                if (d.get("edge_type") == "cross_paper") or ("cross_paper" in str(d.get("relationship", "")))
            )
            print(f"📊 更新后的图统计:")
            print(f"   总节点数: {total_nodes}")
            print(f"   跨论文边数: {cross_edges}")
        except Exception as e:
            # 回退到轻量统计
            graph = rag.chunk_entity_relation_graph
            all_labels = await graph.get_all_labels()
            print(f"📊 更新后的图统计:")
            print(f"   总节点数: {len(all_labels)}")
            print(f"   注意: GraphML文件不可用，跳过边数统计")
        
    except Exception as e:
        print(f"\n❌ 跨论文关联分析失败: {e}")
        import traceback
        traceback.print_exc()


async def demonstrate_node_query_methods(rag):
    """演示三种节点属性查询方法的区别"""
    print("\n" + "="*70)
    print("🚀 演示三种节点属性查询方法")
    print("="*70)
    
    if not API_AVAILABLE:
        print("❌ 节点属性API不可用，跳过演示")
        return
    
    # 🔧 检查知识图谱文件是否存在
    try:
        from node_attributes_api import get_kg_file_path
        kg_file_path = get_kg_file_path()
        print(f"✅ 找到知识图谱文件: {kg_file_path}")
    except Exception as e:
        print(f"⚠️ 无法访问知识图谱文件: {e}")
        print("📋 将只演示方法2（LightRAG环境查询）")
        # 只演示LightRAG查询，然后返回
        await _demonstrate_lightrag_query_only(rag)
        return
    
    # ===================
    # 方法1: 直接从JSON文件查询 (最快速)
    # ===================
    print("\n📁 方法1: 直接从JSON文件查询 (最快速)")
    print("-" * 50)
    
    # 获取第一篇论文的属性
    try:
        paper = get_paper_attributes(0)
    except Exception as e:
        print(f"❌ 获取paper属性失败: {e}")
        paper = None
    if paper:
        print(f"✅ 成功获取论文: {paper['title']}")
        print(f"   作者: {paper.get('authors', 'N/A')}")
        print(f"   年份: {paper.get('year', 'N/A')}")
        print(f"   会议: {paper.get('conference', 'N/A')}")
        
        # 获取该论文的研究问题
        rqs = get_paper_research_questions(paper['entity_name'])
        print(f"   研究问题数: {len(rqs)}")
        for i, rq in enumerate(rqs, 1):  # 显示所有研究问题
            print(f"     RQ{i}: {rq.get('research_question', 'N/A')}")
    
    # 根据关键词搜索论文
    attention_papers = find_paper_by_title("Attention")
    print(f"\n🔍 包含'Attention'的论文: {len(attention_papers)}篇")
    for paper in attention_papers[:2]:  # 显示前2篇
        print(f"   📄 {paper['title']}")
    
    # ===================
    # 方法2: 在LightRAG环境中查询 (最完整)
    # ===================
    print(f"\n🔄 方法2: 在LightRAG环境中查询 (最完整)")
    print("-" * 50)
    
    try:
        graph = rag.chunk_entity_relation_graph
        all_labels = await graph.get_all_labels()
        
        # 找到第一个paper节点
        paper_nodes = []
        for label in all_labels:
            node_data = await graph.get_node(label)
            if node_data and node_data.get("entity_type") == "paper":
                paper_nodes.append((label, node_data))
                break
        
        if paper_nodes:
            label, node_data = paper_nodes[0]
            print(f"✅ 从LightRAG获取论文: {node_data.get('title', 'N/A')}")
            print(f"   节点ID: {label}")
            print(f"   实体类型: {node_data.get('entity_type', 'N/A')}")
            
            # 获取该节点的所有关系
            edges = await graph.get_node_edges(label)
            print(f"   关联关系数: {len(edges)}")
            
            # 显示前3个关系
            for i, (src, tgt) in enumerate(edges[:3], 1):
                edge_data = await graph.get_edge(src, tgt)
                if edge_data:
                    keywords = edge_data.get("keywords", "无关键词")
                    # 确定关系方向
                    if src == label:
                        print(f"     关系{i}: 该论文 → {tgt[:30]}... (关键词: {keywords})")
                    else:
                        print(f"     关系{i}: {src[:30]}... → 该论文 (关键词: {keywords})")
    
    except Exception as e:
        print(f"❌ LightRAG查询失败: {e}")
    
    # ===================
    # 方法3: 特定节点查询和分析 (最灵活)
    # ===================
    print(f"\n🎯 方法3: 特定节点查询和分析 (最灵活)")
    print("-" * 50)
    
    try:
        # 演示复杂查询：找到包含"domain adaptation"的论文及其完整子图
        print("🔍 搜索'domain adaptation'相关论文...")
        
        # 方法3a: 通过JSON进行复杂分析
        papers, rqs, solutions = [], [], []
        
        # 读取所有数据进行分析
        import json
        # 使用API的路径检测功能
        from node_attributes_api import get_kg_file_path
        kg_file_path = get_kg_file_path()
        with open(kg_file_path, 'r', encoding='utf-8') as f:
            kg_data = json.load(f)
        
        entities = kg_data["entities"]
        
        # 查找相关论文
        domain_papers = []
        for entity in entities:
            if entity["entity_type"] == "paper":
                title = entity.get("title", "").lower()
                abstract = entity.get("abstract", "").lower()
                if "domain" in title or "adaptation" in title or "domain" in abstract:
                    domain_papers.append(entity)
        
        print(f"✅ 找到 {len(domain_papers)} 篇域适应相关论文")
        
        # 对每篇论文进行子图分析
        for i, paper in enumerate(domain_papers[:2], 1):  # 分析前2篇
            print(f"\n   📄 论文{i}: {paper['title']}")
            
            # 找到该论文的研究问题
            paper_rqs = get_paper_research_questions(paper['entity_name'])
            print(f"      研究问题数: {len(paper_rqs)}")
            
            for j, rq in enumerate(paper_rqs, 1):
                print(f"        RQ{j}: {rq.get('simplified_research_question', 'N/A')[:50]}...")
                
                # 找到对应的解决方案
                from node_attributes_api import get_research_question_solutions
                solutions = get_research_question_solutions(rq['entity_name'])
                for k, sol in enumerate(solutions, 1):
                    print(f"          SOL{k}: {sol.get('simplified_solution', 'N/A')[:50]}...")
        
        # 方法3b: 在LightRAG中进行关系分析
        print(f"\n🔗 在LightRAG中分析节点关系...")
        graph = rag.chunk_entity_relation_graph
        
        # 统计不同类型节点的连接度
        node_connections = {"paper": [], "research_question": [], "solution": []}
        
        for label in all_labels[:20]:  # 分析前20个节点
            node_data = await graph.get_node(label)
            if node_data:
                entity_type = node_data.get("entity_type", "unknown")
                if entity_type in node_connections:
                    edges = await graph.get_node_edges(label)
                    node_connections[entity_type].append(len(edges))
        
        for node_type, connections in node_connections.items():
            if connections:
                avg_connections = sum(connections) / len(connections)
                print(f"   {node_type}节点平均连接数: {avg_connections:.1f}")
    
    except Exception as e:
        print(f"❌ 特定查询失败: {e}")
    
    # ===================
    # 总结三种方法的优劣
    # ===================
    print(f"\n📊 三种方法对比总结:")
    print("-" * 50)
    print("📁 方法1 (JSON查询):   速度快 | 简单 | 功能基础")
    print("🔄 方法2 (LightRAG):   功能全 | 实时 | 需初始化")  
    print("🎯 方法3 (特定查询):   灵活 | 复杂 | 性能中等")
    print()
    print("💡 建议使用场景:")
    print("   - 快速查看属性 → 使用方法1")
    print("   - 分析图结构关系 → 使用方法2") 
    print("   - 复杂搜索分析 → 使用方法3")


async def main():
    rag = None  # 初始化变量以避免finally块中的引用错误
    try:
        #Clear old data files
        files_to_delete = [
            "graph_chunk_entity_relation.graphml",
            "kv_store_doc_status.json",
            "kv_store_full_docs.json",
            "kv_store_text_chunks.json",
            "vdb_chunks.json",
            "vdb_entities.json",
            "vdb_relationships.json",
        ]

        for file in files_to_delete:
            file_path = os.path.join(WORKING_DIR, file)
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Deleting old file:: {file_path}")

        # Initialize RAG instance
        rag = await initialize()

        custom_kg = load_custom_kg()

        await rag.ainsert_custom_kg(custom_kg)


        # 📊 验证知识图谱插入结果
        print("\n=====================")
        print("📊 验证论文知识图谱插入结果")
        print("=====================")
        
        # 检查插入的实体数量
        graph = rag.chunk_entity_relation_graph
        all_labels = await graph.get_all_labels()
        print(f"✅ 总实体数: {len(all_labels)}")
        
        # 统计实体类型
        entity_types = {}
        for label in all_labels:
            node_data = await graph.get_node(label)
            if node_data:
                entity_type = node_data.get("entity_type", "unknown")
                entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
        
        print("📋 实体类型分布:")
        for entity_type, count in entity_types.items():
            print(f"   {entity_type}: {count}个")

        # 🚀 演示三种节点属性查询方法
        #await demonstrate_node_query_methods(rag)

        # 🔗 跨论文关联分析
        #await perform_cross_paper_analysis(rag)


        print("\n🎉 论文知识图谱查询测试完成！")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if rag:
            await rag.llm_response_cache.index_done_callback()
            await rag.finalize_storages()

            # 使用传统GraphML可视化
            try:
                from visualize import generate_html
                generate_html(add_timestamp=True)
            except Exception as e:
                print(f"⚠️ 可视化生成失败: {e}")



if __name__ == "__main__":
    # Configure logging before running the main function
    configure_logging()
    asyncio.run(main())
    print("\nDone!")