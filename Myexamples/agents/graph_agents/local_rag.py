from typing import List, Optional

import os
import json

from camel.retrievers import AutoRetriever
from camel.types import StorageType
from camel.storages import FaissStorage, VectorRecord, VectorDBQuery
from camel.embeddings import OpenAICompatibleEmbedding
from camel.storages import Neo4jGraph
from camel.agents import ChatAgent, KnowledgeGraphAgent
from camel.messages import BaseMessage
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.configs import QwenConfig


def run_local_rag(
    query: str,
    json_file_path: str = 'Myexamples/data/final_custom_kg_papers.json',
    base_vdb_path: str = 'Myexamples/vdb/camel_faiss_storage',
    paper_attributes: List[str] | None = None,
    neo4j_url: Optional[str] = None,
    neo4j_username: Optional[str] = None,
    neo4j_password: Optional[str] = None,
    build_if_missing: bool = True,
) -> str:
    """运行本地向量库检索并返回结构化文本结果。

    - 若本地 FAISS 库已存在，则直接查询；
    - 若不存在，将跳过构建并返回提示信息（避免工具在无数据时阻塞）。
    - 仅依赖 DashScope/OpenAI 兼容 embedding 接口（通过环境变量提供）。
    """
    
    if paper_attributes is None:
        paper_attributes = [
            "abstract",
            "core_problem",
            "related_work",
            "preliminary_innovation_analysis",
            "basic_problem",
            "datasets",
            "experimental_results",
            "framework_summary",
        ]

    # 显式硬编码 API Key/URL（与 test_graph.py 一致），确保本函数独立可用
    os.environ["OPENAI_COMPATIBILITY_API_KEY"] = "sk-c1a6b588f7d543adb0412c5bc61bdd7b"
    os.environ["OPENAI_COMPATIBILITY_API_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    os.environ["QWEN_API_KEY"] = os.environ["OPENAI_COMPATIBILITY_API_KEY"]
    os.environ["QWEN_API_BASE_URL"] = os.environ["OPENAI_COMPATIBILITY_API_BASE_URL"]

    # 初始化嵌入模型（要求外部已设置 OPENAI_COMPATIBILITY_API_KEY/URL）
    DEBUG = str(os.environ.get("LOCAL_RAG_DEBUG", "")).lower() in ("1", "true", "yes", "y")
    if DEBUG:
        print(f"[RAG][DEBUG] run_local_rag called with query: {query}")
    try:
        embedding_model = OpenAICompatibleEmbedding(
            model_type="text-embedding-v2",
            api_key=os.environ.get("OPENAI_COMPATIBILITY_API_KEY") or os.environ.get("QWEN_API_KEY"),
            url=os.environ.get("OPENAI_COMPATIBILITY_API_BASE_URL") or os.environ.get("QWEN_API_BASE_URL"),
        )
    except Exception as e:
        if DEBUG:
            print("[RAG][DEBUG] Embedding model init failed:", repr(e))
        raise

    # 载入或跳过各属性向量库
    attribute_storages = {}
    built_attrs: List[str] = []
    loaded_attrs: List[str] = []
    for attr in paper_attributes:
        storage_path = os.path.join(base_vdb_path, attr)
        collection_name = f"paper_{attr}"
        index_file_path = os.path.join(storage_path, f"{collection_name}.index")
        if os.path.exists(index_file_path):
            storage = FaissStorage(
                vector_dim=embedding_model.get_output_dim(),
                storage_path=storage_path,
                collection_name=collection_name,
            )
            storage.load()
            attribute_storages[attr] = storage
        # 若不存在则不尝试构建，避免工具副作用

    if not attribute_storages:
        return (
            "No local FAISS vector databases found. "
            "Please build storages first or provide a valid base path."
        )

    # 若本地库缺失且允许，构建 FAISS 索引（对齐 test_graph 的构建流程，批量25）
    attribute_storages = {}
    for attr in paper_attributes:
        storage_path = os.path.join(base_vdb_path, attr)
        collection_name = f"paper_{attr}"
        index_file_path = os.path.join(storage_path, f"{collection_name}.index")
        if os.path.exists(index_file_path):
            storage = FaissStorage(
                vector_dim=embedding_model.get_output_dim(),
                storage_path=storage_path,
                collection_name=collection_name,
            )
            storage.load()
            attribute_storages[attr] = storage
            loaded_attrs.append(attr)
        elif build_if_missing:
            os.makedirs(storage_path, exist_ok=True)
            storage = FaissStorage(
                vector_dim=embedding_model.get_output_dim(),
                storage_path=storage_path,
                collection_name=collection_name,
            )
            # 从 JSON 构建
            try:
                with open(json_file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                # 如果无法构建则跳过该属性
                continue
            texts_to_embed: List[str] = []
            metadata_for_records: List[dict] = []
            for entity in data.get("entities", []):
                if entity.get("entity_type") == "paper":
                    paper_id = entity.get("source_id")
                    attribute_text = entity.get(attr)
                    if paper_id and attribute_text and isinstance(attribute_text, str) and attribute_text.strip():
                        texts_to_embed.append(attribute_text)
                        metadata_for_records.append({
                            "paper_id": paper_id,
                            "attribute_name": attr,
                            "text": attribute_text,
                        })
            if texts_to_embed:
                batch_size = 25
                embeddings: List[List[float]] = []
                for i in range(0, len(texts_to_embed), batch_size):
                    batch_texts = texts_to_embed[i:i + batch_size]
                    batch_embeddings = embedding_model.embed_list(objs=batch_texts)
                    embeddings.extend(batch_embeddings)
                records_to_add: List[VectorRecord] = []
                for i, embedding in enumerate(embeddings):
                    records_to_add.append(
                        VectorRecord(vector=embedding, payload=metadata_for_records[i])
                    )
                if records_to_add:
                    storage.add(records_to_add)
            attribute_storages[attr] = storage
            built_attrs.append(attr)

    if not attribute_storages:
        # 既无可用索引亦不允许构建，后续直接尝试图检索与答案生成
        if DEBUG:
            print("[RAG][DEBUG] No FAISS storages available (none loaded/built)")
        pass
    elif DEBUG:
        print(f"[RAG][DEBUG] FAISS storages -> loaded: {loaded_attrs}, built: {built_attrs}")

    # 向量检索查询
    try:
        query_embedding = embedding_model.embed(obj=query)
    except Exception as e:
        if DEBUG:
            print("[RAG][DEBUG] Embedding query failed:", repr(e))
        query_embedding = None
    all_results: List[str] = []
    for attr, storage in attribute_storages.items():
        if storage.status().vector_count > 0:
            results = []
            if query_embedding is not None:
                db_query = VectorDBQuery(query_vector=query_embedding, top_k=1)
                try:
                    results = storage.query(query=db_query)
                except Exception as e:
                    if DEBUG:
                        print(f"[RAG][DEBUG] Vector query failed for attr='{attr}':", repr(e))
            if results:
                for res in results:
                    res_text = (
                        f"Found in '{res.record.payload.get('attribute_name', attr)}':\n"
                        f"  - Similarity: {res.similarity:.4f}\n"
                        f"  - Paper ID: {res.record.payload.get('paper_id', 'N/A')}\n"
                        f"  - Content: {res.record.payload.get('text', '')}"
                    )
                    all_results.append(res_text)

    vector_result = "\n\n".join(all_results) if all_results else "No relevant documents found in the local vector database."
    if DEBUG:
        print(f"[RAG][DEBUG] Vector search hits: {len(all_results)}")

    # 图数据库检索（可选），并支持通过 KnowledgeGraphAgent 先抽取查询实体
    graph_result = ""
    try:
        url = neo4j_url or os.environ.get("NEO4J_URL")
        user = neo4j_username or os.environ.get("NEO4J_USERNAME")
        pwd = neo4j_password or os.environ.get("NEO4J_PASSWORD")
        # 若未提供，则回退到 test_graph.py 的硬编码配置
        if not (url and user and pwd):
            url = url or "neo4j+s://b3980610.databases.neo4j.io"
            user = user or "neo4j"
            pwd = pwd or "ta_T6_9gzxTfrTiWjRuUhO7Lm6fBbQG8TwxnSqHpoqk"
            if DEBUG:
                print("[RAG][DEBUG] Using default Neo4j credentials from test_graph.py")
        if url and user and pwd:
            n4j = Neo4jGraph(url=url, username=user, password=pwd)
            # 构造模型与 KG Agent（复用 test_graph 方案）
            qwen_model = ModelFactory.create(
                model_platform=ModelPlatformType.QWEN,
                model_type=ModelType.QWEN_MAX,
                model_config_dict=QwenConfig(temperature=0.2).as_dict(),
                api_key=os.environ.get("QWEN_API_KEY") or os.environ.get("OPENAI_COMPATIBILITY_API_KEY"),
                url=os.environ.get("QWEN_API_BASE_URL") or os.environ.get("OPENAI_COMPATIBILITY_API_BASE_URL"),
            )
            uio = None
            try:
                from camel.loaders import UnstructuredIO
                uio = UnstructuredIO()
                query_element = uio.create_element_from_text(text=query, element_id="1")
                kg_agent = KnowledgeGraphAgent(model=qwen_model)
                ans_element = kg_agent.run(query_element, parse_graph_elements=True)
                kg_nodes = [node.id for node in getattr(ans_element, 'nodes', [])]
                if DEBUG:
                    print(f"[RAG][DEBUG] KG extracted nodes: {kg_nodes}")
            except Exception:
                kg_nodes = []

            lines: List[str] = []
            if kg_nodes:
                for node_id in kg_nodes:
                    safe_node = str(node_id).replace("'", "\\'")
                    n4j_query = f"""
                    MATCH (n)
                    WHERE any(key IN keys(n) WHERE toString(n[key]) CONTAINS '{safe_node}')
                    MATCH (n)-[r]-(m)
                    RETURN 
                    'Node ' + coalesce(n.id, n.name, n.title, elementId(n)) +
                    ' (label: ' + labels(n)[0] + ')' +
                    ' has relationship ' + type(r) +
                    ' with Node ' + coalesce(m.id, m.name, m.title, elementId(m)) +
                    ' (label: ' + labels(m)[0] + ')' AS Description
                    LIMIT 5
                    """
                    try:
                        recs = n4j.query(query=n4j_query)
                        lines.extend([rec['Description'] for rec in recs])
                    except Exception:
                        continue
            else:
                # 回退：直接使用原始 query 进行图检索
                safe_query = query.replace("'", "\\'")
                n4j_query = f"""
                MATCH (n)
                WHERE any(key IN keys(n) WHERE toString(n[key]) CONTAINS '{safe_query}')
                MATCH (n)-[r]-(m)
                RETURN 
                'Node ' + coalesce(n.id, n.name, n.title, elementId(n)) +
                ' (label: ' + labels(n)[0] + ')' +
                ' has relationship ' + type(r) +
                ' with Node ' + coalesce(m.id, m.name, m.title, elementId(m)) +
                ' (label: ' + labels(m)[0] + ')' AS Description
                LIMIT 10
                """
                records = n4j.query(query=n4j_query)
                lines = [rec['Description'] for rec in records] if records else []
            graph_result = "\n".join(lines) if lines else "No relevant graph relations found."
            if DEBUG:
                print(f"[RAG][DEBUG] Graph relations count: {0 if not graph_result or graph_result.startswith('No ') else len(lines)}")
    except Exception:
        graph_result = "Graph retrieval skipped due to configuration or connection error."
        if DEBUG:
            print("[RAG][DEBUG] Graph retrieval skipped.")

    # 组合证据，构建高级系统提示并用 ChatAgent 生成最终答案
    structured_context = (
        "=== PRIMARY EVIDENCE (Local Vector Search) ===\n"
        f"{vector_result}\n\n"
        "=== SUPPLEMENTARY CONTEXT (Knowledge Graph Relations) ===\n"
        f"{graph_result}"
    )

    advanced_system_prompt = (
        """
        You are an expert AI research assistant specialized in computer vision, machine learning, and related technical domains. Your mission is to provide comprehensive, accurate, and well-structured answers to technical research questions.

        ## Core Capabilities:
        - Deep understanding of computer vision, machine learning, and AI research methodologies
        - Ability to synthesize information from multiple sources with different reliability levels
        - Expert knowledge in technical concepts, algorithms, frameworks, and experimental designs
        - Skilled in identifying and prioritizing the most relevant information for answering complex queries

        ## Response Strategy:
        1. **Information Prioritization**: 
           - PRIMARY: Focus primarily on "Primary Evidence" which contains highly relevant, semantically matched content
           - SUPPLEMENTARY: Use "Supplementary Context" only if it adds valuable insights not covered in primary evidence
           - FILTERING: Ignore any information that is clearly irrelevant to the query domain or topic

        2. **Answer Structure**:
           - Begin with a direct, concise answer based *strictly* on the provided evidence.
           - Then, create a clearly marked "### Expert Elaboration" section. In this section, if the evidence mentions a technical concept (e.g., 'channel attention') but lacks implementation details, elaborate on its typical mechanism and how it plausibly works in this specific context.
           - Use formatting (bullet points, emphasis) to structure technical details for clarity.
           - Conclude with a summary of key insights.

        3. **Quality Assurance**:
           - In the main answer, ensure technical accuracy based *only* on the evidence.
           - In the "Expert Elaboration" section, clearly state that this is a reasoned inference based on established principles, as the source text lacks full detail.
           - Avoid speculation beyond what is plausible for an expert in the field.

        4. **Communication Style**:
           - Use clear, professional academic language
           - Structure information logically with proper technical terminology
           - Provide sufficient detail for technical understanding while remaining accessible
           - Use formatting (bullet points, emphasis) to enhance clarity when helpful

        ## Critical Guidelines:
        - Never fabricate technical details not present in the provided evidence
        - Prioritize evidence quality over quantity
        - Focus on answering the specific question asked rather than providing general background
        - If multiple approaches are mentioned in evidence, compare and contrast them appropriately
        """
    ).strip()

    try:
        qwen_model = ModelFactory.create(
            model_platform=ModelPlatformType.QWEN,
            model_type=ModelType.QWEN_MAX,
            model_config_dict=QwenConfig(temperature=0.2).as_dict(),
            api_key=os.environ.get("QWEN_API_KEY") or os.environ.get("OPENAI_COMPATIBILITY_API_KEY"),
            url=os.environ.get("QWEN_API_BASE_URL") or os.environ.get("OPENAI_COMPATIBILITY_API_BASE_URL"),
        )
        sys_msg = BaseMessage.make_assistant_message(
            role_name="Expert Research Assistant",
            content=advanced_system_prompt,
        )
        camel_agent = ChatAgent(system_message=sys_msg, model=qwen_model)
        user_prompt = (
            f"## Research Query:\n{query}\n\n"
            f"## Available Evidence:\n{structured_context}\n\n"
            "## Task:\n"
            "Please provide a comprehensive answer to the research query using the evidence above. "
            "Follow your response strategy to prioritize the most relevant information and provide a well-structured, technically accurate answer."
        )
        user_msg = BaseMessage.make_user_message(role_name="CAMEL User", content=user_prompt)
        if DEBUG:
            print("[RAG][DEBUG] Invoking LLM to synthesize final answer...")
        agent_response = camel_agent.step(user_msg)
        final_answer = agent_response.msg.content if agent_response and agent_response.msg else None
        if final_answer and isinstance(final_answer, str) and final_answer.strip():
            if DEBUG:
                print("[RAG][DEBUG] Final answer generated by LLM (length):", len(final_answer))
                print(final_answer)
                return "[RAG_CALLED]\n" + final_answer
            return final_answer
    except Exception:
        # 回退到返回证据，供上层综合
        if DEBUG:
            print("[RAG][DEBUG] LLM synthesis failed, falling back to evidence.")

    # 回退：返回结构化证据文本
    if DEBUG:
        print("[RAG][DEBUG] Returning structured evidence only (no LLM answer)")
        return "[RAG_CALLED]\n" + structured_context + "\n"
    return structured_context + "\n"


