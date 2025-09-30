import sys
print(f"Python Version: {sys.version}")
print(f"Sys Path: {sys.path}")

from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType
from camel.configs import MistralConfig, OllamaConfig, QwenConfig
from camel.loaders import UnstructuredIO
from camel.storages import Neo4jGraph
from camel.retrievers import AutoRetriever
from camel.embeddings import OpenAICompatibleEmbedding
from camel.types import StorageType
from camel.agents import ChatAgent, KnowledgeGraphAgent
from camel.messages import BaseMessage
from camel.storages import FaissStorage, VectorRecord, VectorDBQuery


import json
import os
from getpass import getpass


# # Prompt for the API key securely
# mistral_api_key = "amPLA3bl3H42UZSZaW9vL1qBEFo8P3KK"
# os.environ["MISTRAL_API_KEY"] = mistral_api_key

os.environ["OPENAI_COMPATIBILITY_API_KEY"] = "sk-c1a6b588f7d543adb0412c5bc61bdd7b"
os.environ["OPENAI_COMPATIBILITY_API_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"


os.environ["QWEN_API_KEY"] = os.environ["OPENAI_COMPATIBILITY_API_KEY"]
os.environ["QWEN_API_BASE_URL"] = os.environ["OPENAI_COMPATIBILITY_API_BASE_URL"]


# Set Neo4j instance
n4j = Neo4jGraph(
    url="neo4j+s://b3980610.databases.neo4j.io",
    username="neo4j",
    password="ta_T6_9gzxTfrTiWjRuUhO7Lm6fBbQG8TwxnSqHpoqk",
)


# Set up model
qwen_model = ModelFactory.create(
    model_platform=ModelPlatformType.QWEN,
    model_type=ModelType.COMETAPI_QWEN3_CODER_PLUS_2025_07_22, # Assuming Qwen3 maps to QWEN_TURBO, adjust if needed
    model_config_dict=QwenConfig(temperature=0.2).as_dict(),
    api_key=os.environ["QWEN_API_KEY"],
    url=os.environ["QWEN_API_BASE_URL"],
)


# Set instance
uio = UnstructuredIO()
kg_agent = KnowledgeGraphAgent(model=qwen_model)

# Set retriever
camel_retriever = AutoRetriever(
    vector_storage_local_path="local_data/embedding_storage",
    storage_type=StorageType.QDRANT,
    embedding_model=OpenAICompatibleEmbedding(
        model_type="text-embedding-v2", # 使用阿里云DashScope支持的embedding模型
        api_key=os.environ["OPENAI_COMPATIBILITY_API_KEY"],
        url=os.environ["OPENAI_COMPATIBILITY_API_BASE_URL"],
    ),
)

# Set one user query
query="How can a gating mechanism be designed to selectively filter and fuse information from auxiliary task towers into a main task tower, ensuring that only useful information is absorbed while irrelevant data is rejected?"

# ================== Vector Search Logic Start ==================

# 1. Setup paths and parameters
JSON_FILE_PATH = 'Myexamples/data/final_custom_kg_papers.json'
BASE_VDB_PATH = 'Myexamples/vdb/camel_faiss_storage'
paper_attributes = [
    "abstract", "core_problem", "related_work", "preliminary_innovation_analysis",
    "basic_problem", "datasets", "experimental_results", "framework_summary"
]

# 2. Build or load FAISS indexes for each attribute
attribute_storages = {}
embedding_model = camel_retriever.embedding_model

print("--- Initializing Vector Database ---")
for attr in paper_attributes:
    storage_path = os.path.join(BASE_VDB_PATH, attr)
    collection_name = f"paper_{attr}"
    
    # Check if storage exists by looking for the index file
    index_file_path = os.path.join(storage_path, f"{collection_name}.index")
    if os.path.exists(index_file_path):
        print(f"Loading existing FAISS storage for '{attr}' from {storage_path}")
        storage = FaissStorage(
            vector_dim=embedding_model.get_output_dim(),
            storage_path=storage_path,
            collection_name=collection_name,
        )
        storage.load()
    else:
        print(f"Building FAISS storage for '{attr}'...")
        os.makedirs(storage_path, exist_ok=True)
        storage = FaissStorage(
            vector_dim=embedding_model.get_output_dim(),
            storage_path=storage_path,
            collection_name=collection_name,
        )
        
        try:
            with open(JSON_FILE_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error: Could not load or parse {JSON_FILE_PATH}: {e}")
            continue

        records_to_add = []
        texts_to_embed = []
        metadata_for_records = []

        for entity in data.get("entities", []):
            if entity.get("entity_type") == "paper":
                paper_id = entity.get("source_id")
                attribute_text = entity.get(attr)
                if paper_id and attribute_text and isinstance(attribute_text, str) and attribute_text.strip():
                    texts_to_embed.append(attribute_text)
                    metadata_for_records.append({
                        "paper_id": paper_id,
                        "attribute_name": attr,
                        "text": attribute_text 
                    })
        
        if texts_to_embed:
            print(f"Embedding {len(texts_to_embed)} texts for '{attr}'...")
            
            # 分批处理，每批最多25个文本（阿里云DashScope的限制）
            batch_size = 25
            embeddings = []
            
            for i in range(0, len(texts_to_embed), batch_size):
                batch_texts = texts_to_embed[i:i + batch_size]
                print(f"  Processing batch {i//batch_size + 1}/{(len(texts_to_embed) + batch_size - 1)//batch_size}")
                batch_embeddings = embedding_model.embed_list(objs=batch_texts)
                embeddings.extend(batch_embeddings)
            
            for i, embedding in enumerate(embeddings):
                records_to_add.append(
                    VectorRecord(vector=embedding, payload=metadata_for_records[i])
                )
            
            print(f"Adding {len(records_to_add)} records to '{attr}' storage...")
            storage.add(records_to_add) # This also saves to disk
            print(f"Successfully built and saved storage for '{attr}'.")
    
    attribute_storages[attr] = storage
print("--- Vector Database Initialized ---\n")

# 3. Perform search and format results
print("--- Performing Vector Search ---")
query_embedding = embedding_model.embed(obj=query)

all_results = []
for attr, storage in attribute_storages.items():
    if storage.status().vector_count > 0:
        db_query = VectorDBQuery(query_vector=query_embedding, top_k=1)
        results = storage.query(query=db_query)
        if results:
            for res in results:
                res_text = (
                    f"Found in paper attribute '{res.record.payload.get('attribute_name', 'N/A')}':\n"
                    f"  - Similarity Score: {res.similarity:.4f}\n"
                    f"  - Paper ID: {res.record.payload.get('paper_id', 'N/A')}\n"
                    f"  - Content Snippet: {res.record.payload.get('text', '')}"
                )
                all_results.append(res_text)

vector_result = "\n\n".join(all_results)
if not vector_result:
    vector_result = "No relevant documents found in the local vector database."

# Show the result from vector search
print(vector_result)
print("====================================================\n")

# =================== Vector Search Logic End ===================

# Create an element from user query
query_element = uio.create_element_from_text(
    text=query, element_id="1"
)
print(query_element)
print("====================================================")
# Let Knowledge Graph Agent extract node and relationship information from the qyery
ans_element = kg_agent.run(query_element, parse_graph_elements=True)
print(ans_element)
print("====================================================")


# Match the entity got from query in the knowledge graph storage content
kg_result = []
for node in ans_element.nodes:
    # 这里的 `node.id` 应该对应你的知识图谱中的某个节点主键，例如 `file_id`
    # Cypher 查询需要根据你的实际数据模型进行修改
    
    # 新的、更强大的 Cypher 查询
    n4j_query = f"""
    // 1. 找到所有属性中包含我们关键词的节点
    MATCH (n)
    WHERE 
    // 遍历节点 n 的所有属性 (key)
    any(key IN keys(n) 
        // 检查属性值 (字符串类型) 是否包含我们的关键词
        WHERE toString(n[key]) CONTAINS '{node.id}'
    )

    // 2. 获取这些节点的邻居信息
    MATCH (n)-[r]-(m)

    // 3. 返回格式化的描述
    RETURN 
    'Node ' + coalesce(n.id, n.name, n.title, elementId(n)) + 
    ' (label: ' + labels(n)[0] + ')' +
    ' has relationship ' + type(r) + 
    ' with Node ' + coalesce(m.id, m.name, m.title, elementId(m)) + 
    ' (label: ' + labels(m)[0] + ')' 
    AS Description
    LIMIT 5 // 每个关键词最多返回5条相关信息，防止信息过载
    """
    
    result = n4j.query(query=n4j_query)
    kg_result.extend(result)

kg_result = [item['Description'] for item in kg_result]

# Show the result from knowledge graph database
print(kg_result)


# 构建结构化的上下文信息
structured_context = f"""
=== PRIMARY EVIDENCE (Vector Search Results) ===
{vector_result}

=== SUPPLEMENTARY CONTEXT (Knowledge Graph Relations) ===
{chr(10).join(kg_result) if kg_result else "No relevant graph relations found."}
"""

# 设计高级系统提示词
advanced_system_prompt = """You are an expert AI research assistant specialized in computer vision, machine learning, and related technical domains. Your mission is to provide comprehensive, accurate, and well-structured answers to technical research questions.

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
- If multiple approaches are mentioned in evidence, compare and contrast them appropriately"""

# Set agent with advanced prompt
sys_msg = BaseMessage.make_assistant_message(
    role_name="Expert Research Assistant",
    content=advanced_system_prompt,
)

camel_agent = ChatAgent(system_message=sys_msg, model=qwen_model)

# 构建结构化的用户查询
user_prompt = f"""
## Research Query:
{query}

## Available Evidence:
{structured_context}

## Task:
Please provide a comprehensive answer to the research query using the evidence above. Follow your response strategy to prioritize the most relevant information and provide a well-structured, technically accurate answer.
"""

user_msg = BaseMessage.make_user_message(role_name="CAMEL User", content=user_prompt)

# Get response
agent_response = camel_agent.step(user_msg)

print("====================================================")
print(agent_response.msg.content)

print("finished!")