# /root/autodl-tmp/LightRAG/Mydemo/知识图谱/visualize.py
import os
import datetime
import random
import networkx as nx
from pathlib import Path

try:
    from pyvis.network import Network
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyvis"])
    from pyvis.network import Network

# 1. 路径常量（自适应当前项目根目录）
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parents[2]  # LightRAG 项目根目录
GRAPH_PATH = Path("/root/autodl-tmp/LightRAG/Mydemo/知识图谱/dickens/graph_chunk_entity_relation.graphml")
SAVE_DIR = PROJECT_ROOT / "Mydemo" / "知识图谱" / "html_reports"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

def generate_html(add_timestamp: bool = False):
    if not GRAPH_PATH.exists():
        raise FileNotFoundError(f"GraphML 文件不存在：{GRAPH_PATH}")

    # 2. 读取 NetworkX 图
    G = nx.read_graphml(GRAPH_PATH)
    # 安全过滤：仅在节点存在且两端不相同的情况下保留边
    # 并保留跨论文边的edge_type标记
    H = nx.Graph()
    for n, d in G.nodes(data=True):
        H.add_node(n, **d)
    for u, v, d in G.edges(data=True):
        if u != v and u in H and v in H:
            # 统一跨论文边标记
            et = d.get("edge_type") or d.get("relationship") or d.get("type")
            if et and "cross_paper" in str(et):
                d = dict(d)
                d["edge_type"] = "cross_paper"
            H.add_edge(u, v, **d)
    G = H

    # 3. 创建 Pyvis 网络（内联资源，避免外网依赖；启用物理引擎，恢复动态效果）
    net = Network(height="100vh", width="100%", cdn_resources="in_line", bgcolor="#ffffff")
    net.set_options(
        '{\n'
        '  "physics": {\n'
        '    "enabled": true,\n'
        '    "stabilization": {"iterations": 100},\n'
        '    "barnesHut": {\n'
        '      "gravitationalConstant": -8000,\n'
        '      "centralGravity": 0.3,\n'
        '      "springLength": 95,\n'
        '      "springConstant": 0.04,\n'
        '      "damping": 0.09\n'
        '    }\n'
        '  },\n'
        '  "edges": { "smooth": true, "arrows": { "to": { "enabled": false } } },\n'
        '  "interaction": {\n'
        '    "hover": true,\n'
        '    "navigationButtons": true,\n'
        '    "dragNodes": true,\n'
        '    "dragView": true,\n'
        '    "zoomView": true\n'
        '  },\n'
        '  "nodes": { "shape": "dot", "size": 10, "font": { "size": 12 } }\n'
        '}'
    )
    net.from_nx(G)

    # 4. 节点与边样式：高亮跨论文边，节点按类型着色
    type_to_color = {
        "paper": "#1f77b4",               # 蓝色
        "research_question": "#ff7f0e",   # 橙色
        "solution": "#2ca02c",            # 绿色
    }

    # 节点样式
    # PyVis节点对象中，data保留了NetworkX的属性
    node_attr_map = {n["id"]: n for n in net.nodes}
    for node_id, data in G.nodes(data=True):
        n = node_attr_map.get(str(node_id))
        if not n:
            continue
        entity_type = data.get("entity_type", "")
        n["color"] = type_to_color.get(entity_type, "#888888")
        # 更丰富的悬浮提示
        title_parts = [f"<b>{str(node_id)}</b>"]
        if entity_type:
            title_parts.append(f"类型: {entity_type}")
        for key in ["title", "abstract", "research_question", "solution", "description"]:
            if data.get(key):
                # 限制tooltip长度，避免前端卡顿
                val = str(data.get(key)).replace("\n", " ")
                if len(val) > 180:
                    val = val[:180] + "..."
                title_parts.append(f"{key}: {val}")
        n["title"] = "<br/>".join(title_parts)

    # 边样式：跨论文边加粗并标红
    cross_edges = 0
    # 在过滤前先打上跨论文标记，便于优先保留
    def is_cross_edge(edge_dict):
        et = None
        for key in ("edge_type", "type", "relationship"):
            if edge_dict.get(key) and "cross_paper" in str(edge_dict.get(key)):
                et = "cross_paper"
                break
        if et == "cross_paper":
            return True
        sid = str(edge_dict.get("from", ""))
        tid = str(edge_dict.get("to", ""))
        def paper_id(x: str) -> str:
            return x.split("_RQ_")[0] if "_RQ_" in x else (x.split("_SOL_")[0] if "_SOL_" in x else x)
        return sid and tid and paper_id(sid) != paper_id(tid)

    for e in net.edges:
        e["_is_cross"] = is_cross_edge(e)

    # 由于大规模边渲染可能导致前端卡顿，限制渲染的最大边数（优先保留跨论文边）
    MAX_EDGES = 20000
    if len(net.edges) > MAX_EDGES:
        cross_list = [e for e in net.edges if e.get("_is_cross")]
        non_cross_list = [e for e in net.edges if not e.get("_is_cross")]
        kept = cross_list[:MAX_EDGES]
        if len(kept) < MAX_EDGES:
            # 简单按节点度保留部分普通边
            degrees = dict(G.degree())
            top_nodes = set(sorted(degrees, key=degrees.get, reverse=True)[:3000])
            for e in non_cross_list:
                sid = str(e.get("from", ""))
                tid = str(e.get("to", ""))
                if sid in top_nodes and tid in top_nodes:
                    kept.append(e)
                    if len(kept) >= MAX_EDGES:
                        break
        net.edges = kept

    for e in net.edges:
        # 从NetworkX读取到的属性会被复制到pyvis edge中
        title = e.get("description", "")
        e["title"] = title
        # 判断是否跨论文：优先依据edge_type，其次依据两端节点前缀
        edge_type = None
        # pyvis将所有属性拍平成简单dict，GraphML中的属性会映射为键
        # 这里尝试多个常见键名
        for key in ("edge_type", "type", "relationship"):
            if e.get(key) and "cross_paper" in str(e.get(key)):
                edge_type = "cross_paper"
                break

        if edge_type != "cross_paper":
            # 回退：根据节点ID是否同论文判断（_RQ_/_SOL_后缀）
            sid = str(e.get("from", ""))
            tid = str(e.get("to", ""))
            def paper_id(x: str) -> str:
                return x.split("_RQ_")[0] if "_RQ_" in x else (x.split("_SOL_")[0] if "_SOL_" in x else x)
            if paper_id(sid) != paper_id(tid):
                edge_type = "cross_paper"

        if edge_type == "cross_paper":
            e["color"] = "#ff0000"  # 更鲜艳的红色
            e["width"] = 3.0        # 更粗的线条
            e["dashes"] = False     # 实线
            cross_edges += 1
        else:
            e["color"] = "#e0e0e0"  # 更淡的灰色，突出跨论文边
            e["width"] = 0.5        # 更细的普通边

    # 5. 生成文件名
    if add_timestamp:
        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_name = f"knowledge_graph_{ts}.html"
    else:
        file_name = "knowledge_graph.html"

    html_path = SAVE_DIR / file_name

    # 6. 写 HTML（绕过 net.show 的 notebook 参数问题）
    # 强制以utf-8写入以避免Windows控制台默认gbk编码问题
    html_str = net.generate_html(notebook=False)
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_str)
    print(f"HTML 已保存：{html_path}")
    print(f"可视化统计：节点 {G.number_of_nodes()}，边 {G.number_of_edges()}，其中跨论文边(标红) {cross_edges}")
    return html_path

# 7. 命令行可直接调用
if __name__ == "__main__":
    generate_html(add_timestamp=True)