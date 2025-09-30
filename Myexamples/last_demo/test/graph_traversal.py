import networkx as nx
import os

def load_graph(graphml_file_path):
    """
    加载GraphML文件并返回NetworkX图对象。
    """
    if not os.path.exists(graphml_file_path):
        raise FileNotFoundError(f"文件未找到: {graphml_file_path}")
    return nx.read_graphml(graphml_file_path)

def get_node_type(graph, node_id):
    """
    获取节点的类型。
    """
    return graph.nodes[node_id].get('entity_type')

def find_paths(graph, start_node_id, path_length, node_type_sequence):
    """
    查找指定长度和节点类型序列的路径。
    """
    paths = []
    
    # 广度优先搜索 (BFS) 寻找路径
    queue = [(start_node_id, [start_node_id])]
    
    while queue:
        current_node, current_path = queue.pop(0)
        
        if len(current_path) == path_length + 1:
            # 检查路径中每个节点的类型是否符合预期序列
            valid_path = True
            for i, node_id in enumerate(current_path):
                expected_type = node_type_sequence[i]
                actual_type = get_node_type(graph, node_id)
                if expected_type == "*" : # * 代表任何类型
                    continue
                if actual_type != expected_type:
                    valid_path = False
                    break
            if valid_path:
                paths.append(current_path)
            continue

        for neighbor in graph.neighbors(current_node):
            if neighbor not in current_path:  # 避免循环
                queue.append((neighbor, current_path + [neighbor]))
    return paths

def format_path(graph, path):
    """
    格式化路径显示，包含节点名称和类型。
    """
    formatted_nodes = []
    for node_id in path:
        entity_name = graph.nodes[node_id].get('entity_name', node_id)
        entity_type = graph.nodes[node_id].get('entity_type', 'unknown')
        formatted_nodes.append(f"{entity_name} ({entity_type})")
    return " -> ".join(formatted_nodes)

if __name__ == "__main__":
    graphml_file = "/root/autodl-tmp/LightRAG/Mydemo/知识图谱/dickens/graph_chunk_entity_relation.graphml"
    
    try:
        graph = load_graph(graphml_file)
        print(f"成功加载知识图谱文件: {graphml_file}")
        print(f"图谱包含 {graph.number_of_nodes()} 个节点和 {graph.number_of_edges()} 条边。")

        # 找到所有 'paper' 类型的节点
        paper_nodes = [node_id for node_id, data in graph.nodes(data=True) if data.get('entity_type') == 'paper']
        
        if not paper_nodes:
            print("未找到 'paper' 类型的节点。")
        else:
            print("\n发现 'paper' 节点：")
            for paper_node in paper_nodes:
                print(f"- {graph.nodes[paper_node].get('entity_name', paper_node)}")

                # 查找并显示正向路径：paper -> research_question -> solution
                print(f"  正向路径 (Paper -> Research_Question -> Solution):")
                positive_paths = find_paths(graph, paper_node, 2, ["paper", "research_question", "solution"])
                if positive_paths:
                    for path in positive_paths:
                        print(f"    - {format_path(graph, path)}")
                else:
                    print(f"    未找到从 {graph.nodes[paper_node].get('entity_name', paper_node)} 开始的正向路径。")

                # 查找并显示反向路径：solution -> research_question -> paper
                # 这里需要从solution开始反向查找，或者从paper开始，然后找符合pattern的路径
                # 为了简化，我们可以从paper开始，查找长度为2的任何类型路径，然后筛选出solution -> research_question -> paper
                # 或者，我们可以从所有的 solution 节点开始，反向查找。
                # 考虑到现有find_paths的实现，我们可以先找到所有的solution节点，然后对它们进行路径查找。
                
                # 简化的反向路径查找（从当前paper节点出发，找到相关的solution，然后构建反向路径）
                # 这种方法更直接，但可能不会找到所有通过其他路径连接的 solution
                
                # 更通用的反向路径查找：找到所有直接或间接与该paper相关的solution节点
                # 这里为了展示正反推，我将从所有solution节点开始寻找反向路径
                all_solution_nodes = [node_id for node_id, data in graph.nodes(data=True) if data.get('entity_type') == 'solution']
                
                print(f"  反向路径 (Solution -> Research_Question -> Paper):")
                reverse_paths_found = []
                for sol_node in all_solution_nodes:
                    paths_from_sol = find_paths(graph, sol_node, 2, ["solution", "research_question", "paper"])
                    for path in paths_from_sol:
                        if path[2] == paper_node: # 确保路径的终点是当前paper节点
                            reverse_paths_found.append(path)

                if reverse_paths_found:
                    for path in reverse_paths_found:
                        print(f"    - {format_path(graph, path)}")
                else:
                    print(f"    未找到与 {graph.nodes[paper_node].get('entity_name', paper_node)} 相关的反向路径。")
                print("-" * 50)

    except FileNotFoundError as e:
        print(f"错误: {e}")
    except Exception as e:
        print(f"发生错误: {e}")
