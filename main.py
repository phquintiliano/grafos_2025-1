import re
import os
import time


def extract_values_from_block(block):
    pattern = r"\d+"
    lines = block.split("\n")[1:]
    values = []
    for line in lines:
        numbers = re.findall(pattern, line)
        values.append([int(num) for num in numbers])
    return values


def remove_first_line_from_blocks(blocks):
    first_block = blocks[0].split("\n")[1:]
    return "\n".join(first_block)


def extract_parameters(blocks):
    pattern = r"-?\d+"
    return re.findall(pattern, blocks)


def compute_betweenness(predecessor_matrix, num_nodes):
    betweenness = [0] * (int(num_nodes) + 1)
    for i in range(1, len(predecessor_matrix)):
        for j in range(1, len(predecessor_matrix[i])):
            if i != j and predecessor_matrix[i][j] and predecessor_matrix[i][j] != i:
                betweenness[predecessor_matrix[i][j]] += 1
    return betweenness[1:]


def compute_degree(distance_matrix, num_nodes):
    degrees = [0] * (int(num_nodes) + 1)
    inf = float("inf")
    for i in range(len(distance_matrix)):
        for j in range(len(distance_matrix[i])):
            if 0 < distance_matrix[i][j] < inf:
                degrees[i] += 1
                degrees[j] += 1
    return min(degrees[1:]), max(degrees)


def compute_avg_and_diameter(distance_matrix, num_nodes):
    total_weight = 0
    inf = float("inf")
    max_weight = -1
    for i in range(len(distance_matrix)):
        for j in range(len(distance_matrix[i])):
            if 0 < distance_matrix[i][j] < inf:
                total_weight += distance_matrix[i][j]
                max_weight = max(max_weight, distance_matrix[i][j])
    average = total_weight / (int(num_nodes) * (int(num_nodes) - 1))
    return average, max_weight


def bfs_connected_component(adj_list, vertex, visited):
    queue = [vertex]
    visited[vertex] = True
    while queue:
        current = queue.pop()
        for neighbor in adj_list[current]:
            if not visited[neighbor]:
                visited[neighbor] = True
                queue.append(neighbor)


def count_connected_components(arcs, num_nodes):
    adj_list = {i: [] for i in range(int(num_nodes) + 1)}
    for arc in arcs:
        adj_list[arc[0]].append(arc[1])
    visited = [False] * (int(num_nodes) + 1)
    components = 0

    for v in range(1, int(num_nodes) + 1):
        if not visited[v]:
            components += 1
            bfs_connected_component(adj_list, v, visited)

    return components


def read_file(filename):
    with open("files\\" + filename, "r") as file:
        content = file.read()

    blocks = content.strip().split("\n\n")

    [
        optimal_value,
        vehicle_count,
        capacity,
        depot_node,
        num_nodes,
        total_edges,
        total_arcs,
        required_nodes,
        required_edges,
        required_arcs,
    ] = extract_parameters(remove_first_line_from_blocks(blocks))

    required_node_list = list(
        map(lambda line: line[0], extract_values_from_block(blocks[1]))
    )

    data_blocks = []
    for block in blocks[2:]:
        lines = extract_values_from_block(block)
        if lines and len(lines[-1]) == 1:
            lines.pop()
        data_blocks.append(list(map(lambda line: [line[1], line[2], line[3]], lines)))

    [req_edges, edges, req_arcs, arcs] = data_blocks

    print("Success! Files have been read.")
    return (
        optimal_value,
        vehicle_count,
        capacity,
        depot_node,
        num_nodes,
        total_edges,
        total_arcs,
        required_nodes,
        required_edges,
        required_arcs,
        required_node_list,
        req_edges,
        edges,
        req_arcs,
        arcs,
    )


def compute_metrics(
    num_nodes,
    total_edges,
    total_arcs,
    required_nodes,
    required_edges,
    required_arcs,
    req_edges,
    edges,
    req_arcs,
    arcs,
):
    all_edges = req_edges + edges
    arcs_from_edges = [[e[0], e[1], e[2]] for e in all_edges] + [
        [e[1], e[0], e[2]] for e in all_edges
    ]
    all_arcs = req_arcs + arcs + arcs_from_edges

    dist = {
        u: {v: float("inf") for v in range(int(num_nodes) + 1)}
        for u in range(int(num_nodes) + 1)
    }
    for u in range(int(num_nodes) + 1):
        dist[u][u] = 0

    pred = {u: {} for u in range(int(num_nodes) + 1)}
    for src, dst, weight in all_arcs:
        dist[src][dst] = weight
        pred[src][dst] = src

    min_deg, max_deg = compute_degree(dist, num_nodes)

    for k in range(int(num_nodes) + 1):
        dist_k = dist[k]
        for i in range(int(num_nodes) + 1):
            dist_i = dist[i]
            for j in range(int(num_nodes) + 1):
                d = dist_i[k] + dist_k[j]
                if dist_i[j] > d:
                    dist_i[j] = d
                    pred[i][j] = pred[k][j]

    betweenness = compute_betweenness(pred, num_nodes)
    avg_weight, diameter = compute_avg_and_diameter(dist, num_nodes)
    components = count_connected_components(all_arcs, num_nodes)
    density = ((int(total_edges) * 2) + int(total_arcs)) / (
        int(num_nodes) * (int(num_nodes) - 1)
    )

    return {
        "Number of Nodes": int(num_nodes),
        "Number of Edges": int(total_edges),
        "Number of Arcs": int(total_arcs),
        "Required Nodes": int(required_nodes),
        "Required Edges": int(required_edges),
        "Required Arcs": int(required_arcs),
        "Graph Density": round(density, 2),
        "Connected Components": components,
        "Minimum Degree": min_deg,
        "Maximum Degree": max_deg,
        "Betweenness": betweenness,
        "Average Path Length": round(avg_weight, 2),
        "Diameter": diameter,
    }


def main():
    start = time.time()
    folder = "files"
    files = os.listdir(folder)
    print(files)

    for file in files:
        if file.endswith(".dat"):
            print(f"\n--- Processing file: {file} ---")
            (
                optimal_value,
                vehicle_count,
                capacity,
                depot_node,
                num_nodes,
                total_edges,
                total_arcs,
                required_nodes,
                required_edges,
                required_arcs,
                required_node_list,
                req_edges,
                edges,
                req_arcs,
                arcs,
            ) = read_file(file)

            compute_metrics(
                num_nodes,
                total_edges,
                total_arcs,
                required_nodes,
                required_edges,
                required_arcs,
                req_edges,
                edges,
                req_arcs,
                arcs,
            )

    end = time.time()
    print(f"\nTotal execution time: {(end - start):.2f} seconds")
