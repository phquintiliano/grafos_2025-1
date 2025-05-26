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


def add_id_in_requireds(required_node_list, req_edges, req_arcs):
    count = 1
    required_node_list = [
        [n[0], n[1], i] for i, n in enumerate(required_node_list, start=count)
    ]
    count += len(required_node_list)
    req_edges = [
        [e[0], e[1], e[2], e[3], i] for i, e in enumerate(req_edges, start=count)
    ]
    count += len(req_edges)
    req_arcs = [
        [a[0], a[1], a[2], a[3], i] for i, a in enumerate(req_arcs, start=count)
    ]

    return required_node_list, req_edges, req_arcs


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
        map(
            lambda line: [line[0], line[1]],
            extract_values_from_block(blocks[1]),
        )
    )
    block_id = 2
    data_blocks = []
    for block in blocks[block_id:]:
        lines = extract_values_from_block(block)

        if lines and len(lines[-1]) == 1:
            lines.pop()
        data_blocks.append(
            list(
                map(
                    lambda line: [
                        line[1],
                        line[2],
                        line[3],
                        line[4] if block_id % 2 == 0 else 0,
                    ],
                    lines,
                )
            )
        )
        block_id += 1
    [req_edges, edges, req_arcs, arcs] = data_blocks

    required_node_list, req_edges, req_arcs = add_id_in_requireds(
        required_node_list, req_edges, req_arcs
    )
    return (
        optimal_value,
        vehicle_count,
        int(capacity),
        int(depot_node),
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


def find_all_arcs(req_edges, edges, req_arcs, arcs):
    all_edges = req_edges + edges
    arcs_from_edges = [[e[0], e[1], e[2]] for e in all_edges] + [
        [e[1], e[0], e[2]] for e in all_edges
    ]
    return [[e[0], e[1], e[2]] for e in req_arcs + arcs] + arcs_from_edges


def floyd_warshall(num_nodes, all_arcs):
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

    for k in range(int(num_nodes) + 1):
        dist_k = dist[k]
        for i in range(int(num_nodes) + 1):
            dist_i = dist[i]
            for j in range(int(num_nodes) + 1):
                d = dist_i[k] + dist_k[j]
                if dist_i[j] > d:
                    dist_i[j] = d
                    pred[i][j] = pred[k][j]
    return pred, dist


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
    all_arcs = find_all_arcs(req_edges, edges, req_arcs, arcs)
    pred, dist = floyd_warshall(num_nodes, all_arcs)
    min_deg, max_deg = compute_degree(dist, num_nodes)
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


def build_routes_by_affinity(services, dist, capacity, depot_node):
    unvisited = services[:]
    routes = []

    while unvisited:
        route = []
        current = depot_node
        current_demand = 0

        while True:
            # Filtrar os serviços que cabem na rota
            candidates = [
                s for s in unvisited if current_demand + s["demand"] <= capacity
            ]
            if not candidates:
                break

            # Escolher o mais próximo do último ponto da rota
            next_service = min(candidates, key=lambda s: dist[current][s["from"]])
            route.append(next_service)
            current_demand += next_service["demand"]
            current = next_service["to"]
            unvisited.remove(next_service)

        if route:
            routes.append(route)

    return routes


def greedy_algorithm(route, dist, depot_node):
    route_copy = route[:]
    lowest_way = []
    current = depot_node

    while len(route_copy) > 0:
        next_service = None
        next_distance = float("inf")

        for service in route_copy:
            distance = dist[current][service["from"]]
            if distance < next_distance:
                next_service = service
                next_distance = distance

        if next_service is None:
            break

        lowest_way.append(next_service)
        route_copy.remove(next_service)
        current = next_service["from"]

    return lowest_way


def create_services(required_node_list, req_edges, req_arcs, dist, depot_node):
    services = []
    for nodes in required_node_list:
        services.append(
            {
                "type": "node",
                "id": nodes[2],
                "from": nodes[0],
                "to": nodes[0],
                "demand": nodes[1],
                "cost": 0,
            }
        )
    for edge in req_edges:
        services.append(
            {
                "type": "edge",
                "id": edge[4],
                "from": edge[0],
                "to": edge[1],
                "demand": edge[3],
                "cost": edge[2],
            }
        )

    for arc in req_arcs:
        services.append(
            {
                "type": "arc",
                "id": arc[4],
                "from": arc[0],
                "to": arc[1],
                "demand": arc[3],
                "cost": arc[2],
            }
        )
    services.sort(
        key=lambda s: (dist[depot_node][s["from"]] + dist[s["from"]][s["to"]])
        / s["demand"]
    )

    return services


def find_lowest_ways(routes, dist, depot_node):
    lowest_ways = []
    for route in routes:
        lowest_ways.append(greedy_algorithm(route, dist, depot_node))

    return lowest_ways


def format_routes_to_save(
    depot_node,
    lowest_ways,
    dist,
):
    depot_line = f"(D 0,{depot_node},{depot_node})"
    distances = []
    lines = []
    for i, way in enumerate(lowest_ways):
        line = []
        services = []
        distance = 0
        way_demand = 0

        last_step = depot_node
        for service in way:
            cost = (
                dist[service["from"]][service["to"]] + dist[last_step][service["from"]]
            )
            services.append(f"(S {service['id']},{service['from']},{service['to']})")
            distance += dist[last_step][service["from"]]
            distance += service["cost"]
            last_step = service["to"]
            way_demand += service["demand"]
        final = dist[last_step][depot_node]
        distance += final
        distances.append(distance)

        line.append(0)
        line.append(1)
        line.append(i)
        line.append(way_demand)
        line.append(distance)
        line.append(f" {len(services) + 2}")
        line = [str(l) for l in line]
        line.append(depot_line)
        line += services
        line.append(depot_line)
        lines.append(" ".join(line))
    return lines, distances


def save_file(
    file_name,
    depot_node,
    lowest_ways,
    dist,
    routes,
    start_all,
    end_all,
    start_solution,
    end_solution,
):
    lines = []
    routes, distances = format_routes_to_save(
        depot_node,
        lowest_ways,
        dist,
    )
    lines.append(sum(distances))
    lines.append(len(routes))
    lines.append(end_all - start_all)
    lines.append(end_solution - start_solution)

    lines = [str(line) for line in lines]
    for route in routes:
        lines.append(route)

    string_to_save = "\n".join(lines)
    path = f"{'G29'}/sol-{file_name}"

    with open(path, "w", encoding="utf-8") as file:
        file.write(string_to_save)

    print("Arquivo salvo com sucesso em:", path)


def main():
    start = time.time()
    folder = "files"
    files = os.listdir(folder)

    for file in files:
        if file.endswith(".dat"):
            start_all = time.perf_counter_ns()
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

            all_arcs = find_all_arcs(req_edges, edges, req_arcs, arcs)
            pred, dist = floyd_warshall(num_nodes, all_arcs)

            start_solution = time.perf_counter_ns()
            services = create_services(
                required_node_list, req_edges, req_arcs, dist, depot_node
            )
            routes = build_routes_by_affinity(services, dist, capacity, depot_node)
            lowest_ways = find_lowest_ways(routes, dist, depot_node)
            end_solution = time.perf_counter_ns()

            end_all = time.perf_counter_ns()

            save_file(
                file,
                depot_node,
                lowest_ways,
                dist,
                routes,
                start_all,
                end_all,
                start_solution,
                end_solution,
            )

    end = time.time()
    print(f"\nTotal execution time: {(end - start):.2f} seconds")


main()
