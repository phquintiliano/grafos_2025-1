from collections import defaultdict
import re


def find_values_in_blocks(bloco):
    pattern = r"\d+"
    linhas = bloco.split("\n")[1:]
    valores = []
    for linha in linhas:
        numeros = re.findall(pattern, linha)
        valores.append([int(num) for num in numeros])
    return valores


def separar_linha(blocos):
    primeiro_bloco = blocos[0].split("\n")[1:]
    bloco_sem_linha = "\n".join(primeiro_bloco)
    return bloco_sem_linha


def find_parameters(blocos):
    pattern = r"-?\d+"
    integers = re.findall(pattern, blocos)
    return integers


def find_intermediation(matrix, nodes):
    intermediations = [0] * (int(nodes) + 1)
    for i in range(len(matrix))[1:]:
        for j in range(len(matrix[i]))[1:]:
            if i != j and matrix[i][j] and matrix[i][j] != i:
                intermediations[matrix[i][j]] += 1
    return intermediations[1:]


def find_degree(matrix, nodes):
    degrees = [0] * (int(nodes) + 1)
    inf = float("inf")
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] < inf and matrix[i][j] > 0:
                degrees[i] += 1
                degrees[j] += 1
    return min(degrees[1:]), max(degrees)


def find_medium_weight(matrix, nodes):
    weight = 0
    inf = float("inf")
    highest_weight = -1
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] < inf and matrix[i][j] > 0:
                weight += matrix[i][j]
                highest_weight = max(highest_weight, matrix[i][j])
    return weight / (int(nodes) * (int(nodes) - 1)), highest_weight


def bfs_component(adjacency_list, v, visited):
    queue = [v]
    visited[v] = True
    while queue:
        node = queue.pop()
        for neighbor in adjacency_list[node]:
            if not visited[neighbor]:
                visited[neighbor] = True
                queue.append(neighbor)


def find_components(arcs, nodes):
    adjacency_list = {i: [] for i in range(int(nodes) + 1)}
    for arc in arcs:
        adjacency_list[arc[0]].append(arc[1])
    visited = [False] * (int(nodes) + 1)
    components = 0

    for v in range(int(nodes) + 1)[1:]:
        if not visited[v]:
            components += 1
            bfs_component(adjacency_list, v, visited)

    return components


def read_file(nome_arquivo):
    with open("files\\" + nome_arquivo, "r") as arquivo:
        conteudo = arquivo.read()
    blocos = conteudo.strip().split("\n\n")
    [
        optimal_value,
        vehicles,
        capacity,
        depot_note,
        nodes,
        total_edges,
        total_arcs,
        required_n,
        required_e,
        required_a,
    ] = find_parameters(separar_linha(blocos))
    required_ns = list(map(lambda linha: linha[0], find_values_in_blocks(blocos[1])))
    values = []
    for bloco in blocos[2:]:
        linhas = find_values_in_blocks(bloco)
        if len(linhas) > 0 and len(linhas[-1]) == 1:
            linhas.pop()
        values.append(
            list(
                map(
                    lambda linha: [linha[1], linha[2], linha[3]],
                    linhas,
                )
            )
        )
    [required_es, edges, required_as, arcs] = values
    print("Sucesso! Os arquivos foram lidos.")
    return (
        optimal_value,
        vehicles,
        capacity,
        depot_note,
        nodes,
        total_edges,
        total_arcs,
        required_n,
        required_e,
        required_a,
        required_ns,
        required_es,
        edges,
        required_as,
        arcs,
    )


def metrics(
    nodes,
    total_edges,
    total_arcs,
    required_n,
    required_e,
    required_a,
    required_es,
    edges,
    required_as,
    arcs,
):

    all_edges = required_es + edges
    arcs_from_edges = []
    for edge in all_edges:
        arcs_from_edges.append([edge[0], edge[1], edge[2]])
        arcs_from_edges.append([edge[1], edge[0], edge[2]])
    all_arcs = required_as + arcs + arcs_from_edges

    dist = {
        u: {v: float("inf") for v in range(int(nodes) + 1)}
        for u in range(int(nodes) + 1)
    }

    for u in range(int(nodes) + 1):
        dist[u][u] = 0

    pred = {u: {} for u in range(int(nodes) + 1)}

    for origem, destino, peso in all_arcs:
        dist[origem][destino] = peso
        pred[origem][destino] = origem

    [lowest_degree, highest_degree] = find_degree(dist, nodes)

    for k in range(int(nodes) + 1):
        dist_k = dist[k]  # save recomputation
        for i in range(int(nodes) + 1):
            dist_i = dist[i]  # save recomputation
            for j in range(int(nodes) + 1):
                d = dist_i[k] + dist_k[j]
                if dist_i[j] > d:
                    dist_i[j] = d
                    pred[i][j] = pred[k][j]

    intermediations = find_intermediation(pred, nodes)
    [medium_weight, highest_weight] = find_medium_weight(dist, nodes)
    conected_components = find_components(all_arcs, nodes)
    density = ((int(total_edges) * 2) + int(total_arcs)) / (
        int(nodes) * (int(nodes) - 1)
    )

    print("Quantidade de vertices:", nodes)
    print("Quantidade de arestas: ", total_edges)
    print("Quantidade de arcos: ", total_arcs)
    print("Quantidade de vertices requeridos: ", required_n)
    print("Quantidade de arestas requeridas: ", required_e)
    print("Quantidade de arcos requeridas: ", required_a)
    print("Densidade do grafo: ", density.__round__(2))
    print("Componentes conectados: ", conected_components)
    print("Grau minimo dos vertices: ", lowest_degree)
    print("Grau maximo dos vertices: ", highest_degree)
    print("Intermediacao: ", intermediations)
    print("Caminho medio: ", medium_weight.__round__(2))
    print("Diametro: ", highest_weight)
