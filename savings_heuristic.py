from node import Node
from route import Route

def savings_heuristic(nodes, origin, destination, battery_limit):
    dummy_solution = generate_dummy_soultion(nodes, origin, destination, battery_limit)
    distance_matrix = _compute_distance_matrix(nodes)
    savings = _compute_savings_list(distance_matrix)

def generate_dummy_soultion(nodes, origin, destination, battery_limit):
    routes = []
    for node in nodes[1:-1]:
        route = Route(origin, destination)
        route.add_node(node)
        if route.is_route_feasible(battery_limit):
            route.add_node(destination)
            routes.append(route)
    return routes

def _compute_distance_matrix(nodes: list[Node]):
    num_nodes = len(nodes)
    distance_matrix = [[0] * num_nodes for _ in range(num_nodes)]
    
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if i != j:
                distance = nodes[i].distance_to(nodes[j])
                distance_matrix[i][j] = distance
                distance_matrix[j][i] = distance

    return distance_matrix

def _compute_savings_list(distance_matrix):
    num_nodes = len(distance_matrix)
    savings_list = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes - 1):
            if i != j:
                savings = distance_matrix[0][i] + distance_matrix[j][-1] - distance_matrix[i][j]
                savings_list.append((i, j, savings))
    return sorted(savings_list, key=lambda x: x[2], reverse=True)