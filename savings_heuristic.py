from node import Node
from route import Route

def savings_heuristic(nodes, origin, destination, battery_limit):
    solution = generate_dummy_soultion(nodes, origin, destination, battery_limit)
    distance_matrix = _compute_distance_matrix(nodes)
    rewards = [node.score for node in nodes]
    savings = _compute_savings_list(distance_matrix, rewards)
    
    while savings:
        arc = savings.pop(0)
        routei, routej = Route(origin, destination), Route(origin, destination)
        routei.add_node(nodes[arc[0]])
        routej.add_node(nodes[arc[1]])
        new_route = routei.merge(routej)
        if new_route.is_route_feasible(battery_limit):
            solution.append(new_route)
    for route in solution:
        route.nodes.append(destination)        
    return sorted(solution, key=lambda x: x.total_score, reverse=True)

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

def _compute_savings_list(distance_matrix, rewards, alpha: float = 0.5):
    num_nodes = len(distance_matrix)
    savings_list = []
    for i in range(1, num_nodes - 1):
        for j in range(i + 1, num_nodes - 1):
            if i != j:
                sij = distance_matrix[i][-1] + distance_matrix[0][j] - distance_matrix[i][j]
                aggregated_reward = rewards[i] + rewards[j]
                savings_ij = alpha * sij + (1 - alpha) * aggregated_reward
                savings_list.append((i, j, savings_ij))
    
    return sorted(savings_list, key=lambda x: x[2], reverse=True)