import numpy as np
from node import Node
from route import Route

def savings_heuristic(nodes, origin, destination, num_paths, battery_limit, alpha: float = 0.5, beta: float|None = None):
    """
    Constructive savings heuristic with optional biased randomization.
    
    :param nodes: List of nodes.
    :param origin: Origin node.
    :param destination: Destination node.
    :param num_paths: Number of paths (vehicles).
    :param battery_limit: Battery limit for each vehicle.
    :param alpha: Tuning parameter for savings calculation.
    :param beta: Parameter for the geometric distribution (if None, no randomization).
    :return: List of Route objects.
    """
    nodes = sorted(nodes, key=lambda x: x.index)
    solution = generate_dummy_solution(nodes, origin, destination, battery_limit)
    distance_matrix = _compute_distance_matrix(nodes)
    rewards = [node.score for node in nodes]
    savings = _compute_savings_list(distance_matrix, rewards, alpha)
    
    while savings:
        if beta is not None:
            arc = _biased_random_choice(savings, beta)
            savings.remove(arc)
        else:
            arc = savings.pop(0)
        
        route_i = _find_route_containing_node(solution, nodes[arc[0]])
        route_j = _find_route_containing_node(solution, nodes[arc[1]])
        
        if route_i is not None and route_j is not None and route_i is not route_j:
            new_route = route_i.merge(route_j)
            if new_route.is_route_feasible(battery_limit):
                solution.remove(route_i)
                solution.remove(route_j)
                solution.append(new_route)
  
  
    for route in solution:
        if route.nodes[-1] != destination:
            route.nodes.append(destination)

    return sorted(solution, key=lambda x: x.total_score, reverse=True)[:num_paths]

def generate_dummy_solution(nodes, origin, destination, battery_limit):
    routes = []
    for node in nodes[1:-1]:
        route = Route(origin, destination)
        route.add_node(node)
        if route.is_route_feasible(battery_limit):
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

def _compute_savings_list(distance_matrix, rewards, alpha):
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

def _biased_random_choice(savings, beta=0.3):
    """
    Select an item from the savings list using biased randomization.
    
    :param savings: List of tuples (saving, i, j).
    :param beta: Parameter for the geometric distribution.
    :return: Selected (saving, i, j).
    """
    probabilities = [beta * (1 - beta) ** i for i in range(len(savings))]
    probabilities = np.array(probabilities) / sum(probabilities)
    return savings[np.random.choice(len(savings), p=probabilities)]

def _find_route_containing_node(solution, node):
    for route in solution:
        if node in route.nodes:
            return route
    return None

def get_best_alpha(nodes, origin, destination, num_paths, battery_limit):
    alphas = np.linspace(0.1, 0.9, 100)
    best_alpha = 0
    best_score = 0
    for alpha in alphas:
        solution = savings_heuristic(nodes, origin, destination, num_paths, battery_limit, alpha)
        new_score = sum([route.total_score for route in solution])
        if new_score > best_score:
            best_score = new_score
            best_alpha = alpha
            
    return best_alpha