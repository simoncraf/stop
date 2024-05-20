import random
import time
import math
import numpy as np
from loguru import logger
from node import Node
from route import Route
from savings_heuristic import savings_heuristic, get_best_alpha

def vns(nodes, origin, destination, num_paths, battery_limit):
    """
    Applies the Variable Neighborhood Search (VNS) algorithm to find the best solution for a given problem.

    :param nodes: List of nodes.
    :param origin: Origin node.
    :param destination: Destination node.
    :param num_paths: Number of paths (vehicles).
    :param battery_limit: Battery limit for each vehicle.
    :return: List of Route objects representing the best solution found.
    """
    alpha = get_best_alpha(nodes, origin, destination, num_paths, battery_limit)
    init_sol = savings_heuristic(nodes, origin, destination, num_paths, battery_limit, alpha)
    base_sol = init_sol
    best_sol = init_sol
    base_stochastic_profit = mc_simulation(base_sol, battery_limit, 1)
    T = 1000
    lambda_ = 0.999
    max_iter = 100
    max_time = 120 
    start_time = time.time()

    pool_of_best_solutions = []
    k = 1
    while (time.time() - start_time) < max_time and k <= max_iter:
        logger.info(f"Current iteration: {k}, with T = {T}")
        new_sol = shaking(base_sol, k, origin, destination, num_paths, battery_limit, alpha)
        new_sol = local_search1(new_sol)
        new_sol = local_search2(new_sol)

        new_sol = local_search3(new_sol, nodes, origin, destination, battery_limit)
        
        base_det_profit = det_profit(base_sol)
        new_det_profit = det_profit(new_sol)
        
        if new_det_profit > base_det_profit:
            base_stochastic_profit = mc_simulation(base_sol, battery_limit, 1)
            new_stochastic_profit = mc_simulation(new_sol, battery_limit, 1)
            if new_stochastic_profit > base_stochastic_profit:
                logger.info(f"New base solution found: {new_stochastic_profit}")
                base_sol = new_sol
                best_stochastic_profit = mc_simulation(best_sol, battery_limit, 1)
                if new_stochastic_profit > best_stochastic_profit:
                    logger.warning(f"New best solution found: {new_stochastic_profit}")
                    best_sol = new_sol
                    pool_of_best_solutions.append(best_sol)
            k = 1
        else:
            update_prob = prob_of_updating(new_det_profit, base_det_profit, T)
            if update_prob >= random.random():
                logger.info(f"Non-improving solution accepted with probability {update_prob}")
                base_sol = new_sol
                k = 1
            else:
                k += 1
        T *= lambda_
    
    best_sol_deep_profit = mc_simulation(best_sol, battery_limit, 1)
    for sol in pool_of_best_solutions:
        deep_profit = mc_simulation(sol, battery_limit, 10, 50000)
        if deep_profit > best_sol_deep_profit:
            best_sol = sol
            best_sol_deep_profit = deep_profit
    
    return best_sol


def mc_simulation(solution, battery_limit, average_speed, num_simulations=1000, mean_factor=1.0, std_dev_factor=0.1):
    """
    Perform a Monte Carlo simulation to evaluate the solution.
    
    :param solution: The solution (a list of Route objects) to evaluate.
    :param battery_limit: Maximum allowable travel time for a route.
    :param average_speed: Average speed of the vehicle.
    :param num_simulations: Number of simulation runs to perform.
    :param mean_factor: Mean factor for travel time distribution.
    :param std_dev_factor: Standard deviation factor for travel time distribution.
    :return: Expected profit or cost of the solution.
    """
    total_profit = 0.0

    for _ in range(num_simulations):
        total_score = 0
        
        for route in solution:
            sim_average_speed = np.random.normal(loc=average_speed * mean_factor, scale=average_speed * std_dev_factor)
            sim_average_speed = max(sim_average_speed, 0)
            if route.is_route_feasible(battery_limit, sim_average_speed):
                total_score += route.total_score
        
        total_profit += total_score

    expected_profit = total_profit / num_simulations
    return expected_profit


def shaking(base_sol: list[Route], k: float, origin: Node, destination: Node, num_paths: int, battery_limit: float, alpha: float, beta: float=0.3):
    """
    Perform the shaking operation to generate a new solution.
    
    :param base_sol: The current solution.
    :param k: The shaking parameter.
    :param origin: Origin node.
    :param destination: Destination node.
    :param num_paths: Number of paths (vehicles).
    :param battery_limit: Battery limit for each vehicle.
    :param alpha: Tuning parameter for savings calculation.
    :param beta: Parameter for the geometric distribution.
    :return: The new solution.
    """
    num_routes_to_delete = max(1,int(len(base_sol) * k/100))
    remaining_routes = base_sol.copy()

    for _ in range(num_routes_to_delete):
        route = random.choice(remaining_routes)
        remaining_routes.remove(route)

    deleted_nodes = {node for route in base_sol if route not in remaining_routes for node in route.nodes}
    new_routes = savings_heuristic(list(deleted_nodes), origin, destination, num_paths, battery_limit, alpha, beta)
    combined_routes = remaining_routes + new_routes
    if len(combined_routes) > num_paths:
        combined_routes = sorted(combined_routes, key=lambda x: x.total_score, reverse=True)[:num_paths]
    
    return combined_routes

def local_search1(solution):
    """
    Apply the 2-opt local search with a cache memory mechanism to each route in the solution.
    
    :param solution: List of routes in the current solution.
    :return: Optimized solution with improved routes.
    """
    optimized_solution = []
    route_cache = {}
    
    for route in solution:
        optimized_route = _two_opt(route, route_cache)
        optimized_solution.append(optimized_route)
    
    return optimized_solution

def _two_opt(route, route_cache):
    """
    Perform a 2-opt local search on a given route with a cache memory mechanism.
    
    :param route: The route to optimize.
    :param route_cache: Hash map to cache best-found-so-far routes.
    :return: Optimized route.
    """
    def route_to_tuple(route_nodes):
        """ Helper function to convert route nodes to a hashable tuple. """
        return tuple(node.index for node in route_nodes)
    
    best_distance = route.total_distance
    best_route = route.nodes[:]
    improved = True
    
    while improved:
        improved = False
        for i in range(1, len(best_route) - 2):
            for j in range(i + 1, len(best_route) - 1):
                if j - i == 1: continue
                
                new_route = best_route[:i] + best_route[i:j][::-1] + best_route[j:]
                new_route_tuple = route_to_tuple(new_route)
                
                if new_route_tuple in route_cache:
                    new_distance = route_cache[new_route_tuple]
                else:
                    new_distance = _calculate_route_distance(new_route)
                    route_cache[new_route_tuple] = new_distance
                
                if new_distance < best_distance:
                    best_distance = new_distance
                    best_route = new_route
                    improved = True
                    break
            if improved:
                break
    
    optimized_route = Route(route.origin, route.destination)
    optimized_route.nodes = best_route
    optimized_route.total_distance = best_distance
    optimized_route.total_score = sum(node.score for node in best_route)
    
    return optimized_route

def _calculate_route_distance(route_nodes):
    """
    Calculate the total distance of a route given its nodes.
    
    :param route_nodes: List of nodes in the route.
    :return: Total distance of the route.
    """
    total_distance = 0
    for i in range(len(route_nodes) - 1):
        total_distance += route_nodes[i].distance_to(route_nodes[i + 1])
    return total_distance



import random

def local_search2(solution):
    """
    Apply the node removal and reinsertion local search to each route in the solution.
    
    :param solution: List of routes in the current solution.
    :return: Optimized solution with improved routes.
    """
    optimized_solution = []
    for route in solution:
        optimized_route = _remove_and_reinsert_nodes(route)
        optimized_solution.append(optimized_route)
    return optimized_solution

def _remove_and_reinsert_nodes(route):
    """
    Perform the node removal and reinsertion local search on a given route.
    
    :param route: The route to optimize.
    :return: Optimized route.
    """
    num_nodes_to_remove = random.randint(max(1, len(route.nodes) // 20), max(1, len(route.nodes) // 10))
    removal_mechanism = random.choice(['random', 'highest_reward', 'lowest_reward'])
    
    match removal_mechanism:
        case 'random':
            nodes_to_remove = random.sample(route.nodes[1:-1], num_nodes_to_remove)
        case 'highest_reward':
            nodes_to_remove = sorted(route.nodes[1:-1], key=lambda x: x.score, reverse=True)[:num_nodes_to_remove]
        case 'lowest_reward':
            nodes_to_remove = sorted(route.nodes[1:-1], key=lambda x: x.score)[:num_nodes_to_remove]
    
    remaining_nodes = [node for node in route.nodes if node not in nodes_to_remove]
    new_route = Route(route.origin, route.destination)
    
    for node in remaining_nodes[1:-1]:
        new_route.add_node(node)
    
    for node in nodes_to_remove:
        best_position = None
        best_distance = float('inf')
        
        for i in range(1, len(new_route.nodes)):
            temp_route = new_route.nodes[:i] + [node] + new_route.nodes[i:]
            temp_distance = _calculate_route_distance(temp_route)
            if temp_distance < best_distance:
                best_distance = temp_distance
                best_position = i
        
        new_route.nodes.insert(best_position, node)
        new_route.total_distance = best_distance
        new_route.total_score += node.score
    
    return new_route



def local_search3(solution, nodes, origin, destination, battery_limit):
    """
    Apply the biased insertion algorithm to each route in the solution.
    
    :param solution: List of routes in the current solution.
    :param nodes: List of all nodes.
    :param origin: Origin node.
    :param destination: Destination node.
    :param battery_limit: Battery limit for each vehicle.
    :return: Optimized solution with improved routes.
    """
    optimized_solution = []
    for route in solution:
        optimized_route = _biased_insertion(route, nodes, origin, destination, battery_limit)
        optimized_solution.append(optimized_route)
    return optimized_solution

def _biased_insertion(route, nodes, origin, destination, battery_limit, beta=0.3, average_speed=1):
    """
    Perform the biased insertion algorithm on a given route.
    
    :param route: The route to optimize.
    :param nodes: List of all nodes.
    :param origin: Origin node.
    :param destination: Destination node.
    :param battery_limit: Battery limit for each vehicle.
    :param beta: Parameter for the geometric distribution.
    :return: Optimized route.
    """
    non_served_nodes = [node for node in nodes if node not in route.nodes and node != origin and node != destination]
    improved = True
    
    while improved and non_served_nodes:
        improved = False
        insertion_costs = []

        for node in non_served_nodes:
            for i in range(len(route.nodes) - 1):
                j = route.nodes[i]
                h = route.nodes[i + 1]
                insertion_cost = ((j.distance_to(node) + node.distance_to(h) - j.distance_to(h)) / node.score, node, i + 1)
                insertion_costs.append(insertion_cost)
        
        insertion_costs.sort(key=lambda x: x[0])
        if not insertion_costs:
            break
        
        probabilities = [beta * (1 - beta) ** i for i in range(len(insertion_costs))]
        probabilities = np.array(probabilities) / sum(probabilities)
        selected_index = np.random.choice(len(insertion_costs), p=probabilities)
        selected_cost, selected_node, insert_position = insertion_costs[selected_index]

        new_route_nodes = route.nodes[:insert_position] + [selected_node] + route.nodes[insert_position:]
        new_total_distance = _calculate_route_distance(new_route_nodes)
        
        if new_total_distance / average_speed <= battery_limit:
            route.nodes = new_route_nodes
            route.total_distance = new_total_distance
            route.total_score += selected_node.score
            non_served_nodes.remove(selected_node)
            improved = True
    
    return route

def det_profit(solution):
    """
    Calculate the deterministic profit of a solution.
    
    :param solution: List of routes in the current solution.
    :return: Total deterministic profit of the solution.
    """
    total_profit = 0
    for route in solution:
        total_profit += route.total_score
    return total_profit

def prob_of_updating(det_profit_new_sol, det_profit_base_sol, temperature):
    """
    Calculate the probability of accepting a non-improving solution.
    
    :param det_profit_new_sol: Deterministic profit of the new solution.
    :param det_profit_base_sol: Deterministic profit of the base solution.
    :param temperature: Current temperature.
    :return: Probability of accepting the new solution.
    """
    delta_profit = det_profit_new_sol - det_profit_base_sol
    return math.exp(delta_profit / temperature)