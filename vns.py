import random
import time
import numpy as np
from node import Node
from route import Route
from savings_heuristic import savings_heuristic, get_best_alpha

T = 1000
lambda_ = 0.999
max_iter = 100
max_time = 60*60

def vns(nodes, origin, destination, num_paths, battery_limit):
    alpha = get_best_alpha(nodes, origin, destination, num_paths, battery_limit)
    init_sol = savings_heuristic(nodes, origin, destination, num_paths, battery_limit, alpha)
    base_sol = init_sol
    best_sol = init_sol
    expected_score = fast_simulation(base_sol, battery_limit, 10)
    T = 1000
    lambda_ = 0.999
    max_iter = 100
    max_time = 3600 
    start_time = time.time()

    pool_of_best_solutions = []

    while (time.time() - start_time) < max_time:
        k = 1
        while k <= max_iter:
            new_sol = shaking(base_sol, k, nodes, origin, destination, num_paths, battery_limit, alpha)
            new_sol = local_search1(new_sol)
            new_sol = local_search2(new_sol)
            new_sol = local_search3(new_sol)
            
            if det_profit(new_sol) > det_profit(base_sol):
                fast_simulation(new_sol)
                if stoch_profit(new_sol) > stoch_profit(base_sol):
                    base_sol = new_sol
                    if stoch_profit(new_sol) > stoch_profit(best_sol):
                        best_sol = new_sol
                        pool_of_best_solutions.append(best_sol)
                k = 1
            else:
                update_prob = prob_of_updating(det_profit(new_sol), det_profit(base_sol), T)
                if update_prob >= random.random():
                    base_sol = new_sol
                    k = 1
                else:
                    k += 1
            T *= lambda_
    
    for sol in pool_of_best_solutions:
        deep_simulation(sol)
        if stoch_profit(sol) > stoch_profit(best_sol):
            best_sol = sol
    
    return best_sol


def fast_simulation(solution, battery_limit, average_speed, num_simulations=100, mean_factor=1.0, std_dev_factor=0.1):
    """
    Perform a fast Monte Carlo simulation to evaluate the solution.
    
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
            if route.is_route_feasible(battery_limit, sim_average_speed):
                total_score += route.total_score
        
        total_profit += total_score

    expected_profit = total_profit / num_simulations
    return expected_profit

def shaking(base_sol: list[Route], k: float, nodes: list[Node], origin: Node, destination: Node, num_paths: int, battery_limit: float, alpha: float, beta: float=0.3):
    """
    Perform the shaking operation to generate a new solution.
    
    :param base_sol: The current solution.
    :param k: The shaking parameter.
    :param nodes
    :return: The new solution.
    """
    num_routes_to_delete = max(1,int(len(base_sol) * k/100))
    remaining_routes = base_sol.copy()

    for _ in range(num_routes_to_delete):
        route = random.choice(remaining_routes)
        remaining_routes.remove(route)

    deleted_nodes = {node for route in base_sol if route not in remaining_routes for node in route.nodes}
    new_routes = savings_heuristic(list(deleted_nodes), origin, destination, num_paths, battery_limit, alpha, beta)
    
    return remaining_routes + new_routes

def local_search1(solution):
    # Implement the first local search method here (e.g., 2-opt local search)
    pass

def local_search2(solution):
    # Implement the second local search method here (e.g., node removal)
    pass

def local_search3(solution):
    # Implement the third local search method here (e.g., biased insertion)
    pass

def det_profit(solution):
    # Calculate the deterministic profit of the solution
    pass

def stoch_profit(solution):
    # Calculate the stochastic profit of the solution
    pass

def prob_of_updating(new_profit, base_profit, temperature):
    # Calculate the probability of updating the base solution
    pass

def deep_simulation(solution):
    # Perform deep Monte Carlo simulation to evaluate the solution
    pass