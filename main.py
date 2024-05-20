from savings_heuristic import savings_heuristic, get_best_alpha
from vns import vns
from utils import load_data

def main():
    num_nodes, num_paths, battery_limit, nodes = load_data('data/p1.2.a.txt')

    print('Number of nodes:', num_nodes)
    print('Number of paths:', num_paths)
    print('Battery limit:', battery_limit)

    origin, destination = nodes[0], nodes[-1]
    print('Origin:', origin)
    print('Destination:', destination)
    solution = vns(nodes, origin, destination, num_paths, battery_limit)
    print('Number of feasible routes:', len(solution))
    print(f"The best {num_paths} routes are:\n")
    for sol in solution:
        nodes = [str(node.index) for node in sol.nodes]
        print(f"Route from {origin.index} to {destination.index} passing through {nodes} with total distance {round(sol.total_distance, 4)} and total score {sol.total_score}")

if __name__ == '__main__':
    main()