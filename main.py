from savings_heuristic import savings_heuristic
from utils import load_data

num_nodes, num_paths, battery_limit, nodes = load_data('data/p1.2.a.txt')

print('Number of nodes:', num_nodes)
print('Number of paths:', num_paths)
print('Battery limit:', battery_limit)

origin, destination = nodes[0], nodes[-1]
print('Origin:', origin)
print('Destination:', destination)

solution = savings_heuristic(nodes, origin, destination, battery_limit)
print('Number of feasible routes:', len(solution))
for i, route in enumerate(solution, 1):
    print(f'Route {i}: {route.nodes}')
    
print(f"The best {num_paths} routes are: {sorted(solution, key=lambda x: x.total_score, reverse=True)[:num_paths]}")
