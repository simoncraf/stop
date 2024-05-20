from node import Node

def load_data(path: str) -> tuple[int, int, int, list[Node]]:
    """
    Load data from a file and return the number of nodes, number of paths, battery limit and a list of nodes.
    
    :param path: Path to the file.
    :return: Tuple containing the number of nodes, number of paths, battery limit and a list of nodes.
    """
    with open(path, 'r') as f:
        data = f.readlines()
        
    num_nodes = int(data[0].split()[1].strip())
    num_paths = int(data[1].split()[1].strip())
    battery_limit = float(data[2].split()[1].strip())
    
    nodes = []
    for i in range(3, 3 + num_nodes):
        node_data = data[i].strip().split()
        is_depot = i == 3
        is_target = i == 3 + num_nodes - 1
        node = Node(i -3, float(node_data[0]), float(node_data[1]), float(node_data[2]), is_depot, is_target)
        nodes.append(node)
        
    return num_nodes, num_paths, battery_limit, nodes