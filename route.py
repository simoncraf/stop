from node import Node


class Route:
    def __init__(self, origin: Node, destination: Node):
        self.origin = origin
        self.destination = destination
        self.nodes = [origin]
        self.total_distance = 0
        self.total_score = origin.score
        
    def __repr__(self) -> str:
        return f"Route(origin={self.origin}, destination={self.destination}, nodes={self.nodes}, total_distance={round(self.total_distance, 4)}, total_score={self.total_score})"
    
    def __str__(self) -> str:
        return f"Route from {self.origin} to {self.destination} passing through {[node.idx for node in self.nodes]} with total distance {round(self.total_distance, 4)} and total score {self.total_score}"
        
    def add_node(self, node: Node):
        self.nodes.append(node)
        self.total_distance += self.nodes[-2].distance_to(node)
        self.total_score += node.score

    def is_route_feasible(self, battery_limit, average_speed=10):
        distance_to_destination = self.nodes[-1].distance_to(self.destination)
        return (self.total_distance + distance_to_destination) / average_speed <= battery_limit
    
    def merge(self, route):
        self.nodes.extend(route.nodes[1:])
        self.total_distance += route.total_distance
        self.total_score += route.total_score
        return self