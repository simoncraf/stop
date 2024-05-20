from node import Node

class Route:
    def __init__(self, origin: Node, destination: Node):
        self.origin = origin
        self.destination = destination
        self.nodes = [origin, destination]
        self.total_distance = 0
        self.total_score = origin.score
        
    def __repr__(self) -> str:
        return (f"Route(origin={self.origin}, destination={self.destination}, nodes={self.nodes}, "
                f"total_distance={round(self.total_distance, 4)}, total_score={self.total_score})")
    
    def __str__(self) -> str:
        return (f"Route from {self.origin} to {self.destination} passing through "
                f"{[node.index for node in self.nodes]} with total distance {round(self.total_distance, 4)} "
                f"and total score {self.total_score}")
        
    def add_node(self, node: Node):
        self.nodes.insert(-1, node)
        self.total_distance += self.nodes[-2].distance_to(node) + node.distance_to(self.nodes[-1])
        self.total_distance -= self.nodes[-3].distance_to(self.nodes[-1])
        self.total_score += node.score

    def is_route_feasible(self, battery_limit, average_speed=1):
        return (self.total_distance / average_speed) <= battery_limit
    
    def copy(self) -> 'Route':
        new_route = Route(self.origin, self.destination)
        new_route.nodes = self.nodes[:]
        new_route.total_distance = self.total_distance
        new_route.total_score = self.total_score
        return new_route
    
    def merge(self, route: 'Route') -> 'Route':
        new_route = self.copy()
        new_route.nodes.pop()
        new_route.nodes.extend(route.nodes[1:])
        new_route.total_distance += route.total_distance
        new_route.total_score += route.total_score
        return new_route
