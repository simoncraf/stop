import math

class Node:
    def __init__(self, index, x, y, score, is_depot = False, is_target = False):
        self.index = index
        self.x = x
        self.y = y
        self.score = score
        self.isdepot = is_depot
        self.istarget = is_target

    def __repr__(self):
        return f"Node(index={self.index}, x={self.x}, y={self.y}, score={self.score})"

    def distance_to(self, other_node) -> float:
        return round(math.sqrt((self.x - other_node.x) ** 2 + (self.y - other_node.y) ** 2),4)