from route import Route

def generate_dummy_soultion(nodes, origin, destination, battery_limit):
    routes = []
    for node in nodes[1:-1]:
        route = Route(origin, destination)
        route.add_node(node)
        if route.is_route_feasible(battery_limit):
            route.add_node(destination)
            routes.append(route)
    return routes