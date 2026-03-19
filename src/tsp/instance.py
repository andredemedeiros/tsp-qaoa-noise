import numpy as np
from itertools import permutations

class TSPInstance:
    """Defines a TSP instance with n cities, [0,1]²"""

    def __init__(self, n_cities: int, seed: int = 42):
        self.n = n_cities
        np.random.seed(seed)
        self.coords = np.random.rand(n_cities, 2)
        self.dist = np.array([[np.linalg.norm(self.coords[i] - self.coords[j]) 
                               for j in range(n_cities)] for i in range(n_cities)])
        # print('coords:', self.coords)
        # print('dist:', self.dist)
        
    def brute_force(self):
        """Solves by brute force for validation (small instances only)"""
        cities = list(range(self.n))
        best_cost = float("inf")
        best_route = None
        for perm in permutations(cities[1:]):
            route = [0] + list(perm)
            cost = self.route_cost(route)
            if cost < best_cost:
                best_cost = cost
                best_route = route[:]
        return best_route, best_cost

    def route_cost(self, route):
        """Calculates the cost of a route"""
        return sum(self.dist[route[i]][route[(i+1) % self.n]]
                   for i in range(self.n))