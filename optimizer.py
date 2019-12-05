import random
from tqdm import tqdm


class Graph:
    def __init__(self, cost_matrix, rank):
        """
        Constructor of graph
        :param cost_matrix: matrix of distances between nodes
        :param rank: number of nodes (to initialise pheromone matrix)
        """
        self.cost_matrix = cost_matrix
        self.rank = rank
        # initialise pheromone as equal value for all edge of graph
        self.pheromone = [[1 / (rank * rank) for _ in range(rank)] for _ in range(rank)]

    def update_pheromone(self, ants, rho):
        """
        Method to update pheromone list
        :param ants: ants which gives pheromone delta to pheromone matrix
        :param rho: parameter of optimization
        """
        for i, row in enumerate(self.pheromone):
            for j, col in enumerate(row):
                self.pheromone[i][j] *= rho
                for ant in ants:
                    self.pheromone[i][j] += ant.pheromone_delta[i][j]


class Optimizer:
    def __init__(self, ant_count: int, generations: int, alpha: float, beta: float, rho: float, q: int):
        """
        Constructor of optimizer
        :param ant_count: count of ants
        :param generations: number of generations
        :param alpha: relative importance of pheromone
        :param beta: relative importance of heuristic information
        :param rho: pheromone residual coefficient
        :param q: pheromone intensity
        """
        self.Q = q
        self.rho = rho
        self.beta = beta
        self.alpha = alpha
        self.ant_count = ant_count
        self.generations = generations

    def __call__(self, graph: Graph):
        """
        This method performs optimization of ant colony
        :param graph:
        """
        # initialize best distance and best solution
        best_cost = float('inf')
        best_solution = []

        # perform optimisation
        print('Optimizing...')
        generator = tqdm(range(self.generations))
        for _ in generator:
            # initialise new population
            ants = [_Ant(self, graph) for _ in range(self.ant_count)]

            # update ants position
            for ant in ants:
                for i in range(graph.rank - 1):
                    ant.select_next_node()
                ant.close_route()
                ant.total_cost += graph.cost_matrix[ant.visited_nodes[-1]][ant.visited_nodes[0]]
                if ant.total_cost < best_cost:
                    best_cost = ant.total_cost
                    best_solution = [] + ant.visited_nodes
                # update pheromone
                ant.update_pheromone_delta()
            graph.update_pheromone(ants, self.rho)
            generator.set_description_str('best cost: {}'.format(best_cost))
        return best_solution, best_cost


class _Ant:
    def __init__(self, aco: Optimizer, graph: Graph):
        """
        Constructor of single ant
        :param aco: AC optimizer which is a mother of ant
        :param graph: graph to optimize
        """
        self.colony = aco
        self.graph = graph
        self.total_cost = 0.0
        self.visited_nodes = []
        self.pheromone_delta = []  # the local change of pheromone
        self.visitable_nodes = [i for i in range(graph.rank)]  # nodes which are allowed for the next selection
        self.eta = [[0 if i == j else 1 / graph.cost_matrix[i][j] for j in range(graph.rank)] for i in
                    range(graph.rank)]  # heuristic information

        # select randomly starting point
        starting_point = random.randint(0, graph.rank - 1)

        # update attributes
        self.visited_nodes.append(starting_point)
        self.current = starting_point
        self.visitable_nodes.remove(starting_point)

    def select_next_node(self):
        """
        Method to select next node by ant
        """
        # calculate probabilities for moving to a node in the next step
        denominator = 0
        for i in self.visitable_nodes:
            denominator += self.graph.pheromone[self.current][i] ** self.colony.alpha * self.eta[self.current][
                                                                                            i] ** self.colony.beta
        probabilities = [0 for _ in range(self.graph.rank)]
        for i in range(self.graph.rank):
            try:
                self.visitable_nodes.index(i)  # test if allowed list contains i
                probabilities[i] = self.graph.pheromone[self.current][i] ** self.colony.alpha * \
                    self.eta[self.current][i] ** self.colony.beta / denominator
            except ValueError:
                pass  # do nothing

        # select next node by probability roulette
        selected = 0
        rand = random.random()
        for i, probability in enumerate(probabilities):
            rand -= probability
            if rand <= 0:
                selected = i
                break

        # update parameters of ant
        self.visitable_nodes.remove(selected)
        self.visited_nodes.append(selected)
        self.total_cost += self.graph.cost_matrix[self.current][selected]
        self.current = selected

    def close_route(self):
        """
        Method to close route by ant
        """
        # update parameters of ant
        selected = self.visited_nodes[0]
        self.visited_nodes.append(selected)
        self.total_cost += self.graph.cost_matrix[self.current][selected]
        self.current = selected

    def update_pheromone_delta(self):
        """
        Method to update change of pheromone after finishing a route
        """
        # initialise list of zeros as starting pheromone delta
        self.pheromone_delta = [[0 for j in range(self.graph.rank)] for i in range(self.graph.rank)]
        # update pheromone in the tabu list
        for index in range(1, len(self.visited_nodes)):
            i = self.visited_nodes[index - 1]
            j = self.visited_nodes[index]
            self.pheromone_delta[i][j] = self.colony.Q
