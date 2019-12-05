from datasets import Datasets
from optimizer import Graph, Optimizer
import operator
import matplotlib.pyplot as plt


path = '/Users/michal/PycharmProjects/MRP/datasets/*.tsp'
data = Datasets(path)

# name = 'ali535'
name = 'berlin11_modified'
# name = 'berlin52'
# name = 'fl417'
# name = 'gr666'
# name = 'kroA100'
# name = 'kroA150'
# name = 'nrw1379'
# name = 'pr2392'

aco = Optimizer(ant_count=100, generations=100, alpha=1.0, beta=10.0, rho=0.5, q=10)
graph = Graph(data.datasets[name]['cost_matrix'], data.datasets[name]['rank'])
points_sequence, distance = aco(graph)
print('Found best distance: {} for sequence: {}'.format(distance, points_sequence))


def paint_graph(nodes, edges, title, cost):
    # plot nodes
    x = []
    y = []
    for point in nodes:
        x.append(point[0])
        y.append(point[1])
    y = list(map(operator.sub, [max(y) for i in range(len(nodes))], y))
    plt.plot(x, y, 'co', color='red')
    plt.grid()

    # plot edges
    for _ in range(1, len(edges)):
        i = edges[_ - 1]
        j = edges[_]
        plt.arrow(x[i], y[i], x[j] - x[i], y[j] - y[i], color='black', length_includes_head=True)

    plt.xlim(0, max(x) * 1.1)
    plt.ylim(0, max(y) * 1.1)
    plt.title('dataset: {}, best distance: {}'.format(title, round(cost, 3)))
    plt.show()


paint_graph(data.datasets[name]['points'], points_sequence, name, distance)

