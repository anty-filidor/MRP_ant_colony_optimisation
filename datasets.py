import numpy as np
from glob import glob


class Datasets:
    def __init__(self, path_to_datasets):
        self.datasets = {}  # dictionary to keep datasets
        paths = glob(path_to_datasets)

        for path in paths:
            # save dataset name
            dataset_name = path.split('/')[-1].split('.')[0]
            print('Reading:', dataset_name)

            # initialise empty containers
            cities = []
            points = []
            cost_matrix = []

            with open(path, 'r') as file:
                # omit trash
                line = file.readline()
                while line and 'NODE_COORD_SECTION' not in line:
                    line = file.readline()

                # save data
                line = file.readline()
                while line and 'EOF' not in line:
                    city = line.split()
                    cities.append(dict(index=int(city[0]), x=float(city[1]), y=float(city[2])))
                    points.append((float(city[1]), float(city[2])))
                    line = file.readline()

            # compute coordinates
            rank = len(cities)
            for i in range(rank):
                row = []
                for j in range(rank):
                    row.append(self._distance_two_cities(cities[i], cities[j]))
                cost_matrix.append(row)

            # append dataset points to dictionary
            self.datasets.update({dataset_name:
                                      {'cities': cities, 'points': points, 'rank': rank, 'cost_matrix': cost_matrix}})

        print('Data-sets loaded successfully.')

    @staticmethod
    def _distance_two_cities(city1, city2):
        return np.sqrt((city1['x'] - city2['x']) ** 2 + (city1['y'] - city2['y']) ** 2)
