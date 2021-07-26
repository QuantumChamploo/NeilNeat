import neat
import os
import visualize
import numpy as np



print("first test")

def find_index(array):
    s = np.array(array)
    sort_index = np.argsort(s)
    return sort_index

def run(config_file):
    """
    runs the NEAT algorithm to train a neural network to play flappy bird.
    :param config_file: location of config file
    :return: None
    """
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    p = neat.Population(config)

    genomes = p.population.items()
    genome1key = next(iter(p.population))
    genome = p.population[genome1key]
    print(genome1key)
    print("the genome is")
    print(genome)
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    print("genome connection values")
    print(genome.connections.values())
    for i in genome.connections.values():
        print(i.weight)
        print(i.key)
        print(genome.connections[i.key].key)
    print(genome.connections.values())
    print(genome.connections.keys())

    for node, act_func, agg_func, bias, response, links in net.node_evals:

        for i, w in links:
            print(node)
            print(i)
            print(w)
            print(genome.connections[(i,node)].weight)


if __name__ == '__main__':
    #print('in start')
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')
    run(config_path)