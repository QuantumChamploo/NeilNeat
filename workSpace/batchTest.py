import neat
import os
import visualize
import numpy as np
import time




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
    print(genome)
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    net = neat.nn.FeedForwardNetwork.create(genome, config)
    inputs = [[0,1],[1,0]]
    outputs = [[1,0],[0,1]]

    start1 = time.time()
    intro = net.backProp_activate(inputs,outputs,.000001)
    print(intro)
    lr = .05
    for i in range(500):
        net.backProp_activate(inputs,outputs,lr)
    look = net.backProp_activate(inputs,outputs,lr)
    print(look)




if __name__ == '__main__':
    #print('in start')
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')
    run(config_path)