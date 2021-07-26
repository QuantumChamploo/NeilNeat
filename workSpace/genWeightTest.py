
import random
import os
import time
import neat
import visualize
import pickle
from bareSnake import snakeGame
import numpy as np
import math
from neat.graphs import feed_forward_layers
from neat.graphs import required_for_output

gen = 0



def makeList(mat):
	hld = []
	for i in mat:
		for j in i:
			#print(j)
			hld.append(j)
	return hld

def find_index(array):
    s = np.array(array)
    sort_index = np.argsort(s)
    return sort_index

def eval_genomes(genomes, config):
    #print('in eval')

    global gen

    gen += 1


    nets = []
    games = []
    ge = []

    for genome_id, genome in genomes:
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        games.append(snakeGame())
        ge.append(genome)

    run = True
    tab = 0

    while run and len(games) > 0:
        tab +=1
        # if tab % 100 == 0:
        #     print('did 100 runs')
        #     print(tab)
        #     print(len(games))

        for x, net in enumerate(nets):
            inputs = [[0, 1], [1, 0]]
            outputs = [[1, 0], [0, 1]]
            lr = 1
            if gen == 0:
                for i in range(200):
                    #print("prebackprop")
                    net.backProp_activate(inputs, outputs, lr)
                look = net.backProp_activate(inputs, outputs, lr)
                # print("final guess")
                # print(look)
            out = net.activate(inputs[0])

            ind_arr = find_index(out)
            max_index = ind_arr[-1]
            nex_index = ind_arr[-2]
            if max_index == 0:
                #tasprint("got the first one")
                ge[x].fitness += 32
            out2 = net.activate(inputs[1])
            ind_arr2 = find_index(out2)
            maxIndex2 = ind_arr2[-1]
            if maxIndex2 == 1:
                #print("got the second one")
                ge[x].fitness += 32


            run = False




def run(config_file):
    """
    runs the NEAT algorithm to train a neural network to play flappy bird.
    :param config_file: location of config file
    :return: None
    """
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    #print('post config')
    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    #p.add_reporter(neat.Checkpointer(5))

    # Run for up to 50 generations.
    winner = p.run(eval_genomes,10)

    net = neat.nn.FeedForwardNetwork.create(winner, config)

    # show final stats
    print('\nBest genome:\n{!s}'.format(winner))

    print(repr(winner))
    print("the fitness is ")
    print(winner.fitness)
    visualize.plot_stats(stats,"Population's average and best fitness: Standard unconnected", ylog=False, view=True)
    visualize.plot_species(stats, view=True)

    connections = [cg.key for cg in winner.connections.values() if cg.enabled]
    print("printing connections")
    print(connections)

    layers = feed_forward_layers(config.genome_config.input_keys, config.genome_config.output_keys, connections)
    print("printing layers")
    print(layers)

    req = required_for_output(config.genome_config.input_keys, config.genome_config.output_keys, connections)
    print("printing required")
    print(req)
    output = net.activate((1, 0))
    output2 = net.activate([0,1])

    print("the sample outputs are")
    print(output)
    print(output2)

    #print(winner.game.history)

if __name__ == '__main__':
    #print('in start')
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')
    run(config_path)