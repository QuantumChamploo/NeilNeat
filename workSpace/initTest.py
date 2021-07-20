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
    print(genome)
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    net = neat.nn.FeedForwardNetwork.create(genome, config)
    #visualize.draw_net(config, genome, True, 'snakeTest')
    output2 = net.activate((1, 0))
    # for node, act_func, agg_func, bias, response, links in net.node_evals:
    #     for i, w in links:
    #         print(w)
    output = net.prop_activate((1,0))
    print("the second weights")
    # for node, act_func, agg_func, bias, response, links in net.node_evals:
    #     for i, w in links:
    #         print(w)
    print("first output")
    print(output)
    print("the original activate function")
    print(output2)

    for i in range(100):
        output = net.prop_activate((1, 0))



    print('the 100th output is ')
    print(output)



    # ind_arr = find_index(output)
    #
    # print(ind_arr)

    #bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)


    # print(genomes)
    # for genome_id, genome in genomes:
    #     print(genome_id)
    #     print(genome)
    #     print("\n \n \n")


if __name__ == '__main__':
    #print('in start')
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')
    run(config_path)