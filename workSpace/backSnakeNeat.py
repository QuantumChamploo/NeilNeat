
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

        for x, game in enumerate(games):
            ge[x].game = game
            moved = False
            #output = nets[games.index(game)].activate((makeList(game.toString())))
            output = nets[games.index(game)].activate((makeList(game.sight)))
            ind_arr = find_index(output)
            max_index = ind_arr[-1]
            nex_index = ind_arr[-2]
            third_index = ind_arr[-3]
            fourth_index = ind_arr[-4]

            initScore = game.score
            initFit = ge[x].fitness
            if max_index == 0:
                game.move_left()
            if max_index == 1:
                game.move_right()
            if max_index == 2:
                game.move_up()
            if max_index == 3:
                game.move_down()

            move_dict = {0:game.propDirect([-10, 0]),1:game.propDirect([10,0]),2:game.propDirect([0, -10]),3:game.propDirect([0,10])}
            newScore = game.score
            # print('gene num is ')
            # print(x)
            # print(max_index)
            # print(game.direction)
            alive = game.checkGame()
            if game.popCount > 400:
                alive = False
                #print("over pop")
            if alive == False:
                if len(games) == 1:
                    #print(game.history)
                    #print(game.sight)
                    hlder = 33

                nets.pop(games.index(game))
                ge.pop(games.index(game))
                games.pop(games.index(game))
                #print('popped one')
            else:
                if game.popCount < 40:
                    ge[x].fitness += 5
                    if move_dict[max_index][1] == 1:
                        ge[x].fitness += 10

                #ge[x].fitness += 4*2**(-1*game.distToFood()/15)
                # if game.d2f < game.d2fPrev:
                #     ge[x].fitness += 3
                #     # if x == 0:
                #     #     print("got closer")
                # else:
                #     ge[x].fitness -= 2
                # # if x == 0:
                # else:
                #     print("greater than 40 pops")
                #     print(game.history)
                #     print("\n")
                #     print("\n")
                #     print("\n")

                #     print(4*2**(-1*game.distToFood()/15))
                if newScore - initScore > 0:
                    ge[x].fitness += 200
                    # print (game.score)
                    # if game.score > 1:
                    #     for i in game.history:
                    #         print(i)
                if ge[x].fitness > initFit and game.popCount >40:
                    print("something is not good")

                    #print('moving towards food')

        #  if tab % 100 == 0:
                # print('did 100 runs')
                # print(tab)
                # print(len(games))
                # print('gene num is ')
                # print(x)
                # print(max_index)
                # print(game.direction)
                # print(game.pos)
                # print(game.popCount)
            # if x == 0:
            # 	print(game.toString())
            # 	print("---")
            # 	print("---")

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
    winner = p.run(eval_genomes,250)

    # show final stats
    print('\nBest genome:\n{!s}'.format(winner))
    node_names = {0:'left',1:'right',2:'up',3:'down',-1:'E body',-2:'E food',-3:'E dist'
                                                    ,-4:'S body',-5:'S food',-6:'S dist'
                                                    ,-7:'N body',-8:'N food',-9:'N dist'
                                                    ,-10:'W body',-11:'W food',-12:'W dist'
                                                    ,-13:'SE body',-14:'SE food',-15:'SE dist'
                                                    ,-16:'NE body',-17:'NE food',-18:'NE dist'
                                                    ,-19:'SW body',-20:'SW food',-21:'SW dist'
                                                    ,-22:'NW body',-23:'NW food',-24:'NW dist'}
    visualize.draw_net(config, winner, True, 'snakeMultLayerStnd',node_names=node_names)
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

    #print(winner.game.history)

if __name__ == '__main__':
    #print('in start')
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'configStandardSnake.txt')
    run(config_path)