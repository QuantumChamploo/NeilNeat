
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
import tensorflow as tf

tf.config.run_functions_eagerly(True)

gen = 0

stillGradTrain = True
stillDo = True

pastGamesStates = [[],[]]

def makeList(mat):
	hld = []
	for i in mat:
		for j in i:
			#print(j)
			hld.append(j)
	return hld

def find_index(array):
    if array == [0,0,0,0]:
        return np.random.choice(4, 4, replace=False)
    s = np.array(array)
    sort_index = np.argsort(s)
    return sort_index

def eval_genomes(genomes, config):
    #print('in eval')

    global gen
    global stillDo
    global stillGradTrain

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
        #print(genome)

    run = True
    tab = 0



    while run and len(games) > 0:
        # print("\n\n")
        # print("got out of enum looop")
        # print("\n")
        tab +=1

        if stillDo == False and stillGradTrain == True:
            print("flipped gradtrain")
            stillGradTrain = False


        for x, game in enumerate(games):
            # print("the games szie is ")
            # print(len(games))
            # print("\n\n\n in top of enumerate")
            assert ge[x] == ge[games.index(game)]

            # print(x)
            ge[x].game = game
            moved = False
            #output = nets[games.index(game)].activate((makeList(game.toString())))
            output = nets[games.index(game)].activate((makeList(game.sight)))
            ind_arr = find_index(output)
            max_index = ind_arr[-1]
            nex_index = ind_arr[-2]
            third_index = ind_arr[-3]
            fourth_index = ind_arr[-4]

            enumTime1 = time.time()

            lr = 1
            batchSize = 100

            # if gen % 50 == 0 and len(pastGamesStates[0]) > batchSize and stillGradTrain:
            #     stillDo = False
            #     print("flipped stillDo")
            #     batchTime1 = time.time()
            #     print("about to start 10 sized training loop on all examples")
            #     for i in range(20):
            #         print(i)
            #         nets[games.index(game)].backProp_GenomeFast(pastGamesStates[0][-batchSize:-1], pastGamesStates[1][-batchSize:-1], lr,ge[x])
            #
            #     print("\n the batch acc is")
            #     print(nets[games.index(game)].batchAcc(pastGamesStates[0][0:10],pastGamesStates[1][0:10]))
            #     batchTime2 = time.time()
            #     print("the post training time for genome")
            #     print(batchTime2-batchTime1)
            # if gen % 51 == 0:
            #     stillDo = True
            #     stillGradTrain = True
            # print("\n\n\n")
            # print("got out of decsent loop")
            # print("\n\n")

            initGame = game
            initScore = game.score
            initFit = ge[x].fitness

            commit = False
            #print(output)
            if max_index == 0:
                #print("went left")
                direction = "left"
                commit = initGame.check4Mem(direction)
                # if commit == True:
                #     print("in indexing")
                #     print(direction)

                #game.move_left()
            if max_index == 1:
                #print("went right")
                direction = "right"
                commit = initGame.check4Mem(direction)
                # if commit == True:
                #     print("in indexing")
                #     print(direction)
                #game.move_right()
            if max_index == 2:
                #print("went up")
                direction = "up"
                commit = initGame.check4Mem(direction)
                # if commit == True:
                #     print("in indexing")
                #     print(direction)
                #game.move_up()
            if max_index == 3:
                #print("went down")
                direction = "down"
                commit = initGame.check4Mem(direction)
                # if commit == True:
                #     print("in indexing")
                #     print(direction)
                #game.move_down()



            # if commit == True:
            #     innies, outties = initGame.commit2Mem(max_index)
            #     pastGamesStates[0].append(makeList(innies))
            #     pastGamesStates[1].append(outties)
            #     # print("\n \n \n")
            #     # print(makeList(innies))
            #     # print(outties)
            #     # print(direction)
            #     # print(initGame.munchMove(direction))
            #     # print(initGame.movedFromWall(direction))
            #     # print(initGame.movedTowardsFood(direction))
            #     # print(initGame.check4Mem(direction))


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

            alive = game.checkGame()

            enumTime2 = time.time()



            # if gen % 5 == 0 and len(pastGamesStates) > batchSize:
            #     print("the full genome training time")
            #     print(enumTime2 - enumTime1)

            if game.popCount > 400:
                alive = False

            if alive == False:
                if len(games) == 1:

                    hlder = 33

                nets.pop(games.index(game))
                ge.pop(games.index(game))
                games.pop(games.index(game))

            else:

                if game.popCount < 40:
                    ge[x].fitness += 5
                    # if move_dict[max_index][1] == 1:
                    #     ge[x].fitness += 10


                if newScore - initScore > 0:
                    ge[x].fitness += 200

                if ge[x].fitness > initFit and game.popCount >40:
                    print("something is not good")



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

    prerunTime = time.time()



    # Run for up to 50 generations.
    winner = p.run(eval_genomes,800)

    postRunTime = time.time()

    print("the entire time is ")
    print(postRunTime-prerunTime)

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
    #visualize.draw_net(config, winner, True, 'snakeMultLayerStnd',node_names=node_names)
    print(repr(winner))
    print("the fitness is ")
    print(winner.fitness)
    visualize.plot_stats(stats,"Population's average and best fitness: Standard noConnected", ylog=False, view=True)
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
    config_path = os.path.join(local_dir, 'configStandardSnake2.txt')
    run(config_path)