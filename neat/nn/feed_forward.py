from neat.graphs import feed_forward_layers
from neat.aggregations import *
from neat.activations import  *
import numpy as np
import time

import tensorflow as tf
import math

aggDict = {}
actDict = {tanh_activation:tf.keras.activations.tanh,sigmoid_activation:tf.keras.activations.sigmoid,
           relu_activation:tf.keras.activations.relu,elu_activation:tf.keras.activations.elu,
           softplus_activation:tf.keras.activations.softplus,identity_activation:tf.keras.activations.linear}

def find_index(array):
    s = np.array(array)
    sort_index = np.argsort(s)
    return sort_index

class FeedForwardNetwork(object):
    def __init__(self, inputs, outputs, node_evals):
        self.input_nodes = inputs
        self.output_nodes = outputs
        self.node_evals = node_evals
        self.values = dict((key, 0.0) for key in inputs + outputs)

    def activate(self, inputs):
        if len(self.input_nodes) != len(inputs):
            raise RuntimeError("Expected {0:n} inputs, got {1:n}".format(len(self.input_nodes), len(inputs)))

        for k, v in zip(self.input_nodes, inputs):
            self.values[k] = v

        for node, act_func, agg_func, bias, response, links in self.node_evals:
            node_inputs = []
            for i, w in links:
                node_inputs.append(self.values[i] * w)
            s = agg_func(node_inputs)
            self.values[node] = act_func(bias + response * s)


        return [self.values[i] for i in self.output_nodes]

    def batchAcc(self,inputs,outputs):
        if len(inputs) != len(outputs):
            print("inputs size does not match output size")
            return 0
        right = 0
        for i in range(len(inputs)):
            out = self.activate(inputs[i])

            if find_index(out)[-1] == find_index(outputs[i])[-1]:
                right += 1
        return right/len(inputs)



    def prop_activate(self, inputs,expOutput):
        if len(self.input_nodes) != len(inputs):
            raise RuntimeError("Expected {0:n} inputs, got {1:n}".format(len(self.input_nodes), len(inputs)))

        for k, v in zip(self.input_nodes, inputs):
            self.values[k] = v

        tensValues = self.values
        tensDict = {}
        tensVars = []
        tensEvals = []

        for node, act_func, agg_func, bias, response, links in self.node_evals:
            tensLinks = []
            for i, w in links:
                hldVar = tf.Variable(w,dtype=tf.float64)
                tensLinks.append((i, hldVar))
                tensVars.append(hldVar)
                tensDict[hldVar.ref()] = w

            tensEvals.append((node, act_func, agg_func, bias, response, tensLinks))

        with tf.GradientTape(persistent=True) as tape:
            bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

            for node, act_func, agg_func, bias, response, links in tensEvals:
                node_inputs = []
                s = 0
                for i, w in links:

                    node_inputs.append(tensValues[i] * w)

                    s += tensValues[i] * w


                tensValues[node] = actDict[act_func](bias + response * s)

                # print(act_func(bias + response * s))
            loss = bce(expOutput,[tensValues[i] for i in self.output_nodes])
            # print("the loss is ")
            # print(loss)

        newEvals = []
        count = 0
        # for i in tensVars:
        #     print(i)
        for node, act_func, agg_func, bias, response, links in self.node_evals:
            newLinks = []
            for i, w in links:
                dif = tape.gradient(loss,tensVars[count])

                w -= dif
                newLinks.append((i, w.numpy()))
                count += 1


            newEvals.append((node, act_func, agg_func, bias, response, newLinks))

        self.node_evals = newEvals



        return [tensValues[i].numpy() for i in self.output_nodes]

    def backProp_activate(self, inputs,outputs,learning_rate):
        if len(self.input_nodes) != len(inputs):
            raise RuntimeError("Expected {0:n} inputs, got {1:n}".format(len(self.input_nodes), len(inputs)))


        tensValues = self.values
        tensDict = {}
        tensVars = []
        tensEvals = []

        outReturn = []

        for node, act_func, agg_func, bias, response, links in self.node_evals:
            tensLinks = []
            for i, w in links:
                hldVar = tf.Variable(w,dtype=tf.float64)
                tensLinks.append((i, hldVar))
                tensVars.append(hldVar)
                tensDict[hldVar.ref()] = w

            tensEvals.append((node, act_func, agg_func, bias, response, tensLinks))
        tensDif = []
        for k in range(len(tensVars)):
            tensDif.append(0)



        for j in range(len(inputs)):
            with tf.GradientTape(persistent=True) as tape:
                bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
                for k, v in zip(self.input_nodes, inputs[j]):
                    self.values[k] = v
                for node, act_func, agg_func, bias, response, links in tensEvals:
                    node_inputs = []
                    s = 0
                    for i, w in links:

                        node_inputs.append(tensValues[i] * w)

                        s += tensValues[i] * w


                    tensValues[node] = actDict[act_func](bias + response * s)


                loss = bce(outputs[j],[tensValues[i] for i in self.output_nodes])

            num = 0
            for node, act_func, agg_func, bias, response, links in tensEvals:
                for i,w in links:
                    dif = tape.gradient(loss,tensVars[num])
                    tensDif[num] += dif
                    num += 1

            outReturn.append([tensValues[i].numpy() for i in self.output_nodes])
        newEvals = []
        count = 0
        # for i in tensVars:
        #     print(i)
        for node, act_func, agg_func, bias, response, links in self.node_evals:
            newLinks = []
            for i, w in links:
                delta = tensDif[count]

                w -= delta*learning_rate
                newLinks.append((i, w.numpy()))
                count += 1


            newEvals.append((node, act_func, agg_func, bias, response, newLinks))

        self.node_evals = newEvals



        return outReturn

    def backProp_Genome(self, inputs,outputs,learning_rate,genome):
        # if len(self.input_nodes) != len(inputs):
        #     raise RuntimeError("Expected {0:n} inputs, got {1:n}".format(len(self.input_nodes), len(inputs)))


        tensValues = self.values
        tensDict = {}
        tensVars = []
        tensEvals = []

        outReturn = []

        for node, act_func, agg_func, bias, response, links in self.node_evals:
            tensLinks = []
            for i, w in links:
                hldVar = tf.Variable(w,dtype=tf.float64)
                tensLinks.append((i, hldVar))
                tensVars.append(hldVar)
                tensDict[hldVar.ref()] = w

            tensEvals.append((node, act_func, agg_func, bias, response, tensLinks))
        tensDif = []
        for k in range(len(tensVars)):
            tensDif.append(0)



        for j in range(len(inputs)):
            with tf.GradientTape(persistent=True) as tape:
                bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
                for k, v in zip(self.input_nodes, inputs[j]):
                    self.values[k] = v
                for node, act_func, agg_func, bias, response, links in tensEvals:
                    node_inputs = []
                    s = 0
                    for i, w in links:

                        node_inputs.append(tensValues[i] * w)

                        s += tensValues[i] * w


                    tensValues[node] = actDict[act_func](bias + response * s)


                loss = bce(outputs[j],[tensValues[i] for i in self.output_nodes])

            num = 0
            dif = tape.gradient(loss,tensVars[num])
            for node, act_func, agg_func, bias, response, links in tensEvals:
                for i,w in links:
                    #dif = tape.gradient(loss,tensVars[num])
                    tensDif[num] += dif[num]
                    num += 1


            outReturn.append([tensValues[i].numpy() for i in self.output_nodes])
        newEvals = []
        count = 0
        # for i in tensVars:
        #     print(i)
        for node, act_func, agg_func, bias, response, links in self.node_evals:
            newLinks = []
            for i, w in links:
                delta = tensDif[count]

                w -= delta*learning_rate
                newLinks.append((i, w.numpy()))
                count += 1
                genome.connections[(i,node)].weight = w


            newEvals.append((node, act_func, agg_func, bias, response, newLinks))

        self.node_evals = newEvals



        return outReturn

    @tf.function
    def backProp_GenomeTimed(self, inputs,outputs,learning_rate,genome):
        # if len(self.input_nodes) != len(inputs):
        #     raise RuntimeError("Expected {0:n} inputs, got {1:n}".format(len(self.input_nodes), len(inputs)))


        pretime1 = time.time()


        tensValues = self.values
        tensDict = {}
        tensVars = []
        tensEvals = []

        outReturn = []

        for node, act_func, agg_func, bias, response, links in self.node_evals:
            tensLinks = []
            for i, w in links:
                hldVar = tf.Variable(w,dtype=tf.float64)
                tensLinks.append((i, hldVar))
                tensVars.append(hldVar)
                tensDict[hldVar.ref()] = w

            tensEvals.append((node, act_func, agg_func, bias, response, tensLinks))
        tensDif = []
        for k in range(len(tensVars)):
            tensDif.append(0)

        pretime2 = time.time()

        print("the pre_eval time is")
        print(pretime2-pretime1)

        for j in range(len(inputs)):
            loopTime1 = time.time()
            with tf.GradientTape(persistent=True) as tape:
                bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
                for k, v in zip(self.input_nodes, inputs[j]):
                    self.values[k] = v
                for node, act_func, agg_func, bias, response, links in tensEvals:
                    node_inputs = []
                    s = 0
                    for i, w in links:

                        node_inputs.append(tensValues[i] * w)

                        s += tensValues[i] * w


                    tensValues[node] = actDict[act_func](bias + response * s)


                loss = bce(outputs[j],[tensValues[i] for i in self.output_nodes])

            num = 0
            for node, act_func, agg_func, bias, response, links in tensEvals:
                for i,w in links:
                    dif = tape.gradient(loss,tensVars[num])
                    tensDif[num] += dif
                    num += 1

            print("the loop time is ")
            loopTime2 = time.time()
            print(loopTime2-loopTime1)
            # for i in self.output_nodes:
            #     print(tensValues[i])

            outReturn.append([tensValues[i].numpy for i in self.output_nodes])


            print("in loop:")
            print(j+1)


        postTime1 = time.time()

        newEvals = []
        count = 0
        # for i in tensVars:
        #     print(i)
        for node, act_func, agg_func, bias, response, links in self.node_evals:
            newLinks = []
            for i, w in links:
                delta = tensDif[count]

                w -= delta*learning_rate
                newLinks.append((i, w.numpy()))
                count += 1
                genome.connections[(i,node)].weight = w


            newEvals.append((node, act_func, agg_func, bias, response, newLinks))

        self.node_evals = newEvals


        postTime2 = time.time()

        print("the post loop time:")
        print(postTime2-postTime1)

        return outReturn

    @tf.function
    def backProp_GenomeTimed2(self, inputs, outputs, learning_rate, genome):
        # if len(self.input_nodes) != len(inputs):
        #     raise RuntimeError("Expected {0:n} inputs, got {1:n}".format(len(self.input_nodes), len(inputs)))

        pretime1 = time.time()

        tensValues = self.values
        tensDict = {}
        tensVars = []
        tensEvals = []

        outReturn = []

        for node, act_func, agg_func, bias, response, links in self.node_evals:
            tensLinks = []
            for i, w in links:
                hldVar = tf.Variable(w, dtype=tf.float64)
                tensLinks.append((i, hldVar))
                tensVars.append(hldVar)
                tensDict[hldVar.ref()] = w

            tensEvals.append((node, act_func, agg_func, bias, response, tensLinks))
        tensDif = []
        for k in range(len(tensVars)):
            tensDif.append(0)

        pretime2 = time.time()

        print("the pre_eval time is")
        print(pretime2 - pretime1)

        loopTime1 = time.time()

        with tf.GradientTape() as tape:
            bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

            logits = []
            for j in range(len(inputs)):
                print("in inner loop")
                innerTime = time.time()


                for k, v in zip(self.input_nodes, inputs[j]):
                    self.values[k] = v
                for node, act_func, agg_func, bias, response, links in tensEvals:
                    node_inputs = []
                    s = 0
                    for i, w in links:
                        node_inputs.append(tensValues[i] * w)

                        s += tensValues[i] * w

                    tensValues[node] = actDict[act_func](bias + response * s)
                logits.append([tensValues[i] for i in self.output_nodes])
                outReturn.append([tensValues[i].numpy for i in self.output_nodes])

                outerTime = time.time()
                print("inner loop time is ")
                print(outerTime - innerTime)

            print("loss calc time is")
            calc1 = time.time()
            loss = bce(outputs, logits)
            calc2 = time.time()
            print(calc2-calc1)

        gradtime1 = time.time()
        num = 0
        dif = tape.gradient(loss, tensVars)
        for node, act_func, agg_func, bias, response, links in tensEvals:
            for i, w in links:
                #dif = tape.gradient(loss, tensVars[num])
                tensDif[num] += dif[num]
                num += 1
        gradtime2 = time.time()
        print("the gradient calc time is ")
        print(gradtime2-gradtime1)
        print("the loop time is \n\n")
        loopTime2 = time.time()
        print(loopTime2 - loopTime1)
        # for i in self.output_nodes:
        #     print(tensValues[i])



        postTime1 = time.time()

        newEvals = []
        count = 0
        # for i in tensVars:
        #     print(i)
        for node, act_func, agg_func, bias, response, links in self.node_evals:
            newLinks = []
            for i, w in links:
                delta = tensDif[count]

                w -= delta * learning_rate
                newLinks.append((i, w.numpy()))
                count += 1
                genome.connections[(i, node)].weight = w

            newEvals.append((node, act_func, agg_func, bias, response, newLinks))

        self.node_evals = newEvals

        postTime2 = time.time()

        print("the post loop time:")
        print(postTime2 - postTime1)

        return outReturn

    @tf.function
    def backProp_GenomeFast(self, inputs, outputs, learning_rate, genome):
        # if len(self.input_nodes) != len(inputs):
        #     raise RuntimeError("Expected {0:n} inputs, got {1:n}".format(len(self.input_nodes), len(inputs)))

        #t1 = time.time()

        tensValues = self.values
        tensDict = {}
        tensVars = []
        tensEvals = []

        outReturn = []

        for node, act_func, agg_func, bias, response, links in self.node_evals:
            tensLinks = []
            for i, w in links:
                hldVar = tf.Variable(w, dtype=tf.float64)
                tensLinks.append((i, hldVar))
                tensVars.append(hldVar)
                tensDict[hldVar.ref()] = w

            tensEvals.append((node, act_func, agg_func, bias, response, tensLinks))
        tensDif = []
        for k in range(len(tensVars)):
            tensDif.append(0)

        #t2 = time.time()

        #print("init time %d",(t2 - t1))


        with tf.GradientTape() as tape:
            bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
            #t3 = time.time()
            #print("gradient initialization time is %d",(t3 - t2))
            logits = []
            for j in range(len(inputs)):
                t4 = time.time()
                for k, v in zip(self.input_nodes, inputs[j]):
                    self.values[k] = v
                for node, act_func, agg_func, bias, response, links in tensEvals:
                    node_inputs = []
                    s = 0
                    for i, w in links:
                        node_inputs.append(tensValues[i] * w)

                        s += tensValues[i] * w

                    tensValues[node] = actDict[act_func](bias + response * s)
                logits.append([tensValues[i] for i in self.output_nodes])
                outReturn.append([tensValues[i] for i in self.output_nodes])
                #t5 = time.time()
                #print("activation time is %d",(t5 - t4))
            #t6 = time.time()
            #print("the full set act time is $d",(t6 - t3))




            loss = bce(outputs, logits)
            #t7 = time.time()
            #print("the loss function time is %d",(t7-t6))

        #t8 = time.time()
        #print("the whole grad tape process is %d",(t8-t2))

        num = 0
        dif = tape.gradient(loss, tensVars)
        #t9 = time.time()
        #print("tape.gradient time spent is %d",(t9 - t8))
        for node, act_func, agg_func, bias, response, links in tensEvals:
            for i, w in links:
                # dif = tape.gradient(loss, tensVars[num])
                tensDif[num] += dif[num]
                num += 1

        # for i in self.output_nodes:
        #     print(tensValues[i])

        #t10 = time.time()
        #print("making dif array time is %d",(t10 - t9))
        newEvals = []
        count = 0
        # for i in tensVars:
        #     print(i)
        for node, act_func, agg_func, bias, response, links in self.node_evals:
            newLinks = []
            for i, w in links:
                delta = tensDif[count]

                w -= delta * learning_rate
                newLinks.append((i, w.numpy()))
                count += 1
                genome.connections[(i, node)].weight = w

            newEvals.append((node, act_func, agg_func, bias, response, newLinks))

        self.node_evals = newEvals

        #t11 = time.time()

        #print("the writing of genome file time is %d",(t11 - t10))
        #print("the total time is %d", (t11 - t1))


        return outReturn

    def backProp_Genome2(self, inputs,outputs,learning_rate,genome):
        # if len(self.input_nodes) != len(inputs):
        #     raise RuntimeError("Expected {0:n} inputs, got {1:n}".format(len(self.input_nodes), len(inputs)))

        if len(inputs) == 0 or len(outputs) == 0:
            return []
        tensValues = self.values
        tensDict = {}
        tensVars = []
        tensEvals = []

        outReturn = []
        # print("the genome is ")
        # print(genome)

        for node, act_func, agg_func, bias, response, links in self.node_evals:
            tensLinks = []
            for i, w in links:
                # print("printing links pre eval")
                # print(i)
                # print(node)
                hldVar = tf.Variable(w,dtype=tf.float64)
                tensLinks.append((i, hldVar))
                tensVars.append(hldVar)
                tensDict[hldVar.ref()] = w

            tensEvals.append((node, act_func, agg_func, bias, response, tensLinks))
        tensDif = []
        for k in range(len(tensVars)):
            tensDif.append(0)



        for j in range(len(inputs)):
            # print("doing a backprop input")
            # print(j)
            with tf.GradientTape(persistent=True) as tape:
                bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
                for k, v in zip(self.input_nodes, inputs[j]):
                    self.values[k] = v
                for node, act_func, agg_func, bias, response, links in tensEvals:
                    node_inputs = []
                    s = 0
                    for i, w in links:

                        node_inputs.append(tensValues[i] * w)

                        s += tensValues[i] * w


                    tensValues[node] = actDict[act_func](bias + response * s)


                loss = bce(outputs[j],[tensValues[i] for i in self.output_nodes])

            num = 0
            for node, act_func, agg_func, bias, response, links in tensEvals:
                for i,w in links:
                    dif = tape.gradient(loss,tensVars[num])
                    tensDif[num] += dif
                    num += 1


            outReturn.append([tensValues[i] for i in self.output_nodes])
        newEvals = []
        count = 0
        # for i in tensVars:
        #     print(i)
        for node, act_func, agg_func, bias, response, links in tensEvals:
            #newLinks = []
            for i, w in links:
                # print("printing links")
                # print(i)
                # print(node)
                wHld = w.numpy()
                #print(wHld)
                delta = tensDif[count]

                wHld = wHld - delta*learning_rate
                #newLinks.append((i, w.numpy()))
                count += 1
                genome.connections[(i,node)].weight = wHld


            #newEvals.append((node, act_func, agg_func, bias, response, newLinks))

        self.node_evals = newEvals



        return outReturn


    def backProp_GenomeNumpy(self, inputs,outputs,learning_rate,genome):
        # if len(self.input_nodes) != len(inputs):
        #     raise RuntimeError("Expected {0:n} inputs, got {1:n}".format(len(self.input_nodes), len(inputs)))

        if len(inputs) == 0 or len(outputs) == 0:
            return np.array([])
        tensValues = self.values
        #tensDict = {}
        tensVars = np.array([])
        tensEvals = np.array([])

        outReturn = np.array([])
        # print("the genome is ")
        # print(genome)

        for node, act_func, agg_func, bias, response, links in self.node_evals:
            tensLinks = np.array([])
            for i, w in links:
                # print("printing links pre eval")
                # print(i)
                # print(node)
                hldVar = tf.Variable(w,dtype=tf.float64)
                tensLinks = np.append(tensLinks,(i, hldVar))
                tensVars = np.append(tensVars,hldVar)
                #tensDict[hldVar.ref()] = w

            tensEvals = np.append(tensEvals,(node, act_func, agg_func, bias, response, tensLinks))
        tensDif = np.zeros(len(tensVars))



        for j in range(len(inputs)):
            # print("doing a backprop input")
            # print(j)
            with tf.GradientTape(persistent=True) as tape:
                bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
                for k, v in zip(self.input_nodes, inputs[j]):
                    self.values[k] = v
                for node, act_func, agg_func, bias, response, links in tensEvals:
                    np.array([])
                    s = 0
                    for i, w in links:

                        node_inputs.append(tensValues[i] * w)

                        s += tensValues[i] * w


                    tensValues[node] = actDict[act_func](bias + response * s)


                loss = bce(outputs[j],[tensValues[i] for i in self.output_nodes])

            num = 0
            for node, act_func, agg_func, bias, response, links in tensEvals:
                for i,w in links:
                    dif = tape.gradient(loss,tensVars[num])
                    tensDif[num] += dif
                    num += 1


            outReturn.append([tensValues[i] for i in self.output_nodes])
        np.array([])
        count = 0
        # for i in tensVars:
        #     print(i)
        for node, act_func, agg_func, bias, response, links in tensEvals:
            #newLinks = []
            for i, w in links:
                # print("printing links")
                # print(i)
                # print(node)
                wHld = w.numpy()
                #print(wHld)
                delta = tensDif[count]

                wHld = wHld - delta*learning_rate
                #newLinks.append((i, w.numpy()))
                count += 1
                genome.connections[(i,node)].weight = wHld


            #newEvals.append((node, act_func, agg_func, bias, response, newLinks))

        self.node_evals = newEvals



        return outReturn

    def backProp_activate2(self, inputs,outputs):
        if len(self.input_nodes) != len(inputs):
            raise RuntimeError("Expected {0:n} inputs, got {1:n}".format(len(self.input_nodes), len(inputs)))


        tensValues = self.values
        tensDict = {}
        tensVars = []
        tensEvals = []

        outReturn = []

        for node, act_func, agg_func, bias, response, links in self.node_evals:
            tensLinks = []
            for i, w in links:
                hldVar = tf.Variable(w,dtype=tf.float64)
                tensLinks.append((i, hldVar))
                tensVars.append(hldVar)
                tensDict[hldVar.ref()] = w

            tensEvals.append((node, act_func, agg_func, bias, response, tensLinks))
        tensDif = []
        for k in range(len(tensVars)):
            tensDif.append(0)

        with tf.GradientTape(persistent=True) as tape:
            bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

            for j in range(len(inputs)):
                for k, v in zip(self.input_nodes, inputs[j]):
                    self.values[k] = v
                for node, act_func, agg_func, bias, response, links in tensEvals:
                    node_inputs = []
                    s = 0
                    for i, w in links:

                        node_inputs.append(tensValues[i] * w)

                        s += tensValues[i] * w


                    tensValues[node] = actDict[act_func](bias + response * s)


                loss = bce(outputs[j],[tensValues[i] for i in self.output_nodes])

                num = 0
                for node, act_func, agg_func, bias, response, links in tensEvals:
                    for i,w in links:
                        dif = tape.gradient(loss,tensVars[num])
                        tensDif[num] += dif
                        num += 1

                outReturn.append([tensValues[i].numpy() for i in self.output_nodes])
        newEvals = []
        count = 0
        # for i in tensVars:
        #     print(i)
        for node, act_func, agg_func, bias, response, links in self.node_evals:
            newLinks = []
            for i, w in links:
                delta = tensDif[count]

                w -= delta
                newLinks.append((i, w.numpy()))
                count += 1


            newEvals.append((node, act_func, agg_func, bias, response, newLinks))

        self.node_evals = newEvals



        return outReturn

    @staticmethod
    def create(genome, config):
        """ Receives a genome and returns its phenotype (a FeedForwardNetwork). """

        # Gather expressed connections.
        connections = [cg.key for cg in genome.connections.values() if cg.enabled]

        layers = feed_forward_layers(config.genome_config.input_keys, config.genome_config.output_keys, connections)
        node_evals = []
        for layer in layers:
            for node in layer:
                inputs = []
                node_expr = [] # currently unused
                for conn_key in connections:
                    inode, onode = conn_key
                    if onode == node:
                        cg = genome.connections[conn_key]
                        inputs.append((inode, cg.weight))
                        node_expr.append("v[{}] * {:.7e}".format(inode, cg.weight))

                ng = genome.nodes[node]
                aggregation_function = config.genome_config.aggregation_function_defs.get(ng.aggregation)
                activation_function = config.genome_config.activation_defs.get(ng.activation)
                node_evals.append((node, activation_function, aggregation_function, ng.bias, ng.response, inputs))

        return FeedForwardNetwork(config.genome_config.input_keys, config.genome_config.output_keys, node_evals)
