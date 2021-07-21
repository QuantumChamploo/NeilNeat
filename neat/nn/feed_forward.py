from neat.graphs import feed_forward_layers
from neat.aggregations import *
from neat.activations import  *

import tensorflow as tf
import math

aggDict = {}
actDict = {tanh_activation:tf.keras.activations.tanh,sigmoid_activation:tf.keras.activations.sigmoid,
           relu_activation:tf.keras.activations.relu,elu_activation:tf.keras.activations.elu,
           softplus_activation:tf.keras.activations.softplus,identity_activation:tf.keras.activations.linear}


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
