from neat.graphs import feed_forward_layers

import tensorflow as tf
import math


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
    def prop_activate(self, inputs):
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
                    #print(w)
                    node_inputs.append(tensValues[i] * w)

                    s += tensValues[i] * w


                tensValues[node] = tf.keras.activations.tanh(bias + response * s)

                # print(act_func(bias + response * s))
            loss = bce([0,1,0,1,0],[tensValues[i] for i in self.output_nodes])
            # print("the loss is ")
            # print(loss)

        # grads = tape.gradient(loss,tensVars)
        # print(tensVars)
        # print("the gradient is ")
        # print(grads)
        # print("the tensDict is ")
        # print(tensDict)
        # print(tensDict[tensVars[0].ref()])
        newEvals = []
        count = 0
        # for i in tensVars:
        #     print(i)
        for node, act_func, agg_func, bias, response, links in self.node_evals:
            newLinks = []
            for i, w in links:
                dif = tape.gradient(loss,tensVars[count])
                #print(w)
                #print(dif)
                w -= dif
                newLinks.append((i, w.numpy()))
                count += 1


            newEvals.append((node, act_func, agg_func, bias, response, newLinks))
        #print(newEvals)
        self.node_evals = newEvals
        # for node, act_func, agg_func, bias, response, links in self.node_evals:
        #     for i,w in links:
        #         print(w)
        # self.node_evals = newEvals
        # for i in tensVars:
        #     grad = tape.gradient(loss,i)
        #     weight = tensDict[i.ref()]
        #     #print(weight)
        #     #weight += grad.numpy()
        #     weight = 0
        # print("zerod weights")
        #
        # for node, act_func, agg_func, bias, response, links in self.node_evals:
        #     for i,w in links:
        #         #print(tensVars[count])
        #         print(w)
        #         dif = tape.gradient(loss,tensVars[count])
                #print(dif)
        #         count += 1
        #         #w += dif.numpy()
        #         w = 0
        #         #print(w)

        # for node, act_func, agg_func, bias, response, links in self.node_evals:
        #     for i, w in links:
        #         print(w)
        #
        # print("printing change")
        # print([tensValues[i].numpy() for i in self.output_nodes])
        #print(self.activate(inputs))


        return [tensValues[i].numpy() for i in self.output_nodes]
    # def prop_activate(self, inputs):
    #     if len(self.input_nodes) != len(inputs):
    #         raise RuntimeError("Expected {0:n} inputs, got {1:n}".format(len(self.input_nodes), len(inputs)))
    #
    #     for k, v in zip(self.input_nodes, inputs):
    #         self.values[k] = v
    #
    #     tensLinks = []
    #     tensVars = []
    #     for node, act_func, agg_func, bias, response, links in self.node_evals:
    #         for i, w in links:
    #             hldVar = tf.Variable(w)
    #             tensLinks.append((i, hldVar))
    #             tensVars.append(hldVar)
    #
    #     with tf.GradientTape(persistent=True) as tape:
    #         bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    #
    #         for node, act_func, agg_func, bias, response, links in self.node_evals:
    #             node_inputs = []
    #
    #             for i, w in tensLinks:
    #
    #                 print(self.values[i])
    #                 node_inputs.append(self.values[i] * w)
    #             print(tensVars)
    #             ders = tape.gradient(loss,tensVars)
    #             print("the derivatives are")
    #             print(ders)
    #             s = agg_func(node_inputs)
    #             self.values[node] = act_func(bias + response * s)
    #
    #
    #     return [self.values[i] for i in self.output_nodes]


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
