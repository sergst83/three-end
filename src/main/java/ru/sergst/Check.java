package ru.sergst;

import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.events.NeuralNetworkEvent;

import java.util.Arrays;
import java.util.function.Function;

import static ru.sergst.Utils.testNeuralNetwork;

public class Check {
    public static void main(String[] args) {
        var dataset = DataSet.createFromFile("dataset.csv", 3, 1, ";", true);
        var net = NeuralNetwork.createFromFile("learn_result_trained.nnet");
        net.addListener(event -> {
            if (event.getEventType() == NeuralNetworkEvent.Type.CALCULATED) {
                var n = (NeuralNetwork) event.getSource();
            }
        });
        System.out.println(net.getOutputNeurons());

        System.out.println(Arrays.toString(net.getWeights()));

        testNeuralNetwork(net, dataset);

        var bias = -0.893310190784654;
        var weights = new double[] {0.11794317088612527, 0.28613058343431585, 0.4904617908398686};
        var inputSet = new double[][] {
                {0, 0, 0},
                {0, 0, 1},
                {0, 1, 0},
                {0, 1, 1},
                {1, 0, 0},
                {1, 0, 1},
                {1, 1, 0},
                {1, 1, 1},
        };
        var transferFunction = (Function<Double, Double>) n -> n > 0d ? 1.0 : 0.0;
        for (double[] input : inputSet) {
            var output = 0d;
            for (int i = 0; i < input.length; i++) {
                output += input[i] * weights[i];
            }
            output += bias;
            var result = transferFunction.apply(output);

            System.out.printf("input %s, output %.10f%n", Arrays.toString(input), result);
        }
    }
}
