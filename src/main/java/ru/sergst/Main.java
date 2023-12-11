package ru.sergst;

import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.nnet.Perceptron;
import org.neuroph.nnet.learning.BackPropagation;
import org.neuroph.util.Neuroph;
import org.neuroph.util.TransferFunctionType;

import java.net.URISyntaxException;
import java.time.Duration;
import java.time.Instant;
import java.util.Arrays;

import static org.neuroph.core.events.LearningEvent.Type.EPOCH_ENDED;

public class Main {

    private static final String nnetFileName = "learn_result.nnet";
    public static void main(String[] args) throws URISyntaxException {
        System.out.printf("%s ver. %s%n%n", Neuroph.class.getSimpleName(), Neuroph.getVersion());
        var dataset = DataSet.createFromFile("dataset.csv", 3, 1, ";", true);
        var dataset_xor = DataSet.createFromFile("dataset_xor.csv", 2, 1, ";", true);
        var currentDataSet = dataset;

//        var perceptron = new MultiLayerPerceptron(List.of(2, 2, 1), new NeuronProperties(TransferFunctionType.SIGMOID, false));
        var perceptron = new Perceptron
                (3, 1, TransferFunctionType.STEP);
        perceptron.setLearningRule(new BackPropagation());
        var learningRule = (BackPropagation) perceptron.getLearningRule();
        learningRule.setMaxIterations(100_000);
        learningRule.addListener(event -> {
            if (event.getEventType() == EPOCH_ENDED) {
                    System.out.printf(
                            "Эпоха: %d, Ошибка: %.10f\n",
                            learningRule.getCurrentIteration(),
                            learningRule.getTotalNetworkError()
                    );
                    if (learningRule.getCurrentIteration() == 1) {
                        System.out.printf("Веса после 1го прохода: %s%n", Arrays.toString(perceptron.getWeights()));
                        testNeuralNetwork(perceptron, dataset);
                    }
            }
        });
        Runtime.getRuntime().addShutdownHook(new Thread(() -> perceptron.save("learn_result.nnet")));

        System.out.printf("Начальные веса: %s%n%n", Arrays.toString(perceptron.getWeights()));
        System.out.println("Обучаем нейросеть...");
        var start = Instant.now();
        do {
            perceptron.randomizeWeights();
            perceptron.learn(currentDataSet);
        } while (learningRule.getTotalNetworkError() > learningRule.getMaxError());
        System.out.printf("Обучение завершено за %s миллисекунд.\n\n", Duration.between(start, Instant.now()).toMillis());
        System.out.printf("Сохраняем результат обучения в %s.%n", nnetFileName);
        perceptron.save(nnetFileName);
        System.out.printf("Веса обученной нейросети: %s%n", Arrays.toString(perceptron.getWeights()));
        System.out.printf("Ошибка обученной нейросети: %.10f\n", learningRule.getTotalNetworkError());
        System.out.printf("Количество эпох: %d%n", learningRule.getCurrentIteration());
        System.out.println("Проверка:");
        testNeuralNetwork(perceptron, currentDataSet);
    }

    public static void testNeuralNetwork(NeuralNetwork nnet, DataSet tset) {

        for (DataSetRow dataRow : tset.getRows()) {

            nnet.setInput(dataRow.getInput());
            nnet.calculate();
            double[ ] networkOutput = nnet.getOutput();
            System.out.print("Input: " + Arrays.toString(dataRow.getInput()) );
            System.out.println(" Output: " + Arrays.toString(Arrays.stream(networkOutput).mapToObj("%.10f"::formatted).toArray()));
        }
        System.out.println();
    }
}