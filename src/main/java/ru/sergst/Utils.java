package ru.sergst;

import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;

import java.util.Arrays;

public class Utils {

    public static void testNeuralNetwork(final NeuralNetwork nnet, final DataSet tset) {

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
