package org.deeplearning4j.examples.feedforward.regression;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.apache.commons.math3.analysis.function.Exp;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.RegressionEvaluation;
import org.deeplearning4j.nn.api.Model;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.deeplearning4j.util.ModelSerializer;
import java.io.File;
import org.nd4j.linalg.dataset.api.preprocessor.serializer.NormalizerMinMaxScalerSerializer;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * Created by Anwar on 3/15/2016.
 * An example of regression neural network for performing addition
 */
public class Regression {
    //Random number generator seed, for reproducability
    public static final int seed = 12345;
    //Number of iterations per minibatch
    public static final int iterations = 1;
    //Number of epochs (full passes of the data)
    public static final int nEpochs = 50000;
    //Number of data points
    //Batch size: i.e., each epoch has nSamples/batchSize parameter updates
    public static final int batchSize = 64;
    //Network learning rate
    public static final double learningRate = 0.0065;
    // The range of the sample data, data in range (0-1 is sensitive for NN, you can try other ranges and see how it effects the results
    // also try changing the range along with changing the activation function
    public static final Random rng = new Random(seed);

    public static void main(String[] args) throws Exception {

        RecordReader rr = new CSVRecordReader();
        rr.initialize(new FileSplit(new File("C:\\Users\\Profein\\Desktop\\dl4j-examples\\dl4j-examples-master\\dl4j-examples\\src\\main\\resources\\regression\\trainset.csv")));
        DataSetIterator trainIter = new RecordReaderDataSetIterator(rr, batchSize, 0, 0, true);
        RecordReader rrTest = new CSVRecordReader();
        rrTest.initialize(new FileSplit(new File("C:\\Users\\Profein\\Desktop\\dl4j-examples\\dl4j-examples-master\\dl4j-examples\\src\\main\\resources\\regression\\testset.csv")));
        DataSetIterator testIter = new RecordReaderDataSetIterator(rrTest, batchSize, 0, 0, true);
        NormalizerMinMaxScaler normalizer = new NormalizerMinMaxScaler(0, 1);
        normalizer.fitLabel(true);

        int numInput = 15;
        int numOutputs = 1;
        int nHidden = 20;
        MultiLayerConfiguration net = new NeuralNetConfiguration.Builder()
            .seed(seed)
            .iterations(iterations)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .learningRate(learningRate)
            .weightInit(WeightInit.XAVIER)
            .updater(Updater.NESTEROVS).momentum(0.8)

            .list()
            .layer(0, new DenseLayer.Builder().nIn(numInput).nOut(nHidden)
                .activation(Activation.RELU)

                .build())





            .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                .activation(Activation.IDENTITY)
                .nIn(nHidden).nOut(numOutputs).build())
            .pretrain(false).backprop(true).build()
            ;
        normalizer.fit(trainIter);
        trainIter.setPreProcessor(normalizer);
        testIter.setPreProcessor(normalizer);


        //Create the network
        MultiLayerNetwork model = new MultiLayerNetwork(net);

        model.init();
        model.setListeners(new ScoreIterationListener(1));
        RegressionEvaluation evaluation = new RegressionEvaluation(1);

        DataSet trainData = trainIter.next();
        DataSet testData = testIter.next();
        for (int i = 0; i < nEpochs; i++) {

            model.fit(trainData);

            INDArray features = testData.getFeatureMatrix();
            INDArray labels = testData.getLabels();
            INDArray predicted = model.output(features, false);
            evaluation.eval(labels, predicted);

            System.out.println(evaluation.stats());}
            INDArray features = testData.getFeatureMatrix();
            INDArray labels = testData.getLabels();
            INDArray prediction = model.output(features,false);
            normalizer.revert(testData);
            normalizer.revertLabels(prediction);
            System.out.println(prediction+" "+labels);
        File locationToSave = new File("MyReg2.zip");      //Where to save the network. Note: the file is in .zip format - can be opened externally
        boolean saveUpdater = true;                                             //Updater: i.e., the state for Momentum, RMSProp, Adagrad etc. Save this if you want to train your network more in the future
        ModelSerializer.writeModel(model, locationToSave, saveUpdater);
        File tmpFile = new File ( "normalizers1.zip");
        NormalizerMinMaxScalerSerializer serializer = new NormalizerMinMaxScalerSerializer();

        serializer.write(normalizer, tmpFile);
        System.out.print("Saved bih");








    }
}




