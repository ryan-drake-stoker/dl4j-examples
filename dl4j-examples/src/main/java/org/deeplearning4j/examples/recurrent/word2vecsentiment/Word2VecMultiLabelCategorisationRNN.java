package org.deeplearning4j.examples.recurrent.word2vecsentiment;

import org.apache.commons.compress.archivers.tar.TarArchiveEntry;
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream;
import org.apache.commons.compress.compressors.gzip.GzipCompressorInputStream;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.*;
import java.net.URL;
import java.util.Date;

/**Example: Given a movie review (raw text), classify that movie review as either positive or negative based on the words it contains.
 * This is done by combining Word2Vec vectors and a recurrent neural network model. Each word in a review is vectorized
 * (using the Word2Vec model) and fed into a recurrent neural network.
 * Training data is the "Large Movie Review Dataset" from http://ai.stanford.edu/~amaas/data/sentiment/
 * This data set contains 25,000 training reviews + 25,000 testing reviews
 *
 * Process:
 * 1. Automatic on first run of example: Download data (movie reviews) + extract
 * 2. Load existing Word2Vec model (for example: Google News word vectors. You will have to download this MANUALLY)
 * 3. Load each each review. Convert words to vectors + reviews to sequences of vectors
 * 4. Train network
 *
 * With the current configuration, gives approx. 83% accuracy after 1 epoch. Better performance may be possible with
 * additional tuning.
 *
 * NOTE / INSTRUCTIONS:
 * You will have to download the Google News word vector model manually. ~1.5GB
 * The Google News vector model available here: https://code.google.com/p/word2vec/
 * Download the GoogleNews-vectors-negative300.bin.gz file
 * Then: set the WORD_VECTORS_PATH field to point to this location.
 *
 * @author Alex Black
 */
public class Word2VecMultiLabelCategorisationRNN {


    /** Location to save and extract the training/testing data */
    private String TRAIN_DATA_PATH;
    private String TEST_DATA_PATH;


    /** Location (local file system) for the word2vec vectors. Set this manually. */
    private String WORD_VECTORS_PATH;// = "/home/ryan/projects/dl_service/deps.words";
    private int number_of_labels;
    private double learning_rate;
    private double regularization_rate;
    private int epochs;
    //Number of examples in each minibatch
    private int batchSize;



    public Word2VecMultiLabelCategorisationRNN(String train_file_name, String test_file_name, String word_vec_file, int number_of_labels ){
        this.TRAIN_DATA_PATH = train_file_name;
        this.TEST_DATA_PATH = test_file_name;
        this.WORD_VECTORS_PATH = word_vec_file;
        this.number_of_labels = number_of_labels;
        this.learning_rate = 0.25;
        this.regularization_rate = 0.001;
        this.epochs = 3;
        this.batchSize = 124;
    }

    private void printSettings(){
        System.out.println("Train data:" + this.TRAIN_DATA_PATH);
        System.out.println("Test data:" + this.TEST_DATA_PATH);
        System.out.println("WORD_VECTORS_PATH:" + this.WORD_VECTORS_PATH);
        System.out.println("number_of_labels:" + this.number_of_labels);
        System.out.println("learning_rate:" + this.learning_rate);
        System.out.println("regularization_rate:" + this.regularization_rate);
        System.out.println("epochs:" + this.epochs);
        System.out.println("batchSize:" + this.batchSize);
    }


    public void trainAndTest() throws Exception {
        printSettings();
        int vectorSize = 300;   //Size of the word vectors. 300 in the Google News model
        int truncateReviewsToLength = 300;  //Truncate reviews with length (# words) greater than this

        //Set up network configuration
        MultiLayerConfiguration conf = getSimpleRNNConfiguration(vectorSize);

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(new ScoreIterationListener(1));



        //Initialize the user interface backend
        UIServer uiServer = UIServer.getInstance();

        //Configure where the network information (gradients, score vs. time etc) is to be stored. Here: store in memory.
        StatsStorage statsStorage = new InMemoryStatsStorage();         //Alternative: new FileStatsStorage(File), for saving and loading later

        //Attach the StatsStorage instance to the UI: this allows the contents of the StatsStorage to be visualized
        uiServer.attach(statsStorage);

        //Then add the StatsListener to collect this information from the network, as it trains
        net.setListeners(new StatsListener(statsStorage));


        //DataSetIterators for training and testing respectively
        WordVectors wordVectors = WordVectorSerializer.loadStaticModel(new File(WORD_VECTORS_PATH));
        MultiLabelSentenceCSVIterator train = new MultiLabelSentenceCSVIterator(TRAIN_DATA_PATH, wordVectors, number_of_labels,batchSize, truncateReviewsToLength);
        MultiLabelSentenceCSVIterator test = new MultiLabelSentenceCSVIterator(TEST_DATA_PATH, wordVectors, number_of_labels,batchSize, truncateReviewsToLength);
        Date startpoint = new Date();
        System.out.println("Starting training at " + startpoint.toString());
        File res_tracker = new File("training_results.csv");
        for (int i = 0; i < epochs; i++) {
            System.out.println("Epoch " + i + " starting at :" + new Date().toString());
            net.fit(train);
            train.reset();
            System.out.println("Epoch " + i + " complete at :" + new Date().toString() + ". Starting evaluation:");

            //Run evaluation. This is on 25k reviews, so can take some time
            Evaluation evaluation = new Evaluation();
            while (test.hasNext()) {
                DataSet t = test.next();
                INDArray features = t.getFeatureMatrix();
                INDArray lables = t.getLabels();
                INDArray inMask = t.getFeaturesMaskArray();
                INDArray outMask = t.getLabelsMaskArray();
                INDArray predicted = net.output(features, false, inMask, outMask);

                evaluation.evalTimeSeries(lables, predicted, outMask);
            }
            test.reset();

            System.out.println(evaluation.stats());
            String out_line = this.learning_rate + ", " + this.regularization_rate + ", " + this.batchSize + ", " + this.epochs + ", " + i + ", " + evaluation.f1() + ", " + evaluation.precision() + ", " + evaluation.accuracy() + "\n";
            FileUtils.write(res_tracker, out_line, true);
            String model_name = "lr-" + this.learning_rate + "_rg-" + this.regularization_rate + "_bs-" + this.batchSize + "_ep-" + this.epochs + "_" + i + "_RNN";
            model_name = model_name.replaceAll("\\.", "") + ".model";
            File f = new File(model_name);
            System.out.println("Writing out model");
            System.out.println(f.getAbsoluteFile());
            ModelSerializer.writeModel(net,f,true);

        }



        System.out.println("----- Example complete -----");
    }

    private MultiLayerConfiguration getSimpleRNNConfiguration(int vectorSize) {
        return new NeuralNetConfiguration.Builder()
            .updater(Updater.ADAM).adamMeanDecay(0.9).adamVarDecay(0.999)
            .regularization(true).l2(regularization_rate)
            .weightInit(WeightInit.XAVIER)
            .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue).gradientNormalizationThreshold(1.0)
            .learningRate(learning_rate)
            .list()
            .layer(0, new GravesLSTM.Builder().nIn(vectorSize).nOut(256)
                .activation(Activation.TANH).build())
            .layer(1, new RnnOutputLayer.Builder().activation(Activation.SOFTMAX)
                .lossFunction(LossFunctions.LossFunction.MCXENT).nIn(256).nOut(number_of_labels).build())
            .pretrain(false).backprop(true).build();
    }

    private MultiLayerConfiguration getBiDirectionalRNNConfiguration(int vectorSize) {
        int tbpttLength = 50;
        return new NeuralNetConfiguration.Builder()
            .updater(Updater.ADAM).adamMeanDecay(0.9).adamVarDecay(0.999)
            .regularization(true).l2(regularization_rate)
            .weightInit(WeightInit.XAVIER)
            .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue).gradientNormalizationThreshold(1.0)
            .learningRate(learning_rate)
            .list()
            .layer(0, new GravesLSTM.Builder().nIn(vectorSize).nOut(256)
                .activation(Activation.TANH).build())
            .layer(1, new GravesLSTM.Builder().nIn(vectorSize).nOut(256)
                .activation(Activation.TANH).build())
            .layer(2, new RnnOutputLayer.Builder().activation(Activation.SOFTMAX)
                .lossFunction(LossFunctions.LossFunction.MCXENT).nIn(256).nOut(number_of_labels).build())
            .backpropType(BackpropType.TruncatedBPTT).tBPTTForwardLength(tbpttLength).tBPTTBackwardLength(tbpttLength)
            .pretrain(false).backprop(true).build();
    }


    public String getTRAIN_DATA_PATH() {
        return TRAIN_DATA_PATH;
    }

    public void setTRAIN_DATA_PATH(String TRAIN_DATA_PATH) {
        this.TRAIN_DATA_PATH = TRAIN_DATA_PATH;
    }

    public String getTEST_DATA_PATH() {
        return TEST_DATA_PATH;
    }

    public void setTEST_DATA_PATH(String TEST_DATA_PATH) {
        this.TEST_DATA_PATH = TEST_DATA_PATH;
    }

    public String getWORD_VECTORS_PATH() {
        return WORD_VECTORS_PATH;
    }

    public void setWORD_VECTORS_PATH(String WORD_VECTORS_PATH) {
        this.WORD_VECTORS_PATH = WORD_VECTORS_PATH;
    }

    public int getNumber_of_labels() {
        return number_of_labels;
    }

    public void setNumber_of_labels(int number_of_labels) {
        this.number_of_labels = number_of_labels;
    }

    public double getLearning_rate() {
        return learning_rate;
    }

    public void setLearning_rate(double learning_rate) {
        this.learning_rate = learning_rate;
    }

    public double getRegularization_rate() {
        return regularization_rate;
    }

    public void setRegularization_rate(double regularization_rate) {
        this.regularization_rate = regularization_rate;
    }

    public int getEpochs() {
        return epochs;
    }

    public void setEpochs(int epochs) {
        this.epochs = epochs;
    }

    public int getBatchSize() {
        return batchSize;
    }

    public void setBatchSize(int batchSize) {
        this.batchSize = batchSize;
    }



}
