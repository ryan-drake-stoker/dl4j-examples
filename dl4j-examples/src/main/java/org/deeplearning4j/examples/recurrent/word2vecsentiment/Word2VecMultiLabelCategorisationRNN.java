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
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.*;
import java.net.URL;

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
    public final String TRAIN_DATA_PATH;
    public final String TEST_DATA_PATH;
    /** Location (local file system) for the Google News vectors. Set this manually. */
    public final String WORD_VECTORS_PATH;// = "/home/ryan/projects/dl_service/deps.words";
    private final int number_of_labels;


    public Word2VecMultiLabelCategorisationRNN(String train_file_name, String test_file_name, String word_vec_file, int number_of_labels){
        this.TRAIN_DATA_PATH = train_file_name;
        this.TEST_DATA_PATH = test_file_name;
        this.WORD_VECTORS_PATH = word_vec_file;
        this.number_of_labels = number_of_labels;
    }


    public void trainAndTest() throws Exception {
        if(WORD_VECTORS_PATH.startsWith("/PATH/TO/YOUR/VECTORS/")){
            throw new RuntimeException("Please set the WORD_VECTORS_PATH before running this example");
        }

        //Download and extract data
        //downloadData();

        int batchSize = 64;     //Number of examples in each minibatch
        int vectorSize = 300;   //Size of the word vectors. 300 in the Google News model
        int nEpochs = 1;        //Number of epochs (full passes of training data) to train on
        int truncateReviewsToLength = 256;  //Truncate reviews with length (# words) greater than this

        //Set up network configuration
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .updater(Updater.ADAM).adamMeanDecay(0.9).adamVarDecay(0.999)
            .regularization(true).l2(1e-5)
            .weightInit(WeightInit.XAVIER)
            .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue).gradientNormalizationThreshold(1.0)
            .learningRate(2e-2)
            .list()
            .layer(0, new GravesLSTM.Builder().nIn(vectorSize).nOut(256)
                .activation(Activation.TANH).build())
            .layer(1, new RnnOutputLayer.Builder().activation(Activation.SOFTMAX)
                .lossFunction(LossFunctions.LossFunction.MCXENT).nIn(256).nOut(3).build())
            .pretrain(false).backprop(true).build();

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
        MultiLabelSentenceCSVIterator train = new MultiLabelSentenceCSVIterator(TRAIN_DATA_PATH, wordVectors, 3,batchSize, truncateReviewsToLength);
        MultiLabelSentenceCSVIterator test = new MultiLabelSentenceCSVIterator(TEST_DATA_PATH, wordVectors, 3,batchSize, truncateReviewsToLength);

        System.out.println("Starting training");
        for (int i = 0; i < nEpochs; i++) {
            net.fit(train);
            train.reset();
            System.out.println("Epoch " + i + " complete. Starting evaluation:");

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
        }



        System.out.println("----- Example complete -----");
    }


}
