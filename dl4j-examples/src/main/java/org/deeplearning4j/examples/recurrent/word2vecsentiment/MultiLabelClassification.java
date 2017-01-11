package org.deeplearning4j.examples.recurrent.word2vecsentiment;

/**
 * Created by ryan on 11/01/17.
 */
public class MultiLabelClassification {

    public static void main(String[] args) throws Exception{
        String train_file = "/home/ryan/projects/dl_service/test_multi_lable.txt";
        String test_file = train_file;
        String word2vec_file = "/home/ryan/projects/dl_service/deps.words";
        int number_of_labels = 3;
        double learning_rate = 0.25;
        double regularization = 0.001;
        int epochs = 3;
        Word2VecMultiLabelCategorisationRNN classifier = new Word2VecMultiLabelCategorisationRNN(train_file, test_file, word2vec_file, number_of_labels, learning_rate, regularization, epochs);
        classifier.trainAndTest();
    }

}
