package org.deeplearning4j.examples.recurrent.word2vecsentiment;

/**
 * Created by ryan on 11/01/17.
 */
public class MultiLabelClassification {

    public static void main(String[] args) throws Exception{
        String train_file = "/home/ryan/projects/dl_service/test_multi_lable.txt";
        String test_file = "/home/ryan/projects/dl_service/test_multi_lable.txt";
        String word2vec_file = "/home/ryan/projects/dl_service/deps.words";
        int number_of_labels = 3;

        Word2VecMultiLabelCategorisationRNN classifier = new Word2VecMultiLabelCategorisationRNN(train_file, test_file, word2vec_file, number_of_labels);

        if(args.length > 0){
            classifier.setLearning_rate(Double.parseDouble(args[0]));
        }
        if(args.length > 1){
            classifier.setRegularization_rate(Double.parseDouble(args[1]));
        }

        if(args.length > 2){
            classifier.setEpochs(Integer.parseInt(args[2]));
        }

        if(args.length > 3){
            classifier.setBatchSize(Integer.parseInt(args[3]));
        }

        classifier.trainAndTest();
    }

}
