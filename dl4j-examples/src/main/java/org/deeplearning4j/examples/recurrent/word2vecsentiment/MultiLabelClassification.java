package org.deeplearning4j.examples.recurrent.word2vecsentiment;

import org.nd4j.jita.conf.CudaEnvironment;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.util.DataTypeUtil;

/**
 * Created by ryan on 11/01/17.
 */
public class MultiLabelClassification {

    public static void main(String[] args) throws Exception{
        String train_file = "/home/ryan/projects/dl_service/test_short.txt";
        String test_file = "/home/ryan/projects/dl_service/test_short.txt";
        String word2vec_file = "/home/ryan/projects/dl_service/deps.words";
        int number_of_labels = 3;

//        DataTypeUtil.setDTypeForContext(DataBuffer.Type.HALF);
//        CudaEnvironment.getInstance().getConfiguration()
//            .setMaximumDeviceCacheableLength(1024 * 1024 * 1024L)
//            .setMaximumDeviceCache(12L * 1024 * 1024 * 1024L)
//            .setMaximumHostCacheableLength(1024 * 1024 * 1024L)
//            .setMaximumHostCache(12L * 1024 * 1024 * 1024L);

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

        if(args.length > 4){
            classifier.setDecay_rate(Double.parseDouble(args[4]));
        }

        classifier.trainAndTest();
    }

}
