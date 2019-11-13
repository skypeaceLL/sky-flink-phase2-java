package sky.tf.multiplemodels;

import com.alibaba.tianchi.garbage_image_util.ConfigConstant;
import com.alibaba.tianchi.garbage_image_util.IdLabel;
import com.alibaba.tianchi.garbage_image_util.ImageData;
import org.apache.flink.api.common.functions.RichFlatMapFunction;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.util.Collector;
import sky.tf.GarbageClassificationModel;
import sky.tf.ImageClassIndex;
import sky.tf.ModelParams;

import java.util.ArrayList;
import java.util.List;

/**
 * @author SkyPeace
 * The Map operator for predict the class of image. Experimental only.
 */
public class ImageFlatMapMultipleModels extends RichFlatMapFunction<ImageData, IdLabel> {
    private String imageModelPath = System.getenv(ConfigConstant.IMAGE_MODEL_PATH);
    private String imageModelPathPackagePath = System.getenv(ConfigConstant.IMAGE_MODEL_PACKAGE_PATH);
    private String modelInferencePath = System.getenv("MODEL_INFERENCE_PATH");
    private List<ModelParams> modelParamsList;
    private transient List<GarbageClassificationModel> modelList;
    private transient ImagePredictSupportiveMultipleModels supportive;

    public ImageFlatMapMultipleModels(List<ModelParams> modelParamsList)
    {
        this.modelParamsList = modelParamsList;
    }

    @Override
    public void open(Configuration parameters) throws Exception
    {
        //For troubleshooting use.
        System.out.println(String.format("ImageFlatMap.open(): imageModelPath is %s", this.imageModelPath));
        System.out.println(String.format("ImageFlatMap.open(): modelInferencePath is %s", this.modelInferencePath));
        System.out.println(String.format("ImageFlatMap.open(): imageModelPathPackagePath is %s", this.imageModelPathPackagePath));

        //Step2: Load optimized OpenVino model from files (HDFS).
        // Cost about 1 seconds each model, quick enough. But it is not good solution to use too many models in client.
        modelList = new ArrayList<GarbageClassificationModel>();
        for(ModelParams modelParams:modelParamsList) {
            GarbageClassificationModel model =
                    ImageModelLoaderMultipleModels.getInstance().loadOpenVINOModelOnce(modelParams);
            modelList.add(model);
        }

        //First time warm-dummy check
        ImageClassIndex.getInsatnce().loadClassIndexMap(modelInferencePath);
        this.supportive = new ImagePredictSupportiveMultipleModels(modelList, modelParamsList);
        this.supportive.firstTimeDummyCheck();
    }

    @Override
    public void flatMap(ImageData value, Collector<IdLabel> out)
            throws Exception
    {
        IdLabel idLabel = new IdLabel();
        idLabel.setId(value.getId());

        long beginTime = System.currentTimeMillis();
        System.out.println(String.format("PREDICT_PROCESS BEGIN %s", beginTime));

        String imageLabelString = supportive.predictHumanString(value);

        long endTime = System.currentTimeMillis();
        System.out.println(String.format("PREDICT_PROCESS END %s (Cost: %s)", endTime, (endTime - beginTime)));

        //Check whether elapsed time >= threshold. Logging it for review.
        if((endTime - beginTime)>498)
            System.out.println(String.format("PREDICT_PROCESS MAYBE EXCEED 500ms %s (Cost: %s)",
                                                endTime, (endTime - beginTime)));

        idLabel.setLabel(imageLabelString);
        out.collect(idLabel);
    }

    @Override
    public void close() throws Exception
    {
        System.out.println(String.format("getNumberOfParallelSubtasks: %s",
                this.getRuntimeContext().getNumberOfParallelSubtasks()));
        for(GarbageClassificationModel model:modelList) {
            model.release();
        }
    }

}

