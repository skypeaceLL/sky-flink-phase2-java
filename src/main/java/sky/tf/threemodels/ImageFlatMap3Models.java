package sky.tf.threemodels;

import com.alibaba.tianchi.garbage_image_util.ConfigConstant;
import com.alibaba.tianchi.garbage_image_util.IdLabel;
import com.alibaba.tianchi.garbage_image_util.ImageData;
import org.apache.flink.api.common.functions.RichFlatMapFunction;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.util.Collector;
import sky.tf.GarbageClassificationModel;
import sky.tf.ImageClassIndex;
import sky.tf.ModelParams;

/**
 * @author SkyPeace
 * The Map operator for predict the class of image.
 */
public class ImageFlatMap3Models extends RichFlatMapFunction<ImageData, IdLabel> {
    private String imageModelPath = System.getenv(ConfigConstant.IMAGE_MODEL_PATH);
    private String imageModelPathPackagePath = System.getenv(ConfigConstant.IMAGE_MODEL_PACKAGE_PATH);
    private String modelInferencePath = System.getenv("MODEL_INFERENCE_PATH");
    private ModelParams model1Params;
    private ModelParams model2Params;
    private ModelParams model3Params;
    private transient GarbageClassificationModel model1;
    private transient GarbageClassificationModel model2;
    private transient GarbageClassificationModel model3;
    private transient ImagePredictSupportive3Models supportive;

    public ImageFlatMap3Models(ModelParams model1Params, ModelParams model2Params, ModelParams model3Params)
    {
        this.model1Params = model1Params;
        this.model2Params = model2Params;
        this.model3Params = model3Params;
    }

    @Override
    public void open(Configuration parameters) throws Exception
    {
        //For troubleshooting use.
        System.out.println(String.format("ImageFlatMap.open(): imageModelPath is %s", this.imageModelPath));
        System.out.println(String.format("ImageFlatMap.open(): modelInferencePath is %s", this.modelInferencePath));
        System.out.println(String.format("ImageFlatMap.open(): imageModelPathPackagePath is %s", this.imageModelPathPackagePath));
        System.out.println(String.format("ImageFlatMap.open(): optimizedOpenVinoModel1Dir is %s",
                this.model1Params.getOptimizedModelDir()));
        System.out.println(String.format("ImageFlatMap.open(): optimizedOpenVinoModel2Dir is %s",
                this.model2Params.getOptimizedModelDir()));
        System.out.println(String.format("ImageFlatMap.open(): optimizedOpenVinoModel3Dir is %s",
                this.model3Params.getOptimizedModelDir()));

        //Step2: Load optimized OpenVino model from files (HDFS). (Cost about 2 seconds each model, quick enough)
        this.model1 = ImageModelLoader3Models.getInstance().loadOpenVINOModel1Once(this.model1Params);
        this.model2 = ImageModelLoader3Models.getInstance().loadOpenVINOModel2Once(this.model2Params);
        this.model3 = ImageModelLoader3Models.getInstance().loadOpenVINOModel3Once(this.model3Params);

        ImageClassIndex.getInsatnce().loadClassIndexMap(modelInferencePath);
        ValidationDatasetAnalyzer validationTrainer = new ValidationDatasetAnalyzer();
        validationTrainer.searchOptimizedThreshold(modelInferencePath);
        float preferableThreshold1 = validationTrainer.getPreferableThreshold1();
        float preferableThreshold2 = validationTrainer.getPreferableThreshold2();
        this.supportive = new ImagePredictSupportive3Models(this.model1, this.model1Params,
                this.model2, this.model2Params, this.model3, this.model3Params,
                preferableThreshold1, preferableThreshold2);
        //First time warm-dummy check
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
        if((endTime - beginTime)>495)
            System.out.println(String.format("PREDICT_PROCESS MAYBE EXCEED THRESHOLD %s (Cost: %s)",
                    endTime, (endTime - beginTime)));

        idLabel.setLabel(imageLabelString);
        out.collect(idLabel);
    }

    @Override
    public void close() throws Exception
    {
        System.out.println(String.format("getNumberOfParallelSubtasks: %s",
                this.getRuntimeContext().getNumberOfParallelSubtasks()));
        model1.release();
        if(model2!=null)
            model2.release();
        if(model3!=null)
            model3.release();
    }

}

