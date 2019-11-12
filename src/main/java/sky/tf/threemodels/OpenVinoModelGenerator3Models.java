package sky.tf.threemodels;

import com.alibaba.tianchi.garbage_image_util.ConfigConstant;
import org.apache.flink.api.java.tuple.Tuple2;
import sky.tf.ModelParams;

import java.io.File;
import java.util.List;

/**
 * The class use for generate optimized OpenVino models (Three type of models for prediction)
 * @author SkyPeace
 */
public class OpenVinoModelGenerator3Models {
    private String imageModelPath = System.getenv(ConfigConstant.IMAGE_MODEL_PATH);
    private String imageModelPackagePath = System.getenv(ConfigConstant.IMAGE_MODEL_PACKAGE_PATH);
    private String modelInferencePath = System.getenv("MODEL_INFERENCE_PATH");
    private ModelParams model1Params;
    private ModelParams model2Params;
    private ModelParams model3Params;

    public void execute()
    {
        if (null == this.modelInferencePath) {
            throw new RuntimeException("ImageFlatMap(): Not set MODEL_INFERENCE_PATH environmental variable");
        }
        if (null == this.imageModelPackagePath) {
            throw new RuntimeException("ImageFlatMap(): Not set imageModelPathPackagePath environmental variable");
        }
        String imageModelPackageDir =
                imageModelPackagePath.substring(0, imageModelPackagePath.lastIndexOf(File.separator));
        if(!imageModelPackageDir.equalsIgnoreCase(modelInferencePath))
        {
            System.out.println("WARN: modelInferencePath NOT EQUAL imageModelPathPackageDir");
            System.out.println(String.format("modelInferencePath: %s", modelInferencePath));
            System.out.println(String.format("imageModelPackageDir: %s", imageModelPackageDir));
            System.out.println(String.format("imageModelPath: %s", imageModelPath));
        }

        try {
            //Generate optimized OpenVino model1. resnet_v1_101
            String tfModel1Path = modelInferencePath + File.separator + "SavedModel/model1";
            String optimizedOpenVinoModel1Dir = imageModelPackageDir;
            model1Params = new ModelParams();
            model1Params.setModelType("resnet");
            model1Params.setModelName("model1");
            model1Params.setInputName("input_1");
            model1Params.setInputShape(new int[]{1, 224, 224, 3});
            model1Params.setMeanValues(new float[]{123.68f,116.78f,103.94f});
            model1Params.setScale(1.0f);
            model1Params.setOptimizedModelDir(optimizedOpenVinoModel1Dir);
            //Call Zoo API generate optimized OpenVino Model
            List<Tuple2<String, byte[]>> optimizedOpenVinoModelData =
                    ImageModelLoader3Models.getInstance().generateOpenVinoModelData(tfModel1Path, model1Params);
            //Write optimized model's bytes into files (HDFS). The optimized model files parent dir is same as TF model package.
            ImageModelLoader3Models.getInstance().
                    saveToOpenVinoModelFile(optimizedOpenVinoModelData, model1Params);

            //Generate optimized OpenVino model2. inception_v4
            String tfModel2Path = modelInferencePath + File.separator + "SavedModel/model2";
            String optimizedOpenVinoMode2Dir = imageModelPackageDir;
            model2Params = new ModelParams();
            model2Params.setModelType("inception");
            model2Params.setModelName("model2");
            model2Params.setInputName("input_1");
            model2Params.setInputShape(new int[]{1, 299, 299, 3});
            model2Params.setMeanValues(new float[]{127.5f,127.5f,127.5f});
            model2Params.setScale(127.5f);
            model2Params.setOptimizedModelDir(optimizedOpenVinoMode2Dir);
            //Call Zoo API generate optimized OpenVino Model
            optimizedOpenVinoModelData =
                    ImageModelLoader3Models.getInstance().generateOpenVinoModelData(tfModel2Path, model2Params);
            ImageModelLoader3Models.getInstance().
                    saveToOpenVinoModelFile(optimizedOpenVinoModelData, model2Params);

            //Generate optimized OpenVino model3. inception_v3
            String tfModel3Path = this.modelInferencePath + File.separator + "SavedModel/model3";
            String optimizedOpenVinoMode3Dir = imageModelPackageDir;
            model3Params = new ModelParams();
            model3Params.setModelType("inception");
            model3Params.setModelName("model3");
            model3Params.setInputName("input_1");
            model3Params.setInputShape(new int[]{1, 299, 299, 3});
            model3Params.setMeanValues(new float[]{127.5f,127.5f,127.5f});
            model3Params.setScale(127.5f);
            model3Params.setOptimizedModelDir(optimizedOpenVinoMode3Dir);
            //Call Zoo API generate optimized OpenVino Model
            optimizedOpenVinoModelData =
                    ImageModelLoader3Models.getInstance().generateOpenVinoModelData(tfModel3Path, model3Params);
            ImageModelLoader3Models.getInstance().
                    saveToOpenVinoModelFile(optimizedOpenVinoModelData, model3Params);

        }catch(Exception ex)
        {
            ex.printStackTrace();
            throw new RuntimeException("WARN: OpenVinoModelGenerator.execute() FAILED.");
        }
    }

    public ModelParams getModel1Params()
    {
        return model1Params;
    }

    public ModelParams getModel2Params()
    {
        return model2Params;
    }

    public ModelParams getModel3Params()
    {
        return model3Params;
    }
}
