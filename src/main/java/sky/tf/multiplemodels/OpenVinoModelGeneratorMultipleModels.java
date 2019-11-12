package sky.tf.multiplemodels;

import com.alibaba.tianchi.garbage_image_util.ConfigConstant;
import org.apache.flink.api.java.tuple.Tuple2;
import sky.tf.ModelParams;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

/**
 * The class use for generate optimized OpenVino models (Multiple different models for prediction)
 * @author SkyPeace
 */
public class OpenVinoModelGeneratorMultipleModels {
    private String imageModelPath = System.getenv(ConfigConstant.IMAGE_MODEL_PATH);
    private String imageModelPackagePath = System.getenv(ConfigConstant.IMAGE_MODEL_PACKAGE_PATH);
    private String modelInferencePath = System.getenv("MODEL_INFERENCE_PATH");
    private List<ModelParams> modelParamsList;

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
            //Generate optimized OpenVino model
            this.modelParamsList = new ArrayList<ModelParams>();
            for(int i=0;i<5;i++) {
                String tfModelPath = modelInferencePath + File.separator + "SavedModel/model" + i;
                String optimizedOpenVinoModelDir = imageModelPackageDir;
                ModelParams modelParams = new ModelParams();
                modelParams.setModelType("inception");
                modelParams.setModelName("model" + i);
                modelParams.setInputName("input_1");
                modelParams.setInputShape(new int[]{1, 299, 299, 3});
                modelParams.setMeanValues(new float[]{127.5f, 127.5f, 127.5f});
                modelParams.setScale(127.5f);
                modelParams.setOptimizedModelDir(optimizedOpenVinoModelDir);
                //Call Zoo API generate optimized OpenVino Model
                List<Tuple2<String, byte[]>> optimizedOpenVinoModelData =
                        ImageModelLoaderMultipleModels.getInstance().generateOpenVinoModelData(tfModelPath, modelParams);
                ImageModelLoaderMultipleModels.getInstance().
                        saveToOpenVinoModelFile(optimizedOpenVinoModelData, modelParams);
                modelParamsList.add(modelParams);
            }
        }catch(Exception ex)
        {
            ex.printStackTrace();
            throw new RuntimeException("WARN: OpenVinoModelGenerator.execute() FAILED.");
        }
    }

    public List<ModelParams> getModelParamsList()
    {
        return this.modelParamsList;
    }
}
