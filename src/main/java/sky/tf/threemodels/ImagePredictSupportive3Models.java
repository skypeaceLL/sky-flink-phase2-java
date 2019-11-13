package sky.tf.threemodels;

import com.alibaba.tianchi.garbage_image_util.ImageData;
import com.intel.analytics.bigdl.opencv.OpenCV;
import com.intel.analytics.zoo.pipeline.inference.JTensor;
import org.opencv.core.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import sky.tf.*;

import java.util.*;

/**
 * @author SkyPeace
 * The supportive for image classification prediction.
 */
public class ImagePredictSupportive3Models
{
    private Logger logger = LoggerFactory.getLogger(ImagePredictSupportive3Models.class);
    final long MAX_PROCESSING_TIME = 400; //ms. 400

    private GarbageClassificationModel model1;
    private ModelParams model1Params;
    private GarbageClassificationModel model2;
    private ModelParams model2Params;
    private GarbageClassificationModel model3;
    private ModelParams model3Params;

    private float preferableThreshold1 = 0.94f;
    private float preferableThreshold2 = 0.81f;

    public ImagePredictSupportive3Models(GarbageClassificationModel model1, ModelParams model1Params,
                                         GarbageClassificationModel model2, ModelParams model2Params,
                                         GarbageClassificationModel model3, ModelParams model3Params,
                                         float preferableThreshold1, float preferableThreshold2)
    {
        this.model1 = model1;
        this.model1Params = model1Params;
        this.model2 = model2;
        this.model2Params = model2Params;
        this.model3 = model3;
        this.model3Params = model3Params;
        this.preferableThreshold1 = preferableThreshold1;
        this.preferableThreshold2 = preferableThreshold2;
    }

    /**
     * Do predict and transfer the label ID to human string.
     * @param imageData
     * @return
     * @throws Exception
     */
    public String predictHumanString(ImageData imageData) throws Exception
    {
        Integer predictId = predict(imageData);
        String humanString = this.getImageHumanString(predictId);
        if (humanString == null) {
            humanString = "class_index_not_available";
            throw new Exception(humanString);
        }
        return humanString;
    }

    /**
     * Do predict using 3 different models
     * @param imageData
     * @return
     * @throws Exception
     */
    public Integer predict(ImageData imageData) throws Exception {

        long processBeginTime = System.currentTimeMillis();
        //Decode jpeg
        //Use turbojpeg as jpeg decoder. It is more fast than OpenCV decoder (OpenCV use libjpeg as decoder).
        //Refer to https://libjpeg-turbo.org/About/Performance for performance comparison.
        Mat matRGB = ImageDataPreprocessing.getRGBMat(imageData, ImageDataPreprocessing.IMAGE_DECODER_TURBOJPEG);

        //Model1 do prediction
        ImageDataPreprocessing dataPreprocessing = new ImageDataPreprocessing(model1Params);
        List<List<JTensor>> inputs = dataPreprocessing.doPreProcessing(matRGB, ImageDataPreprocessing.PREPROCESSING_VGG, false);
        long t1 = System.currentTimeMillis();
        System.out.println(String.format("%s MODEL1_DO_PREDICT BEGIN %s", "###", t1));
        List<List<JTensor>> tensorResults = this.model1.predict(inputs);
        if (tensorResults == null && tensorResults.get(0) == null || tensorResults.get(0).size() == 0) {
            throw new Exception(String.format("ERROR: %s Model1 predict result is null.", imageData.getId()));
        }
        long t2 = System.currentTimeMillis();
        System.out.println(String.format("%s MODEL1_DO_PREDICT END %s (Cost: %s)", "###", t2, (t2 - t1)));

        List<PredictionResult> primaryResults = this.convertToPredictionResults(tensorResults);
        PredictionResult primaryResult = this.getSingleResult(primaryResults);
        Integer primaryPredictionId = primaryResult.getPredictionId();

        if (primaryResult.getProbability() >= preferableThreshold1) {
            System.out.println("Model1's HP predictionId saved time.");
            return primaryPredictionId;
        }

        long timeUsed = System.currentTimeMillis() - processBeginTime;
        if(timeUsed>=MAX_PROCESSING_TIME)
        {
            System.out.println("There maybe no enough time to try multiple modles to do predict.");
            return primaryPredictionId;
        }

        System.out.println("There is enough time to try multiple modles to do predict.");

        //Use model2 do prediction
        dataPreprocessing = new ImageDataPreprocessing(model2Params);
        List<List<JTensor>> secondaryInputs = dataPreprocessing.doPreProcessing(matRGB, ImageDataPreprocessing.PREPROCESSING_INCEPTION, false);
        t1 = System.currentTimeMillis();
        System.out.println(String.format("%s MODEL2_DO_PREDICT BEGIN %s", "###", t1));
        List<List<JTensor>> secondaryTensorResults = this.model2.predict(secondaryInputs);
        if (secondaryTensorResults == null && secondaryTensorResults.get(0) == null || secondaryTensorResults.get(0).size() == 0) {
            throw new Exception(String.format("ERROR: %s Model2 predict result is null.", imageData.getId()));
        }
        t2 = System.currentTimeMillis();
        System.out.println(String.format("%s MODEL2_DO_PREDICT END %s (Cost: %s)", "###", t2, (t2 - t1)));

        List<PredictionResult> secondaryResults = this.convertToPredictionResults(secondaryTensorResults);
        PredictionResult secondaryResult = this.getSingleResult(secondaryResults);
        Integer secondaryPredictionId = secondaryResult.getPredictionId();

        if (secondaryResult.getProbability() >= preferableThreshold1)
        {
            if (!secondaryPredictionId.equals(primaryPredictionId)) {
                System.out.println("Force use model2's HP predictionId.");
            } else {
                System.out.println("Model2's HP predictionId saved time.");
            }
            return secondaryPredictionId;
        }

        //Use model3 do prediction
        dataPreprocessing = new ImageDataPreprocessing(model3Params);
        List<List<JTensor>> thirdInputs = dataPreprocessing.doPreProcessing(matRGB, ImageDataPreprocessing.PREPROCESSING_INCEPTION, false);
        t1 = System.currentTimeMillis();
        System.out.println(String.format("%s MODEL3_DO_PREDICT BEGIN %s", "###", t1));
        List<List<JTensor>> thirdTensorResults = this.model3.predict(thirdInputs);
        if (thirdTensorResults == null && thirdTensorResults.get(0) == null || thirdTensorResults.get(0).size() == 0) {
            throw new Exception(String.format("ERROR: %s Model3 predict result is null.", imageData.getId()));
        }
        t2 = System.currentTimeMillis();
        System.out.println(String.format("%s MODEL3_DO_PREDICT END %s (Cost: %s)", "###", t2, (t2 - t1)));
        List<PredictionResult> thirdResults = this.convertToPredictionResults(thirdTensorResults);
        PredictionResult thirdResult = this.getSingleResult(thirdResults);
        Integer thirdPredictionId = thirdResult.getPredictionId();

        if (thirdResult.getProbability() >= preferableThreshold1)
        {
            if (!thirdPredictionId.equals(primaryPredictionId)) {
                System.out.println("Force use model3's HP predictionId.");
            } else {
                System.out.println("Model3's HP predictionId saved time.");
            }
            return thirdPredictionId;
        }

        if(thirdPredictionId.equals(primaryPredictionId)&&thirdPredictionId.equals(secondaryPredictionId))
        {
            System.out.println("Model1, model2 and model3 get agreement.");
            return primaryPredictionId;
        }
        if (primaryPredictionId.equals(secondaryPredictionId)) {
            System.out.println("Model1 and model2 get agreement.");
            return primaryPredictionId;
        }
        if (primaryPredictionId.equals(thirdPredictionId)) {
            System.out.println("Model1 and model3 get agreement.");
            return primaryPredictionId;
        }
        if (thirdPredictionId.equals(secondaryPredictionId)) {
            System.out.println("Model3 and model2 get agreement.");
            return secondaryPredictionId;
        }
        System.out.println("Model1, model2 and model3 does NOT get any agreement.");

        if (secondaryResult.getProbability() >= preferableThreshold2)
        {
            System.out.println("Model2(HP) is the last choice.");
            return secondaryPredictionId;
        }
        if(thirdResult.getProbability() >= preferableThreshold2){
            System.out.println("Model3(LP) is the last choice.");
            return thirdPredictionId;
        }

        System.out.println("Model1(LP) is the last choice.");
        return primaryPredictionId;
    }

    /**
     * Get single result
     * @param pResults
     * @return
     */
    private PredictionResult getSingleResult(List<PredictionResult> pResults)
    {
        if(pResults.size()==1)
            return pResults.get(0);
        System.out.println(String.format("There are %s multiple results", pResults.size()));
        Map<Integer, PredictionResult> distinctResults = new HashMap<Integer, PredictionResult>();
        for(PredictionResult pResult:pResults)
        {
            Integer predictionId = pResult.getPredictionId();
            PredictionResult existResult = distinctResults.get(predictionId);
            if(existResult == null){
                distinctResults.put(predictionId, pResult);
            }else{
                existResult.setCount(existResult.getCount() + 1);
                existResult.setProbability( Math.max(pResult.getProbability(), existResult.getProbability()));
                distinctResults.put(predictionId, existResult);
            }
        }
        PredictionResult maxResult = null;
        for(Integer key:distinctResults.keySet())
        {
            PredictionResult pResult = distinctResults.get(key);
            float ratio = (float)pResult.getCount() / pResults.size();
            if(ratio>0.5f)
            {
                System.out.println("Majority win.");
                maxResult = pResult;
                break;
            }
        }
        if(maxResult==null)
            maxResult = pResults.get(0);
        return maxResult;
    }

    /**
     * Convert JTensor list to prediciton results.
     * @param result
     * @return
     * @throws Exception
     */
    private List<PredictionResult> convertToPredictionResults(List<List<JTensor>> result) throws Exception
    {
        long beginTime = System.currentTimeMillis();
        List<PredictionResult> pResults = new ArrayList<PredictionResult>();
        System.out.println("result.size(): " + result.size());
        System.out.println("result.get(0).size(): " + result.get(0).size());
        for(int j=0; j<result.size(); j++) {
            JTensor resultTensor = result.get(j).get(0);
            float[] predictData = resultTensor.getData();
            if (predictData.length > 0) {
                float maxProbability = predictData[0];
                int maxNo = 0;
                for (int i = 1; i < predictData.length; i++) {
                    if (predictData[i] > maxProbability) {
                        maxProbability = predictData[i];
                        maxNo = i;
                    }
                }
                PredictionResult pResult = new PredictionResult();
                pResult.setPredictionId(maxNo);
                pResult.setProbability(maxProbability);
                pResult.setCount(1);
                pResults.add(pResult);
            }else{
                throw new Exception("ERROR: predictData.length=0");
            }
        }
        long endTime = System.currentTimeMillis();
        if((endTime-beginTime)>5)
            System.out.println(String.format("%s CONVERT_TO_PREDICTION_RESULT END %s (Cost: %s)",
                    "###", endTime, (endTime - beginTime)));
        return pResults;
    }

    /**
     * Load turbojpeg library.
     * @throws Exception
     */
    private void loadTurboJpeg()
    {
        long beginTime = System.currentTimeMillis();
        if(!TurboJpegLoader.isTurbojpegLoaded()) {
            throw new RuntimeException("LOAD_TURBOJPEG library failed. Please check.");
        }
        long endTime = System.currentTimeMillis();
        if((endTime - beginTime) > 1)
            System.out.println(String.format("LOAD_TURBOJPEG END %s (Cost: %s)", endTime, (endTime - beginTime)));
    }

    /**
     * Load OpenCV library.
     * @throws Exception
     */
    private void loadOpenCV()
    {
        long beginTime = System.currentTimeMillis();
        if(!OpenCV.isOpenCVLoaded()) {
            throw new RuntimeException("LOAD_OPENCV library failed. Please check.");
        }
        long endTime = System.currentTimeMillis();
        if((endTime - beginTime) > 10)
            System.out.println(String.format("LOAD_OPENCV END %s (Cost: %s)", endTime, (endTime - beginTime)));

        //if(!OpenvinoNativeLoader.load())
        //  throw new RuntimeException("LOAD_Openvino library failed. Please check.");
        //if(!TFNetNative.isLoaded())
        //    throw new RuntimeException("LOAD_TFNetNative library failed. Please check.");
    }

    /**
     * Get image class index
     * @return
     */
    private String getImageHumanString(Integer id)
    {
        return ImageClassIndex.getInsatnce().getImageHumanstring(String.valueOf(id));
    }

    /**
     * For first time warm-dummy check only.
     * @param model
     * @throws Exception
     */
    public void firstTimeDummyCheck() throws Exception
    {
        this.checkModel(model1, model1Params);
        if(this.model2!=null)
            this.checkModel(model2, model2Params);
        if(this.model3!=null)
            this.checkModel(model3, model3Params);
        this.getImageHumanString(-1);
        this.loadOpenCV();
        this.loadTurboJpeg();
    }

    //For first time warm-dummy check only.
    private void checkModel(GarbageClassificationModel pModel, ModelParams pModelParams) throws Exception
    {
        if(pModel == null) {
            throw new Exception("ERROR: The model is null. Aborted the predict.");
        }
        long beginTime = System.currentTimeMillis();
        System.out.println(String.format("FIRST_PREDICT BEGIN %s", beginTime));

        JTensor tensor = new JTensor();
        tensor.setData(new float[pModelParams.getInputSize()]);
        tensor.setShape(pModelParams.getInputShape());
        List list = new ArrayList<JTensor>();
        list.add(tensor);
        List<List<JTensor>> inputs = new ArrayList<List<JTensor>>();
        inputs.add(list);
        pModel.predict(inputs);

        long endTime = System.currentTimeMillis();
        System.out.println(String.format("FIRST_PREDICT END %s (Cost: %s)", endTime, (endTime - beginTime)));
    }

}
