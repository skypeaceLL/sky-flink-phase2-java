package sky.tf.multiplemodels;

import com.alibaba.tianchi.garbage_image_util.ImageData;
import com.intel.analytics.bigdl.opencv.OpenCV;
import com.intel.analytics.zoo.pipeline.inference.JTensor;
import org.opencv.core.Mat;
import sky.tf.*;

import java.util.*;

/**
 * @author SkyPeace
 * The supportive for image classification prediction.
 */
public class ImagePredictSupportiveMultipleModels
{
    //private Logger logger = LoggerFactory.getLogger(ImageModelLoader3Models.class);
    private List<ModelParams> modelParamsList;
    private List<GarbageClassificationModel> modelList;

    public ImagePredictSupportiveMultipleModels(List<GarbageClassificationModel> modelList, List<ModelParams> modelParamsList)
    {
        this.modelList = modelList;
        this.modelParamsList = modelParamsList;
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
     * Using multiple models do predict.
     * @param imageData
     * @return
     * @throws Exception
     */
    private Integer predict(ImageData imageData) throws Exception {
        //Decode jpeg
        //Use turbojpeg as jpeg decoder. It is more fast than OpenCV decoder (OpenCV use libjpeg as decoder).
        //Refer to https://libjpeg-turbo.org/About/Performance for performance comparison.
        Mat matRGB = ImageDataPreprocessing.getRGBMat(imageData, ImageDataPreprocessing.IMAGE_DECODER_TURBOJPEG);

        long t1 = System.currentTimeMillis();
        System.out.println(String.format("%s DO_PREDICT BEGIN %s", "###", t1));
        List<PredictionResult> modelsPredictionResults = new ArrayList<PredictionResult>();
        for(int i=0;i<modelParamsList.size();i++)
        {
            PredictionResult pResult = this.doPredict(
                    modelList.get(i), modelParamsList.get(i), matRGB, ImageDataPreprocessing.PREPROCESSING_INCEPTION, false);
            modelsPredictionResults.add(pResult);
        }
        PredictionResult pResult = this.getSingleResult(modelsPredictionResults);
        Integer predictionId = pResult.getPredictionId();
        long t2 = System.currentTimeMillis();
        System.out.println(String.format("%s DO_PREDICT END %s (Cost: %s)", "###", t2, (t2 - t1)));
        return predictionId;
    }

    /**
     * Specified a model to do predict.
     * @param model
     * @param modelParams
     * @param matRGB
     * @param preprocessingType
     * @param enableMultipleVoters
     * @return
     * @throws Exception
     */
    private PredictionResult doPredict(GarbageClassificationModel model,
                                        ModelParams modelParams, Mat matRGB, int preprocessingType, boolean enableMultipleVoters) throws Exception
    {
        ImageDataPreprocessing dataPreprocessing = new ImageDataPreprocessing(modelParams);
        List<List<JTensor>> inputs = dataPreprocessing.doPreProcessing(matRGB, preprocessingType, enableMultipleVoters);
        List<List<JTensor>> tensorResults = model.predict(inputs);
        List<PredictionResult> pResults = this.convertToPredictionResults(tensorResults);
        PredictionResult pResult = this.getSingleResult(pResults);
        return pResult;
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
                existResult.setProbability(Math.max(pResult.getProbability(), existResult.getProbability()));
                distinctResults.put(predictionId, existResult);
            }
        }
        PredictionResult maxCountResult = null;
        for(Integer key:distinctResults.keySet())
        {
            PredictionResult pResult = distinctResults.get(key);
            float ratio = (float)pResult.getCount() / pResults.size();
            if(ratio>0.5f)
            {
                System.out.println("Majority get agreement.");
                maxCountResult = pResult;
                return maxCountResult;
            }else if(pResult.getCount()>1)
                maxCountResult = pResult;
        }
        if(maxCountResult!=null) {
            System.out.println("Few models get agreement.");
            return maxCountResult;
        }

        System.out.println("The models can NOT get agreement.");
        float maxProbability = pResults.get(0).getProbability();
        int maxIdx = 0;
        for (int i = 1; i < pResults.size(); i++) {
            if (pResults.get(i).getProbability() > maxProbability) {
                maxProbability = pResults.get(i).getProbability();
                maxIdx = i;
            }
        }
        return pResults.get(maxIdx);
    }

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
     * @throws Exception
     */
    public void firstTimeDummyCheck() throws Exception
    {
        for(int i=0; i<modelParamsList.size(); i++)
            this.checkModel(modelList.get(i), modelParamsList.get(i));
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
