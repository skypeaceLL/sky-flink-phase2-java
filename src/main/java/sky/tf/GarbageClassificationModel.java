package sky.tf;

import com.intel.analytics.zoo.pipeline.inference.*;

import java.util.List;
import java.util.concurrent.LinkedBlockingQueue;

/**
 * @author SkyPeace
 * The models for predict. For this case, use Zoo OpenVINOModel directly instead of InferenceModel.
 */
public class GarbageClassificationModel implements IModel, java.io.Serializable
{
    // Support Zoo InferenceModel and Zoo OpenVINOModel directly
    // For this case, the final decision is use Zoo OpenVINOModel directly instead of Zoo InferenceModel.
    private OpenVINOModel openVINOModel;
    private LinkedBlockingQueue<Integer> referenceQueue = new LinkedBlockingQueue<Integer>(12);

    public GarbageClassificationModel(byte[] modelXml, byte[] modelBin)
    {
        this.openVINOModel = this.loadOpenVINOModel(modelXml, modelBin);
    }

    private OpenVINOModel loadOpenVINOModel(byte[] modelXml, byte[] modelBin){
        long beginTime = System.currentTimeMillis();
        System.out.println(String.format("LOAD_OPENVION_MODEL_FROM_BYTES BEGIN %s", beginTime));
        OpenVINOModel model = OpenVinoInferenceSupportive$.MODULE$.loadOpenVinoIR(modelXml, modelBin,
                com.intel.analytics.zoo.pipeline.inference.DeviceType.CPU(), 0);
        //OpenVinoInferenceSupportive.loadOpenVinoIRFromTempDir("modelName", "tempDir");
        long endTime = System.currentTimeMillis();
        System.out.println(String.format("LOAD_OPENVION_MODEL_FROM_BYTES END %s (Cost: %s)",
                endTime, (endTime - beginTime)));
        return model;
    }

    /**
     * Predict by inputs
     * @param inputs
     * @return
     * @throws Exception
     */
    public List<List<JTensor>> predict(List<List<JTensor>> inputs) throws Exception {
        if(openVINOModel!=null)
            return openVINOModel.predict(inputs);
        else if(inferenceModel!=null)
            return inferenceModel.doPredict(inputs);
        else
            throw new RuntimeException("inferenceModel and openVINOModel are both null.");
    }

    public void addRefernce() throws InterruptedException
    {
        referenceQueue.put(1);
    }

    /**
     * Release model
     * @throws Exception
     */
    public synchronized void release() throws Exception
    {
        if(referenceQueue.peek()==null)
            return;
        referenceQueue.poll();
        if(referenceQueue.peek()==null)
        {
            //Thread.sleep(100);
            if (inferenceModel != null) {
                System.out.println("Release inferenceModel ...");
                inferenceModel.doRelease();
                System.out.println("inferenceModel released");
            }
            else if (openVINOModel != null) {
                System.out.println("Release openVINOModel ...");
                openVINOModel.release();
                System.out.println("openVINOModel released");
            }
            else
                throw new RuntimeException("inferenceModel and openVINOModel are both null.");
        }
    }


    //Below code is reserved for test. Do not delete them.
    private InferenceModel inferenceModel;
    /**
     * Reserved for experimental test
     * @param savedModelBytes
     */
    public GarbageClassificationModel(byte[] savedModelBytes, ModelParams modelParams) {
        this.inferenceModel = this.loadInferenceModel(savedModelBytes, modelParams);
    }

    /**
     * Reserved for experimental test
     * @param savedModelBytes
     * @return
     */
    private InferenceModel loadInferenceModel(byte[] savedModelBytes, ModelParams modelParams)
    {
        long beginTime = System.currentTimeMillis();
        System.out.println(String.format("loadInferenceModel BEGIN %s", beginTime));
        InferenceModel model = new InferenceModel(1);
        model.doLoadTF(savedModelBytes, modelParams.getInputShape(), false,
                modelParams.getMeanValues(), modelParams.getScale(), modelParams.getInputName());
        long endTime = System.currentTimeMillis();
        System.out.println(String.format("loadInferenceModel END %s (Cost: %s)",
                endTime, (endTime - beginTime)));
        return model;
    }
}
