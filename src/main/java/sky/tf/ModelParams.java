package sky.tf;

/**
 * @author SkyPeace
 * The class for configure model's parameters.
 */
public class ModelParams implements java.io.Serializable
{
    private String modelType; //resnet, inception
    private String modelName;
    private String inputName;
    private int[] inputShape;
    private int inputSize;
    private float[] meanValues;
    private float scale;

    private String optimizedModelDir;

    public String getModelName() {
        return modelName;
    }

    public void setModelName(String modelName) {
        this.modelName = modelName;
    }

    public String getOptimizedModelDir() {
        return optimizedModelDir;
    }

    public void setOptimizedModelDir(String optimizedModelDir) {
        this.optimizedModelDir = optimizedModelDir;
    }

    public String getModelType() {
        return modelType;
    }

    public void setModelType(String modelType) {
        this.modelType = modelType;
    }

    public String getInputName() {
        return inputName;
    }

    public void setInputName(String inputName) {
        this.inputName = inputName;
    }

    public int[] getInputShape() {
        return inputShape;
    }

    public void setInputShape(int[] inputShape) {
        this.inputShape = inputShape;
    }

    public int getInputSize() {
        int size = 1;
        for(int i=1;i<inputShape.length;i++)
            size = size * inputShape[i];
        return size;
    }

    public float[] getMeanValues() {
        return meanValues;
    }

    public void setMeanValues(float[] meanValues) {
        this.meanValues = meanValues;
    }

    public float getScale() {
        return scale;
    }

    public void setScale(float scale) {
        this.scale = scale;
    }
}
