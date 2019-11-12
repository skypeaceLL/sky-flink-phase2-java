package sky.tf;

public class PredictionResult {

    private Integer predictionId;
    private float probability;
    private Integer count;

    public Integer getPredictionId() {
        return predictionId;
    }

    public void setPredictionId(Integer predictionId) {
        this.predictionId = predictionId;
    }

    public float getProbability() {
        return probability;
    }

    public void setProbability(float probability) {
        this.probability = probability;
    }

    public Integer getCount() {
        return count;
    }

    public void setCount(Integer count) {
        this.count = count;
    }
}
