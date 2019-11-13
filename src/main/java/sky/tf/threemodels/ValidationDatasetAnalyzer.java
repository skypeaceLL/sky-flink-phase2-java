package sky.tf.threemodels;

import sky.tf.PredictionResult;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

/**
 * Experimental. Analysis validation data, search preferable probability thresholds for chose prediction result.
 */
public class ValidationDatasetAnalyzer {
    private float preferableThreshold1 = 0.94f;
    private float preferableThreshold2 = 0.81f;

    public void searchOptimizedThreshold(String validationPredictionDir)
    {
        String model1ValidationFilePath = validationPredictionDir + File.separator + "model1_validation_record.txt";
        String model2ValidationFilePath = validationPredictionDir + File.separator + "model2_validation_record.txt";
        String model3ValidationFilePath = validationPredictionDir + File.separator + "model3_validation_record.txt";
        List<Integer> labelList = getValidationLabels(model1ValidationFilePath);
        List<PredictionResult> model1Results = getPredictionResults(model1ValidationFilePath);
        List<PredictionResult> model2Results = getPredictionResults(model2ValidationFilePath);
        List<PredictionResult> model3Results = getPredictionResults(model3ValidationFilePath);
        if(labelList.size()==0) {
            System.out.println("WARN: Validation labels count is 0.");
            return;
        }
        int model1Success=0;
        int model2Success=0;
        int model3Success=0;
        for (int i = 0; i < labelList.size(); i++) {
            if (labelList.get(i).equals(model1Results.get(i).getPredictionId())) {
                model1Success++;
            }
            if (labelList.get(i).equals(model2Results.get(i).getPredictionId())) {
                model2Success++;
            }
            if (labelList.get(i).equals(model3Results.get(i).getPredictionId())) {
                model3Success++;
            }
        }
        float model1SucessRatio = (float) model1Success / (float)labelList.size();
        float model2SucessRatio = (float) model2Success / (float)labelList.size();
        float model3SucessRatio = (float) model3Success / (float)labelList.size();
        System.out.println("The model1 success ratio: " + model1SucessRatio);
        System.out.println("The model2 success ratio: " + model2SucessRatio);
        System.out.println("The model3 success ratio: " + model3SucessRatio);

        float totalRatio = model1SucessRatio*model2SucessRatio*model3SucessRatio +
                model1SucessRatio*model2SucessRatio*(1f-model3SucessRatio) +
                model1SucessRatio*(1f-model2SucessRatio)*model3SucessRatio +
                (1f-model1SucessRatio)*model2SucessRatio*model3SucessRatio;
        System.out.println("The total success ratio: " + totalRatio);
        long t1 = System.currentTimeMillis();
        float tmaxThreshold1 = 0f;
        float tmaxThreshold2 = 0f;
        float maxSuccessRatio = 0f;
        for (int j = 0; j < 20; j++) {
            float threshold1 = 0.01f * j + 0.80f;
            for (int k = 0; k < 80; k++) {
                float threshold2 = 0.01f * k + 0.2f;
                int success = 0;
                List<Integer> finalResults = new ArrayList<Integer>();
                for (int i = 0; i < labelList.size(); i++) {
                    PredictionResult model1Result = model1Results.get(i);
                    PredictionResult model2Result = model2Results.get(i);
                    PredictionResult model3Result = model3Results.get(i);

                    if (model1Result.getProbability() >= threshold1) {
                        finalResults.add(model1Result.getPredictionId());
                        continue;
                    }

                    if (model2Result.getProbability() >= threshold1)
                    {
                        finalResults.add(model2Result.getPredictionId());
                        continue;
                    }

                    /*
                    if (model3Result.getProbability() >= threshold1)
                    {
                        finalResults.add(model3Result.getPredictionId());
                        continue;
                    }
                    */

                    if (model1Result.getPredictionId().equals(model2Result.getPredictionId())
                            && model1Result.getPredictionId().equals(model3Result.getPredictionId())) {
                        finalResults.add(model1Result.getPredictionId());
                        continue;
                    }
                    if (model2Result.getPredictionId().equals(model3Result.getPredictionId())) {
                        finalResults.add(model2Result.getPredictionId());
                        continue;
                    }
                    if (model1Result.getPredictionId().equals(model2Result.getPredictionId())) {
                        finalResults.add(model1Result.getPredictionId());
                        continue;
                    }
                    if (model1Result.getPredictionId().equals(model3Result.getPredictionId())) {
                        finalResults.add(model1Result.getPredictionId());
                        continue;
                    }

                    if (model2Result.getProbability() >= threshold2)
                    {
                        finalResults.add(model2Result.getPredictionId());
                        continue;
                    }
                    if (model3Result.getProbability() >= threshold2) {
                        finalResults.add(model3Result.getPredictionId());
                        continue;
                    }

                    finalResults.add(model1Result.getPredictionId());

                }
                for (int i = 0; i < labelList.size(); i++) {
                    if (labelList.get(i).equals(finalResults.get(i))) {
                        success++;
                    }
                }
                float successRatio = (float) success / (float)labelList.size();
                if(successRatio > maxSuccessRatio) {
                    maxSuccessRatio = successRatio;
                    tmaxThreshold2 = threshold2;
                    tmaxThreshold1 = threshold1;
                }
            }
        }
        long t2 = System.currentTimeMillis();
        System.out.println("Val elapsed time: " + (t2-t1));
        System.out.println("Max threshold1: " + tmaxThreshold1);
        System.out.println("Max threshold2: " + tmaxThreshold2);
        System.out.println("Max successRation: " + maxSuccessRatio);
        this.preferableThreshold1 = tmaxThreshold1;
        this.preferableThreshold2 = tmaxThreshold2;
    }

    /**
     * Get validation data prediction results.
     */
    private List<PredictionResult> getPredictionResults(String filePath)
    {
        List<PredictionResult> pResults = new ArrayList<PredictionResult>();
        try {
            FileReader fr = new FileReader(filePath);
            BufferedReader br = new BufferedReader(fr);
            String line;
            int count1 = 0;
            while ((line = br.readLine()) != null) {
                String columns[] = line.split(",");
                //Format like as: 69 XXX => 69:YYY(P=0.74892)
                String aString = columns[0];
                String aStrings[] = aString.split("=>");
                String src = aStrings[0].replace(" ","");
                String tgt = aStrings[1].trim();
                //tgt = tgt.replace(":", "");
                String pString = tgt.substring(tgt.indexOf("P=") +2 );
                tgt = tgt.substring(0, tgt.indexOf(":"));
                pString = pString.replace(")","");
                float P = Float.parseFloat(pString);
                PredictionResult result = new PredictionResult();
                result.setPredictionId(Integer.valueOf(tgt));
                result.setProbability(P);
                pResults.add(result);
                count1++;
            }
            //System.out.println("Count1: " + count1);
            fr.close();
        }catch (Exception ex)
        {
            String errMsg = "Read model error index file FAILED. " + ex.getMessage();
            System.out.println("WRAN: " + errMsg);
            //throw new RuntimeException(errMsg);
        }
        return pResults;
    }

    /**
     * Get validation dataset labels.
     * @param filePath
     * @return
     */
    private List<Integer> getValidationLabels(String filePath)
    {
        List<Integer> pLabels = new ArrayList<Integer>();
        try {
            FileReader fr = new FileReader(filePath);
            BufferedReader br = new BufferedReader(fr);
            String line;
            int count1 = 0;
            while ((line = br.readLine()) != null) {
                String columns[] = line.split(",");
                //Format like as: 69 XXX => 69:YYY(P=0.74892)
                String aString = columns[0];
                String sourceId = aString.substring(0, aString.indexOf(" "));
                pLabels.add(Integer.valueOf(sourceId));
            }
            System.out.println("Count1: " + count1);
            fr.close();
        }catch (Exception ex)
        {
            String errMsg = "Read model error index file FAILED. " + ex.getMessage();
            System.out.println("WRAN: " + errMsg);
            //throw new RuntimeException(errMsg);
        }
        return pLabels;
    }

    public float getPreferableThreshold1()
    {
        return this.preferableThreshold1;
    }
    public float getPreferableThreshold2()
    {
        return this.preferableThreshold2;
    }

    public static void main(String args[])
    {
        ValidationDatasetAnalyzer validationDatasetTrainer = new ValidationDatasetAnalyzer();
        validationDatasetTrainer.searchOptimizedThreshold("/Users/xyz/PycharmProjects/Inference");
        System.out.println(validationDatasetTrainer.getPreferableThreshold1());
        System.out.println(validationDatasetTrainer.getPreferableThreshold2());
    }

}
