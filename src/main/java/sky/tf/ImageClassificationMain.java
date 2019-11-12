package sky.tf;
import com.alibaba.tianchi.garbage_image_util.ImageClassSink;
import com.alibaba.tianchi.garbage_image_util.ImageDirSource;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import sky.tf.threemodels.ImageFlatMap3Models;
import sky.tf.threemodels.OpenVinoModelGenerator3Models;
/**
 * The main class of Image inference for garbage classification
 * @author SkyPeace
 */
public class ImageClassificationMain {
	public static void main(String[] args) throws Exception {
		StreamExecutionEnvironment flinkEnv = StreamExecutionEnvironment.getExecutionEnvironment();

		//Step 1. Generate optimized OpenVino models (Three type of models) for later prediction.
		OpenVinoModelGenerator3Models modelGenerator = new OpenVinoModelGenerator3Models();
		modelGenerator.execute();
		ImageFlatMap3Models imageFlatMap = new ImageFlatMap3Models(modelGenerator.getModel1Params(),
				modelGenerator.getModel2Params(), modelGenerator.getModel3Params());

		ImageDirSource source = new ImageDirSource();
		//IMPORTANT: Operator chaining maybe hit the score log issue (Tianchi ENV) when parallelism is set to N_N_x.
		//Use statistic tag PREDICT_PROCESS to get prediction's real elapsed time
		//flinkEnv.disableOperatorChaining();
		flinkEnv.addSource(source).setParallelism(1)
				.flatMap(imageFlatMap).setParallelism(2)
				.addSink(new ImageClassSink()).setParallelism(2);
		flinkEnv.execute("Image inference for garbage classification-PhaseII-1.0 - SkyPeace");
	}
}
