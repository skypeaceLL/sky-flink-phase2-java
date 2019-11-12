package sky.tf;

import com.intel.analytics.zoo.pipeline.inference.JTensor;

import java.util.List;

public interface IModel {
    List<List<JTensor>> predict(List<List<JTensor>> inputs) throws Exception;
    void release() throws Exception;
}
