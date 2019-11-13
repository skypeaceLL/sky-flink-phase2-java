package sky.tf.multiplemodels;

import com.google.common.io.Files;
import com.intel.analytics.zoo.pipeline.inference.OpenVinoInferenceSupportive$;
import org.apache.commons.io.FileUtils;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.*;
import org.apache.hadoop.hdfs.DistributedFileSystem;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import sky.tf.GarbageClassificationModel;
import sky.tf.ModelParams;

import java.io.*;
import java.net.URI;
import java.nio.channels.Channels;
import java.nio.channels.FileChannel;
import java.nio.channels.ReadableByteChannel;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;


/**
 * @author SkyPeace
 * The model loader. Experimental only.
 */
public class ImageModelLoaderMultipleModels {
    private Logger logger = LoggerFactory.getLogger(ImageModelLoaderMultipleModels.class);
    private static ImageModelLoaderMultipleModels instance = new ImageModelLoaderMultipleModels();
    private volatile Map<String, GarbageClassificationModel> modelsMap =
            new HashMap<String, GarbageClassificationModel>();

    private ImageModelLoaderMultipleModels() {}

    public static ImageModelLoaderMultipleModels getInstance()
    {
        return instance;
    }

    /**
     * Generate optimized OpenVino model's data (bytes). -- First step.
     * @param savedModelPath
     * @return
     */
    public List<Tuple2<String, byte[]>> generateOpenVinoModelData(String savedModelPath, ModelParams modelParams) throws Exception
    {
        List<Tuple2<String, byte[]>> modelData = new ArrayList<Tuple2<String, byte[]>>();
        File modelFile = new File(savedModelPath);
        String optimizeModelPath = null;
        if(modelFile.isDirectory())
            optimizeModelPath =
                optimizeModelFromModelDir(savedModelPath, modelParams);
        else
            optimizeModelPath =
                    optimizeModelFromModelPackage(savedModelPath, modelParams);
        byte[] xml = readDFSFile(optimizeModelPath + File.separator + "saved_model.xml");
        byte[] bin = readDFSFile(optimizeModelPath + File.separator + "saved_model.bin");
        logger.info("Size of optimized saved_model.xml: " + xml.length);
        logger.info("Size of optimized saved_model.bin: " + bin.length);
        Tuple2<String, byte[]> xmlData = new Tuple2("xml", xml);
        Tuple2<String, byte[]> binData = new Tuple2("bin", bin);
        modelData.add(xmlData);
        modelData.add(binData);
        return modelData;
    }

    /**
     * Optimize model from saved model dir. Return optimized model temp directory.
     * @param savedModelDir
     * @return
     * @throws Exception
     */
    private String optimizeModelFromModelDir(String savedModelDir, ModelParams modelParams) throws Exception
    {
        File tmpDir = Files.createTempDir();
        String optimizeModelTmpDir = tmpDir.getCanonicalPath();
        //OpenVinoInferenceSupportive.optimizeTFImageClassificationModel();
        OpenVinoInferenceSupportive$.MODULE$.optimizeTFImageClassificationModel(
                savedModelDir + File.separator + "SavedModel", modelParams.getInputShape(), false,
                modelParams.getMeanValues(), modelParams.getScale(), modelParams.getInputName(), optimizeModelTmpDir);
        return optimizeModelTmpDir;
    }

    /**
     * Optimize model from saved model package (tar.gz file). Return optimized model temp dir.
     * @param savedModelPackagePath
     * @return
     * @throws Exception
     */
    private String optimizeModelFromModelPackage(String savedModelPackagePath, ModelParams modelParams) throws Exception
    {
        byte[] savedModelBytes = readDFSFile(savedModelPackagePath);
        File tmpDir = Files.createTempDir();
        String tempDirPath = tmpDir.getCanonicalPath();
        String tarFileName = "saved-model.tar";
        File tarFile = new File(tempDirPath + File.separator + tarFileName);
        ByteArrayInputStream tarFileInputStream = new ByteArrayInputStream(savedModelBytes);
        ReadableByteChannel tarFileSrc = Channels.newChannel(tarFileInputStream);
        FileChannel tarFileDest = (new FileOutputStream(tarFile)).getChannel();
        tarFileDest.transferFrom(tarFileSrc, 0L, 9223372036854775807L);
        tarFileDest.close();
        tarFileSrc.close();
        String tarFileAbsolutePath = tarFile.getAbsolutePath();
        String modelRootDir = tempDirPath + File.separator + "saved-model";
        File modelRootDirFile = new File(modelRootDir);
        FileUtils.forceMkdir(modelRootDirFile);
        //tar -xvf -C
        Process proc = Runtime.getRuntime().exec(new String[]{"tar", "-xvf", tarFileAbsolutePath, "-C", modelRootDir});
        //Runtime.getRuntime().exec(new String[]{"ls", "-l", modelRootDir});
        BufferedReader insertReader = new BufferedReader(new InputStreamReader(proc.getInputStream()));
        BufferedReader errorReader = new BufferedReader(new InputStreamReader(proc.getErrorStream()));
        proc.waitFor();
        if(insertReader!=null)
            insertReader.close();
        if(errorReader!=null)
            errorReader.close();
        File[] files = modelRootDirFile.listFiles();
        String savedModelTmpDir = files[0].getAbsolutePath();
        logger.info("Saved model temp dir will be used for optimization: " + savedModelTmpDir);
        String optimizeModelTmpDir = optimizeModelFromModelDir(savedModelTmpDir, modelParams);
        return optimizeModelTmpDir;
    }

    /**
     * Save optimized OpenVino model's data (bytes) into HDFS files
     * The specified dir is also the parent dir of saved model package.
     * @param openVinoModelData
     * @param modelParams
     * @return
     */
    public void saveToOpenVinoModelFile(List<Tuple2<String, byte[]>> openVinoModelData, ModelParams modelParams) throws Exception
    {
        writeOptimizedModelToDFS(openVinoModelData, modelParams);
    }

    /**
     * Load OpenVino model with singleton pattern. -- Second step.
     * In this case, use this solution by default because it only cost about 2 seconds to load in Map.open().
     * @param modelParams
     * @return
     */
    public synchronized GarbageClassificationModel loadOpenVINOModelOnce(ModelParams modelParams) throws Exception
    {
        GarbageClassificationModel model = modelsMap.get(modelParams.getModelName());
        if(model == null)
        {
            model = this.getModel(modelParams);
            modelsMap.put(modelParams.getModelName(), model);
        }
        model.addRefernce();
        return model;
    }

    private GarbageClassificationModel getModel(ModelParams modelParams)  throws Exception
    {
        List<Tuple2<String, byte[]>> openVinoModelData =
                getModelDataFromOptimizedModelDir(modelParams);
        byte[] modelXml = openVinoModelData.get(0).f1;
        byte[] modelBin = openVinoModelData.get(1).f1;
        return new GarbageClassificationModel(modelXml, modelBin);
    }

    /**
     * Get OpenVino model data from optimized model files
     * @param modelParams
     * @return
     */
    private List<Tuple2<String, byte[]>> getModelDataFromOptimizedModelDir(ModelParams modelParams) throws Exception
    {
        String optimizedModelDir = modelParams.getOptimizedModelDir();
        String modelName = modelParams.getModelName();
        List<Tuple2<String, byte[]>> modelData = new ArrayList<Tuple2<String, byte[]>>();
        byte[] xml = readDFSFile(optimizedModelDir + File.separator + "optimized_openvino_" + modelName + "_KEEPME.xml");
        byte[] bin = readDFSFile(optimizedModelDir + File.separator + "optimized_openvino_" + modelName + "_KEEPME.bin");
        logger.info("Size of optimized_openvino_" + modelName + "_KEEPME.xml: " + xml.length);
        logger.info("Size of optimized_openvino_" + modelName + "_KEEPME.bin: " + bin.length);
        Tuple2<String, byte[]> xmlData = new Tuple2("xml", xml);
        Tuple2<String, byte[]> binData = new Tuple2("bin", bin);
        modelData.add(xmlData);
        modelData.add(binData);
        return modelData;
    }

    /**
     * Read saved model files (HDFS) into byte array.
     * @param filePath
     * @return
     */
    private byte[] readDFSFile(String filePath) throws Exception
    {
        try {
            long beginTime = System.currentTimeMillis();
            System.out.println(String.format("READ_MODEL_FILE from %s BEGIN %s", filePath, beginTime));
            Path imageRoot = new Path(filePath);
            org.apache.hadoop.conf.Configuration hadoopConfig = new org.apache.hadoop.conf.Configuration();
            hadoopConfig.set("fs.hdfs.impl", DistributedFileSystem.class.getName());
            hadoopConfig.set("fs.file.impl", LocalFileSystem.class.getName());
            FileSystem fileSystem = FileSystem.get(new URI(filePath), hadoopConfig);
            FileStatus fileStatus = fileSystem.getFileStatus(imageRoot);
            //RemoteIterator<LocatedFileStatus> it = fileSystem.listFiles(imageRoot, false);
            long fileLength = fileStatus.getLen();
            FSDataInputStream in = fileSystem.open(imageRoot);
            byte[] buffer = new byte[(int) fileLength];
            in.readFully(buffer);
            in.close();
            long endTime = System.currentTimeMillis();
            System.out.println(String.format("READ_MODEL_FILE from %s END %s (Cost: %s)",
                    filePath, endTime, (endTime - beginTime)));
            fileSystem.close();
            return buffer;
        }catch(Exception ex)
        {
            String msg = "Read DFS file FAILED. " + ex.getMessage();
            logger.error(msg, ex);
            throw ex;
        }
    }

    /**
     * Write the optimized model's data to files (HDFS)
     * @param openVinoModelData
     * @param modelParams
     */
    private void writeOptimizedModelToDFS(List<Tuple2<String, byte[]>> openVinoModelData, ModelParams modelParams) throws Exception
    {
        String optimizedModelDir = modelParams.getOptimizedModelDir();
        String modelName = modelParams.getModelName();
        try {
            long beginTime = System.currentTimeMillis();
            System.out.println(String.format("WRITE_OPTIMIZED_MODEL_FILE %s BEGIN %s", optimizedModelDir, beginTime));
            org.apache.hadoop.conf.Configuration hadoopConfig = new org.apache.hadoop.conf.Configuration();
            hadoopConfig.set("fs.hdfs.impl", DistributedFileSystem.class.getName());
            hadoopConfig.set("fs.file.impl", LocalFileSystem.class.getName());
            FileSystem fileSystem = FileSystem.get(new URI(optimizedModelDir), hadoopConfig);

            String xmlFileName = optimizedModelDir + File.separator + "optimized_openvino_"+ modelName + "_KEEPME.xml";
            System.out.println(String.format("Model xmlFileName: %s", xmlFileName));
            Path xmlFilePath = new Path(xmlFileName);
            FSDataOutputStream xmlFileOut = fileSystem.create(xmlFilePath, true);
            xmlFileOut.write(openVinoModelData.get(0).f1);
            xmlFileOut.flush();
            xmlFileOut.close();

            String binFileName = optimizedModelDir + File.separator + "optimized_openvino_"+ modelName + "_KEEPME.bin";
            System.out.println(String.format("Model binFileName: %s", xmlFileName));
            Path binFilePath = new Path(binFileName);
            FSDataOutputStream binFileOut = fileSystem.create(binFilePath, true);
            binFileOut.write(openVinoModelData.get(1).f1);
            binFileOut.flush();
            binFileOut.close();

            long endTime = System.currentTimeMillis();
            System.out.println(String.format("WRITE_OPTIMIZED_MODEL_FILE %s END %s (Cost: %s)",
                    optimizedModelDir, endTime, (endTime - beginTime)));
            fileSystem.close();
        }catch(Exception ex)
        {
            String msg = "Write DFS file FAILED. " + ex.getMessage();
            logger.error(msg, ex);
            throw ex;
        }
    }

}
