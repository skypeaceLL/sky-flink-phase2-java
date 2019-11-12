package sky.tf;

import java.io.*;
import java.util.HashMap;
import java.util.Map;

/**
 * @author SkyPeace
 *  Image class index
 */
public class ImageClassIndex {

    private Map<String, String> mapClassIndex;
    private static ImageClassIndex instance = new ImageClassIndex();

    private ImageClassIndex()
    {
    }
    public static ImageClassIndex getInsatnce()
    {
        return instance;
    }

    public synchronized void loadClassIndexMap(String modelInferencePath)
    {
        String labelFilePath = modelInferencePath + File.separator + "labels.txt";
        if(this.mapClassIndex == null) {
            long beginTime = System.currentTimeMillis();
            System.out.println(String.format("READ_CLASS_INDEX BEGIN %s", beginTime));
            this.mapClassIndex = this.read(labelFilePath);
            long endTime = System.currentTimeMillis();
            System.out.println(String.format("READ_CLASS_INDEX END %s (Cost: %s)", endTime, (endTime - beginTime)));
        }
    }

    public String getImageHumanstring(String id)
    {
        return mapClassIndex.get(id);
    }

    /**
     * Get class index.
     * For test only.
     * @param filePath
     * @return
     */
    private Map<String, String> read(String filePath)
    {
        Map<String, String> map = new HashMap<String, String>();
        try {
            FileReader fr = new FileReader(filePath);
            BufferedReader br = new BufferedReader(fr);
            String line;
            while ((line = br.readLine()) != null) {
                String columns[] = line.split(":");
                map.put(columns[0], columns[1]);
            }
            fr.close();
        }catch (IOException ex)
        {
            String errMsg = "Read class index file FAILED. " + ex.getMessage();
            System.out.println("ERROR: " + errMsg);
            throw new RuntimeException(errMsg);
        }
        return map;
    }

}
