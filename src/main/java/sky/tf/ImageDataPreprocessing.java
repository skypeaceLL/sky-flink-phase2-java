package sky.tf;

import com.alibaba.tianchi.garbage_image_util.ImageData;
import com.intel.analytics.bigdl.transform.vision.image.opencv.OpenCVMat;
import com.intel.analytics.zoo.pipeline.inference.JTensor;
import org.libjpegturbo.turbojpeg.TJ;
import org.libjpegturbo.turbojpeg.TJDecompressor;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

/**
 * @author SkyPeace
 * The class for image data preprocessing.
 */
public class ImageDataPreprocessing {
    public static final int IMAGE_DECODER_OPENCV = 0;
    public static final int IMAGE_DECODER_TURBOJPEG = 1;

    public static final int PREPROCESSING_VGG = 11;
    public static final int PREPROCESSING_INCEPTION = 12;
    public static final int PREPROCESSING_TIANCHI = 13;
    private ModelParams modelParams;

    public ImageDataPreprocessing(ModelParams modelParams)
    {
        this.modelParams = modelParams;
    }

    /**
     * Get RGB Mat data
     * @param imageData
     * @param decodeType
     * @return
     * @throws Exception
     */
    public static Mat getRGBMat(ImageData imageData, int decodeType) throws Exception
    {
        //********************** Decode image **********************//
        long beginTime = System.currentTimeMillis();
        System.out.println(String.format("%s IMAGE_BYTES = %s", "###", imageData.getImage().length));
        System.out.println(String.format("%s IMAGE_BYTES_IMDECODE BEGIN %s", "###", beginTime));
        Mat matRGB = null;
        if(decodeType == IMAGE_DECODER_OPENCV)
            matRGB = decodeByOpenCV(imageData);
        else if(decodeType == IMAGE_DECODER_TURBOJPEG)
            matRGB = decodeByTurboJpeg(imageData);
        else
            throw new Exception(String.format("Not support such decodeType: %s", decodeType));
        System.out.println(String.format("%s IMAGE_BYTES_IMDECODE END %s (Cost: %s)",
                "###", System.currentTimeMillis(), (System.currentTimeMillis() - beginTime)));
        return matRGB;
    }

    /**
     * Decode image data by turbojpeg. It is more fast than OpenCV decoder.
     * Please refer to https://libjpeg-turbo.org/About/Performance for performance comparison.
     * @param imageData
     * @return
     * @throws Exception
     */
    private static Mat decodeByTurboJpeg(ImageData imageData) throws Exception
    {
        TJDecompressor tjd = new TJDecompressor(imageData.getImage());
        //TJ.FLAG_FASTDCT can get more performance. TJ.FLAG_ACCURATEDCT can get more accuracy.
        byte bytes[] = tjd.decompress(tjd.getWidth(), 0, tjd.getHeight(), TJ.PF_RGB, TJ.FLAG_ACCURATEDCT);
        Mat matRGB = new Mat(tjd.getHeight(), tjd.getWidth(), CvType.CV_8UC3);
        matRGB.put(0, 0, bytes);
        return matRGB;
    }

    /**
     * Decode image data by OpenCV
     * @param imageData
     * @return
     */
    private static Mat decodeByOpenCV(ImageData imageData)
    {
        Mat srcmat = new MatOfByte(imageData.getImage());
        Mat mat = Imgcodecs.imdecode(srcmat, Imgcodecs.CV_LOAD_IMAGE_COLOR);
        Mat matRGB = new MatOfByte();
        Imgproc.cvtColor(mat, matRGB, Imgproc.COLOR_BGR2RGB);
        System.out.println(String.format("%s image_width, image_height, image_depth, image_dims = %s, %s, %s, %s",
                "###", mat.width(), mat.height(), mat.depth(), mat.dims()));
        return matRGB;
    }

    /**
     * Preprocess image data and convert to float array (RGB) for JTensor list need.
     * @param matRGB
     * @param preprocessType
     * @param enableMultipleVoters
     * @return
     * @throws Exception
     */
    public List<List<JTensor>> doPreProcessing(Mat matRGB, int preprocessType, boolean enableMultipleVoters) throws Exception
    {
        //********************** Resize, crop and other pre-processing **********************//
        long beginTime = System.currentTimeMillis();
        System.out.println(String.format("%s IMAGE_RESIZE BEGIN %s", "###", beginTime));
        List<Mat> matList = new ArrayList<Mat>();
        int image_size = modelParams.getInputShape()[2];

        //VGG_PREPROCESS, INCEPTION_PREPROCESS and PREPROCESSING_TIANCHI
        if(preprocessType==PREPROCESSING_VGG) {
            matList = this.resizeAndCenterCropImage(matRGB, 256, image_size, image_size, enableMultipleVoters);
        }else if(preprocessType==PREPROCESSING_INCEPTION) {
            matList = this.cropAndResizeImage(matRGB, image_size, image_size, enableMultipleVoters);
        }else if(preprocessType==PREPROCESSING_TIANCHI) {
            matList = this.resize(matRGB, image_size);
        }
        else
            throw new Exception(String.format("Not support such preprocessType: %s", preprocessType));
        System.out.println(String.format("%s IMAGE_RESIZE END %s (Cost: %s)",
                "###", System.currentTimeMillis(), (System.currentTimeMillis() - beginTime)));

        //********** Convert Mat to float array. R channel, G channel, B channel **********//
        beginTime = System.currentTimeMillis();
        System.out.println(String.format("%s CONVERT_TO_RGB_FLOAT BEGIN %s", "###", beginTime));
        List<List<JTensor>> inputs = new ArrayList<List<JTensor>>();
        for(int i=0;i<matList.size();i++) {
            List list = new ArrayList<JTensor>();
            float data[] = new float[modelParams.getInputSize()];
            List<Mat> rgbMatList = new ArrayList<Mat>();
            //Split to R channel, G channel and B channel
            org.opencv.core.Core.split(matList.get(i), rgbMatList);
            MatOfByte matFloat = new MatOfByte();
            //VConcat R,G,B and convert to float array. Native operation, quick enough (AVG less than 1ms).
            org.opencv.core.Core.vconcat(rgbMatList, matFloat);
            OpenCVMat.toFloatPixels(matFloat, data);
            //System.arraycopy(data, 0, datat, i*data.length, data.length);

            /* Below code is the example of worst performance, DO NOT coding as that.
            for (int row = 0; row < 224; row++) {
                for (int col = 0; col < 224; col++) {
                    data[(col + row * 224) + 224 * 224 * 0] = (float) (dst.get(row, col)[0]);
                    data[(col + row * 224) + 224 * 224 * 1] = (float) (dst.get(row, col)[1]);
                    data[(col + row * 224) + 224 * 224 * 2] = (float) (dst.get(row, col)[2]);
                }
            }
            */

            //Create a JTensor
            JTensor tensor = new JTensor();
            tensor.setData(data);
            tensor.setShape(modelParams.getInputShape());
            list.add(tensor);
            inputs.add(list);
        }
        System.out.println(String.format("%s CONVERT_TO_RGB_FLOAT END %s (Cost: %s)",
                "###", System.currentTimeMillis(), (System.currentTimeMillis() - beginTime)));
        return inputs;
    }

    /**
     * Just resize to specified size. Use for TianChi model
     * @param src
     * @param image_size
     * @return
     */
    private List<Mat> resize(Mat src, int image_size)
    {
        Mat resizedMat = new MatOfByte();
        Imgproc.resize(src, resizedMat, new Size(image_size, image_size), 0, 0, Imgproc.INTER_CUBIC);
        List<Mat> matList = new ArrayList<Mat>();
        matList.add(resizedMat);
        return matList;
    }

    /**
     * Resize and center crop. For VGG_PREPROCESS
     * @param src
     * @param smallestSide
     * @param outWidth
     * @param outHeight
     * @param enableMultipleVoters
     */
    private List<Mat> resizeAndCenterCropImage(Mat src, int smallestSide, int outWidth, int outHeight, boolean enableMultipleVoters)
    {
        //OpenCV resize. Use INTER_LINEAR, INTER_CUBIC or INTER_LANCZOS4
        //Imgproc.resize(src, dst, new Size(outWidth, outHeight), 0, 0, Imgproc.INTER_CUBIC);
        float scale = 0f;
        if(src.height()>src.width())
            scale = (float)smallestSide / src.width();
        else
            scale = (float)smallestSide / src.height();
        float newHeight = src.height() * scale;
        float newWidth = src.width() * scale;
        Mat resizedMat = new MatOfByte();

        Imgproc.resize(src, resizedMat, new Size(newWidth, newHeight), 0, 0, Imgproc.INTER_LINEAR);
        int offsetHeight = (int)((resizedMat.height() - outHeight) / 2);
        int offsetWidth = (int)((resizedMat.width() - outWidth) / 2);

        //Center
        Mat matCenter = resizedMat.submat(offsetHeight, offsetHeight + outHeight, offsetWidth, offsetWidth + outWidth);
        List<Mat> matList = new ArrayList<Mat>();
        matList.add(matCenter);

        if(enableMultipleVoters) {
            //matFlip
            Mat matCenterFlip = new MatOfByte();
            Core.flip(matCenter, matCenterFlip, 0);
            //Mat matTop = resizedMat.submat(0, outHeight, offsetWidth, offsetWidth + outWidth);
            Mat matBottom = resizedMat.submat((int) resizedMat.height() - outHeight, (int) resizedMat.height(), offsetWidth, offsetWidth + outWidth);
            Mat matLeft = resizedMat.submat(offsetHeight, offsetHeight + outHeight, 0, outWidth);
            Mat matRight = resizedMat.submat(offsetHeight, offsetHeight + outHeight, (int) resizedMat.width() - outWidth, (int) resizedMat.width());
            matList.add(matCenterFlip);
            //matList.add(matTop);
            matList.add(matBottom);
            matList.add(matRight);
            matList.add(matLeft);
        }
        return matList;
    }

    /**
     * Crop and resize. For INCEPTION_PREPROCESS
     * @param src
     * @param outWidth
     * @param outHeight
     * @param enableMultipleVoters
     * @return
     */
    private List<Mat> cropAndResizeImage(Mat src, int outWidth, int outHeight, boolean enableMultipleVoters)
    {
        float scale = 0.875f;
        float newHeight = src.height() * scale;
        float newWidth = src.width() * scale;

        int offsetHeight = (int)((src.height() - newHeight) / 2);
        int offsetWidth = (int)((src.width() - newWidth) / 2);
        Mat centerCropMat = src.submat(offsetHeight, offsetHeight + (int)newHeight, offsetWidth, offsetWidth + (int)newWidth);

        Mat resizedMat = new MatOfByte();
        Imgproc.resize(centerCropMat, resizedMat, new Size(outWidth, outHeight), 0, 0, Imgproc.INTER_LINEAR);
        List<Mat> matList = new ArrayList<Mat>();
        matList.add(resizedMat);

        if(enableMultipleVoters) {
            Mat matCenterFlip = new MatOfByte();
            Core.flip(resizedMat, matCenterFlip, 0);
            matList.add(matCenterFlip);

            Mat matBottom = src.submat(src.height()-(int)newHeight, src.height(), offsetWidth, offsetWidth + (int)newWidth);
            Mat resizedMatBottom = new MatOfByte();
            Imgproc.resize(matBottom, resizedMatBottom, new Size(outWidth, outHeight), 0, 0, Imgproc.INTER_LINEAR);
            matList.add(resizedMatBottom);

            Mat matLeft = src.submat(offsetHeight, offsetHeight + (int) newHeight, 0, (int) newWidth);
            Mat resizedMatLeft = new MatOfByte();
            Imgproc.resize(matLeft, resizedMatLeft, new Size(outWidth, outHeight), 0, 0, Imgproc.INTER_LINEAR);
            matList.add(resizedMatLeft);

            Mat matRight = src.submat(offsetHeight, offsetHeight + (int)newHeight, src.width() - (int)newWidth, src.width());
            Mat resizedMatRight = new MatOfByte();
            Imgproc.resize(matRight, resizedMatRight, new Size(outWidth, outHeight), 0, 0, Imgproc.INTER_LINEAR);
            matList.add(resizedMatRight);
        }
        return matList;
    }

    /**
     * Another implements of INCEPTION_PREPROCESS. For experimental only.
     * @param src
     * @param outWidth
     * @param outHeight
     * @param dst
     */
    private void cropAndResizeImageReserved(Mat src, int outWidth, int outHeight, Mat dst)
    {
        Mat srcFloat = new MatOfFloat();
        src.convertTo(srcFloat, CvType.CV_32F, 1.0f/255.0f);
        float scale = 0.875f;
        float newHeight = srcFloat.height() * scale;
        float newWidth = srcFloat.width() * scale;

        int offsetHeight = (int)((srcFloat.height() - newHeight) / 2);
        int offsetWidth = (int)((srcFloat.width() - newWidth) / 2);
        Mat cropMat = srcFloat.submat(offsetHeight, offsetHeight + (int)newHeight, offsetWidth, offsetWidth + (int)newWidth);

        Mat resizedMat = new MatOfFloat();
        Imgproc.resize(cropMat, resizedMat, new Size(outWidth, outHeight), 0, 0, Imgproc.INTER_LINEAR);

        Mat resizedMat1 = new MatOfFloat();
        Core.subtract(resizedMat, Scalar.all(0.5), resizedMat1);
        Core.multiply(resizedMat1, Scalar.all(2.0), dst);
    }

}
