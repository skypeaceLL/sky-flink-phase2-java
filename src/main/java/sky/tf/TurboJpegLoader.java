package sky.tf;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.net.URL;
import java.nio.channels.FileChannel;
import java.nio.channels.ReadableByteChannel;

import static java.io.File.createTempFile;
import static java.nio.channels.Channels.newChannel;

/**
 * @author SkyPeace
 * The class for load turbojpeg native library
 */

public class TurboJpegLoader {
  private static boolean isLoaded = false;
  private static File tmpFile = null;

  static {
    try {
      String jturboJpegFileName = "META-INF/lib/linux_64/libturbojpeg.so";
      if (System.getProperty("os.name").toLowerCase().contains("mac")) {
        jturboJpegFileName = "META-INF/lib/osx_64/libturbojpeg.dylib";
      } else if (System.getProperty("os.name").toLowerCase().contains("win")) {
        jturboJpegFileName = "META-INF/lib/windows_64/turbojpeg.dll";
      }
      tmpFile = extract(jturboJpegFileName);
      System.load(tmpFile.getAbsolutePath());
      tmpFile.delete(); // delete so temp file after loaded
      isLoaded = true;
    } catch (Exception e) {
      isLoaded = false;
      e.printStackTrace();
      throw new RuntimeException("Failed to load Turbjpeg");
    }
  }

  /**
   * Check if libturbojpeg is loaded
   * @return
   */
  public static boolean isTurbojpegLoaded() {
    return isLoaded;
  }

  // Extract so file from jar to a temp path
  private static File extract(String path) {
    try {
      URL url = TurboJpegLoader.class.getResource("/" + path);
      if (url == null) {
        throw new Error("Can't find dynamic lib file in jar, path = " + path);
      }

      InputStream in = TurboJpegLoader.class.getResourceAsStream("/" + path);
      File file = null;

      // Windows won't allow to change the dll name, so we keep the name
      // It's fine as windows is consider in a desktop env, so there won't multiple instance
      // produce the dynamic lib file
      if (System.getProperty("os.name").toLowerCase().contains("win")) {
        file = new File(System.getProperty("java.io.tmpdir") + File.separator + path);
      } else {
        String targetFileTempPath = path.substring(path.lastIndexOf(File.separator) + 1);
        file = createTempFile("dlNativeLoader", targetFileTempPath);
      }

      ReadableByteChannel src = newChannel(in);
      FileChannel dest = new FileOutputStream(file).getChannel();
      dest.transferFrom(src, 0, Long.MAX_VALUE);
      dest.close();
      src.close();
      return file;
    } catch (Throwable e) {
      throw new Error("Can't extract dynamic lib file to /tmp dir.\n" + e);
    }
  }
}
