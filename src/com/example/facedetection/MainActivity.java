package com.example.facedetection;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.ImageFormat;
import android.graphics.Paint;
import android.hardware.Camera;
import android.hardware.Camera.Size;
import android.os.Bundle;
import android.os.Environment;
import android.util.Log;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.Toast;

import java.io.File;
import java.io.FilenameFilter;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import com.googlecode.javacpp.Loader;
import com.googlecode.javacv.cpp.opencv_contrib.FaceRecognizer;
import com.googlecode.javacv.cpp.opencv_objdetect;

import static com.googlecode.javacv.cpp.opencv_core.*;
import static com.googlecode.javacv.cpp.opencv_imgproc.*;
import static com.googlecode.javacv.cpp.opencv_objdetect.*;
import static com.googlecode.javacv.cpp.opencv_highgui.*;

// ----------------------------------------------------------------------

public class MainActivity extends Activity {
    private FrameLayout layout;
    private FaceView faceView;
    private Preview mPreview;
    private static final int CAMERA_PIC_REQUEST = 1337; 

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN);

        // Hide the window title.
        requestWindowFeature(Window.FEATURE_NO_TITLE);
    }
    
    @Override 
    protected void onStart() {
    	super.onResume();
    	setContentView(R.layout.activity_main);
    }
    
    public void takePhoto(View view) {
    	Intent cameraIntent = new Intent(android.provider.MediaStore.ACTION_IMAGE_CAPTURE);
    	startActivityForResult(cameraIntent, CAMERA_PIC_REQUEST); 
    	System.out.println("Photo Taken!");
    }
    
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        if (requestCode == CAMERA_PIC_REQUEST) {
        	System.out.println("CHECKED!");
        	Bitmap thumbnail = (Bitmap) data.getExtras().get("data");
        	ImageView image = (ImageView) findViewById(R.id.photoResultView);  
        	image.setImageBitmap(thumbnail); 
        }
    }
    
    public void startCamera(View view) {
    	
        // Create our Preview view and set it as the content of our activity.
        try {
            layout = new FrameLayout(this);
            faceView = new FaceView(this);
            mPreview = new Preview(this, faceView);
            layout.addView(mPreview);
            layout.addView(faceView);
            setContentView(layout);
        } catch (IOException e) {
            e.printStackTrace();
            new AlertDialog.Builder(this).setMessage(e.getMessage()).create().show();
        }
    }
}

// ----------------------------------------------------------------------

class FaceView extends View implements Camera.PreviewCallback {
    public static final int SUBSAMPLING_FACTOR = 8;
    
    private Map<String, Integer> names;

    private IplImage grayImage;
    private CvHaarClassifierCascade classifier;
    private CvMemStorage storage;
    private CvSeq faces;
    private Bitmap myBitmap;
    
    private Context context;

    public FaceView(MainActivity context) throws IOException {
        super(context);
        this.context = context;

        // Load the classifier file from Java resources.
        File classifierFile = Loader.extractResource(getClass(),
            "/com/example/facedetection/haarcascade_frontalface_alt.xml",
            context.getCacheDir(), "classifier", ".xml");
        if (classifierFile == null || classifierFile.length() <= 0) {
            throw new IOException("Could not extract the classifier file from Java resource.");
        }

        // Preload the opencv_objdetect module to work around a known bug.
        Loader.load(opencv_objdetect.class);
        classifier = new CvHaarClassifierCascade(cvLoad(classifierFile.getAbsolutePath()));
        classifierFile.delete();
        if (classifier.isNull()) {
            throw new IOException("Could not load the classifier file.");
        }
        storage = CvMemStorage.create();
        
        loadFaceRecognizer();
    }
    
    private void loadFaceRecognizer() {
    	File imgFile = new  File( Environment.getExternalStorageDirectory().getPath() + "/faceRecognizerTest/test.png");
    	IplImage testImage = cvLoadImage(imgFile.getAbsolutePath());
    	File root = new File(Environment.getExternalStorageDirectory().getPath() + "/faceRecognizer/");

        FilenameFilter pngFilter = new FilenameFilter() {
          public boolean accept(File dir, String name) {
            return name.toLowerCase().endsWith(".png");
          }
        };

        File[] imageFiles = root.listFiles(pngFilter);

        MatVector images = new MatVector(imageFiles.length);

        int[] labels = new int[imageFiles.length];

        int counter = 0;
        int label;

        IplImage img;
        IplImage grayImg;
        
        names = new HashMap<String, Integer>();
        names.put("unknown", -1);
        names.put("2", 2);
        names.put("3", 3);
        names.put("4", 4);
        names.put("andy", 1000);
        names.put("erin", 5);
        names.put("gabe", 6);
        names.put("hill", 7);
        names.put("jack", 8);
        names.put("maciek", 9);
        names.put("zach", 10);

        for (File image : imageFiles) {
        
        	
          img = cvLoadImage(image.getAbsolutePath());
          label = names.get(image.getName().split("\\-")[0]);
          // Convert image to grayscale:
          grayImg = IplImage.create(img.width(), img.height(), IPL_DEPTH_8U, 1);
          cvCvtColor(img, grayImg, CV_BGR2GRAY);
          // Append it in the image list:
          images.put(counter, grayImg);
          // And in the labels list:
          labels[counter] = label;
          // Increase counter for next image:
          counter++;
          Log.i("BUFFER", String.valueOf(label));
        }

        //FaceRecognizer faceRecognizer = new FaceRecognizer();
         FaceRecognizer faceRecognizer = com.googlecode.javacv.cpp.opencv_contrib.createFisherFaceRecognizer();
        // FaceRecognizer faceRecognizer = createLBPHFaceRecognizer()
         faceRecognizer.set("threshold", 1000.0);
        faceRecognizer.train(images, labels);

        // Load the test image:
        IplImage greyTestImage = IplImage.create(testImage.width(), testImage.height(), IPL_DEPTH_8U, 1);
        cvCvtColor(testImage, greyTestImage, CV_BGR2GRAY);

        // And get a prediction:
        int predictedLabel = faceRecognizer.predict(greyTestImage);
        for (Entry<String, Integer> entry : names.entrySet()) {
            if (entry.getValue().equals(predictedLabel)) {
            	Toast.makeText(this.context, entry.getKey(), Toast.LENGTH_LONG).show();
            }
        }
        
      }

    public void onPreviewFrame(final byte[] data, final Camera camera) {
        try {
            Camera.Size size = camera.getParameters().getPreviewSize();
            processImage(data, size.width, size.height);
            camera.addCallbackBuffer(data);
        } catch (RuntimeException e) {
            // The camera has probably just been released, ignore.
        }
    }

    protected void processImage(byte[] data, int width, int height) {
        // First, downsample our image and convert it into a grayscale IplImage
        int f = SUBSAMPLING_FACTOR;
        if (grayImage == null || grayImage.width() != width/f || grayImage.height() != height/f) {
            grayImage = IplImage.create(width/f, height/f, IPL_DEPTH_8U, 1);
        }
        int imageWidth  = grayImage.width();
        int imageHeight = grayImage.height();
        int dataStride = f*width;
        int imageStride = grayImage.widthStep();
        ByteBuffer imageBuffer = grayImage.getByteBuffer();
        for (int y = 0; y < imageHeight; y++) {
            int dataLine = y*dataStride;
            int imageLine = y*imageStride;
            for (int x = 0; x < imageWidth; x++) {
                imageBuffer.put(imageLine + x, data[dataLine + f*x]);
            }
        }

        faces = cvHaarDetectObjects(grayImage, classifier, storage, 1.1, 3, CV_HAAR_DO_CANNY_PRUNING);
        postInvalidate();
        cvClearMemStorage(storage);
    }

    @Override
    protected void onDraw(Canvas canvas) {
        Paint paint = new Paint();
        paint.setColor(Color.RED);
        paint.setTextSize(20);

        String s = "FacePreview - This side up.";
        float textWidth = paint.measureText(s);
        canvas.drawText(s, (getWidth()-textWidth)/2, 20, paint);
        
        if (faces != null) {
        	Log.i("faces", String.valueOf(faces.total()));
            paint.setStrokeWidth(2);
            paint.setStyle(Paint.Style.STROKE);
            float scaleX = (float)getWidth()/grayImage.width();
            float scaleY = (float)getHeight()/grayImage.height();
            int total = faces.total();
            for (int i = 0; i < total; i++) {
                CvRect r = new CvRect(cvGetSeqElem(faces, i));
                int x = r.x(), y = r.y(), w = r.width(), h = r.height();
                Log.i("faces", String.valueOf(getWidth()-x*scaleX));
                if(getWidth()-x*scaleX < getWidth()-(x+w)*scaleX) {
                	canvas.drawRect(getWidth()-x*scaleX, y*scaleY, getWidth()-(x+w)*scaleX, (y+h)*scaleY, paint);
                } else {
                	canvas.drawRect(getWidth()-(x+w)*scaleX, y*scaleY, getWidth()-x*scaleX, (y+h)*scaleY, paint);
                }
            }
        }
    }
}

// ----------------------------------------------------------------------

class Preview extends SurfaceView implements SurfaceHolder.Callback {
    SurfaceHolder mHolder;
    Camera mCamera;
    Camera.PreviewCallback previewCallback;

    Preview(Context context, Camera.PreviewCallback previewCallback) {
        super(context);
        this.previewCallback = previewCallback;

        // Install a SurfaceHolder.Callback so we get notified when the
        // underlying surface is created and destroyed.
        mHolder = getHolder();
        mHolder.addCallback(this);
        mHolder.setType(SurfaceHolder.SURFACE_TYPE_PUSH_BUFFERS);
    }

    public void surfaceCreated(SurfaceHolder holder) {
        // The Surface has been created, acquire the camera and tell it where
        // to draw.
        mCamera = Camera.open(Camera.getNumberOfCameras()-1);
        try {
           mCamera.setPreviewDisplay(holder);
        } catch (IOException exception) {
            mCamera.release();
            mCamera = null;
            // TODO: add more exception handling logic here
        }
    }

    public void surfaceDestroyed(SurfaceHolder holder) {
        // Surface will be destroyed when we return, so stop the preview.
        // Because the CameraDevice object is not a shared resource, it's very
        // important to release it when the activity is paused.
        mCamera.stopPreview();
        mCamera.release();
        mCamera = null;
    }


    private Size getOptimalPreviewSize(List<Size> sizes, int w, int h) {
        final double ASPECT_TOLERANCE = 0.05;
        double targetRatio = (double) w / h;
        if (sizes == null) return null;

        Size optimalSize = null;
        double minDiff = Double.MAX_VALUE;

        int targetHeight = h;

        // Try to find an size match aspect ratio and size
        for (Size size : sizes) {
            double ratio = (double) size.width / size.height;
            if (Math.abs(ratio - targetRatio) > ASPECT_TOLERANCE) continue;
            if (Math.abs(size.height - targetHeight) < minDiff) {
                optimalSize = size;
                minDiff = Math.abs(size.height - targetHeight);
            }
        }

        // Cannot find the one match the aspect ratio, ignore the requirement
        if (optimalSize == null) {
            minDiff = Double.MAX_VALUE;
            for (Size size : sizes) {
                if (Math.abs(size.height - targetHeight) < minDiff) {
                    optimalSize = size;
                    minDiff = Math.abs(size.height - targetHeight);
                }
            }
        }
        return optimalSize;
    }

    public void surfaceChanged(SurfaceHolder holder, int format, int w, int h) {
        // Now that the size is known, set up the camera parameters and begin
        // the preview.
        Camera.Parameters parameters = mCamera.getParameters();

        List<Size> sizes = parameters.getSupportedPreviewSizes();
        Size optimalSize = getOptimalPreviewSize(sizes, w, h);
        parameters.setPreviewSize(optimalSize.width, optimalSize.height);

        mCamera.setParameters(parameters);
        if (previewCallback != null) {
            mCamera.setPreviewCallbackWithBuffer(previewCallback);
            Camera.Size size = parameters.getPreviewSize();
            byte[] data = new byte[size.width*size.height*
                    ImageFormat.getBitsPerPixel(parameters.getPreviewFormat())/8];
            mCamera.addCallbackBuffer(data);
        }
        mCamera.startPreview();
    }

}