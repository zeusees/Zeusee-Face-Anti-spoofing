package alive.zeusees.activedetection;

import android.Manifest;
import android.app.Activity;
import android.content.Context;
import android.content.pm.PackageManager;
import android.graphics.PixelFormat;
import android.hardware.Camera;
import android.os.Handler;
import android.os.HandlerThread;
import android.support.v4.app.ActivityCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.Display;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.WindowManager;
import android.widget.LinearLayout;
import android.widget.RelativeLayout;
import android.widget.TextView;

import java.io.IOException;
import java.util.Iterator;
import java.util.List;

public class Detection extends Activity implements SurfaceHolder.Callback,Camera.PreviewCallback {


    Camera camera = null;
    SurfaceView camerasurface = null;
    SurfaceHolder holder;
    public static final String FOCUS_MODE_MACRO = "macro";
    Camera.Parameters parameters;
    HandlerThread handleThread = null;
    Handler detectHandler = null;
    private int width = 480;
    private int height = 270;
    int PreviewWidth = 0;
    int PreviewHeight = 0;
    public AliveDetection aliveDetection;
    public TextView detectTime;
    public TextView stateBox;
    public static TextView warning;






    /**
     * 获取最适合屏幕的照片 尺寸
     *
     * @param sizes
     * @param w
     * @param h
     * @return
     */

    private static final int REQUEST_EXTERNAL_STORAGE = 1;
    private static String[] PERMISSIONS_STORAGE = {
            "android.permission.READ_EXTERNAL_STORAGE",
            "android.permission.WRITE_EXTERNAL_STORAGE" };


    public static void verifyStoragePermissions(Activity activity) {

        try {
            //检测是否有写的权限
            int permission = ActivityCompat.checkSelfPermission(activity,
                    "android.permission.WRITE_EXTERNAL_STORAGE");
            if (permission != PackageManager.PERMISSION_GRANTED) {
                // 没有写的权限，去申请写的权限，会弹出对话框
                ActivityCompat.requestPermissions(activity, PERMISSIONS_STORAGE,REQUEST_EXTERNAL_STORAGE);
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static Camera.Size getOptimalPreviewSize(List<Camera.Size> sizes, int w, int h) {
        final double ASPECT_TOLERANCE = 0.1;
        double targetRatio = (double) w / h;
        if (sizes == null)
            return null;

        Camera.Size optimalSize = null;
        double minDiff = Double.MAX_VALUE;

        int targetHeight = h;

        // Try to find an size match aspect ratio and size
        for ( Camera.Size size : sizes) {
            double ratio = (double) size.width / size.height;
            if (Math.abs(ratio - targetRatio) > ASPECT_TOLERANCE)
                continue;
            if (Math.abs(size.height - targetHeight) < minDiff) {
                optimalSize = size;
                minDiff = Math.abs(size.height - targetHeight);
            }
        }

        // Cannot find the one match the aspect ratio, ignore the requirement
        if (optimalSize == null) {
            minDiff = Double.MAX_VALUE;
            for (Camera.Size size : sizes) {
                if (Math.abs(size.height - targetHeight) < minDiff) {
                    optimalSize = size;
                    minDiff = Math.abs(size.height - targetHeight);
                }
            }
        }
        return optimalSize;
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_detection);
        verifyStoragePermissions(this);

        detectTime = (TextView)findViewById(R.id.textView);
        warning= (TextView)findViewById(R.id.textView3);
        stateBox = (TextView)findViewById(R.id.textView2);

        camerasurface = (SurfaceView) findViewById(R.id.surfaceView2);

        if (savedInstanceState == null) {
            savedInstanceState = new Bundle();
        }
        camerasurface.getHolder().addCallback(this);
        camerasurface.setKeepScreenOn(true);
        RelativeLayout.LayoutParams linearParams = (RelativeLayout.LayoutParams) camerasurface.getLayoutParams();


        camera = Camera.open(1);//打开当前选中的摄像头

        WindowManager wm = (WindowManager) getSystemService(Context.WINDOW_SERVICE);//获取窗口的管理器
        Display display = wm.getDefaultDisplay();//获得窗口里面的屏幕
        parameters  = camera.getParameters();
        // 选择合适的预览尺寸
        List<Camera.Size> sizeList = parameters.getSupportedPreviewSizes();

        // 如果sizeList只有一个我们也没有必要做什么了，因为就他一个别无选择
        if (sizeList.size() > 1) {
            Iterator<Camera.Size> itor = sizeList.iterator();
            while (itor.hasNext()) {
                Camera.Size cur = itor.next();
                if(cur.height<=
                        360)
                {
                    PreviewWidth = cur.width;
                    PreviewHeight = cur.height;
                    break;

                }
            }
        }

//        PreviewWidth = 640;
//        PreviewHeight = 360;

        parameters.setPreviewSize(PreviewWidth, PreviewHeight); //获得摄像区域的大小
     //   parameters.setPreviewFrameRate(3);//每秒3帧  每秒从摄像头里面获得3个画面
        parameters.setPictureFormat(PixelFormat.JPEG);//设置照片输出的格式
        parameters.set("jpeg-quality", 85);//设置照片质量
        parameters.setPictureSize(PreviewWidth, PreviewHeight);//设置拍出来的屏幕大小
        //
        camera.setParameters(parameters);//把上面的设置 赋给摄像头
        try {
            camera.setPreviewDisplay(camerasurface.getHolder());//把摄像头获得画面显示在SurfaceView控件里面
        } catch (IOException e) {
            e.printStackTrace();
        }
        //  camera.setDisplayOrientation(90);
        camera.startPreview();
        camera.setPreviewCallback(Detection.this);

        aliveDetection  = new AliveDetection("/sdcard/AliveDetection");


    }

    /**
     * A native method that is implemented by the 'native-lib' native library,
     * which is packaged with this application.
     */
    @Override
    protected void onResume() {
        super.onResume();
        Log.i("MainActivity","Activity On onResume");
        camera  = Camera.open(1);

        Camera.Parameters para = camera.getParameters();
        para.setPreviewSize(camerasurface.getWidth(),camerasurface.getHeight());
//        camera.setParameters(para);
    }

    @Override
    protected void onPause() {
        super.onPause();
        Log.i("MainActivity","Activity On Pasue");

        if (camera != null) {
            camera.setPreviewCallback(null);
            camera.stopPreview();
            camera.release();
            //finish();
        }
    }

    @Override
    public void surfaceCreated(SurfaceHolder surfaceHolder) {

    }

    @Override
    public void surfaceChanged(SurfaceHolder surfaceHolder, int i, int i1, int i2) {

        camera.stopPreview();//停掉原来摄像头的预览
        camera.release();//释放资源
        camera = null;//取消原来摄像头
        camera = Camera.open(1);//打开当前选中的摄像头
//        para.setPreviewSize(width, height);
//        camera.setParameters(para);
        PreviewWidth = 640;
        PreviewHeight = 360;

        parameters = camera.getParameters();
        parameters.setFocusMode("macro");
        parameters.setFocusMode(Camera.Parameters.FOCUS_MODE_CONTINUOUS_PICTURE);
//        parameters.setZoom(parameters.getMaxZoom());
        camera.setParameters(parameters);

        try {
            camera.setPreviewDisplay(camerasurface.getHolder());

        } catch (IOException e) {
            e.printStackTrace();
        }

        WindowManager wm = (WindowManager) getSystemService(Context.WINDOW_SERVICE);//获取窗口的管理器
        Display display = wm.getDefaultDisplay();//获得窗口里面的屏幕
         parameters  = camera.getParameters();
        // 选择合适的预览尺寸
        List<Camera.Size> sizeList = parameters.getSupportedPreviewSizes();

        Camera.Size size;

        // 如果sizeList只有一个我们也没有必要做什么了，因为就他一个别无选择

        if (sizeList.size() > 1) {
            Iterator<Camera.Size> itor = sizeList.iterator();
            while (itor.hasNext()) {
                Camera.Size cur = itor.next();
                if(cur.height<=
                        360)
                {
                    PreviewWidth = cur.width;
                    PreviewHeight = cur.height;
                    break;

                }
            }
        }

        parameters.setPreviewSize(PreviewWidth, PreviewHeight); //获得摄像区域的大小
        parameters.setPictureSize(PreviewWidth, PreviewHeight);//设置拍出来的屏幕大小
        //
        camera.setParameters(parameters);//把上面的设置 赋给摄像头
        try {
            camera.setPreviewDisplay(camerasurface.getHolder());//把摄像头获得画面显示在SurfaceView控件里面
        } catch (IOException e) {
            e.printStackTrace();
        }


        camera.setDisplayOrientation(90);
        camera.startPreview();
        camera.setPreviewCallback(Detection.this);
//        isFront = true;

    }

    @Override
    public void surfaceDestroyed(SurfaceHolder surfaceHolder) {

    }
    @Override
    public void onPreviewFrame(final byte[] data, final Camera camera) {
        long currentTime1 = System.currentTimeMillis();

        int state = aliveDetection.AliveDetection(data,PreviewHeight,PreviewWidth);
        long diff = System.currentTimeMillis() - currentTime1;
        detectTime.setText("detectTime:"+ String.valueOf(diff)+"ms");

        if(state == -1)
        {
            stateBox.setText("不能检测到人脸");
        }
        else if(state==0){
            stateBox.setText("正常");
        }
        else if(state==1){
            stateBox.setText("摇头");
        }
        else if(state==2){
            stateBox.setText("抬头");
        }
        else if(state==3){
            stateBox.setText("低头");
        }
        else if(state==-2 || state==-3)
        {
            finish();
        }

        Log.d("STATE_log",String.valueOf(state));
//        System.out.println("p");
    }

}
