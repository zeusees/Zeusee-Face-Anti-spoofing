package alive.zeusees.activedetection;

import android.Manifest;
import android.app.Activity;
import android.content.Context;
import android.content.pm.PackageManager;
import android.hardware.Camera;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.util.Log;
import android.view.Display;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.WindowManager;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.RequiresApi;
import androidx.core.app.ActivityCompat;

import java.io.IOException;
import java.util.Iterator;
import java.util.List;

public class Detection extends Activity implements SurfaceHolder.Callback, Camera.PreviewCallback {


    private Camera camera = null;
    private SurfaceView camerasurface = null;
    private SurfaceHolder holder;
    public static final String FOCUS_MODE_MACRO = "macro";
    private Camera.Parameters parameters;
    private HandlerThread handleThread = null;
    private Handler detectHandler = null;
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
            "android.permission.WRITE_EXTERNAL_STORAGE"};


    @RequiresApi(api = Build.VERSION_CODES.M)
    public void callCamera(SurfaceHolder surfaceHolde) {
        if (Build.VERSION.SDK_INT > 21) {
            String callCamera = Manifest.permission.CAMERA;
            String callRecord = Manifest.permission.RECORD_AUDIO;
            String[] permissions = new String[]{callCamera, callRecord};
            int selfsound = ActivityCompat.checkSelfPermission(this, callRecord);
            int selfPermission = ActivityCompat.checkSelfPermission(this, callCamera);
//            int selfwrite = ActivityCompat.checkSelfPermission(this, writestorage);
//            int selfread = ActivityCompat.checkSelfPermission(this, readstorage);
            if (selfPermission != PackageManager.PERMISSION_GRANTED || selfsound != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(this, permissions, 1);
            } else {
                initCamera(surfaceHolde);
            }
        } else {
            initCamera(surfaceHolde);
        }
    }

    //处理申请权限的结果
    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        switch (requestCode) {
            case 1:

                if (grantResults.length > 0
                        && grantResults[0] == PackageManager.PERMISSION_GRANTED) {

                    initCamera(camerasurface.getHolder());
                } else {
                    Toast.makeText(this, "you refused the camera function", Toast.LENGTH_SHORT).show();
                }
                break;
        }
    }


    private void initCamera(SurfaceHolder surfaceHolde) {
        camera = Camera.open(1);
        WindowManager wm = (WindowManager) getSystemService(Context.WINDOW_SERVICE);//获取窗口的管理器
        Display display = wm.getDefaultDisplay();
        parameters = camera.getParameters();
        List<Camera.Size> sizeList = parameters.getSupportedPreviewSizes();
        Camera.Size size = getOptimalPreviewSize(sizeList, display.getWidth(), display.getHeight());
        if (sizeList.size() > 1) {
            Iterator<Camera.Size> itor = sizeList.iterator();
            while (itor.hasNext()) {
                Camera.Size cur = itor.next();
                if (cur.height <=
                        360) {

                    height = cur.height;
                    width = cur.width;

                    break;

                }
            }
        }
//        parameters.setPreviewFpsRange(30000,30000);
        parameters.setPreviewFpsRange(30000, 30000);
        parameters.setVideoStabilization(true);

        parameters.setPreviewSize(width, height);
        try {
            camera.setPreviewDisplay(camerasurface.getHolder());//把摄像头获得画面显示在SurfaceView控件里面
        } catch (IOException e) {
            e.printStackTrace();
        }
        List<String> focusModes = parameters.getSupportedFocusModes();
        if (focusModes.contains(Camera.Parameters.FOCUS_MODE_CONTINUOUS_PICTURE)) {
            parameters.setFocusMode(Camera.Parameters.FOCUS_MODE_CONTINUOUS_PICTURE);
        } else if (focusModes.contains(Camera.Parameters.FOCUS_MODE_AUTO)) {
            parameters.setFocusMode(Camera.Parameters.FOCUS_MODE_AUTO);
        }

//        parameters.setFlashMode(Camera.Parameters.FLASH_MODE_TORCH);

        camera.setParameters(parameters);
        camera.setDisplayOrientation(90);
        camera.startPreview();
        camera.setPreviewCallback(Detection.this);
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
        for (Camera.Size size : sizes) {
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
        detectTime = (TextView) findViewById(R.id.textView);
        warning = (TextView) findViewById(R.id.textView3);
        stateBox = (TextView) findViewById(R.id.textView2);

        camerasurface = (SurfaceView) findViewById(R.id.surfaceView2);
        camerasurface.getHolder().addCallback(this);

        aliveDetection = new AliveDetection("/sdcard/AliveDetection");
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (camera != null) {
            camera.setPreviewCallback(null);
            camera.stopPreview();
            camera.release();
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
    }

    @RequiresApi(api = Build.VERSION_CODES.M)
    @Override
    public void surfaceCreated(SurfaceHolder surfaceHolder) {
        callCamera(camerasurface.getHolder());
    }

    @Override
    public void surfaceChanged(SurfaceHolder surfaceHolder, int i, int i1, int i2) {
        if (surfaceHolder.getSurface() == null) {
            return;
        }

        try {
            camera.stopPreview();
        } catch (Exception e) {
            // ignore: tried to stop a non-existent preview
        }

        try {
            camera.setPreviewCallback(this);
            camera.setPreviewDisplay(surfaceHolder);
            camera.startPreview();

        } catch (Exception e) {
            Log.d("TAG", "Error starting camera preview: " + e.getMessage());
        }

    }

    @Override
    public void surfaceDestroyed(SurfaceHolder surfaceHolder) {

    }

    @Override
    public void onPreviewFrame(final byte[] data, final Camera camera) {
        long currentTime1 = System.currentTimeMillis();
        final Camera.Size previewSize = camera.getParameters().getPreviewSize();

        int state = aliveDetection.AliveDetection(data, previewSize.height, previewSize.width);
        long diff = System.currentTimeMillis() - currentTime1;
        detectTime.setText("detectTime:" + String.valueOf(diff) + "ms");

        if (state == -1) {
            stateBox.setText("不能检测到人脸");
        } else if (state == 0) {
            stateBox.setText("正常");
        } else if (state == 1) {
            stateBox.setText("摇头");
        } else if (state == 2) {
            stateBox.setText("抬头");
        } else if (state == 3) {
            stateBox.setText("低头");
        }

//        else if (state == -2 || state == -3) {
//            finish();
//        }

        Log.e("STATE_log", String.valueOf(state));
    }
}
