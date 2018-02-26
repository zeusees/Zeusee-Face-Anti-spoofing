package alive.zeusees.activedetection;

import android.Manifest;
import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.Environment;
import android.support.v4.app.ActivityCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.TextView;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;

public class MainActivity extends Activity {

    // Used to load the 'native-lib' library on application startup.
    private static final int REQUEST_EXTERNAL_STORAGE = 1;

    private static String[] PERMISSIONS_STORAGE = {
            "android.permission.READ_EXTERNAL_STORAGE",
            "android.permission.WRITE_EXTERNAL_STORAGE" };


    Button startBtn;
//    public void requestAllPower() {
//        if (ContextCompat.checkSelfPermission(this,
//                Manifest.permission.WRITE_EXTERNAL_STORAGE)
//                != PackageManager.PERMISSION_GRANTED) {
//            if (ActivityCompat.shouldShowRequestPermissionRationale(this,
//                    Manifest.permission.WRITE_EXTERNAL_STORAGE)) {
//            } else {
//                ActivityCompat.requestPermissions(this,
//                        new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE,
//                                Manifest.permission.READ_EXTERNAL_STORAGE}, 1);
//            }
//        }
//    }


    public void copyFilesFromAssets(Context context, String oldPath, String newPath) {
        try {
            String[] fileNames = context.getAssets().list(oldPath);
            if (fileNames.length > 0) {
                // directory
                File file = new File(newPath);
                if (!file.mkdir())
                {
                    Log.d("mkdir","can't make folder");

                }
//                    return false;                // copy recursively
                for (String fileName : fileNames) {
                    copyFilesFromAssets(context, oldPath + "/" + fileName,
                            newPath + "/" + fileName);
                }
            } else {
                // file
                InputStream is = context.getAssets().open(oldPath);
                FileOutputStream fos = new FileOutputStream(new File(newPath));
                byte[] buffer = new byte[1024];
                int byteCount;
                while ((byteCount = is.read(buffer)) != -1) {
                    fos.write(buffer, 0, byteCount);
                }
                fos.flush();
                is.close();
                fos.close();
            }
        } catch (Exception e) {
            // TODO Auto-generated catch block
            e.printStackTrace();
        }
    }

    public void init()
    {

        String assetPath = "AliveDetection";
        String sdcardPath = Environment.getExternalStorageDirectory()
                + File.separator + assetPath;
        copyFilesFromAssets(this, assetPath, sdcardPath);

    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        init();


        startBtn = (Button)findViewById(R.id.button);
        startBtn.setOnClickListener(
                new View.OnClickListener() {
                    @Override
                    public void onClick(View view) {
                        Intent intent = new Intent(MainActivity.this,Detection.class);
                        MainActivity.this.startActivity(intent);
                    }
                }
        );




        // Example of a call to a native method
//        TextView tv = (TextView) findViewById(R.id.sample_text);
//        tv.setText(stringFromJNI());
    }

    /**
     * A native method that is implemented by the 'native-lib' native library,
     * which is packaged with this application.
     */
}
