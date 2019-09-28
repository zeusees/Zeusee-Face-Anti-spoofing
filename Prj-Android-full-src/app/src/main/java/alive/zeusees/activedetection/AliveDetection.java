package alive.zeusees.activedetection;

import android.os.CountDownTimer;
import android.util.Log;

import java.util.Random;
import java.util.Timer;
import java.util.TimerTask;

/**
 * Created by yujinke on 10/02/2018.
 */

public class AliveDetection {

    static {
        System.loadLibrary("native-lib");
        Log.d("loadlib","load lib");
    }

    public native static int detect(byte[] data,int height,int width,long handle);
    public native static long init(String folder);
    public native static void release(long handle);
    private long handle ;
    public String[] strings = {"请摇头","请抬头","请低头"};
    private int expectOrder;
    private int controlState;
//    Timer timer = new Timer();
//    CountDownTimer timer;
    int flag = 0;
    int state = -1;
    int stage = 0;
    int time = 2 ;

    int poseCode;



    CountDownTimer taskWaiting = new CountDownTimer(14000, 2000) {
        @Override
        public void onTick(long millisUntilFinished) {
            Detection.warning.setText("请等待" + String.valueOf(time*2)+"s");
            time-=1;
            if(time==0)
            {
                poseRequest();
                state=1;
            }
            if(time<0)
            {


                if(flag == 1){
                    if(stage==1)
                    {
//                            poseRequest();
//                            startAliveDetection();
                        SuccessInfo();
                        taskWaiting.cancel();


                    }
                    else{
                        stage+=1;
                        poseRequest();
//                        taskWaiting.cancel();

                        //startAliveDetection();
                    }
                }else
                {
                    FailedInfo();
                    taskWaiting.cancel();

                    //检测失败
                }

            }
        }

        @Override
        public void onFinish() {
            //startAliveDetection();
            time = 3;

        }
    };






    Timer waiting;

    CountDownTimer taskTnterval = new CountDownTimer(2000,1000) {
        @Override
        public void onTick(long millisUntilFinished) {


        }

        @Override
        public void onFinish() {
            poseRequest();
            startAliveDetection();
        }
    };


    Timer interval;


    TimerTask taskTimeOut = new TimerTask() {
        public void run() {

        }
    };

    Timer TimeOut;


    void SuccessInfo()
    {
        Detection.warning.setText("检测成功");

    }

    void FailedInfo()
    {

        Detection.warning.setText("检测失败");

    }

    void PoseInfo(String info)
    {
        Detection.warning.setText(info);
    }

    void poseRequest(){
        Random rand = new Random();
        poseCode =rand.nextInt(3)+1;
        PoseInfo(strings[poseCode-1]);
        state =1;
        flag = 0;
    }
    void startAliveDetection()
    {
        taskTnterval.start();

    }



    AliveDetection(String folder)
    {

        controlState = 0 ;
        expectOrder = -1;
        handle = init(folder);
        waiting = new Timer(true);
        TimeOut= new Timer(true);
        interval= new Timer(true);



    }

    protected void finalize() throws java.lang.Throwable {
        super.finalize();
        release(handle);
    }

    public int StateDetection(byte[] data,int height,int width )
    {
        return detect(data,height,width,handle);
    }

    public void startDetection(){
        taskWaiting.start();

        //开始 1.5秒 开始计时让对方做动作
    }



    public int AliveDetection(byte[] data,int height,int width )
    {
        int res = detect(data,height,width,handle);
        if (res == 0 && controlState ==0)
        {
            //启动 计时器
            startDetection();
            controlState = 1;
        }
        if(poseCode ==res && state == 1)
        {
            flag = 1;
            state=-1;

        }
        Log.i("detection","contorlState:"+String.valueOf(controlState));
        Log.i("detection","expectOrder:"+String.valueOf(poseCode));
        Log.i("detection","res:"+String.valueOf(res));

        return res;

    }
}
