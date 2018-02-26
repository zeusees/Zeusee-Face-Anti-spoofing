package alive.zeusees.activedetection;


import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.RectF;
import android.util.AttributeSet;
import android.util.Log;
import android.view.View;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ListView;

//import com.faceplusplus.api.FaceDetecter.Face;

public class FaceMask extends View {
    Paint localPaint = null;
//    Face[] faceinfos = null;
    RectF rect = null;
    LinearLayout container  = null;
    byte[] data_= null;
    int w_size;
    int h_size;
    String Name  = null;
    int Age = -1;

    public FaceMask(Context context, AttributeSet atti) {
        super(context, atti);
        rect = new RectF();
        localPaint = new Paint();
        localPaint.setColor(0xff00b4ff);
        localPaint.setStrokeWidth(5);
        localPaint.setStyle(Paint.Style.STROKE);


    }


    @Override
    protected void onDraw(Canvas canvas) {
        super.onDraw(canvas);



        }

}
