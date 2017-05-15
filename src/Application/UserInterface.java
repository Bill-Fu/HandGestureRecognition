package Application;

import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_imgproc;
import org.bytedeco.javacpp.opencv_core.Point;
import org.bytedeco.javacpp.opencv_core.Scalar;

import static org.bytedeco.javacpp.opencv_imgproc.*;
import org.bytedeco.javacv.*;
import org.opencv.core.*;

public class UserInterface {
	private HandClassification HC;
	
	private CanvasFrame canvasResult;
	private CanvasFrame canvasHandRegion;
	private CanvasFrame canvasForeground;
	
	private opencv_core.Mat resultImg;
	private opencv_core.Mat handRegion;
	private opencv_core.Mat foreground;
	
	private String[] gestureName = {"Fist Gesture", "Stop Gesture", "Five Gesture", "Point Gesture", "Victory Gesture"};
	
	public UserInterface(HandClassification HC) {
		this.HC = HC;
		resultImg = new opencv_core.Mat();
		canvasResult = new CanvasFrame("手势识别");
	}
	
	public void showResult() {
        OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();
        
		opencv_core.Rect HSVRectRoi = HC.getHFE().getHD().getHSVRectROI();
        opencv_core.RectVector palms = HC.getHFE().getHD().getPalms();
        
        HC.getHFE().getHD().getCam().getCurImg().copyTo(resultImg);
        
        if (palms.size() != 0) {
        	for (int i = 0; i < palms.size(); ++i) {
            	opencv_core.Rect palm = HC.getHFE().getHD().getPalm(i);
            	HC.getHFE().getHD().setROI(palm);
            	opencv_imgproc.rectangle(resultImg, palm, new opencv_core.Scalar(0, 255, 0, 0));
            	putText(resultImg, HC.getGesture(), new opencv_core.Point(palm.x(), palm.y()),CV_FONT_HERSHEY_COMPLEX,0.7,new opencv_core.Scalar(255,0,0,0));
            }
        }
        else if (HSVRectRoi!=null) {
        	HC.getHFE().getHD().setROI(HSVRectRoi);
			putText(resultImg, HC.getGesture(), new opencv_core.Point(HSVRectRoi.x(), HSVRectRoi.y()),CV_FONT_HERSHEY_COMPLEX,0.7,new opencv_core.Scalar(255,0,0,0));
			opencv_imgproc.rectangle(resultImg, HSVRectRoi, new opencv_core.Scalar(0, 255, 0, 0));
		}
        
		canvasResult.showImage(converter.convert(resultImg));
	}
	
	// NOT USE NOW, FOR DEBUG HSV MODEL
	public void showHandRegion() {
		OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();
        
		this.handRegion = HC.getHFE().getHD().getHSVHandArea();
		
		canvasHandRegion.showImage(converter.convert(handRegion));
	}
	
	// NOT USE NOW, FOR BACKGROUNDSUBTRACTORMOG2 USE
	public void showForeground() {
		OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();
        
		this.foreground = HC.getHFE().getHD().getForegroundHand();
		
		canvasForeground.showImage(converter.convert(foreground));
	}
	
	public void closeFrame() {
		canvasHandRegion.dispose();
		canvasResult.dispose();
	}
	
	public CanvasFrame getCanvas() {
		return canvasResult;
	}
	
}
