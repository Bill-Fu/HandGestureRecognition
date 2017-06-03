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
	private opencv_core.Rect HSVRectRoi;
	private opencv_core.RectVector palms;
	
	private String[] gestureName = {"Fist Gesture", "Stop Gesture", "Five Gesture", "Point Gesture", "Victory Gesture"};
	
	private OpenCVFrameConverter.ToMat converter;
	
	private opencv_core.Rect palm;
	
	public UserInterface(HandClassification HC) {
		this.HC = HC;
		this.resultImg = new opencv_core.Mat();
		this.canvasResult = new CanvasFrame("手势识别");
		this.converter = new OpenCVFrameConverter.ToMat();
	}
	
	public void showResult() {
		this.HSVRectRoi = HC.getHFE().getHD().getHSVRectROI();
        this.palms = HC.getHFE().getHD().getPalms();
        
        HC.getHFE().getHD().getCam().getCurImg().copyTo(this.resultImg);
        
        if (this.palms.size() > 1) {
        	for (int i = 0; i < palms.size(); ++i) {
            	palm = HC.getHFE().getHD().getPalm(i);
            	HC.getHFE().getHD().setROI(palm);
            	opencv_imgproc.rectangle(resultImg, palm, new opencv_core.Scalar(0, 255, 0, 0));
            	putText(resultImg, HC.getGesture(), new opencv_core.Point(palm.x(), palm.y()),CV_FONT_HERSHEY_COMPLEX,0.7,new opencv_core.Scalar(0,0,255,0));
            }
        }
        else if (HSVRectRoi!=null) {
        	HC.getHFE().getHD().setROI(HSVRectRoi);
			putText(resultImg, HC.getGesture(), new opencv_core.Point(HSVRectRoi.x(), HSVRectRoi.y()),CV_FONT_HERSHEY_COMPLEX,0.7,new opencv_core.Scalar(0,0,255,0));
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
