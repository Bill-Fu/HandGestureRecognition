package Application;

import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_imgproc;

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
	
	public UserInterface(HandClassification HC) {
		
		this.HC = HC;
		resultImg = new opencv_core.Mat();
		canvasResult = new CanvasFrame("手势识别");
	}
	
	public void showResult() {
        OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();
        
		opencv_core.Rect brect = HC.getHFE().getHD().getRect();
        
        resultImg = HC.getHFE().getHD().getCam().getCurImg();
        
		if(brect!=null) {
			putText(resultImg, HC.getGesture(), new opencv_core.Point(450, 20),CV_FONT_HERSHEY_COMPLEX,0.7,new opencv_core.Scalar(0,255,0,0));
			opencv_imgproc.rectangle(resultImg, brect, new opencv_core.Scalar(0, 255, 0, 0));
		}
		
		canvasResult.showImage(converter.convert(resultImg));
		

	}
	
	public void showHandRegion() {
		OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();
        
		this.handRegion = HC.getHFE().getHD().getHSVHandArea();
		
		canvasHandRegion.showImage(converter.convert(handRegion));
	}
	
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
