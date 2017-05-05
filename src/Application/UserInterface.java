package Application;

import org.bytedeco.javacpp.opencv_core;
import static org.bytedeco.javacpp.opencv_imgproc.*;
import org.bytedeco.javacv.*;
import org.opencv.core.*;

public class UserInterface {
	
	private CanvasFrame canvasResult;
	private CanvasFrame canvasHandRegion;
	private HandClassification HC;
	private opencv_core.Mat resultImg;
	private opencv_core.Mat handRegion;
	
	public UserInterface(HandClassification HC) {
		
		this.HC = HC;
		resultImg = new opencv_core.Mat();
		canvasResult = new CanvasFrame("手势识别", CanvasFrame.getDefaultGamma()/HC.getHFE().getHD().getCam().getFrameGrabber().getGamma());
		canvasHandRegion = new CanvasFrame("手部区域", CanvasFrame.getDefaultGamma()/HC.getHFE().getHD().getCam().getFrameGrabber().getGamma());
	}
	
	public void showResult() {
        OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();
        
        resultImg = converter.convert(HC.getHFE().getHD().getCam().getCurFrame());
        
		putText(resultImg, HC.getGesture(), new opencv_core.Point(450, 20),CV_FONT_HERSHEY_COMPLEX,0.7,new opencv_core.Scalar(0,255,0,0));
		
		canvasResult.showImage(converter.convert(resultImg));
	}
	
	public void showHandRegion() {
		OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();
        
		handRegion = HC.getHFE().getHD().getHandArea();
		
		canvasHandRegion.showImage(converter.convert(handRegion));
	}
	
	public void closeFrame() {
		canvasHandRegion.dispose();
		canvasResult.dispose();
	}
	
	public CanvasFrame getCanvas() {
		
		return canvasResult;
	}
}
