package Application;

import org.bytedeco.javacpp.opencv_core;
import static org.bytedeco.javacpp.opencv_imgproc.*;
import org.bytedeco.javacv.*;
import org.opencv.core.*;

public class UserInterface {
	
	private CanvasFrame Canvas;
	private HandClassification HC;
	private opencv_core.Mat resultImg;
	
	public UserInterface(HandClassification HC) {
		
		this.HC = HC;
		resultImg = new opencv_core.Mat();
		Canvas = new CanvasFrame("手势识别", CanvasFrame.getDefaultGamma()/HC.getHFE().getHD().getCam().getFrameGrabber().getGamma());
	}
	
	public void showFrame() {
        OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();
        
        resultImg = converter.convert(HC.getHFE().getHD().getCam().getCurFrame());
        
		putText(resultImg, HC.getGesture(), new opencv_core.Point(450, 20),CV_FONT_HERSHEY_COMPLEX,0.7,new opencv_core.Scalar(0,255,0,0));
		
		Canvas.showImage(converter.convert(resultImg));
	}
	
	public void closeFrame() {
		
		Canvas.dispose();
	}
	
	public CanvasFrame getCanvas() {
		
		return Canvas;
	}
}
