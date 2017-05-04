package Application;

import java.io.File;

import org.bytedeco.javacv.*;
import org.opencv.core.*;
import org.opencv.imgproc.*;
import org.opencv.highgui.*;
import org.opencv.objdetect.HOGDescriptor;
import org.opencv.ml.*;

public class HandDetection {
	
	private Mat DetectedHand;
	private Camera Cam;
	
	public HandDetection(Camera Cam) {
		this.Cam = Cam;
		DetectedHand = new Mat();
	}
	
	public Mat getDetectedHand() {
		OpenCVFrameConverter.ToMat Frame2Mat = new OpenCVFrameConverter.ToMat();
		
		DetectedHand = new Mat(Frame2Mat.convert(Cam.getCurFrame()).address());
		
		return DetectedHand;
	}
	
	public Camera getCam() {
		
		return Cam;
	}
	
}
