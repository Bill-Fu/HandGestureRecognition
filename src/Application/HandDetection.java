package Application;

import java.io.File;

import org.bytedeco.javacv.*;
import org.opencv.core.*;
import org.opencv.imgproc.*;
import org.opencv.highgui.*;
import org.opencv.objdetect.HOGDescriptor;
import org.opencv.ml.*;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_imgproc;

public class HandDetection {
	
	private opencv_core.Mat DetectedHand;
	private Camera Cam;
	private opencv_core.Mat Background;

	public HandDetection(Camera Cam) {
		OpenCVFrameConverter.ToMat Frame2Mat = new OpenCVFrameConverter.ToMat();
		this.Cam = Cam;
		
		this.DetectedHand = Frame2Mat.convert(Cam.getCurFrame());
		//opencv_imgproc.cvtColor(this.DetectedHand, this.DetectedHand,opencv_imgproc.CV_BGR2GRAY);
		
		// Set background
		this.Background = Frame2Mat.convert(Cam.getCurFrame());
		opencv_imgproc.cvtColor(this.Background, this.Background,opencv_imgproc.CV_BGR2GRAY);
	}
	
	public Mat getDetectedHand() {
		OpenCVFrameConverter.ToMat Frame2Mat = new OpenCVFrameConverter.ToMat();
		
		this.DetectedHand = Frame2Mat.convert(Cam.getCurFrame());
		//opencv_imgproc.cvtColor(this.DetectedHand, this.DetectedHand,opencv_imgproc.CV_BGR2GRAY);
		
		return new Mat(this.DetectedHand.address());
	}
	
	public Camera getCam() {
		return this.Cam;
	}
	
	public Mat getForegroundHand() {
		OpenCVFrameConverter.ToMat Frame2Mat = new OpenCVFrameConverter.ToMat();
		
		opencv_core.Mat Img = Frame2Mat.convert(Cam.getCurFrame());
		opencv_core.Mat Foreground = new opencv_core.Mat();
		
		
		
		return new Mat(Foreground.address());
	}
	
}
