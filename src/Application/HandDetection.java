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
import org.bytedeco.javacpp.opencv_core.Scalar;

public class HandDetection {

	private Camera Cam;
	
	private opencv_core.Mat DetectedHand;
	private opencv_core.Mat Background;
	
	private opencv_core.Mat hsvLower;
	private opencv_core.Mat hsvUpper;
	private opencv_core.Mat hsvLower2;
	private opencv_core.Mat hsvUpper2;
	
	private opencv_core.Mat imgThreshed,imgThreshed2;
	
	private OpenCVFrameConverter.ToMat Frame2Mat;
	
	int height, width;
	
	public HandDetection(Camera Cam) {
		Frame2Mat = new OpenCVFrameConverter.ToMat();
		this.Cam = Cam;
		
		this.DetectedHand = Frame2Mat.convert(Cam.getCurFrame());
		//opencv_imgproc.cvtColor(this.DetectedHand, this.DetectedHand,opencv_imgproc.CV_BGR2GRAY);
		
		// Set background
		this.Background = Frame2Mat.convert(Cam.getCurFrame());
		opencv_imgproc.cvtColor(this.Background, this.Background,opencv_imgproc.CV_BGR2GRAY);
		
		//Set HSV
		setHSV(10,25,105,55,180,75);
	}
	
	public Mat getDetectedHand() {
		this.DetectedHand = Frame2Mat.convert(Cam.getCurFrame());
		//opencv_imgproc.cvtColor(this.DetectedHand, this.DetectedHand,opencv_imgproc.CV_BGR2GRAY);
		
		return new Mat(this.DetectedHand.address());
	}
	
	public Camera getCam() {
		return this.Cam;
	}
	
	public Mat getForegroundHand() {
		opencv_core.Mat Img = Frame2Mat.convert(Cam.getCurFrame());
		opencv_core.Mat Foreground = new opencv_core.Mat();
		
		return new Mat(Foreground.address());
	}
	
	
	public opencv_core.Mat getHandArea() {
		opencv_core.Mat Img = Frame2Mat.convert(Cam.getCurFrame());
		
		imgThreshed = new opencv_core.Mat(height,width,opencv_core.CV_8UC1);
		imgThreshed2 = new opencv_core.Mat(height,width,opencv_core.CV_8UC1);
		
		opencv_imgproc.cvtColor(Img, Img, opencv_imgproc.CV_BGR2HSV);

		opencv_core.inRange(Img, hsvLower, hsvUpper, imgThreshed);
		opencv_core.inRange(Img, hsvLower2, hsvUpper2, imgThreshed2);
		opencv_core.add(imgThreshed,imgThreshed2,imgThreshed);
		
		return imgThreshed;
	}
	
	private void setHSV(int midH,int varH,int midS,int varS,int midV,int varV)
	{
		int huelower2=181;
		int huelower1=midH-varH;
		if(huelower1<0)
		{
			huelower2=180+midH-varH;
			huelower1=0;
		}
		
		height = Frame2Mat.convert(Cam.getCurFrame()).rows();
		width = Frame2Mat.convert(Cam.getCurFrame()).cols();

		hsvLower = new opencv_core.Mat(height,width,opencv_core.CV_8UC3,new Scalar(huelower1, midS-varS, midV-varV,0));
		hsvUpper = new opencv_core.Mat(height,width,opencv_core.CV_8UC3,new Scalar(midH+varH, midS+varS, midV+varV,0));
		hsvLower2 = new opencv_core.Mat(height,width,opencv_core.CV_8UC3,new Scalar(huelower2, midS-varS, midV-varV,0));
		hsvUpper2 = new opencv_core.Mat(height,width,opencv_core.CV_8UC3,new Scalar(255, midS+varS, midV+varV,0));
	}
	
}
