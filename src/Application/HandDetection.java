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
import org.bytedeco.javacpp.opencv_video;
import org.bytedeco.javacpp.opencv_core.MatVector;
import org.bytedeco.javacpp.opencv_core.Scalar;

public class HandDetection {

	private Camera Cam;

	//private opencv_video.BackgroundSubtractorKNN BGS;
	private opencv_video.BackgroundSubtractorMOG2 BGS;
	
	private opencv_core.Mat DetectedHand;
	
	private opencv_core.Mat hsvLower;
	private opencv_core.Mat hsvUpper;
	private opencv_core.Mat hsvLower2;
	private opencv_core.Mat hsvUpper2;
	
	private opencv_core.Mat imgThreshed,imgThreshed2;
	
	private opencv_core.Mat foreground;
	
	private OpenCVFrameConverter.ToMat Frame2Mat;
	
	private opencv_core.Mat kernel;
	
	private opencv_core.Rect rect;
	
	int height, width;
	
	public opencv_core.Rect getRect() {
		return this.rect;
	}
	
	public HandDetection(Camera Cam) {
		this.Frame2Mat = new OpenCVFrameConverter.ToMat();
		this.Cam = Cam;
		
		//this.DetectedHand = Frame2Mat.convert(Cam.getCurFrame());
		//opencv_imgproc.cvtColor(this.DetectedHand, this.DetectedHand,opencv_imgproc.CV_BGR2GRAY);
		this.DetectedHand = Cam.getCurImg();
		
		//this.BGS = opencv_video.createBackgroundSubtractorKNN();
		this.BGS = opencv_video.createBackgroundSubtractorMOG2();
		
		this.foreground = new opencv_core.Mat();
		
		this.kernel = new opencv_core.Mat(8, 8, opencv_core.CV_8U, new opencv_core.Scalar(1d));
		
		//Set HSV
		setHSV(10,25,105,55,180,75);
		
		imgThreshed = new opencv_core.Mat(height,width,opencv_core.CV_8UC3);
		imgThreshed2 = new opencv_core.Mat(height,width,opencv_core.CV_8UC3);
		
		rect = null;
	}
	
	public Mat getDetectedHand() {
		//this.DetectedHand = Frame2Mat.convert(Cam.getCurFrame());
		this.DetectedHand = Cam.getCurImg();
		
		this.DetectedHand = new opencv_core.Mat(DetectedHand, this.rect);
		
		return new Mat(this.DetectedHand.address());
	}
	
	public Camera getCam() {
		return this.Cam;
	}
	
	// not use
	public opencv_core.Mat getForegroundHand() {
		opencv_core.Mat Img = Cam.getCurImg();
		
		BGS.apply(Img, this.foreground);
		
		//opencv_imgproc.erode(foreground, foreground, kernel);
		//opencv_imgproc.dilate(foreground, foreground, kernel);
		
		return this.foreground;
	}
	
	public opencv_core.Mat getHSVHandArea() {
		//opencv_core.Mat Img = Frame2Mat.convert(Cam.getCurFrame());
		opencv_core.Mat Img = Cam.getCurImg();

		opencv_imgproc.cvtColor(Img, Img, opencv_imgproc.CV_BGR2HSV);
		
		opencv_core.inRange(Img, hsvLower, hsvUpper, imgThreshed);
		opencv_core.inRange(Img, hsvLower2, hsvUpper2, imgThreshed2);
		opencv_core.add(imgThreshed,imgThreshed2,imgThreshed);
		
		// erode and dilate operation, can be removed if need
		opencv_imgproc.erode(imgThreshed, imgThreshed, kernel);
		opencv_imgproc.dilate(imgThreshed, imgThreshed, kernel);
		
		return imgThreshed;
	}
	
	public void getContour() {
		float maxArea = 1200.0f;
		opencv_core.RotatedRect box;
		opencv_core.RotatedRect maxbox = null;
		opencv_core.Point2f vertices = new opencv_core.Point2f(4);
		opencv_core.Rect brect = null;
		opencv_core.Rect brectnew = null;
		
		opencv_core.Mat tmp = new opencv_core.Mat();
		
		getHSVHandArea().copyTo(tmp);
		
		opencv_core.MatVector contours = new opencv_core.MatVector();
		
		opencv_imgproc.findContours(tmp, contours, opencv_imgproc.RETR_LIST, opencv_imgproc.CHAIN_APPROX_NONE);
		
		for(int i = 0; i < contours.size(); ++i) {
			box = opencv_imgproc.minAreaRect(contours.get(i));
			float area = box.size().height()*box.size().width();
			if(area > maxArea) {
				maxArea = area;
				maxbox = box;
			}
		}

		// TODO rewrite it
		if(maxbox!=null) {
			maxbox.points(vertices);
			brect = maxbox.boundingRect();
			
			int x = brect.x();
			int y = brect.y();
			int h = brect.size().height();
			int w = brect.size().width();
			int l = (h>w)?h:w;
			
			int xp = (int)(x + 0.5 * w - 0.7 * l);
			int yp = (int)(y + 0.5 * h - 0.7 * l);
			
			xp = (xp > 0)? xp : 0;
			yp = (yp > 0)? yp : 0;
			l = (int)(1.4 * l);
			
			if((xp + l) > imgThreshed.size().width()) {
				l = imgThreshed.size().width() - xp;
			}
			
			if((yp + l) > imgThreshed.size().height()) {
				l = imgThreshed.size().height() - yp;
			}
			
			brectnew = new opencv_core.Rect(xp, yp, l, l);
		}
		
		this.rect = brectnew;
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
		
		height = Cam.getCurImg().rows();
		//height = Frame2Mat.convert(Cam.getCurFrame()).rows();
		width = Cam.getCurImg().cols();
		//width = Frame2Mat.convert(Cam.getCurFrame()).cols();

		hsvLower = new opencv_core.Mat(height,width,opencv_core.CV_8UC3,new Scalar(huelower1, midS-varS, midV-varV,0));
		hsvUpper = new opencv_core.Mat(height,width,opencv_core.CV_8UC3,new Scalar(midH+varH, midS+varS, midV+varV,0));
		hsvLower2 = new opencv_core.Mat(height,width,opencv_core.CV_8UC3,new Scalar(huelower2, midS-varS, midV-varV,0));
		hsvUpper2 = new opencv_core.Mat(height,width,opencv_core.CV_8UC3,new Scalar(255, midS+varS, midV+varV,0));
	}
	
	public void updateDetectedHand() {
		getContour();
	}
	
}