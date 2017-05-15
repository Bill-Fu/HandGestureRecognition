package Application;

import java.io.File;

import org.bytedeco.javacv.*;
import org.opencv.core.*;
import org.opencv.imgproc.*;
import org.opencv.highgui.*;
import org.opencv.objdetect.HOGDescriptor;
import org.bytedeco.javacpp.opencv_objdetect.CascadeClassifier;
import org.opencv.ml.*;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_imgproc;
import org.bytedeco.javacpp.opencv_objdetect;
import org.bytedeco.javacpp.opencv_video;
import org.bytedeco.javacpp.opencv_core.MatVector;
import org.bytedeco.javacpp.opencv_core.Scalar;

public class HandDetection {

	private static final float minDetectedArea = 1200.0f;
	private static final float ROIRatio = 1.4f;
	private static final String HAARModel = "palmCascadeClassifier.xml";
	
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
	
	private opencv_core.Mat kernel;
	
	private opencv_core.Rect ROI;
	
	// FOR HSV MODEL USE
	private opencv_core.Rect HSVRectROI;
	
	// FOR HAAR MODEL USE
	private opencv_core.RectVector palms;
	
	private int height, width;
	
	private CascadeClassifier palmCascade;
	
	public HandDetection(Camera Cam) {
		this.Cam = Cam;
		
		this.DetectedHand = Cam.getCurImg();
		
		// NOT USE NOW, FOR BACKGROUNDSUBTRACTOR USE
		//this.BGS = opencv_video.createBackgroundSubtractorKNN();
		this.BGS = opencv_video.createBackgroundSubtractorMOG2();
		
		this.foreground = new opencv_core.Mat();
		
		this.kernel = new opencv_core.Mat(8, 8, opencv_core.CV_8U, new opencv_core.Scalar(1d));
		
		//Set HSV
		setHSV(10,25,105,55,180,75);
		
		this.imgThreshed = new opencv_core.Mat(height,width,opencv_core.CV_8UC3);
		this.imgThreshed2 = new opencv_core.Mat(height,width,opencv_core.CV_8UC3);
		
		this.ROI = null;
		this.HSVRectROI = null;
		
		// load HAAR Model
		File palmCascadeClassifierFile = new File(HAARModel);
		palmCascade = new CascadeClassifier(palmCascadeClassifierFile.getAbsolutePath());
		
		if (!palmCascade.load(palmCascadeClassifierFile.getAbsolutePath())){
			System.out.println("Can't load file!");
		}
		
		this.palms = new opencv_core.RectVector();
		
	}
	
	public opencv_core.Rect getHSVRectROI() {
		return this.HSVRectROI;
	}
	
	public void updateDetectedHand() {
		Cam.updateCurImg();
		getHSVROI();
		getHaarROI();
	}
	
	public void getHaarROI() {
		opencv_core.Mat Img = new opencv_core.Mat();
		Cam.getCurImg().copyTo(Img);
		
		opencv_imgproc.cvtColor(Img, Img, opencv_imgproc.CV_BGR2GRAY);
		
		palmCascade.detectMultiScale(Img, palms, 1.1, 2, opencv_objdetect.CV_HAAR_SCALE_IMAGE, new opencv_core.Size(40, 40), new opencv_core.Size(500, 500));
	}
	
	public void setROI(opencv_core.Rect ROI) {
		this.ROI = ROI;
	}
	
	public opencv_core.RectVector getPalms() {
		return this.palms;
	}
	
	public opencv_core.Rect getPalm(int idx) {
		return reshapeROIRect(this.palms.get(idx), ROIRatio);
	}
	
	// Interface for HandFeatureExtraction ONLY
	public Mat getDetectedHand() {
		Cam.getCurImg().copyTo(DetectedHand);
		
		this.DetectedHand = new opencv_core.Mat(DetectedHand, this.ROI);
		
		return new Mat(this.DetectedHand.address());
	}
	
	public Camera getCam() {
		return this.Cam;
	}
	
	public opencv_core.Mat getHSVHandArea() {
		opencv_core.Mat Img = new opencv_core.Mat();
		Cam.getCurImg().copyTo(Img);

		opencv_imgproc.cvtColor(Img, Img, opencv_imgproc.CV_BGR2HSV);
		
		opencv_core.inRange(Img, hsvLower, hsvUpper, imgThreshed);
		opencv_core.inRange(Img, hsvLower2, hsvUpper2, imgThreshed2);
		opencv_core.add(imgThreshed,imgThreshed2,imgThreshed);
		
		// erode and dilate operation, can be removed if need
		opencv_imgproc.erode(imgThreshed, imgThreshed, kernel);
		opencv_imgproc.dilate(imgThreshed, imgThreshed, kernel);
		
		return imgThreshed;
	}
	
	// NOT USE NOW
	public opencv_core.Mat getForegroundHand() {
		opencv_core.Mat Img = Cam.getCurImg();
		
		BGS.apply(Img, this.foreground);
		
		
		//opencv_imgproc.erode(foreground, foreground, kernel);
		//opencv_imgproc.dilate(foreground, foreground, kernel);
		
		return this.foreground;
	}
	
	private void getHSVROI() {
		opencv_core.RotatedRect maxBox;
		opencv_core.Rect ROIRect = null;
		
		maxBox = findMaxContourRect();
		if (maxBox != null) {
			ROIRect = maxBox.boundingRect();
			ROIRect = reshapeROIRect(ROIRect, ROIRatio);
		}
		
		this.HSVRectROI = ROIRect;
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
		width = Cam.getCurImg().cols();

		hsvLower = new opencv_core.Mat(height,width,opencv_core.CV_8UC3,new Scalar(huelower1, midS-varS, midV-varV,0));
		hsvUpper = new opencv_core.Mat(height,width,opencv_core.CV_8UC3,new Scalar(midH+varH, midS+varS, midV+varV,0));
		hsvLower2 = new opencv_core.Mat(height,width,opencv_core.CV_8UC3,new Scalar(huelower2, midS-varS, midV-varV,0));
		hsvUpper2 = new opencv_core.Mat(height,width,opencv_core.CV_8UC3,new Scalar(255, midS+varS, midV+varV,0));
	}
	
	
	private opencv_core.RotatedRect findMaxContourRect() {
		float maxArea = minDetectedArea;
		opencv_core.RotatedRect maxBox = null;
		opencv_core.Mat handArea = new opencv_core.Mat();
		opencv_core.MatVector contours = new opencv_core.MatVector();
		
		getHSVHandArea().copyTo(handArea);
		
		opencv_imgproc.findContours(handArea, contours, opencv_imgproc.RETR_LIST, opencv_imgproc.CHAIN_APPROX_NONE);
		
		for (int i = 0; i < contours.size(); ++i) {
			opencv_core.RotatedRect box = opencv_imgproc.minAreaRect(contours.get(i));
			float area = box.size().height()*box.size().width();
			if(area > maxArea) {
				maxArea = area;
				maxBox = box;
			}
		}
		
		return maxBox;
	}
	
	private opencv_core.Rect reshapeROIRect(opencv_core.Rect ROIRect, float ratio) {
		int old_x = ROIRect.x();
		int old_y = ROIRect.y();
		int old_height = ROIRect.size().height();
		int old_width = ROIRect.size().width();
		
		int center_x = (int)(old_x + 0.5 * old_width);
		int center_y = (int)(old_y + 0.5 * old_height);
		
		int new_height = (int)(ratio * (old_height > old_width ? old_height : old_width));
		int new_width = new_height;
		
		if ((int)(center_x - 0.5 * new_width) < 0) {
			new_width = center_x * 2;
		}
		
		if ((int)(center_x + 0.5 * new_width) > this.width) {
			new_width = (this.width - center_x) * 2;
		}
		
		if ((int)(center_y - 0.5 * new_height) < 0) {
			new_height = center_y * 2;
		}
		
		if ((int)(center_y + 0.5 * new_height) > this.height) {
			new_height = (this.height - center_y) * 2;
		}
		
		int new_x = (int)(center_x - 0.5 * new_width);
		int new_y = (int)(center_y - 0.5 * new_height);
		
		ROIRect = new opencv_core.Rect(new_x, new_y, new_width, new_height);
	
		return ROIRect;
	}
}