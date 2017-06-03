package Application;

import java.io.File;

import org.opencv.core.*;
import org.opencv.imgproc.*;
import org.opencv.highgui.*;
import org.opencv.objdetect.HOGDescriptor;
import org.opencv.ml.*;

import org.bytedeco.javacv.*;


public class HandFeatureExtraction {
	
	private HandDetection HD;
	private HOGDescriptor desc;
	private MatOfFloat descVals;
	private Mat grabbedImage;
	
	public HandFeatureExtraction(HandDetection HD) {
		this.HD = HD;
		this.desc = new HOGDescriptor();
		this.descVals = new MatOfFloat();
		this.grabbedImage = new Mat();
	}
	
	public MatOfFloat getDescVals() {
		HD.getDetectedHand().copyTo(grabbedImage);
		
		Imgproc.cvtColor(grabbedImage, grabbedImage, Imgproc.COLOR_BGR2GRAY);
		Imgproc.resize(grabbedImage, grabbedImage, new Size(64,128));
		
		desc.compute(grabbedImage, descVals);
		
		return descVals;
	}
	
	public HandDetection getHD() {
		return HD;
	}
}
