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
	
	public HandFeatureExtraction(HandDetection HD) {
		this.HD = HD;
		desc = new HOGDescriptor();
		descVals = new MatOfFloat();
	}
	
	public MatOfFloat getDescVals() {
		Mat grabbedImage = new Mat();
		//grabbedImage = HD.getDetectedHand();
		Imgproc.cvtColor(HD.getDetectedHand(), grabbedImage, Imgproc.COLOR_BGR2GRAY);
		Imgproc.resize(grabbedImage, grabbedImage, new Size(64,128));
		
		desc.compute(grabbedImage, descVals);
		
		return descVals;
	}
	
	public HandDetection getHD() {
		return HD;
	}
}
