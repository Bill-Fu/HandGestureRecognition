package Application;

import java.io.File;

import org.bytedeco.javacv.*;
import org.opencv.core.*;
import org.opencv.imgproc.*;
import org.opencv.highgui.*;
import org.opencv.objdetect.HOGDescriptor;
import org.opencv.ml.*;

public class HandClassification {
	
	private static final int Ges_A = 0;
	private static final int Ges_B = 1;
	private static final int Ges_Five = 2;
	private static final int Ges_Point = 3;
	private static final int Ges_V = 4;
	
	private static final String Gesture_A = "Fist Gesture";
	private static final String Gesture_B = "Stop Gesture";
	private static final String Gesture_Five = "Five Gesture";
	private static final String Gesture_Point = "Point Gesture";
	private static final String Gesture_V = "Victory Gesture";
	private static final String Gesture_Err = "Unknown Gesture";
	
	private HandFeatureExtraction HFE;
	private CvSVM SVM;
	private float Score;
	
	public HandClassification(HandFeatureExtraction HFE, String Model) {
		this.HFE = HFE;
		SVM = new CvSVM();
		SVM.load(Model);
	}
	
	public float getScore() {
		
		return SVM.predict(HFE.getDescVals().reshape(1, 1));
	}
	
	public HandFeatureExtraction getHFE() {
		
		return HFE;
	}
	
	public String getGesture() {
		
		int curGesture = (int)getScore();
		
		switch(curGesture){
			case Ges_A:
				return Gesture_A;
			case Ges_B:
				return Gesture_B;
			case Ges_Five:
				return Gesture_Five;
			case Ges_Point:
				return Gesture_Point;
			case Ges_V:
				return Gesture_V;
			default:
				return Gesture_Err;
		}
		
	}
}
