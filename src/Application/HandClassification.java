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
	private static final int Ges_C = 2;
	private static final int Ges_Five = 3;
	private static final int Ges_Point = 4;
	private static final int Ges_V = 5;
	
	private static final String Gesture_A = "握拳手势";
	private static final String Gesture_B = "Stop手势";
	private static final String Gesture_C = "虎口手势";
	private static final String Gesture_Five = "五指手势";
	private static final String Gesture_Point = "上指手势";
	private static final String Gesture_V = "胜利手势";
	private static final String Gesture_Err = "错误手势";
	
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
			case Ges_C:
				return Gesture_C;
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
