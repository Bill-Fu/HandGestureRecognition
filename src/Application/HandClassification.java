package Application;

import java.io.File;
import java.util.*;

import org.bytedeco.javacv.*;
import org.opencv.core.*;
import org.opencv.imgproc.*;
import org.opencv.highgui.*;
import org.opencv.objdetect.HOGDescriptor;
import org.opencv.ml.*;

public class HandClassification {
	private static final int maxBufferSize = 20;
	private static final int gestureNum = 5;
	
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
	private LinkedList<Integer> Buffer;
	private int[] Stat;
	
	public HandClassification(HandFeatureExtraction HFE, String Model) {
		this.HFE = HFE;
		SVM = new CvSVM();
		Buffer = new LinkedList<Integer>();
		Stat = new int[gestureNum];
		
		for(int i = 0; i < Stat.length; ++i) {
			Stat[i] = 0;
		}
		
		SVM.load(Model);
	}
	
	private float getScore() {
		return SVM.predict(HFE.getDescVals().reshape(1, 1));
	}
	
	public HandFeatureExtraction getHFE() {
		return HFE;
	}
	
	public String getGesture() {
		int curGesture = (int)getScore();
		
		switch(getStableGesture(curGesture)){
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
	
	private int getStableGesture(int curGesture) {
		Buffer.add(curGesture);
		int index = -1;
		int tmpMax = -1;
		
		if(Buffer.size() > maxBufferSize) {
			Buffer.remove();
		}
		
		for (int i = 0; i < Stat.length; ++i) {
			Stat[i] = 0;
		}
		
		for (int i: Buffer) {
			Stat[i]++;
		}
		
		for (int i = 0; i < Stat.length; ++i) {
			if(Stat[i] > tmpMax) {
				tmpMax = Stat[i];
				index = i;
			}
		}
		
		return index;
	}
}
