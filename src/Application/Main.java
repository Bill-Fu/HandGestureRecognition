package Application;

import java.io.File;

import org.opencv.core.*;
import org.opencv.imgproc.*;
import org.opencv.highgui.*;
import org.opencv.objdetect.HOGDescriptor;
import org.opencv.ml.*;

import org.bytedeco.javacv.*;

public class Main {
	
	private static final int virtualCam = 0;
	private static final int webCam = 1;

	public static void main(String[] args) throws Exception {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		
		Camera Cam = new Camera(virtualCam);
		HandDetection HD = new HandDetection(Cam);
		HandFeatureExtraction HFE = new HandFeatureExtraction(HD);
		HandClassification HC = new HandClassification(HFE, "model.xml");
		UserInterface UI = new UserInterface(HC);
		
		System.out.println(HC.getGesture());
		
		UI.showFrame();
		
		while (UI.getCanvas().isVisible() && HD.getDetectedHand() != null) {
			Cam.updateFrame();
			UI.showFrame();
			System.out.println(HC.getGesture());
		}
		
		Cam.closeCamera();
		UI.closeFrame();
	}

}
