package Application;

import java.io.File;

import org.opencv.core.*;
import org.opencv.imgproc.*;
import org.opencv.highgui.*;
import org.opencv.objdetect.HOGDescriptor;
import org.opencv.ml.*;
import org.bytedeco.javacpp.*;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacv.*;

public class Main {
	
	private static final int virtualCam = 0;
	private static final int webCam = 1;

	public static void main(String[] args) throws Exception {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		
		Camera Cam = new Camera(webCam);
		HandDetection HD = new HandDetection(Cam);
		HandFeatureExtraction HFE = new HandFeatureExtraction(HD);
		HandClassification HC = new HandClassification(HFE, "model.xml");
		UserInterface UI = new UserInterface(HC);
		
		
		//----------------------Test----------------------
		UI.showResult();
		UI.showHandRegion();
		//UI.showForeground();
		
		opencv_core.Mat tmpMat = HD.getHSVHandArea();
		BytePointer Ptr;
		
		for (int i = 0; i < tmpMat.rows(); ++i) {
			for (int j = 0; j< tmpMat.cols(); ++j) {
				Ptr = tmpMat.ptr(i, j);
				System.out.print(Ptr.get(0));
			}
		}
			
		//-------------------------------------------------
			
		while (UI.getCanvas().isVisible() && HD.getDetectedHand() != null) {
			UI.showResult();
			UI.showHandRegion();
			//UI.showForeground();
			//System.out.println("height: " + Cam.getCurImg().rows());
			//System.out.println("width: " + Cam.getCurImg().cols());
		}
		
		Cam.closeCamera();
		UI.closeFrame();
	}

}
