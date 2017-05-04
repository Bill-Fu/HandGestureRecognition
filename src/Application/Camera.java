package Application;

import java.io.File;

import org.bytedeco.javacv.*;
import org.opencv.core.*;
import org.opencv.imgproc.*;
import org.opencv.highgui.*;
import org.opencv.objdetect.HOGDescriptor;
import org.opencv.ml.*;

public class Camera {

	private FrameGrabber grabber;
	private Frame curFrame;
	
	public Camera(int type) throws Exception {
		
		//default type is 0, it depends on your computer device settings, you can also add virtual camera for test
		grabber = FrameGrabber.createDefault(type);

		grabber.start();
		
		curFrame = grabber.grab();
	}
	
	public void updateFrame() throws Exception {
		
		curFrame = grabber.grab();
	}
	
	public Frame getCurFrame() {
		
		return curFrame;
	}
	
	public FrameGrabber getFrameGrabber() {
		
		return grabber;
	}
	
	public void closeCamera() throws Exception {
		
		grabber.stop();
	}
}
