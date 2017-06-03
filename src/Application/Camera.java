package Application;

import java.io.File;

import org.bytedeco.javacv.*;
import org.bytedeco.javacpp.opencv_videoio;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_video;
import org.bytedeco.javacpp.opencv_videostab;
import org.opencv.core.*;
import org.opencv.imgproc.*;
import org.opencv.highgui.*;
import org.opencv.objdetect.HOGDescriptor;
import org.opencv.ml.*;

public class Camera {

	private opencv_videoio.VideoCapture capture;
	private opencv_core.Mat curImg;
	
	public Camera(int type) throws Exception {
		this.curImg = new opencv_core.Mat();
		this.capture = new opencv_videoio.VideoCapture(type);
		
		this.capture.read(this.curImg);
	}
	
	public opencv_core.Mat getCurImg() {
		return this.curImg;
	}
	
	public void updateCurImg() {
		this.capture.read(this.curImg);
	}
	
	public void closeCamera() throws Exception {
		capture.close();
	}
	
}
