package TestModel;

import java.io.File;
import org.opencv.core.*;
import org.opencv.imgproc.*;
import org.opencv.highgui.*;
import org.opencv.objdetect.HOGDescriptor;
import org.opencv.ml.*;

public class Test {
	
	private static final String[] pathsUniform = {"C:\\Users\\fuhao\\Desktop\\dataset_clean\\test\\A\\uniform",
			 									  "C:\\Users\\fuhao\\Desktop\\dataset_clean\\test\\B\\uniform",
			 									  "C:\\Users\\fuhao\\Desktop\\dataset_clean\\test\\C\\uniform",
			 									  "C:\\Users\\fuhao\\Desktop\\dataset_clean\\test\\Five\\uniform",
			 									  "C:\\Users\\fuhao\\Desktop\\dataset_clean\\test\\V\\uniform",
			 									  };
	
	private static final String[] pathsComplex = {"C:\\Users\\fuhao\\Desktop\\dataset_clean\\test\\A\\complex",
			 									  "C:\\Users\\fuhao\\Desktop\\dataset_clean\\test\\B\\complex",
			 									  "C:\\Users\\fuhao\\Desktop\\dataset_clean\\test\\C\\complex",
			 									  "C:\\Users\\fuhao\\Desktop\\dataset_clean\\test\\Five\\complex",
			 									  "C:\\Users\\fuhao\\Desktop\\dataset_clean\\test\\V\\complex",
			 									  };
	
	private static Mat testImages;
	private static Mat testData;
	private static Mat testLabels;
	private static Mat classes;
	private static Mat results;
	
	private static HOGDescriptor desc;
	private static CvSVM classifier;
	
	private static void init() {
		
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

		testImages = new Mat();
		testData = new Mat();
		testLabels = new Mat();
		classes = new Mat();
		results = new Mat();
		desc = new HOGDescriptor();
		classifier = new CvSVM();
		
		System.out.println("Init successfully!");
	}
	
	private static void loadModel() {

		classifier.load("model.xml");
		
		System.out.println("Model loaded successfully!");
	}
	
	private static void loadData(String[] Paths) {
		
		for (int i = 0; i < Paths.length; ++i) {
			for (File file : new File(Paths[i]).listFiles()) {
				Mat img = new Mat();
				img = Highgui.imread(file.getAbsolutePath(), Highgui.CV_LOAD_IMAGE_GRAYSCALE);
				
				Imgproc.resize(img, img, new Size(64, 128));
				
				MatOfFloat descVals = new MatOfFloat();
				desc.compute(img, descVals);
				
				testImages.push_back(descVals.reshape(1, 1));
				
				Mat tmp = new Mat(1, 1, CvType.CV_32FC1);
				tmp.put(0, 0, (int)i);
				
				testLabels.push_back(tmp);
			}
		}

		testLabels.copyTo(classes);
		testImages.copyTo(testData);
		testData.convertTo(testData, CvType.CV_32FC1);
		
		System.out.println("Data loaded successfully!");
	}
	
	private static void predict() {
		
		classifier.predict_all(testData, results);
		
		float correct = 0;
		float sum = results.height();
		
		for (int i = 0; i < results.height(); ++i) {
			if (results.get(i, 0)[0] == classes.get(i, 0)[0]) {
				correct++;
			}
		}
		
		System.out.println("Accuracy: " + 100 * correct / sum + "%");
		System.out.println("Predict successfully!");
	}
	
	public static void main(String[] args) {
		
		init();

		loadModel();
		
		loadData(pathsComplex);
		
		predict();
	}

}
