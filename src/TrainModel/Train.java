package TrainModel;

import java.io.File;
import org.opencv.core.*;
import org.opencv.imgproc.*;
import org.opencv.highgui.*;
import org.opencv.objdetect.HOGDescriptor;
import org.opencv.ml.*;

public class Train {

	private static final String[] paths1 = {
			"C:\\Users\\fuhao\\Desktop\\dataset_clean\\train\\A",
			"C:\\Users\\fuhao\\Desktop\\dataset_clean\\train\\B",
			"C:\\Users\\fuhao\\Desktop\\dataset_clean\\train\\Five",
			"C:\\Users\\fuhao\\Desktop\\dataset_clean\\train\\Point",
			"C:\\Users\\fuhao\\Desktop\\dataset_clean\\train\\V",
	};
	
	private static final String[] paths2 = {
			"C:\\Users\\wb-fh265231\\Dropbox\\graduation_project\\dataset_clean\\train\\A",
			"C:\\Users\\wb-fh265231\\Dropbox\\graduation_project\\dataset_clean\\train\\B",
			"C:\\Users\\wb-fh265231\\Dropbox\\graduation_project\\dataset_clean\\train\\Five",
			"C:\\Users\\wb-fh265231\\Dropbox\\graduation_project\\dataset_clean\\train\\Point",
			"C:\\Users\\wb-fh265231\\Dropbox\\graduation_project\\dataset_clean\\train\\V",
	};
	
	private static Mat trainingImages;
	private static Mat trainingData;
	private static Mat trainingLabels;
	private static Mat classes;
	
	private static CvSVM classifier;
	private static CvSVMParams params;
	private static HOGDescriptor desc;
	
	private static void init() {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		
		classes = new Mat();
		trainingData = new Mat();
		trainingImages = new Mat();
		trainingLabels = new Mat();
		classifier = new CvSVM();
		desc = new HOGDescriptor();
		params = new CvSVMParams();
		
		System.out.println("Init successfully!");
	}
	
	private static void loadData() {
		
		for (int i = 0; i < paths2.length; ++i) {
			for (File file : new File(paths2[i]).listFiles()) {
				Mat img = new Mat();
				img = Highgui.imread(file.getAbsolutePath(), Highgui.CV_LOAD_IMAGE_GRAYSCALE);
				
				Imgproc.resize(img, img, new Size(64, 128));
				
				MatOfFloat descVals = new MatOfFloat();
				desc.compute(img, descVals);
				
				trainingImages.push_back(descVals.reshape(1, 1));
				
				Mat tmp = new Mat(1, 1, CvType.CV_32FC1);
				tmp.put(0, 0, (int)i);
				
				trainingLabels.push_back(tmp);
			}
		}
		
		trainingLabels.copyTo(classes);
		trainingImages.copyTo(trainingData);
		trainingData.convertTo(trainingData, CvType.CV_32FC1);
		
		System.out.println("Data loaded successfully!");
	}
	
	private static void trainModel() {
		params.set_kernel_type(CvSVM.LINEAR);
		params.set_svm_type(CvSVM.C_SVC);
		
		classifier.train_auto(trainingData, classes, new Mat(), new Mat(), params);
		
		System.out.println("Model trained successfully!");
	}
	
	private static void saveModel() {
		classifier.save("model.xml");
		
		System.out.println("Model saved successfully!");
	}
	
	public static void main(String[] args) {
		init();
		
		loadData();
		
		trainModel();
		
		saveModel();
		
	}

}
