package Application;

import org.bytedeco.javacv.CanvasFrame;

public class UserInterface {
	
	private CanvasFrame Canvas;
	private HandClassification HC;
	
	public UserInterface(HandClassification HC) {
		
		this.HC = HC;
		Canvas = new CanvasFrame("手势识别", CanvasFrame.getDefaultGamma()/HC.getHFE().getHD().getCam().getFrameGrabber().getGamma());
	}
	
	public void showFrame() {
		
		Canvas.showImage(HC.getHFE().getHD().getCam().getCurFrame());
	}
	
	public void closeFrame() {
		
		Canvas.dispose();
	}
	
	public CanvasFrame getCanvas() {
		
		return Canvas;
	}
}
