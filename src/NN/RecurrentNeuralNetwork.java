package NN;

import Jama.Matrix;
import misc.Mat;

public class RecurrentNeuralNetwork {
	
	
	public Matrix[] theta;
	
	//These are actually vectors
	public Matrix[] prevNodes;
	
	public Matrix predict(Matrix X){
		for(int i = 0; i < X.getRowDimension(); i++){
			Matrix a = X.getMatrix(i, i, 0, X.getColumnDimension());
		}
		return null;
	}
	
	public void reset(){
		
	}
}
