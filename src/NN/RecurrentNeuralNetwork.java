package NN;

import Jama.Matrix;
import misc.Mat;
import regression.LogisticRegression;

public class RecurrentNeuralNetwork {
	
	
	public Matrix[] theta;
	
	//These are actually vectors
	public Matrix[] prevNodes;
	
	public Matrix predict(Matrix X){
		X = Mat.add1sColumn(X);
		for(int i = 0; i < X.getRowDimension(); i++){
			Matrix a = X.getMatrix(i, i, 0, X.getColumnDimension());
			for(int j = 0; j < theta.length; j++){
				a = Mat.append(a, prevNodes[i]);
				Matrix a2 = a.times(theta[j]);
				a2 = LogisticRegression.sigmoid(a2);
				a2 = Mat.add1sColumn(a2);
				a = a2;
			}
			a = Mat.remove1stColumn(a);
		}
		return a;

	}
	
	
	/*
	 * THIS IS ONLY FOR ONE INPUT SET!
	 */
	//Each row represents a new input time
	//Each column is a feature at said input time
	public Matrix[] grad(Matrix X, Matrix y){
		Matrix[] gradient = new Matrix[theta.length];
		return null;
	}
	
	
	public void reset(){
		for(int i = 0; i < prevNodes.length; i++){
			prevNodes[i] = new Matrix(prevNodes[i].getRowDimension(), prevNodes[i].getColumnDimension());
		}
	}
}
