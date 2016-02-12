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
