package regression;

import Jama.Matrix;

public class LogisticRegression {
	/*
	 * I think this works!
	 */
	public static Matrix sigmoid(Matrix m){
		Matrix s = new Matrix(m.getRowDimension(), m.getColumnDimension());
		for(int i = 0; i < s.getRowDimension(); i++){
			for(int j = 0; j < s.getColumnDimension(); j++){
				s.set(i, j, (1/(1+Math.pow(Math.E,-m.get(i,j)))));
			}
		}
		return s;
	}
	
	public static Matrix sigmoidGradient(Matrix m){
		Matrix ones = new Matrix(m.getRowDimension(), m.getColumnDimension());
		for(int i = 0; i < ones.getRowDimension(); i++){
			for(int j = 0; j < ones.getColumnDimension(); j++){
				ones.set(i, j, 1);
			}
		}
		return (sigmoid(m).times(-1).plus(ones).arrayTimes(sigmoid(m)));
	}
	
	public static double J(Matrix X, Matrix y, Matrix theta, double lambda){
		Matrix hypothesis = sigmoid(X.times(theta));
		double j = 0;
		for(int i = 0; i < X.getRowDimension(); i++){
			j += (-y.get(i, 0) * Math.log(hypothesis.get(i, 0))) - ((1-y.get(i, 0)) * Math.log(1-hypothesis.get(i, 0)));
		}
		j = j / X.getRowDimension() / 2;
		double reg = 0;
		for(int i = 1; i < theta.getRowDimension(); i++){
			reg += theta.get(i, 0) * theta.get(i, 0);
		}
		reg = reg * lambda / 2 / X.getRowDimension();
		return j;
	}
	
	public static Matrix grad(Matrix X, Matrix y, Matrix theta, double lambda){
		Matrix hypothesis = sigmoid(X.times(theta));
		Matrix buffer = hypothesis.minus(y);
		buffer = buffer.transpose().times(X).transpose();
		buffer = buffer.times(1.0/X.getRowDimension());
		for(int i = 1; i < theta.getRowDimension(); i++){
			buffer.set(i, 0, buffer.get(i,0) + lambda * theta.get(i,0) / X.getRowDimension());
		}
		return buffer;
	}
	
	public static Matrix logRegression(Matrix X, Matrix y, Matrix theta, double alpha, double lambda, int iterations){
		Matrix newTheta = new Matrix(theta.getArray());
		for(int i = 0; i < iterations; i++){
			System.out.println("Error after " + i + ":" + J(X,y,newTheta,lambda));
			newTheta = newTheta.minus(grad(X,y,newTheta,lambda).times(alpha));
		}
		return newTheta;
	}
	
	public static Matrix predict(Matrix X, Matrix theta){
		return sigmoid(X.times(theta));
	}
	
}
