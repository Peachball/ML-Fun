package regression;

import Jama.Matrix;

public class Regression {
	
	public static double ALPHA = 0.0001;
	public static double[] multivariateRegression(Data dataset){
		
		return null;
	}
	
	public static Matrix gradDescent(Matrix X, Matrix initialTheta, Matrix y, int iterations, double alpha, double lambda){
		Matrix theta = initialTheta;
		for(int i = 0; i < iterations; i++){
			theta = theta.minus(Regression.grad(X, theta, y, lambda));
		}
		return theta;
	}
	public static double[] linearRegression(Data dataset){
		double slope = 0;
		double yInt = 0;
		while(error(slope, yInt, dataset) > 0.1){
			double slopeBuffer = slope;
			double yIntBuffer = yInt;
			double change = 0;
			for(int i = 0; i < dataset.length(); i++){
				change += (dataset.getX(i) * slopeBuffer + yIntBuffer - dataset.getY(i)) * dataset.getX(i);
			}
			change *= 1.0 / dataset.length();
			slope -= change * ALPHA;
			change = 0;
			for(int i = 0; i < dataset.length(); i++){
				change += (dataset.getX(i) * slopeBuffer + yIntBuffer - dataset.getY(i));
			}
			change *= 1.0 / dataset.length();
			yInt -= change * ALPHA;
		}
		
		double[] a = new double[2];
		a[0] = slope;
		a[1] = yInt;
		return a;
	}
	
	/*
	 * Assuming theta is a column vector and that each row of X is a different example
	 */
	public static Matrix predict(Matrix X, Matrix theta){
		Matrix z = X.times(theta);
		return z;
	}
	
	public static Matrix grad(Matrix X, Matrix theta, Matrix y, double lambda){
		Matrix hypothesis = X.times(theta);
		Matrix errors = hypothesis.minus(y);
		errors = X.times(errors).times(1/X.getRowDimension());
		return errors;
	}
	public static double J(Matrix X, Matrix theta, Matrix y, double lambda){
		Matrix hypothesis = X.times(theta);
		hypothesis.minusEquals(y);
		hypothesis = hypothesis.arrayTimes(hypothesis).times(1/2/hypothesis.getRowDimension());
		double j = 0;
		double reg = 0;
		for(int i = 0; i < hypothesis.getColumnDimension(); i++){
			j += hypothesis.get(i, 0);
		}
		for(int i = 0; i < theta.getRowDimension(); i++){
			reg += theta.get(i, 0);
		}
		return j + (Math.pow(reg, 2) * lambda / X.getRowDimension());
	}
	
	@Deprecated
	private static double error(double slope, double yInt, Data dataset){
		double ans = 0;
		for(int i = 0; i < dataset.length(); i++){
			ans += Math.pow((dataset.getX(i) * slope + yInt - dataset.getY(i)), 2);
		}
		ans *= 1.0 / 2.0 / dataset.length();
		return ans;
	}
	
	public static void printMatrix(Matrix m){
		for(int i = 0; i < m.getRowDimension(); i++){
			for(int j = 0; j < m.getColumnDimension(); j++){
				System.out.print(m.get(i, j) + " ");
			}
			System.out.print('\n');
		}
	}
	
	public static Matrix findMean(Matrix X){
		Matrix means = new Matrix(1, X.getColumnDimension());
		for(int col = 0; col < X.getColumnDimension(); col++){
			for(int row = 0; row < X.getRowDimension(); row++){
				means.set(0, col, means.get(0, col) + X.get(row, col));
			}
			means.set(0, col, means.get(0, col) * 1.0 / X.getColumnDimension());
		}
		return means;
	}
	
	public static Matrix findDeviation(Matrix X, Matrix means){
		Matrix sigma = new Matrix(1, X.getColumnDimension());
		for(int col = 0; col < X.getColumnDimension(); col++){
			for(int row = 0; row < X.getRowDimension(); row++){
				sigma.set(0, col, sigma.get(0, col) + Math.abs((X.get(row, col) - means.get(0, col))));
			}
			sigma.set(0, col, sigma.get(0, col) * 1.0 / X.getColumnDimension());
		}
		return sigma;
	}
	
	public static Matrix rescale(Matrix X, Matrix means, Matrix sigma){
		Matrix newX = X.minus(means).arrayRightDivide(sigma);
		return newX;
	}
	
	public static Matrix rescale(Matrix X){
		Matrix means = findMean(X);
		return rescale(X, means, findDeviation(X, means));
	}
}