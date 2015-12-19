package NN;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;

import org.apache.commons.io.FileUtils;

import Jama.Matrix;
import misc.Mat;
import regression.LogisticRegression;

public class NeuralNetwork {

	/*
	 * it is assumed that the bias unit is not added in X
	 * Theta should be a ton of column vectors
	 */
	public static Matrix predict(Matrix X, Matrix[] theta, int type){
		Matrix a = X;
		a = Mat.add1sColumn(a);
		for(int i = 0; i < theta.length; i++){
			a = a.times(theta[i]);
			if(type == 1)
			a = LogisticRegression.sigmoid(a); //idk maybe I'll do linear things later
			if(type == 2)
				;
			a = Mat.add1sColumn(a); // Add bias unit
		}
		a = Mat.remove1stColumn(a); // Remove bias unit
		return a;
	}
	
	public static Matrix[] grad(Matrix X, Matrix[] theta, Matrix y){
		return grad(X,theta,y,0);
	}
	public static Matrix[] grad(Matrix X, Matrix[] theta, Matrix y, double lambda){ // Theta is 0 indexed
		Matrix[] grad = new Matrix[theta.length]; //0 indexed
		Matrix[] a = new Matrix[theta.length+1]; //1 indexed
		Matrix[] z = new Matrix[theta.length]; //0 indexed
		Matrix[] delta = new Matrix[theta.length]; //0 indexed
		a[0] = X;
		a[0] = Mat.add1sColumn(a[0]);
		for(int i = 0; i < theta.length; i++){
			z[i] = a[i].times(theta[i]);
			a[i+1] = LogisticRegression.sigmoid(z[i]);
			a[i+1] = Mat.add1sColumn(a[i+1]);
		}
		//Calculated activation functions!
		
		a[a.length-1] = Mat.remove1stColumn(a[a.length-1]);
		delta[delta.length-1] = a[a.length-1].minus(y);
		for(int i = delta.length-2; i >= 0; i--){
			delta[i] = delta[i+1].times(theta[i+1].transpose());
			delta[i] = Mat.remove1stColumn(delta[i]);
			delta[i] = delta[i].arrayTimes(LogisticRegression.sigmoidGradient(z[i]));
		}
		
		for(int i = 0; i < grad.length; i++){
			grad[i] = delta[i].transpose().times(a[i]).transpose();
			for(int j = 0; j < grad[i].getRowDimension(); j++){
				for(int k = 1; k < grad[i].getColumnDimension(); k++){
					grad[i].set(j, k, grad[i].get(j, k) + theta[i].get(j, k) * lambda);
				}
			}
			grad[i] = grad[i].times(1.0 / X.getRowDimension());
		}
		return grad;
	}
	
	public static double J(Matrix X, Matrix[] theta, Matrix y, double lambda){
		double m = X.getRowDimension();
		double j = 0;
		Matrix p = predict(X, theta, 1);
		for(int i = 0; i < p.getRowDimension(); i++){
			for(int k = 0; k < p.getColumnDimension(); k++){
				j += (-y.get(i, k) * Math.log(p.get(i, k)) - (1-y.get(i, k)) * Math.log(1 - p.get(i, k))) / m;
			}
		}
		double reg = 0;
		for(int i = 0; i < theta.length; i++){
			for(int a = 1; a < theta[i].getRowDimension(); a++){
				for(int b = 0; b < theta[i].getColumnDimension(); b++){
					reg += theta[i].get(a,b) * theta[i].get(a, b) * lambda / 2.0 / m;
				}
			}
		}
		return reg + j;
	}
	
	public static Matrix[] trainNN(Matrix X, Matrix[] inittheta, Matrix y, double alpha, double lambda, int iterations){
		Matrix[] theta = inittheta;
		System.out.println(J(X, theta, y, lambda));
		for(int i = 0; i < iterations; i++){
			Matrix[] grad = grad(X, theta, y);
			for(int j = 0; j < grad.length; j++){
				theta[j] = theta[j].minus(grad[j].times(alpha));
			}
			System.out.println(J(X, theta, y, lambda));
		}
		return theta;
	}

	public static void write(Matrix[] m, String file) throws IOException{
		PrintWriter writer = new PrintWriter(new FileWriter(file));
		writer.print("");
		writer.close();
		for(int i = 0; i < m.length; i++){
			PrintWriter out = new PrintWriter(new FileWriter(file, true));
			out.println("Matrix " + i);
			m[i].print(out, 1, 10);
			out.close();
		}
	}
	
	public static Matrix[] read(File file) throws IOException{
		if(file.isDirectory()){
			throw new IOException("Can't split directories");
		}
		File temp = new File("temp");
		if(temp.exists()){
			int i = 0;
			while(temp.exists()){
				temp = new File("temp" + i);
				i++;
			}
		}
		temp.mkdir();

		BufferedReader in = new BufferedReader(new FileReader(file));
		String name = in.readLine();
		while((name) != null){
			PrintWriter out = new PrintWriter(new FileWriter(temp.getPath() + File.separator + name));
			String buffer;
			while((buffer = in.readLine()) != null){
				if(buffer.contains("Matrix")){
					break;
				}
				out.println(buffer);
			}
			out.close();
			name = buffer;
		}
		
		ArrayList<Matrix> m = new ArrayList<Matrix>();
		File[] files = temp.listFiles();
		for(int i = 0; i < files.length; i++){
			BufferedReader reader = new BufferedReader(new FileReader(files[i]));
			m.add(Matrix.read(reader));
			reader.close();
		}
		in.close();
		FileUtils.deleteDirectory(temp);
		

		return m.toArray(new Matrix[m.size()]);
	}
	
	public static void testGradient(Matrix X, Matrix y){
		double epsilon = 0.00000001;
		Matrix[] theta1 = new Matrix[1];
		theta1[0] = Mat.ones(X.getColumnDimension() + 1, y.getColumnDimension());
		theta1[0].set(0, 0, 1+epsilon);
		Matrix[] theta2 = new Matrix[1];
		theta2[0] = Mat.ones(X.getColumnDimension() + 1, y.getColumnDimension());
		theta2[0].set(0, 0, 1-epsilon);

		double testGrad = J(X, theta1, y, 0) - J(X, theta2, y, 0);
		testGrad = testGrad / 2 / epsilon;
		Matrix[] test3 = new Matrix[1];
		test3[0] = Mat.ones(X.getColumnDimension() + 1, y.getColumnDimension());
		Matrix[] ograd = grad(X, test3, y);
		System.out.println(ograd[0].get(0, 0) + " vs " + testGrad);
		System.out.println(ograd[0].get(0, 0) - testGrad);
	}
	
	public static void testGradient(){
		testGradient(Matrix.random(3, 6), Matrix.random(3, 2));
	}
}