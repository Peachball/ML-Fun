package main;

import java.awt.Graphics;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;

import Jama.Matrix;
import NN.NeuralNetwork;
import misc.Mat;

public class Main {
	public static void main(String[] args){
		trafficLights("C:\\Users\\s-xuch\\Pictures\\Traffic Lights");
	}
	
	private static Matrix[] trafficLights(String dir){
		File d = new File(dir);
		File[] files = d.listFiles();
		File red, yellow, green;
		red = yellow = green = null;
		Matrix X = new Matrix(0, 250 * 250 * 3);
		Matrix y = new Matrix(0, 3);
		for(int i = 0; i < files.length; i++){
			if(files[i].isDirectory() && files[i].getName().contentEquals("red")){
				red = files[i];
			}
			if(files[i].isDirectory() && files[i].getName().contentEquals("yellow")){
				yellow = files[i];
			}
			if(files[i].isDirectory() && files[i].getName().contentEquals("green")){
				green = files[i];
			}
		}
		File[] reds, yellows, greens;
		reds = red.listFiles();
		yellows = yellow.listFiles();
		greens = green.listFiles();
		for(int r = 0; r < 1; r++){
			X = addExample(X, convertImage(reds[r].getAbsolutePath()));
			y = addExample(y, createExample(0, 3));
		}
		for(int r = 0; r < 1; r++){
			X = addExample(X, convertImage(yellows[r].getAbsolutePath()));
			y = addExample(y, createExample(1, 3));
		}
		for(int r = 0; r < 0; r++){
			X = addExample(X, convertImage(greens[r].getAbsolutePath()));
			y = addExample(y, createExample(2, 3));
		}
		
		Matrix[] theta = new Matrix[5];
		theta[0] = Matrix.random(X.getColumnDimension() + 1, 25);
		theta[1] = Matrix.random(26,25);
		theta[2] = Matrix.random(26,25);
		theta[3] = Matrix.random(26,25);
		theta[4] = Matrix.random(26,3);
		theta = NeuralNetwork.trainNN(X, theta, y, 0.03, 0, 1000);
		return theta;
	}
	
	private static Matrix createExample(int option, int totalOptions){
		Matrix y = new Matrix(totalOptions, 1);
		y.set(option, 0, 1);
		return y;
	}
	
	private static Matrix addExample(Matrix X, Matrix x){
		Matrix buffer = new Matrix(X.getRowDimension() + 1, X.getColumnDimension());
		/*
		buffer.setMatrix(0, X.getRowDimension() - 1, 0, X.getColumnDimension() - 1, X);
		Matrix j = x.transpose();
		buffer.setMatrix(X.getRowDimension(), X.getRowDimension(),0, x.getColumnDimension()-1, x.transpose());
		*/
		buffer = Mat.addBotMatrix(X, x.transpose());
		return buffer;
	}
	
	private static Matrix convertImage(String image){
		try{
		File im = new File(image);
		Image bu = ImageIO.read(im).getScaledInstance(250, 250, Image.SCALE_DEFAULT);
		BufferedImage bi = new BufferedImage(bu.getWidth(null),bu.getHeight(null),BufferedImage.TYPE_INT_ARGB);
		Graphics g = bi.createGraphics();
		g.drawImage(bu,0,0,null);
		g.dispose();
		Matrix x = new Matrix(3 * bi.getHeight() * bi.getWidth(), 1);
		System.out.println("Converting " + image + "...");
		for(int i = 0; i < bi.getWidth(); i++){
			for(int j = 0; j < bi.getHeight(); j++){
				int rgb = bi.getRGB(i, j);
				int red = (rgb >> 16) & 0xFF;
				int green = (rgb >> 8) & 0xFF;
				int blue = (rgb) & 0xFF;
				x.set(3 * (j * bi.getWidth() + i) + 0, 0, red);
				x.set(3 * (j * bi.getWidth() + i) + 1, 0, green);
				x.set(3 * (j * bi.getWidth() + i) + 2, 0, blue);
			}
		}
		return x;
		}
		catch(IOException e){
			e.printStackTrace();
			return null;
		}
	}

	private static void logRegTest(){
		//Generate sample data
		/*
		Matrix sampleX = new Matrix(100,2);
		Matrix y = new Matrix(100, 1);
		for(int i = 0; i < 100; i++){
			sampleX.set(i, 0, 1);
			sampleX.set(i, 1, Math.random() * 50);
			if(sampleX.get(i, 1) > 25){
				y.set(i, 0, 1);
			}
			else{
				y.set(i, 0, 0);
			}
		}
		
		Matrix initialTheta = new Matrix( sampleX.getColumnDimension() ,1);
		Matrix finalTheta = LogisticRegression.logRegression(sampleX, y, initialTheta, .01, 0, 1000000);
		Regression.printMatrix(initialTheta);
		Regression.printMatrix(finalTheta);
		*/
	}
}
