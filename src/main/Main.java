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
import regression.Regression;

public class Main {
	static String trafficLightWeights = "tf.txt";
	static String trafficLightDir = "C:\\Users\\s-xuch\\Pictures\\Traffic Lights";
	static int finePeter = 250;

	static final String Xtest = "t10k-images.idx3-ubyte";
	static final String ytest = "t10k-labels.idx1-ubyte";
	static final String Xlegit = "train-images.idx3-ubyte";
	static final String yLegit = "train-labels.idx1-ubyte";

	public static void main(String[] args) {
		Matrix y = Mat.readidx(yLegit, false, 0, 20000);
		Matrix X = Mat.readidx(Xlegit, true, 0, 20000);
		System.out.println("Loaded Images and labels");
		Matrix[] theta = new Matrix[4];
		try {
			theta = NeuralNetwork.read(new File("handwriting.nn"));
		} catch (IOException e) {
			theta[0] = Matrix.random(X.getColumnDimension() + 1, 100);
			theta[1] = Matrix.random(101, 100);
			theta[2] = Matrix.random(101, 100);
			theta[3] = Matrix.random(101, 10);
		}
		for (int i = 0; i < 1000; i++) {
			theta = NeuralNetwork.trainNN(X, theta, y, 0.01, 0, 10);
			try {
				System.out.println("saved");
				NeuralNetwork.write(theta, "handwriting.nn");
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
	}

	private static Matrix[] trafficLights(String dir) {
		File d = new File(dir);
		File[] files = d.listFiles();
		File red, yellow, green;
		red = yellow = green = null;
		Matrix X = new Matrix(0, finePeter * finePeter * 3);
		Matrix y = new Matrix(0, 3);
		for (int i = 0; i < files.length; i++) {
			if (files[i].isDirectory() && files[i].getName().contentEquals("red")) {
				red = files[i];
			}
			if (files[i].isDirectory() && files[i].getName().contentEquals("yellow")) {
				yellow = files[i];
			}
			if (files[i].isDirectory() && files[i].getName().contentEquals("green")) {
				green = files[i];
			}
		}
		File[] reds, yellows, greens;
		reds = red.listFiles();
		yellows = yellow.listFiles();
		greens = green.listFiles();
		for (int r = 0; r < reds.length; r++) {
			X = addExample(X, convertImage(reds[r].getAbsolutePath()));
			y = addExample(y, createExample(0, 3));
		}
		for (int r = 0; r < yellows.length; r++) {
			X = addExample(X, convertImage(yellows[r].getAbsolutePath()));
			y = addExample(y, createExample(1, 3));
		}
		for (int r = 0; r < greens.length; r++) {
			X = addExample(X, convertImage(greens[r].getAbsolutePath()));
			y = addExample(y, createExample(2, 3));
		}
		Regression.rescale(X);

		Matrix[] theta = new Matrix[3];
		theta[0] = Matrix.random(X.getColumnDimension() + 1, 100);
		theta[1] = Matrix.random(101, 100);
		theta[2] = Matrix.random(101, 3);
		theta = NeuralNetwork.trainNN(X, theta, y, 0.1, 0, 100);
		try {
			NeuralNetwork.write(theta, trafficLightWeights);
		} catch (IOException e) {
			e.printStackTrace();
		}
		Matrix p = NeuralNetwork.predict(X, theta, 1);
		p.print(1, 3);
		return theta;
	}

	private static Matrix createExample(int option, int totalOptions) {
		Matrix y = new Matrix(totalOptions, 1);
		y.set(option, 0, 1);
		return y;
	}

	private static Matrix addExample(Matrix X, Matrix x) {
		Matrix buffer = new Matrix(X.getRowDimension() + 1, X.getColumnDimension());
		buffer = Mat.addBotMatrix(X, x.transpose());
		return buffer;
	}

	private static Matrix convertImage(String image) {
		try {
			File im = new File(image);
			Image bu = ImageIO.read(im).getScaledInstance(finePeter, finePeter, Image.SCALE_DEFAULT);
			BufferedImage bi = new BufferedImage(bu.getWidth(null), bu.getHeight(null), BufferedImage.TYPE_INT_ARGB);
			Graphics g = bi.createGraphics();
			g.drawImage(bu, 0, 0, null);
			g.dispose();
			Matrix x = new Matrix(3 * bi.getHeight() * bi.getWidth(), 1);
			System.out.println("Converting " + image + "...");
			for (int i = 0; i < bi.getWidth(); i++) {
				for (int j = 0; j < bi.getHeight(); j++) {
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
		} catch (IOException e) {
			e.printStackTrace();
			return null;
		}
	}

	private static void logRegTest() {
		// Generate sample data
		/*
		 * Matrix sampleX = new Matrix(100,2); Matrix y = new Matrix(100, 1);
		 * for(int i = 0; i < 100; i++){ sampleX.set(i, 0, 1); sampleX.set(i, 1,
		 * Math.random() * 50); if(sampleX.get(i, 1) > 25){ y.set(i, 0, 1); }
		 * else{ y.set(i, 0, 0); } }
		 * 
		 * Matrix initialTheta = new Matrix( sampleX.getColumnDimension() ,1);
		 * Matrix finalTheta = LogisticRegression.logRegression(sampleX, y,
		 * initialTheta, .01, 0, 1000000); Regression.printMatrix(initialTheta);
		 * Regression.printMatrix(finalTheta);
		 */
	}
}
