package misc;

import java.awt.Color;

import Jama.Matrix;
import NN.NeuralNetwork;

public class Snake {
	
	/**
	 * Test things here
	 * @param args
	 */
	public static void main(String[] args){
		int[][] board = {{1, 0, 1},
				{0, 1, 0},
				{1, 0, 1}};
		display(board);

	}
	/**
	 * Assuming that you are using a multilayer neural network
	 */
	public static double F(Matrix[] theta, int x, int y, boolean display){
		boolean alive = true;
		int[][] board = new int[x][y];
		addApple(board);
		board[0][0] = 1;
		int status = 0;
		int xHead = 0;
		int yHead = 0;
		while(status >= 0){
			Matrix X = convert(board);
			int nextDir = 0;
			double max = 0;
			Matrix result = NeuralNetwork.predict(X, theta, 1);
			for(int i = 0; i < 4; i++){
				if(result.get(0, i) > max){
					max = result.get(0, i);
					nextDir = i + 1;
				}
			}
			if(display){
				display(board);
			}
			int s = nextIteration(board, nextDir);
			if(s < 0){
				return -s;
			}
		}
		return 0;
	}
	
	/**
	 * dir: 1 is up, 2, is right, 3 is down, 4 is left
	 * exit statuses: 0 is nothing, + is got an apple, and - is loss (died)
	 * If the return value is not 0, then it is returning the length of the snake
	 * @return Exit status
	 * @param board
	 * @param dir
	 */
	private static int nextIteration(int[][] board, int dir){
		int size = 0;
		int xHead = 0;
		int yHead = 0;
		for(int i = 0; i < board.length; i++){
			for(int j = 0; j < board[i].length; j++){
				if(board[i][j] > 0){
					if(board[i][j] > size){
						size = board[i][j];
						xHead = i;
						yHead = j;
					}
					board[i][j]--;
				}
			}
		}
		int xSize = board.length;
		int ySize = board[0].length;
		switch(dir){
		case 1:
			if(yHead + 1 >= ySize){
				yHead = 0;
			}
			else{
				yHead++;
			}
			break;
		case 2:
			if(xHead + 1 >= xSize){
				xHead = 0;
			}
			else{
				xHead++;
			}
			break;
		case 3:
			if(yHead - 1 < 0){
				yHead = ySize - 1;
			}
			else{
				yHead--;
			}
			break;
		case 4:
			if(xHead - 1 < 0){
				xHead = ySize - 1;
			}
			else{
				xHead--;
			}
			break;
		}
		if(board[xHead][yHead] == -1){
			board[xHead][yHead] = size + 1;
			return size;
		}
		else if(board[xHead][yHead] > 0){
			return -size;
		}
		else{
			board[xHead][yHead] = size;
		}
		return 0;
	}
	
	private static void addApple(int[][] board){
		int spaces = 0;
		for(int i = 0; i < board.length; i++){
			for(int j = 0; j < board[i].length; j++){
				if(board[i][j] == 1){
					spaces++;
				}
			}
		}
		double rng = Math.random() * (board.length * board[0].length - spaces);
		spaces = 0;
		for(int i = 0; i < board.length; i++){
			for(int j = 0; j < board[i].length; j++){
				if(spaces - rng < 1){
					board[i][j] = -1;
					break;
				}
			}
		}
	}

	private static Matrix convert(int[][] board){
		Matrix m = new Matrix(3 * board.length * board[0].length, 1);
		double xSize = board.length;
		double ySize = board[0].length;
		int maxSize = 0;
		int maxX = 0;
		int maxY = 0;
		for(int i = 0; i < board.length; i++){
			for(int j = 0; j < board[i].length; j++){
				if(board[i][j] > 0) m.set(3 * (i * board[i].length + j), 0, 1);
				if(board[i][j] == -1) m.set(3 * (i * board[i].length + j) + 1, 0, 1);
				if(board[i][j] > maxSize){
					maxSize = board[i][j];
					maxX = i;
					maxY = j;
				}
			}
		}
		m.set(3 * (maxX * board[0].length + maxY) + 2, 0, 1);
		return m;
	}
	
	public static void display(int[][] board){
		double xSize = board.length;
		double ySize = board[0].length;
		StdDraw.setScale();
		for(int i = 0; i < board.length; i++){
			for(int j = 0; j < board[i].length; j++){
				switch(board[i][j]){
				case 1:
					StdDraw.setPenColor();
					StdDraw.filledRectangle((i + 0.5) / xSize, (j + 0.5) / ySize, 0.5 / xSize, 0.5 / ySize);
					break;
				case 2:
					StdDraw.setPenColor(Color.RED);
					StdDraw.filledRectangle((i + 0.5) / xSize, (j + 0.5) / ySize, 0.5 / xSize, 0.5 / ySize);
					break;
				}
			}
		}
	}
}
