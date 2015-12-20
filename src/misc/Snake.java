package misc;

import java.awt.Color;

import Jama.Matrix;
import evolutionary.NEAT;
import evolutionary.NEAT.FitnessFunction;
import evolutionary.NEAT.Genome;
import evolutionary.NEAT.NEATException;

public class Snake {

	public static int defaultspeed = 50;
	public static int maxIdleTime = 100;

	/**
	 * Test things here
	 * 
	 * @param args
	 * @throws NEATException
	 */
	public static void main(String[] args) throws NEATException {
		int boardX, boardY;
		boardX = boardY = 10;
		NEAT ne = new NEAT(3 * boardX * boardY, 4, (Genome g) -> F(g, boardX, boardY, true , 0));
		while(true){
			ne.reproduce();
		}
	}

	public static double F(Genome theta, int x, int y, boolean display, int speed) {
		int idleTime = 0;
		boolean alive = true;
		int[][] board = new int[x][y];
		addApple(board);
		board[0][0] = 1;
		int status = 0;
		int xHead = 0;
		int yHead = 0;
		while (true) {
			Matrix X = convert(board);
			int nextDir = 0;
			double max = 0;
			try {
				Matrix result = theta.predict(X);
				for (int i = 0; i < 4; i++) {
					if (result.get(0, i) > max) {
						max = result.get(0, i);
						nextDir = i + 1;
					}
				}
			} catch (NEATException e) {
				e.printStackTrace();
				return 0;
			}
			if (display) {
				try {
					Thread.sleep(speed);
				} catch (InterruptedException e) {
					e.printStackTrace();
				}
				StdDraw.clear();
				display(board);
			}
			int s = nextIteration(board, nextDir);
			if (s < 0) {
				System.out.println("You died with " + (-s) + " points");
				return -s;
			}
			if (s > 0) {
				idleTime = 0;
				if (status < s) {
					status = s;
				}
				addApple(board);
			} else {
				idleTime++;
			}
			if (idleTime > 50) {
				System.out.println("Didn't get the apple enough...");
				return status;
			}
		}
	}

	/**
	 * dir: 1 is up, 2, is right, 3 is down, 4 is left exit statuses: 0 is
	 * nothing, + is got an apple, and - is loss (died) If the return value is
	 * not 0, then it is returning the length of the snake
	 * 
	 * @return Exit status
	 * @param board
	 * @param dir
	 */
	private static int nextIteration(int[][] board, int dir) {
		int size = 0;
		int xHead = 0;
		int yHead = 0;
		for (int i = 0; i < board.length; i++) {
			for (int j = 0; j < board[i].length; j++) {
				if (board[i][j] > 0) {
					if (board[i][j] > size) {
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
		switch (dir) {
		case 1:
			if (yHead + 1 >= ySize) {
				yHead = 0;
			} else {
				yHead++;
			}
			break;
		case 2:
			if (xHead + 1 >= xSize) {
				xHead = 0;
			} else {
				xHead++;
			}
			break;
		case 3:
			if (yHead - 1 < 0) {
				yHead = ySize - 1;
			} else {
				yHead--;
			}
			break;
		case 4:
			if (xHead - 1 < 0) {
				xHead = ySize - 1;
			} else {
				xHead--;
			}
			break;
		}
		if (board[xHead][yHead] == -1) {
			board[xHead][yHead] = size + 1;
			return size;
		} else if (board[xHead][yHead] > 0) {
			return -size;
		} else {
			board[xHead][yHead] = size;
		}
		return 0;
	}

	private static void addApple(int[][] board) {
		int spaces = 0;
		for (int i = 0; i < board.length; i++) {
			for (int j = 0; j < board[i].length; j++) {
				if (board[i][j] == 1) {
					spaces++;
				}
			}
		}
		double rng = Math.random() * (board.length * board[0].length - spaces);
		spaces = 0;
		for (int i = 0; i < board.length; i++) {
			for (int j = 0; j < board[i].length; j++) {
				if(board[i][j] > 0){
					continue;
				}
				if (rng - spaces < 1) {
					board[i][j] = -1;
					return;
				}
				spaces++;
			}
		}
	}

	private static Matrix convert(int[][] board) {
		Matrix m = new Matrix(3 * board.length * board[0].length, 1);
		double xSize = board.length;
		double ySize = board[0].length;
		int maxSize = 0;
		int maxX = 0;
		int maxY = 0;
		for (int i = 0; i < board.length; i++) {
			for (int j = 0; j < board[i].length; j++) {
				if (board[i][j] > 0)
					m.set(3 * (i * board[i].length + j), 0, 1);
				if (board[i][j] == -1)
					m.set(3 * (i * board[i].length + j) + 1, 0, 1);
				if (board[i][j] > maxSize) {
					maxSize = board[i][j];
					maxX = i;
					maxY = j;
				}
			}
		}
		m.set(3 * (maxX * board[0].length + maxY) + 2, 0, 1);
		return m.transpose();
	}

	public static void display(int[][] board) {
		double xSize = board.length;
		double ySize = board[0].length;
		StdDraw.setScale();
		for (int i = 0; i < board.length; i++) {
			for (int j = 0; j < board[i].length; j++) {
				if(board[i][j] > 0){
					StdDraw.setPenColor(Color.BLACK);
					StdDraw.filledRectangle((i + 0.5) / xSize, (j + 0.5) / ySize, 0.5 / xSize, 0.5 / ySize);
				}
				else if(board[i][j] < 0){
					StdDraw.setPenColor(Color.RED);
					StdDraw.filledRectangle((i + 0.5) / xSize, (j + 0.5) / ySize, 0.5 / xSize, 0.5 / ySize);
					break;
				}
			}
		}
	}
}
