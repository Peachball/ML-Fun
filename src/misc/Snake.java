package misc;

import java.awt.Color;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;

import Jama.Matrix;
import evolutionary.NEAT;
import evolutionary.NEAT.Genome;
import evolutionary.NEAT.NEATException;

public class Snake {

	public static int defaultspeed = 100;

	/**
	 * Test things here
	 * 
	 * @param args
	 * @throws NEATException
	 */
	public static void main(String[] args) throws NEATException, IOException {
		int boardX, boardY;
		boardX = boardY = 5;
		int numOfTrials = 2 * boardX * boardY;
		NEAT ne;
		try {
			ne = NEAT.readNEAT(new BufferedReader(new FileReader("neat.neat")),
					(Genome g) -> F(g, boardX, boardY, false, 0, numOfTrials));
			System.out.println("Loaded old NEAT");
		} catch (IOException e) {
			ne = new NEAT(3 * boardX * boardY + 1, 4, (Genome g) -> F(g, boardX, boardY, false, 0, numOfTrials));
			Genome s = ne.new Genome();
			s.mutate();
			ne.genePool.add(s);
			ne.updateGenes();
			System.out.println("Created new NEAT");
		}

		switch (2) {
		case 1:
			// Play top
			Genome g = ne.getTop();
			F(g, boardX, boardY, true, defaultspeed, 10);
			break;
		case 2:
			for (int i = 1; i <= 10000; i++) {
				if (i % 100 == 0) {
					System.out.println("Saving...Do not close");
					PrintWriter out = new PrintWriter(new FileWriter("neat.neat"));
					ne.writeNEAT(out);
					out.close();
					System.out.println("Best avg: " + ne.preBest);
					System.out.println("Auto saved");
					System.out.println("Best: " + ne.bestFitness());
					System.out.println("Generation: " + i);
					if (ne.getTop() != null) {
						F(ne.getTop(), boardX, boardY, true, defaultspeed, 10);
					}
				}
				ne.reproduce();
			}
			break;
		}
	}

	public static double F(Genome theta, int x, int y, boolean display, int speed, int trials) {
		double sum = 0;
		for (int i = 0; i < trials; i++) {
			sum += F(theta, x, y, display, speed);
		}
		return sum / trials;
	}

	public static double F(Genome theta, int x, int y, boolean display, int speed) {
		int idleTime = 0;
		boolean alive = true;
		int[][] board = new int[x][y];
		addApple(board);
		board[0][0] = 1;
		int status = 0;
		while (true) {
			Matrix X = convert(board);
			int nextDir = 1;
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
			int s = nextIteration(board, nextDir);
			if (display) {
				StdDraw.clear();
				display(board);
				StdDraw.show(speed);
			}
			if (s < 0) {
				// System.out.println("You died with " + (-s) + " points");
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
			if (idleTime > (x * y / 2)) {
				// System.out.println("Didn't get the apple enough...With " +
				// status + " Points");
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
		if(dir < 1 || dir > 4)
			System.out.println("error");
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
		if (board[xHead][yHead] <= -1) {
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
				if (board[i][j] > 0) {
					spaces++;
				}
			}
		}
		double rng = Math.random() * (board.length * board[0].length - spaces) + 1;
		spaces = 0;
		for (int i = 0; i < board.length; i++) {
			for (int j = 0; j < board[i].length; j++) {
				if (board[i][j] > 0) {
					continue;
				}
				if (rng - spaces < 1) {
					board[i][j] = -1;
					return;
				}
				spaces++;
			}
			if (i == board.length - 1)
				i = 0;
		}
		System.out.println("problem here");
	}

	private static Matrix convert(int[][] board) {
		Matrix m = new Matrix(3 * board.length * board[0].length + 1, 1);
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
		m.set(3 * board.length * board[0].length, 0, 1);
		return m.transpose();
	}

	public static void display(int[][] board) {
		double xSize = board.length;
		double ySize = board[0].length;
		StdDraw.setScale();
		for (int i = 0; i < board.length; i++) {
			for (int j = 0; j < board[i].length; j++) {
				if (board[i][j] > 0) {
					StdDraw.setPenColor(Color.BLACK);
					StdDraw.filledRectangle((i + 0.5) / xSize, (j + 0.5) / ySize, 0.5 / xSize, 0.5 / ySize);
				} else if (board[i][j] < 0) {
					StdDraw.setPenColor(Color.RED);
					StdDraw.filledRectangle((i + 0.5) / xSize, (j + 0.5) / ySize, 0.5 / xSize, 0.5 / ySize);
					break;
				}
			}
		}
	}
}
