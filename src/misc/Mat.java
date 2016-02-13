package misc;

import java.io.DataInputStream;
import java.io.FileInputStream;

import Jama.Matrix;

public class Mat {

	public static Matrix remove1stColumn(Matrix m) {
		Matrix a = new Matrix(m.getRowDimension(), m.getColumnDimension() - 1);
		for (int i = 0; i < a.getRowDimension(); i++) {
			for (int j = 0; j < a.getColumnDimension(); j++) {
				a.set(i, j, m.get(i, j + 1));
			}
		}
		return a;
	}

	public static Matrix ones(int rows, int columns) {
		Matrix m = new Matrix(rows, columns);
		for (int i = 0; i < rows; i++) {
			for (int j = 0; j < columns; j++) {
				m.set(i, j, 1);
			}
		}

		return m;
	}

	public static Matrix addBotMatrix(Matrix a, Matrix b) {
		Matrix c = new Matrix(a.getRowDimension() + b.getRowDimension(), a.getColumnDimension());
		for (int i = 0; i < a.getRowDimension(); i++) {
			for (int j = 0; j < a.getColumnDimension(); j++) {
				c.set(i, j, a.get(i, j));
			}
		}
		for (int i = 0; i < b.getRowDimension(); i++) {
			for (int j = 0; j < b.getColumnDimension(); j++) {
				c.set(i + a.getRowDimension(), j, b.get(i, j));
			}
		}

		return c;
	}

	public static Matrix add1sColumn(Matrix m) {
		Matrix a = new Matrix(m.getRowDimension(), m.getColumnDimension() + 1);
		for (int i = 0; i < a.getRowDimension(); i++) {
			a.set(i, 0, 1);
			for (int j = 1; j < a.getColumnDimension(); j++) {
				a.set(i, j, m.get(i, j - 1));
			}
		}
		return a;
	}

	public static Matrix add1sRow(Matrix m) {
		Matrix a = new Matrix(m.getRowDimension() + 1, m.getColumnDimension());
		for (int i = 0; i < a.getColumnDimension(); i++) {
			a.set(0, i, 1);
		}
		return a;
	}

	public static Matrix readidx(String filename, boolean mode, int offset, int length) {
		Matrix X = null;
		try {
			DataInputStream in = new DataInputStream(new FileInputStream(filename));
			in.readInt();
			int examples = in.readInt();
			if(mode){
				int width = in.readInt();
				int height = in.readInt();
				if(offset + length > examples){
					offset = examples - length;
				}
				in.skip(offset * width * height);
				X = new Matrix(length , width * height);
				for(int i = 0; i < length; i++){
					for(int j = 0; j < width*height ;j++){
						X.set(i , j, in.read());
					}
				}
			}
			else{
				if(offset + length > examples){
					offset = examples - length;
				}
				in.skip(offset);
				X = new Matrix(length , 10);
				for(int i = 0; i < length; i++){
					X.set(i , in.read(), 1);
				}
			}
		} catch (Exception e) {
			e.printStackTrace();
		}
		return X;
	}
	
	public static Matrix readidx(String filename, boolean mode){
		return readidx(filename, mode, 0, 10000);
	}

	public static Matrix append(Matrix a, Matrix b){
		if(a.getRowDimension() != b.getRowDimension()){
			System.out.println("UNEQUAL DIMENSIONS FOR APPEND");
			return null;
		}
		Matrix ans = new Matrix(a.getRowDimension(), a.getColumnDimension()+ b.getColumnDimension());
		for(int i = 0; i < ans.getRowDimension(); i++){
			for(int j = 0; j < a.getColumnDimension(); j++){
				ans.set(i, j, a.get(i, j));
			}
			for(int j = 0; j < b.getColumnDimension(); j++){
				ans.set(i, j + a.getColumnDimension(), b.get(i, j));
			}
		}
		return ans;
	}
}
