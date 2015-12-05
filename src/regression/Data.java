package regression;

import java.util.ArrayList;

public class Data {
	private ArrayList<Double> x;
	private ArrayList<Double> y;

	public Data(){
		x = new ArrayList<Double>();
		y = new ArrayList<Double>();
	}
	
	public double getX(int index){
		return x.get(index).doubleValue();
	}
	
	public double getY(int index){
		return y.get(index).doubleValue();
	}
	
	public int length(){
		return x.size();
	}

	public void add(double x, double y){
		this.x.add(x);
		this.y.add(y);
	}
	
	public void delete(int index){
		x.remove(index);
		y.remove(index);
	}
	
	public static Data generateLinearData(double yInt, double slope){
		return generateLinearData(yInt, slope, 100);
	}
	public static Data generateLinearData(double yInt, double slope, int amount){
		Data data = new Data();
		for(int i = 0; i < amount; i++){
			data.add(i, slope * i + yInt + Math.random() - 0.5);
		}
		return data;
	}
}
