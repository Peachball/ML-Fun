package evolutionary;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedList;
import java.util.Queue;

import Jama.Matrix;

public class NEAT {
	public double deltaThreshold = 3;
	public double randomInitMean = 0;
	public double randomInitRange = 5;
	public double stepSize = 1;
	public double excessImportance = 1;
	public double disjointImportance = 1;
	public double weightImportance = 1;
	public ArrayList<Genome> genePool;
	
	public NEAT(){
		genePool = new ArrayList<Genome>();
	}
	
	public void reproduce(){
		
	}

	class Link {
		public int startNode;
		public int endNode;
		public double weight;
		public int innovationNumber;
		public boolean enabled;

		public Link(int startNode, int endNode, double weight, int innovationNumber) {
			this.startNode = startNode;
			this.endNode = endNode;
			this.weight = weight;
			this.innovationNumber = innovationNumber;
			enabled = true;
		}

		@Override
		public Link clone() {
			Link l = new Link(startNode, endNode, weight, innovationNumber);
			return l;
		}

		public void mutate(double stepSize) {
			this.weight = Math.random() * stepSize - 1.0 / 2 * stepSize;
		}
	}

	class Node implements Comparable<Node> {
		public ArrayList<Link> incoming = new ArrayList<Link>();
		public int id;

		public Node(ArrayList<Link> incoming, int id) {
			for (int i = 0; i < incoming.size(); i++) {
				this.incoming.add(incoming.get(i).clone());
			}
			this.id = id;
		}

		public Node(int id) {
			this.id = id;
		}

		public void addLink(Link l) {
			incoming.add(l);
		}

		@Override
		public Node clone() {
			Node newNode = new Node(incoming, id);
			return newNode;
		}

		@Override
		public int compareTo(Node o) {
			return id - o.id;
		}

	}

	class Genome {
		public ArrayList<Node> nodes;
		public ArrayList<Link> links;
		public int numOfNodes = 0;
		public int inputSize;
		public int outputSize;

		/**
		 * Create a new genome based on the data Will not have any references to
		 * the old genome
		 * 
		 * @param l
		 * @param n
		 * @param numOfNodes
		 */
		public Genome(ArrayList<Node> n, int numOfNodes, int inputSize, int outputSize) {
			nodes = new ArrayList<Node>();
			links = new ArrayList<Link>();
			for (int i = 0; i < n.size(); i++) {
				Node a = n.get(i).clone();
				nodes.add(a);
				for (int j = 0; j < a.incoming.size(); j++) {
					links.add(a.incoming.get(j));
				}
			}
			this.numOfNodes = numOfNodes;
			this.inputSize = inputSize;
			this.outputSize = outputSize;
		}

		@Override
		public Genome clone() {
			Genome newGenome = new Genome(nodes, numOfNodes, inputSize, outputSize);
			return newGenome;
		}

		/**
		 * Adds a node randomly, and the i1 and i2 are innovation numbers for
		 * each link
		 * 
		 * Can only split open a link
		 * @param i1
		 * @param i2
		 */
		public void addNode(int i1, int i2){
			numOfNodes++;
			int l = (int) Math.random() * links.size();
			links.get(l).enabled = false;
			Link l1 = new Link(links.get(l).startNode, numOfNodes, 1, i1);
			Link l2 = new Link(numOfNodes, links.get(1).endNode, links.get(l).weight, i2);

			Node a = new Node(numOfNodes);
			a.addLink(l1);
			for (int i = 0; i < nodes.size(); i++) {
				if (nodes.get(i).id == links.get(l).endNode) {
					nodes.get(i).addLink(l2);
					break;
				}
			}

			// Add new nodes and links
			addNodeIntoList(a);
			links.add(l1);
			links.add(l2);
		}
		
		public void addLink(int node1, int node2, int innovationNumber) throws Exception{
			Link l = new Link(node1, node2, 0, innovationNumber);
			Node n = getNode(node2);
			n.addLink(l);
		}

		public Matrix predict(Matrix X) throws Exception {
			if (X.getColumnDimension() != inputSize) {
				throw new Exception("This genome is not made for that size");
			}
			Matrix y = new Matrix(X.getRowDimension(), X.getColumnDimension());
			for (int ex = 0; ex < X.getRowDimension(); ex++) {
				double[] nodeValues = new double[numOfNodes + 1];
				boolean[] set = new boolean[numOfNodes+1];
				Queue<Node> f = new LinkedList<Node>();
				for (int i = 1; i <= inputSize; i++) {
					nodeValues[i] = X.get(0, i - 1);
				}
				for (int i = inputSize + 1; i <= inputSize + outputSize; i++) {
					f.add(getNode(i));
				}
				while (!f.isEmpty()) {
					double sum = 0;
					Node buffer = f.poll();
					//Iterate through all links to sum up things
					for(Link n : buffer.incoming){
						if(!set[n.startNode]){
							f.add(getNode(n.startNode));
						}
						else{
						}
					}
					set[buffer.id] = true;
				}
				for(int i = inputSize + 1; i <= inputSize + outputSize; i++){
					y.set(ex, i - inputSize - 1, nodeValues[i]);
				}
			}
			return y;
		}

		private Node getNode(int id) throws Exception {
			Collections.sort(nodes);
			int index = Collections.binarySearch(nodes, new Node(id));
			if (index == -1) {
				throw new Exception("Unable to find node");
			} else {
				return nodes.get(index);
			}
		}

		private void addNodeIntoList(Node n) {
			if (nodes.size() == 0) {
				nodes.add(n);
			} else if (nodes.get(0).id > n.id) {
				nodes.add(0, n);
			} else if (nodes.get(nodes.size() - 1).id < n.id) {
				nodes.add(nodes.size(), n);
			} else {
				int i = 0;
				while (nodes.get(i).id < n.id) {
					i++;
				}
				nodes.add(i, n);
			}
		}
		
		@Deprecated
		public void mutate(){
			//Determine whether or not to add a node
			//Determine what percentage of links to mutate
		}
		
		@Deprecated
		public double distance(Genome g){
			
			return 0;
		}
	}

	/**
	 * Defines the sigmoid function that the NEAT network uses (or any function,
	 * really)
	 * 
	 * @param z
	 */
	private static double sigmoid(double z) {
		return 1.0 / (1 + Math.pow(Math.E, -4.9 * z));
	}

	public static double distance(Genome a, Genome b) {

		return 0;
	}
}