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
	public double weightImportance = 0.4;
	public double linkMutateChance = 0.5;
	public int currentInnovation = 1;
	public double addLinkChance = 0.1;
	public double addNodeChance = 0.1;
	public ArrayList<Genome> genePool;

	public NEAT() {
		genePool = new ArrayList<Genome>();
	}

	@Deprecated
	public void reproduce() {

	}

	class Link implements Comparable<Link> {
		public int startNode;
		public int endNode;
		public double weight;
		public int innovationNumber;
		public boolean enabled;

		/**
		 * Creates a new link (no references to anything)
		 * 
		 * @param startNode
		 * @param endNode
		 * @param weight
		 * @param innovationNumber
		 */
		public Link(int startNode, int endNode, double weight, int innovationNumber) {
			this.startNode = startNode;
			this.endNode = endNode;
			this.weight = weight;
			this.innovationNumber = innovationNumber;
			enabled = true;
		}

		/**
		 * USE THIS ONLY WHEN LOOKING FOR ANOTHER LINK BY ID
		 * 
		 * @param innovation
		 */
		@Deprecated
		public Link(int innovation) {
			this.innovationNumber = innovationNumber;
		}

		/**
		 * Create a new link with the same paramaters but different reference
		 */
		@Override
		public Link clone() {
			Link l = new Link(startNode, endNode, weight, innovationNumber);
			return l;
		}

		public void mutate(double stepSize) {
			this.weight += Math.random() * stepSize - 1.0 / 2 * stepSize;
		}

		@Override
		public int compareTo(Link o) {
			return this.innovationNumber - o.innovationNumber;
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

		public void addLink(Link l) throws NEATException {
			if (l.endNode != id) {
				throw new NEATException("link node mismatch:" + l.endNode + " " + id);
			}
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

		public int getLinkIndex(int id) {
			for (int i = 0; i < incoming.size(); i++) {
				if (incoming.get(i).innovationNumber == id)
					return i;
			}
			return -1;
		}

	}

	class Genome {
		public ArrayList<Node> nodes;
		public ArrayList<Link> links;
		public int numOfNodes = 0; // Total number of nodes, so that
		public int inputSize;
		public int outputSize;
		public int maxInnovation;

		@Override
		public Genome clone() {
			// Copy links
			Genome g = new Genome(inputSize, outputSize, links);
			return g;
		}

		public Genome(int inputSize, int outputSize, ArrayList<Link> l) {
			nodes = new ArrayList<Node>();
			links = new ArrayList<Link>();
			this.numOfNodes = inputSize + outputSize;
			this.inputSize = inputSize;
			this.outputSize = outputSize;
			maxInnovation = 0;
			for (int i = 0; i < l.size(); i++) {
				Link a = l.get(i);
				Link newLink = a.clone();
				links.add(newLink);
				Node s = getNode(a.startNode);
				Node e = getNode(a.endNode);
				if (a.startNode < inputSize + outputSize) {
					if (s == null) {
						s = new Node(newLink.startNode);
					}
					s.incoming.add(newLink);
				}
				if (a.endNode < inputSize + outputSize) {
					if (e == null) {
						e = new Node(newLink.endNode);
					}
					e.incoming.add(newLink);
				}
			}
		}

		/**
		 * Create a new genome based on the data Will not have any references to
		 * the old genome
		 * 
		 * @param l
		 * @param n
		 * @param numOfNodes
		 */
		public Genome(ArrayList<Node> n, int inputSize, int outputSize) {
			nodes = new ArrayList<Node>();
			links = new ArrayList<Link>();
			for (int i = 0; i < n.size(); i++) {
				Node a = n.get(i).clone();
				nodes.add(a);
				for (int j = 0; j < a.incoming.size(); j++) {
					links.add(a.incoming.get(j));
				}
			}
			this.numOfNodes = inputSize + outputSize;
			this.inputSize = inputSize;
			this.outputSize = outputSize;
			maxInnovation = 0;
		}

		public Genome(int inputSize, int outputSize) {
			this(new ArrayList<Node>(), inputSize, outputSize);
		}

		/**
		 * Adds a node randomly, and the i1 and i2 are innovation numbers for
		 * each link
		 * 
		 * Can only split open a link
		 * 
		 * @param i1
		 * @param i2
		 */
		public void addNode(int i1, int i2) throws NEATException {
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
			maxInnovation = i1 > maxInnovation ? i1 : maxInnovation;
			maxInnovation = i2 > maxInnovation ? i2 : maxInnovation;
		}

		public void addLink(int node1, int node2, int innovationNumber) throws NEATException {
			Link l = new Link(node1, node2, 0, innovationNumber);
			Node n = getNode(node2);
			n.addLink(l);
			maxInnovation = innovationNumber > maxInnovation ? innovationNumber : maxInnovation;
		}

		public Matrix predict(Matrix X) throws Exception {
			if (X.getColumnDimension() != inputSize) {
				throw new Exception("This genome is not made for that size");
			}
			Matrix y = new Matrix(X.getRowDimension(), X.getColumnDimension());
			for (int ex = 0; ex < X.getRowDimension(); ex++) {
				double[] nodeValues = new double[numOfNodes + 1];
				boolean[] set = new boolean[numOfNodes + 1];
				Queue<Node> f = new LinkedList<Node>();
				for (int i = 1; i <= inputSize; i++) {
					nodeValues[i] = X.get(0, i - 1);
					set[i] = true;
				}
				for (int i = inputSize + 1; i <= inputSize + outputSize; i++) {
					f.add(getNode(i));
				}
				while (!f.isEmpty()) {
					double sum = 0;
					Node buffer = f.poll();
					// Iterate through all links to sum up things
					if (set[buffer.id]) {
						continue;
					}
					boolean summed = true;
					for (Link n : buffer.incoming) {
						if (!set[n.startNode]) {
							f.add(getNode(n.startNode));
							summed = false;
						} else {
							nodeValues[n.endNode] += n.weight * nodeValues[n.startNode];
						}
					}
					if (!summed) {
						continue;
					} else {
						set[buffer.id] = true;
						nodeValues[buffer.id] = sigmoid(nodeValues[buffer.id]);
					}
				}
				for (int i = inputSize + 1; i <= inputSize + outputSize; i++) {
					y.set(ex, i - inputSize - 1, nodeValues[i]);
				}
			}
			return y;
		}

		private Node getNode(int id) {
			Collections.sort(nodes);
			int index = Collections.binarySearch(nodes, new Node(id));
			if (index == -1) {
				return null;
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
		public void mutate() throws NEATException {
			// Determine whether or not to add a node
			if(addNodeChance > Math.random()){
				this.addNode(++currentInnovation, ++currentInnovation);
			}
			// Determine what percentage of links to mutate
			for(Link l : this.links){
				if(linkMutateChance > Math.random()){
					l.mutate(stepSize);
				}
			}
			
			//Determine whether or not to add a link
			if(addLinkChance > Math.random()){
				int startNode = (int) (Math.random() * numOfNodes + 1);
				int endNode;
				if(startNode <= inputSize + outputSize){
					if(startNode <= inputSize){
						endNode = inputSize + (int) (Math.random() * outputSize + 1);
					}
					else{
						endNode = (int) (Math.random() * inputSize + 1);
					}
				}
				else{
					while(true){
						endNode = (int) (Math.random() * numOfNodes + 1);
						if(endNode == startNode || isConnected(endNode, startNode)){
							continue;
						}
						else{
							break;
						}
					}
				}
				this.addLink(startNode, endNode, ++currentInnovation);
			}
		}

		public double distance(Genome g) {
			double wI = 0;
			double eI = 0;
			double dI = 0;
			int numW = 0;
			if (this.numOfNodes > g.numOfNodes) {
				for (int i = 0; i < links.size(); i++) {
					Link b = links.get(i);
					Link index = g.getLink(b.innovationNumber);
					if (index == null) {
						if (b.innovationNumber > g.maxInnovation) {
							eI += excessImportance;
						} else {
							dI += disjointImportance;
						}
					} else {
						wI += weightImportance * Math.abs(b.weight - index.weight);
						numW++;
					}
				}
			} else {
				for (int i = 0; i < g.links.size(); i++) {
					Link b = g.links.get(i);
					Link index = this.getLink(b.innovationNumber);
					if (index == null) {
						if (b.innovationNumber > this.maxInnovation) {
							eI += excessImportance;
						} else {
							dI += disjointImportance;
						}
					} else {
						wI += weightImportance * Math.abs(b.weight - index.weight);
						numW++;
					}
				}
			}
			double d = dI + eI;
			if (this.links.size() > g.links.size()) {
				return d / this.links.size() + (wI / numW);
			} else {
				return d / g.links.size() + (wI / numW);
			}
		}

		public int getLinkIndex(int id) {
			for (int i = 0; i < links.size(); i++) {
				if (links.get(i).innovationNumber == id) {
					return i;
				}
			}
			return -1;
		}

		public Link getLink(int id) {
			for (int i = 0; i < links.size(); i++) {
				if (links.get(i).innovationNumber == id)
					return links.get(i);
			}
			return null;
		}
		
		private boolean isConnected(int n1, int n2) throws NEATException{
			Queue<Node> q = new LinkedList<Node>();
			Node s = getNode(n1);
			Node e = getNode(n2);
			if(s == null || e == null){
				return false;
			}
			q.add(e);
			while(!q.isEmpty()){
				Node cur = q.poll();
				if(cur == s){
					return true;
				}
				for(Link l : cur.incoming){
					Node add = getNode(l.startNode);
					if(add == null){
						throw new NEATException("isConnected failure (not sure why)");
					}
					q.add(getNode(l.startNode));
				}
			}
			return false;
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

class NEATException extends Exception {
	public NEATException(String s) {
		super(s);
	}
}