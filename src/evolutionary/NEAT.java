package evolutionary;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedList;
import java.util.Queue;
import java.util.StringTokenizer;

import Jama.Matrix;

public class NEAT {
	public double randomInitMean = 0;
	public double randomInitRange = 0.1;
	public double minStepSize = 0;
	public double maxStepSize = 0.1;
	public double excessImportance = 2;
	public double disjointImportance = 2;
	public double weightImportance = 0.4;
	public double linkMutateChance = 0.25;
	public int currentInnovation = 1;
	public double addLinkChance = 0.02;
	public double addNodeChance = 0.005;
	public ArrayList<Genome> genePool;
	public int inputSize = 0;
	public int outputSize = 0;
	public int stableGenePoolSize = 100;
	public int maxGenePoolSize = 500;
	public FitnessFunction f;
	public int numOfNodes;
	public double distanceImportance = 0.1;
	public double fitnessImportance = 1;
	public Genome best;

	public NEAT(int inputSize, int outputSize, FitnessFunction f) {
		genePool = new ArrayList<Genome>();
		this.inputSize = inputSize;
		this.outputSize = outputSize;
		numOfNodes = inputSize + outputSize;
		this.f = f;
	}

	public void loadFromFile(String filename) throws IOException, NEATException {
		BufferedReader b = new BufferedReader(new FileReader(filename));
		NEAT buffer = readNEAT(b, f);
		if (buffer.inputSize != this.inputSize || buffer.outputSize != this.outputSize)
			throw new NEATException("Incompatible neats");
		this.genePool = buffer.genePool;
		this.numOfNodes = this.numOfNodes > buffer.numOfNodes ? this.numOfNodes : buffer.numOfNodes;
		this.currentInnovation = this.currentInnovation > buffer.currentInnovation ? this.currentInnovation
				: buffer.currentInnovation;

		updateGenes();
		b.close();
	}

	public void updateGenes() {
		double totalDistance = 0;
		double totalFitness = 0;
		if(best == null){
			best = genePool.get(0);
		}
		for (Genome g : genePool) {
			g.fitness = f.getFitness(g);
			if (g.fitness > best.fitness) {
				best = g;
			}
			totalFitness += g.fitness;
			for (Genome ge : genePool) {
				g.distance += g.distance(ge);
			}
			g.distance /= genePool.size();
			totalDistance += g.distance;
		}
		if(totalFitness == 0)
			totalFitness = 1;
		if(totalDistance == 0)
			totalDistance = 1;
		for (Genome g : genePool) {
			g.adjFitness = (fitnessImportance * g.fitness / totalFitness)
					+ (distanceImportance * g.distance / totalDistance);
			g.reproduceChance = g.adjFitness / 2 / (fitnessImportance + distanceImportance);
		}
	}

	/**
	 * We start off simple, and we go more complicated later Need to make it so
	 * that the code isn't dependent on predefined fitnesses
	 */
	public double preBest = 0;
	public int generation = 0;

	@Deprecated
	public void reproduce() {
		// Get fitnesses
		for (int j = 0; j < genePool.size(); j++) {
			Genome g = genePool.get(j);
			// int likelyOffspring = Math.max((int) (g.reproduceChance *
			// maxGenePoolSize), 1);
			int likelyOffspring = (int) (maxGenePoolSize / genePool.size());
			for (int i = 0; i < likelyOffspring; i++) {
				Genome newG = g.clone();
				try {
					newG.mutate();
				} catch (NEATException e) {
					e.printStackTrace();
				}
				newG.fitness = f.getFitness(newG);
				genePool.add(0, newG);
				j++;
			}
		}
		updateGenes();
		double totalFitness = 0;
		for (Genome g : genePool) {
			totalFitness += g.fitness;
		}
		if (totalFitness / genePool.size() > preBest) {
			System.out.println("Avg fitness = " + (totalFitness / genePool.size()));
			preBest = totalFitness / genePool.size();
			System.out.println("Generation: " + generation);
		}

		// Kill off the weak
		while (genePool.size() > stableGenePoolSize) {
			for (int i = 0; i < genePool.size(); i++) {
				Genome g = genePool.get(i);
				if (g.adjFitness * stableGenePoolSize < Math.random()) {
					genePool.remove(g);
					i--;
				}
			}
		}
		generation++;
	}

	public Genome getTop() {
		return best;
	}

	public double bestFitness() {
		if(getTop() != null){
			return getTop().fitness;
		}
		else{
			return 0;
		}
	}

	public class Link implements Comparable<Link> {
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

		public void mutate() {
			this.weight += (Math.random() - 0.5) * (maxStepSize - minStepSize) + minStepSize;
		}

		@Override
		public int compareTo(Link o) {
			return this.innovationNumber - o.innovationNumber;
		}
	}

	public class Node implements Comparable<Node> {
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

	public class Genome implements Comparable<Genome> {
		public ArrayList<Node> nodes = new ArrayList<Node>();
		public ArrayList<Link> links = new ArrayList<Link>();
		public int maxInnovation;
		public double fitness;
		public double adjFitness;
		public double distance;
		public double reproduceChance;

		@Override
		public Genome clone() {
			// Copy links
			Genome g = null;
			try {
				g = new Genome(links);
			} catch (NEATException e) {
				e.printStackTrace();
				System.exit(0);
			}
			return g;
		}

		public Genome(ArrayList<?> z) throws NEATException {
			for (int i = 1; i <= inputSize + outputSize; i++) {
				nodes.add(new Node(i));
			}
			if (z.isEmpty()) {
				maxInnovation = 0;
			} else if (z.get(0) instanceof Link) {
				ArrayList<Link> l = (ArrayList<Link>) z;
				maxInnovation = 0;
				for (int i = 0; i < l.size(); i++) {
					Link a = l.get(i);
					Link newLink = a.clone();
					links.add(newLink);
					Node s = getNode(a.startNode);
					Node e = getNode(a.endNode);
					if (newLink.innovationNumber > maxInnovation) {
						maxInnovation = newLink.innovationNumber;
					}
					if (s == null) {
						s = new Node(newLink.endNode);
					}
					s.incoming.add(newLink);
					if (e == null) {
						e = new Node(newLink.endNode);
					}
					e.incoming.add(newLink);
				}
			} else if (z.get(0) instanceof Node) {
				System.out.println("dafuq is this");
				ArrayList<Node> n = (ArrayList<Node>) z;
				for (int i = 0; i < n.size(); i++) {
					Node a = n.get(i).clone();
					nodes.add(a);
					for (int j = 0; j < a.incoming.size(); j++) {
						links.add(a.incoming.get(j));
					}
				}
				maxInnovation = 0;
			}
		}

		public Genome() {
			for (int i = 1; i <= inputSize + outputSize; i++) {
				nodes.add(new Node(i));
			}
			maxInnovation = 0;
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
			if (links.isEmpty()) {
				return;
			}
			numOfNodes++;
			int l = (int) Math.random() * links.size();
			links.get(l).enabled = false;
			Link l1 = new Link(links.get(l).startNode, numOfNodes, 1, i1);
			Link l2 = new Link(numOfNodes, links.get(l).endNode, links.get(l).weight, i2);

			Node a = new Node(numOfNodes);
			a.addLink(l1);
			Node old = getNode(links.get(l).endNode);
			if (old == null) {
				old = new Node(links.get(l).endNode);
				nodes.add(old);
			}
			old.addLink(l2);

			// Add new nodes and links
			addNodeIntoList(a);
			links.add(l1);
			links.add(l2);
			maxInnovation = i1 > maxInnovation ? i1 : maxInnovation;
			maxInnovation = i2 > maxInnovation ? i2 : maxInnovation;
		}

		public void addLink(int node1, int node2, int innovationNumber) throws NEATException {
			Link l = new Link(node1, node2, randomInitMean + ((Math.random() - 0.5) * randomInitRange),
					innovationNumber);
			Node n = getNode(node2);
			if (n == null) {
				n = new Node(node2);
			}
			n.addLink(l);
			links.add(l);
			maxInnovation = innovationNumber > maxInnovation ? innovationNumber : maxInnovation;
		}

		public Matrix predict(Matrix X) throws NEATException {
			if (X.getColumnDimension() != inputSize) {
				throw new NEATException("This genome is not made for that size");
			}
			Matrix y = new Matrix(X.getRowDimension(), outputSize);
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
							Node addn = getNode(n.startNode);
							if (addn == null) {
								addn = new Node(n.startNode);
							}
							f.add(addn);
							summed = false;
						} else if (n.enabled) {
							nodeValues[n.endNode] += n.weight * nodeValues[n.startNode];
						}
					}
					if (!summed) {
						nodeValues[buffer.id] = 0;
						set[buffer.id] = false;
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

		private Node getNode(int id) throws NEATException {
			if (id > numOfNodes) {
				throw new NEATException("Invalid node id: " + id);
			}
			Collections.sort(nodes);
			int index = Collections.binarySearch(nodes, new Node(id));
			if (index < 0) {
				Node n = new Node(id);
				return n;
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

		public void mutate() throws NEATException {
			// Determine whether or not to add a node
			if (addNodeChance > Math.random()) {
				this.addNode(currentInnovation + 1, currentInnovation + 2);
				currentInnovation += 2;
			}
			// Determine what percentage of links to mutate
			for (Link l : this.links) {
				if (linkMutateChance > Math.random()) {
					l.mutate();
				}
			}

			// Determine whether or not to add a link
			if (addLinkChance > Math.random() || links.isEmpty()) {
				int startNode = (int) (Math.random() * (numOfNodes - outputSize) + 1);
				int endNode;
				if (startNode <= 0) {
					throw new NEATException("WHAT THE HELL");
				}
				if (startNode <= inputSize) {
					endNode = inputSize + (int) (Math.random() * (numOfNodes - inputSize) + 1);
				} else {
					while (true) {
						startNode = (int) (Math.random() * (numOfNodes - inputSize - outputSize) + 1 + inputSize
								+ outputSize);
						if (startNode > numOfNodes) {
							continue;
						}
						endNode = (int) (Math.random() * (numOfNodes - inputSize) + 1 + inputSize);
						if (endNode == startNode || isConnected(endNode, startNode)
								|| isDirConnected(startNode, endNode)) {
							continue;
						} else {
							break;
						}
					}
				}
				if (startNode > numOfNodes || endNode > numOfNodes)
					System.out.println("Captain we got a problem: " + startNode + "," + endNode);
				this.addLink(startNode, endNode, currentInnovation + 1);
				currentInnovation += 1;
			}
		}

		public double distance(Genome g) {
			double wI = 0;
			double eI = 0;
			double dI = 0;
			int numW = 0;
			if (this.links.size() > g.links.size()) {
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
			double wa = wI / (numW != 0 ? numW : 1);
			if (this.links.size() > g.links.size()) {
				return d / this.links.size() + (wI / (numW != 0 ? numW : 1));
			} else {
				return d / (g.links.size() != 0 ? g.links.size() : 1) + (wI / (numW != 0 ? numW : 1));
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

		private boolean isConnected(int n1, int n2) throws NEATException {
			Queue<Node> q = new LinkedList<Node>();
			Node s = getNode(n1);
			Node e = getNode(n2);
			if (s == null || e == null) {
				return false;
			}
			q.add(e);
			while (!q.isEmpty()) {
				Node cur = q.poll();
				if (cur == s) {
					return true;
				}
				for (Link l : cur.incoming) {
					Node add = getNode(l.startNode);
					if (add == null) {
						throw new NEATException("isConnected failure (not sure why)");
					}
					q.add(getNode(l.startNode));
				}
			}
			return false;
		}

		private boolean isDirConnected(int n1, int n2) throws NEATException {
			Node e = getNode(n2);
			if (e == null) {
				return false;
			} else {
				for (int i = 0; i < e.incoming.size(); i++) {
					if (n1 == e.incoming.get(i).innovationNumber) {
						return true;
					}
				}
			}
			return false;
		}

		@Override
		public int compareTo(Genome o) {
			if (this.adjFitness == o.adjFitness) {
				return 0;
			}
			return this.adjFitness < o.adjFitness ? -1 : 1;
		}

		/*
		 * public double complexity(){ double sum = links.size(); for(Link l :
		 * links){ sum += l.weight * l.weight * cWeightImportance; } return sum;
		 * }
		 */
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

	public Genome generateRandomGenome() {
		Genome g = new Genome();
		return g;
	}

	public static interface FitnessFunction {
		public double getFitness(Genome g);
	}

	public static class NEATException extends Exception {
		public NEATException(String s) {
			super(s);
		}
	}

	public void writeNEAT(PrintWriter p) {
		p.println("NEAT START");
		p.println(randomInitMean);
		p.println(randomInitRange);
		p.println(minStepSize);
		p.println(maxStepSize);
		p.println(excessImportance);
		p.println(disjointImportance);
		p.println(weightImportance);
		p.println(linkMutateChance);
		p.println(currentInnovation);
		p.println(addLinkChance);
		p.println(addNodeChance);
		p.println(inputSize);
		p.println(outputSize);
		p.println(stableGenePoolSize);
		p.println(maxGenePoolSize);
		p.println(numOfNodes);
		p.println(distanceImportance);
		p.println(this.numOfNodes);
		p.println(genePool.size());
		writeGenome(p, best);
		for (Genome g : genePool) {
			writeGenome(p, g);
		}
		p.println("NEAT END");
	}

	public static NEAT readNEAT(BufferedReader b, FitnessFunction f) throws NEATException, IOException {
		if (!b.readLine().contentEquals("NEAT START"))
			throw new NEATException("Not a NEAT file");
		NEAT n = new NEAT(0, 0, f);
		n.randomInitMean = Double.parseDouble(b.readLine());
		n.randomInitRange = Double.parseDouble(b.readLine());
		n.minStepSize = Double.parseDouble(b.readLine());
		n.maxStepSize = Double.parseDouble(b.readLine());
		n.excessImportance = Double.parseDouble(b.readLine());
		n.disjointImportance = Double.parseDouble(b.readLine());
		n.weightImportance = Double.parseDouble(b.readLine());
		n.linkMutateChance = Double.parseDouble(b.readLine());
		n.currentInnovation = Integer.parseInt(b.readLine());
		n.addLinkChance = Double.parseDouble(b.readLine());
		n.addNodeChance = Double.parseDouble(b.readLine());
		n.inputSize = Integer.parseInt(b.readLine());
		n.outputSize = Integer.parseInt(b.readLine());
		n.stableGenePoolSize = Integer.parseInt(b.readLine());
		n.maxGenePoolSize = Integer.parseInt(b.readLine());
		n.numOfNodes = Integer.parseInt(b.readLine());
		n.distanceImportance = Double.parseDouble(b.readLine());
		n.numOfNodes = Integer.parseInt(b.readLine());
		int numOfGenomes = Integer.parseInt(b.readLine());
		n.best = readGenome(b, n);
		for (int i = 0; i < numOfGenomes; i++) {
			n.genePool.add(readGenome(b, n));
		}
		n.updateGenes();
		return n;
	}

	public static void writeGenome(PrintWriter p, Genome g) {
		p.println("Genome Start");
		p.println(g.links.size());
		for (int i = 0; i < g.links.size(); i++) {
			Link l = g.links.get(i);
			p.println(l.startNode + " " + l.endNode + " " + l.weight + " " + l.innovationNumber + " " + l.enabled);
		}
		p.println("Genome End");
	}

	public static Genome readGenome(BufferedReader b, NEAT n) throws NEATException, IOException {
		if (!b.readLine().contentEquals("Genome Start"))
			throw new NEATException("file is not a genome");
		ArrayList<Link> links = new ArrayList<Link>();
		int size = Integer.parseInt(b.readLine());
		for (int i = 0; i < size; i++) {
			StringTokenizer read = new StringTokenizer(b.readLine());
			Link readLink = n.new Link(Integer.parseInt(read.nextToken()), Integer.parseInt(read.nextToken()),
					Double.parseDouble(read.nextToken()), Integer.parseInt(read.nextToken()));
			readLink.enabled = read.nextToken() == "true";
			links.add(readLink);
		}
		if (!b.readLine().contentEquals("Genome End"))
			throw new NEATException("genome part doesn't end...");
		return n.new Genome(links);
	}
}