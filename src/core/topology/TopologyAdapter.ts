import { defineComponent, Types } from 'bitecs';
import { Matrix, matrix, eigen, multiply, add, subtract, inv, norm, zeros, ones } from 'mathjs';
import { NeuralSynchronizer } from '../neural/NeuralSynchronizer';
import { QuantumStateEncoder } from '../quantum/QuantumStateEncoder';
import { MLPredictor } from '../MLPredictor';
import { CognitiveWebSocketServer } from '../CognitiveWebSocketServer';

// Enhanced Topology Node Component
const TopologyNode = defineComponent({
  influence: Types.f32,          // 0-1
  centrality: Types.f32,         // 0-1
  stability: Types.f32,          // 0-1
  connectivityDegree: Types.ui32,// Number of connections
  clusterCoefficient: Types.f32, // 0-1
  pathLength: Types.f32,         // Average path length to others
  adaptationRate: Types.f32,     // 0-1
  lastUpdateTime: Types.ui32,    // Timestamp (ms)
  quantumEntanglement: Types.f32,// Quantum entanglement score (0-1)
  neuralCoherence: Types.f32     // Neural sync coherence (0-1)
});

interface NetworkMetrics {
  globalEfficiency: number;       // Efficiency of information flow
  clusteringCoefficient: number;  // Average local clustering
  averagePathLength: number;      // Mean shortest path length
  smallWorldIndex: number;        // Small-world property measure
  modularityScore: number;        // Community structure strength
}

interface AdaptationStrategy {
  type: 'merge' | 'split' | 'rewire' | 'strengthen' | 'weaken';
  nodes: number[];
  confidence: number;             // 0-1
  impact: number;                 // Expected network improvement (0-1)
  quantumImpact: number;          // Quantum state influence (0-1)
}

export class TopologyAdapter {
  private adjacencyMatrix: Matrix;
  private neuralSync: NeuralSynchronizer;
  private quantumEncoder: QuantumStateEncoder;
  private predictor: MLPredictor;
  private wsServer: CognitiveWebSocketServer;
  private nodeStates: Map<number, any> = new Map();
  private readonly ADAPTATION_THRESHOLD = 0.7;
  private readonly MIN_CLUSTER_SIZE = 3;
  private readonly MAX_CLUSTER_SIZE = 12;
  private readonly PREDICTION_HORIZON = 6; // Steps for ML prediction

  constructor(
    initialSize: number,
    neuralSync: NeuralSynchronizer,
    quantumEncoder: QuantumStateEncoder,
    wsPort: number = 8080
  ) {
    this.adjacencyMatrix = this.initializeAdjacencyMatrix(initialSize);
    this.neuralSync = neuralSync;
    this.quantumEncoder = quantumEncoder;
    this.predictor = new MLPredictor(wsPort);
    this.wsServer = new CognitiveWebSocketServer(wsPort);
    this.initializeComponents();
  }

  private async initializeComponents() {
    await Promise.all([
      this.neuralSync.initialize(),
      this.quantumEncoder.initialize(),
      this.predictor.initialize()
    ]);
  }

  private initializeAdjacencyMatrix(size: number): Matrix {
    const matrix = zeros(size, size) as Matrix;
    const smallWorldFactor = 0.1;

    // Ring connectivity
    for (let i = 0; i < size; i++) {
      matrix.set([i, (i + 1) % size], 1);
      matrix.set([(i + 1) % size, i], 1);
    }

    // Random long-range connections
    const longRangeConnections = Math.floor(size * smallWorldFactor);
    for (let i = 0; i < longRangeConnections; i++) {
      const source = Math.floor(Math.random() * size);
      const target = Math.floor(Math.random() * size);
      if (source !== target && matrix.get([source, target]) === 0) {
        matrix.set([source, target], 1);
        matrix.set([target, source], 1);
      }
    }

    return matrix;
  }

  async adaptTopology(
    nodes: number[],
    performanceMetrics: Map<number, any>
  ): Promise<{
    adaptations: AdaptationStrategy[];
    metrics: NetworkMetrics;
  }> {
    const currentMetrics = await this.calculateNetworkMetrics();
    const strategies = await this.identifyAdaptationStrategies(nodes, performanceMetrics, currentMetrics);
    const appliedStrategies = await this.applyAdaptations(strategies.filter(s => s.confidence > this.ADAPTATION_THRESHOLD));
    const updatedMetrics = await this.calculateNetworkMetrics();

    this.updateNodeStates(nodes, performanceMetrics);
    this.wsServer.broadcastStateUpdate("topology", {
      event: 'topology_update',
      adaptations: appliedStrategies,
      metrics: updatedMetrics,
      visualization: this.visualizeTopology()
    });

    return { adaptations: appliedStrategies, metrics: updatedMetrics };
  }

  private async identifyAdaptationStrategies(
    nodes: number[],
    performanceMetrics: Map<number, any>,
    networkMetrics: NetworkMetrics
  ): Promise<AdaptationStrategy[]> {
    const strategies: AdaptationStrategy[] = [];
    const clusters = this.identifyClusters();

    // Merge opportunities
    for (let i = 0; i < clusters.length; i++) {
      for (let j = i + 1; j < clusters.length; j++) {
        const cluster1 = clusters[i];
        const cluster2 = clusters[j];
        const mergeConfidence = await this.evaluateMergeStrategy(cluster1, cluster2, performanceMetrics);
        if (mergeConfidence > this.ADAPTATION_THRESHOLD) {
          strategies.push({
            type: 'merge',
            nodes: [...cluster1, ...cluster2],
            confidence: mergeConfidence,
            impact: this.estimateStrategyImpact('merge', [...cluster1, ...cluster2], performanceMetrics),
            quantumImpact: this.calculateQuantumImpact([...cluster1, ...cluster2])
          });
        }
      }
    }

    // Split opportunities
    clusters.filter(c => c.length > this.MAX_CLUSTER_SIZE).forEach(cluster => {
      const splitConfidence = this.evaluateSplitStrategy(cluster, performanceMetrics);
      if (splitConfidence > this.ADAPTATION_THRESHOLD) {
        strategies.push({
          type: 'split',
          nodes: cluster,
          confidence: splitConfidence,
          impact: this.estimateStrategyImpact('split', cluster, performanceMetrics),
          quantumImpact: this.calculateQuantumImpact(cluster)
        });
      }
    });

    // Rewiring opportunities
    for (const node of nodes) {
      const rewireConfidence = await this.evaluateRewireStrategy(node, performanceMetrics);
      if (rewireConfidence > this.ADAPTATION_THRESHOLD) {
        strategies.push({
          type: 'rewire',
          nodes: [node],
          confidence: rewireConfidence,
          impact: this.estimateStrategyImpact('rewire', [node], performanceMetrics),
          quantumImpact: this.calculateQuantumImpact([node])
        });
      }
    }

    return this.rankStrategies(strategies);
  }

  private identifyClusters(): number[][] {
    const { values, vectors } = eigen(this.adjacencyMatrix);
    const k = this.estimateOptimalClusters(values);
    return this.spectralClustering(vectors, k);
  }

  private estimateOptimalClusters(values: any): number {
    const sortedValues = Array.from(values).sort((a: number, b: number) => b - a);
    let maxGap = 0;
    let optimalK = 2;
    for (let i = 0; i < sortedValues.length - 1; i++) {
      const gap = sortedValues[i] - sortedValues[i + 1];
      if (gap > maxGap) {
        maxGap = gap;
        optimalK = i + 1;
      }
    }
    return Math.max(this.MIN_CLUSTER_SIZE, Math.min(optimalK, this.adjacencyMatrix.size()[0] / 2));
  }

  private spectralClustering(eigenvectors: Matrix, k: number): number[][] {
    const points = Array.from({ length: eigenvectors.size()[0] }, (_, i) =>
      Array.from({ length: k }, (_, j) => eigenvectors.get([i, j]))
    );
    return this.kMeansClustering(points, k);
  }

  private kMeansClustering(points: number[][], k: number): number[][] {
    let centroids = Array.from({ length: k }, () => points[Math.floor(Math.random() * points.length)].slice());
    let clusters: number[][] = Array(k).fill(null).map(() => []);
    let converged = false;
    let iterations = 0;
    const maxIterations = 100;

    while (!converged && iterations < maxIterations) {
      const newClusters: number[][] = Array(k).fill(null).map(() => []);
      points.forEach((point, index) => {
        const distances = centroids.map(c => this.euclideanDistance(point, c));
        const nearest = distances.indexOf(Math.min(...distances));
        newClusters[nearest].push(index);
      });

      converged = this.clustersEqual(clusters, newClusters);
      clusters = newClusters;

      centroids = clusters.map(cluster => {
        if (cluster.length === 0) return centroids[0];
        return cluster.reduce((sum, idx) => sum.map((v, i) => v + points[idx][i]), Array(points[0].length).fill(0))
          .map(v => v / cluster.length);
      });
      iterations++;
    }
    return clusters;
  }

  private euclideanDistance(point1: number[], point2: number[]): number {
    return Math.sqrt(point1.reduce((sum, val, i) => sum + Math.pow(val - point2[i], 2), 0));
  }

  private clustersEqual(clusters1: number[][], clusters2: number[][]): boolean {
    return clusters1.length === clusters2.length && clusters1.every((c1, i) => 
      c1.length === clusters2[i].length && c1.every(v => clusters2[i].includes(v))
    );
  }

  private async evaluateMergeStrategy(cluster1: number[], cluster2: number[], performanceMetrics: Map<number, any>): Promise<number> {
    const connectivity = this.calculateInterClusterConnectivity(cluster1, cluster2);
    const similarity = await this.calculateClusterSimilarity(cluster1, cluster2, performanceMetrics);
    const neuralCoherence = await this.calculateNeuralCoherence([...cluster1, ...cluster2]);
    return (connectivity * 0.4 + similarity * 0.4 + neuralCoherence * 0.2);
  }

  private evaluateSplitStrategy(cluster: number[], performanceMetrics: Map<number, any>): number {
    const connectivity = this.calculateIntraClusterConnectivity(cluster);
    const variance = this.calculateClusterVariance(cluster, performanceMetrics);
    return (1 - connectivity) * 0.6 + variance * 0.4;
  }

  private async evaluateRewireStrategy(node: number, performanceMetrics: Map<number, any>): Promise<number> {
    const isolation = this.calculateNodeIsolation(node);
    const potential = await this.calculateConnectionPotential(node, performanceMetrics);
    const quantumWeight = this.calculateQuantumImpact([node]);
    return (isolation * 0.4 + potential * 0.4 + quantumWeight * 0.2);
  }

  private calculateInterClusterConnectivity(cluster1: number[], cluster2: number[]): number {
    let connections = 0;
    const maxConnections = cluster1.length * cluster2.length;
    cluster1.forEach(node1 => {
      cluster2.forEach(node2 => {
        if (this.adjacencyMatrix.get([node1, node2]) === 1) connections++;
      });
    });
    return maxConnections > 0 ? connections / maxConnections : 0;
  }

  private calculateIntraClusterConnectivity(cluster: number[]): number {
    let connections = 0;
    const maxConnections = cluster.length * (cluster.length - 1) / 2;
    for (let i = 0; i < cluster.length; i++) {
      for (let j = i + 1; j < cluster.length; j++) {
        if (this.adjacencyMatrix.get([cluster[i], cluster[j]]) === 1) connections++;
      }
    }
    return maxConnections > 0 ? connections / maxConnections : 0;
  }

  private calculateNodeIsolation(node: number): number {
    const degree = this.adjacencyMatrix.get([node]).reduce((sum: number, val: number) => sum + val, 0);
    return 1 - (degree / (this.adjacencyMatrix.size()[0] - 1));
  }

  private async calculateConnectionPotential(node: number, performanceMetrics: Map<number, any>): Promise<number> {
    const nodeState = performanceMetrics.get(node);
    if (!nodeState) return 0.5;
    const register = this.quantumEncoder.encodeState(nodeState, node.toString());
    const entanglement = this.quantumEncoder.calculateEntanglementMetrics(register).score;
    const syncResult = await this.neuralSync.synchronizeStates(nodeState, nodeState); // Self-sync for coherence
    return (entanglement + syncResult.coherenceScore) / 2;
  }

  private async applyAdaptations(strategies: AdaptationStrategy[]): Promise<AdaptationStrategy[]> {
    const appliedStrategies: AdaptationStrategy[] = [];
    const size = this.adjacencyMatrix.size()[0];
    const newMatrix = matrix(this.adjacencyMatrix.toArray()) as Matrix;

    for (const strategy of strategies) {
      let success = false;
      switch (strategy.type) {
        case 'merge':
          success = await this.mergeClusters(strategy.nodes, newMatrix);
          break;
        case 'split':
          success = await this.splitCluster(strategy.nodes, newMatrix);
          break;
        case 'rewire':
          success = await this.rewireNode(strategy.nodes[0], newMatrix);
          break;
        case 'strengthen':
          success = await this.strengthenConnections(strategy.nodes, newMatrix);
          break;
        case 'weaken':
          success = await this.weakenConnections(strategy.nodes, newMatrix);
          break;
      }
      if (success) {
        appliedStrategies.push(strategy);
        for (const node of strategy.nodes) {
          TopologyNode.lastUpdateTime[node] = Date.now();
        }
      }
    }

    this.adjacencyMatrix = newMatrix;
    this.updateNetworkMetrics(appliedStrategies);
    return appliedStrategies;
  }

  private async mergeClusters(nodes: number[], matrix: Matrix): Promise<boolean> {
    const clusterNodes = new Set(nodes);
    for (let i = 0; i < matrix.size()[0]; i++) {
      if (!clusterNodes.has(i)) {
        const hasConnection = nodes.some(n => matrix.get([i, n]) === 1);
        if (!hasConnection) {
          const target = nodes[Math.floor(Math.random() * nodes.length)];
          matrix.set([i, target], 1);
          matrix.set([target, i], 1);
        }
      }
    }
    return true;
  }

  private async splitCluster(nodes: number[], matrix: Matrix): Promise<boolean> {
    const mid = Math.floor(nodes.length / 2);
    const subCluster1 = nodes.slice(0, mid);
    const subCluster2 = nodes.slice(mid);
    for (const n1 of subCluster1) {
      for (const n2 of subCluster2) {
        matrix.set([n1, n2], 0);
        matrix.set([n2, n1], 0);
      }
    }
    return true;
  }

  private async rewireNode(node: number, matrix: Matrix): Promise<boolean> {
    const currentConnections = matrix.get([node]).map((val: number, idx: number) => val === 1 ? idx : -1).filter((idx: number) => idx !== -1);
    const candidates = Array.from({ length: matrix.size()[0] }, (_, i) => i)
      .filter(i => i !== node && !currentConnections.includes(i) && matrix.get([node, i]) === 0);
    
    if (candidates.length > 0) {
      const oldTarget = currentConnections[Math.floor(Math.random() * currentConnections.length)];
      const newTarget = candidates[Math.floor(Math.random() * candidates.length)];
      matrix.set([node, oldTarget], 0);
      matrix.set([oldTarget, node], 0);
      matrix.set([node, newTarget], 1);
      matrix.set([newTarget, node], 1);
      return true;
    }
    return false;
  }

  private async strengthenConnections(nodes: number[], matrix: Matrix): Promise<boolean> {
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        if (matrix.get([nodes[i], nodes[j]]) === 0) {
          matrix.set([nodes[i], nodes[j]], 1);
          matrix.set([nodes[j], nodes[i]], 1);
        }
      }
    }
    return true;
  }

  private async weakenConnections(nodes: number[], matrix: Matrix): Promise<boolean> {
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        if (matrix.get([nodes[i], nodes[j]]) === 1) {
          matrix.set([nodes[i], nodes[j]], 0);
          matrix.set([nodes[j], nodes[i]], 0);
        }
      }
    }
    return true;
  }

  private async calculateNetworkMetrics(): Promise<NetworkMetrics> {
    const size = this.adjacencyMatrix.size()[0];
    const efficiency = this.calculateGlobalEfficiency();
    const clustering = this.calculateClusteringCoefficient();
    const pathLength = this.calculateAveragePathLength();
    const smallWorld = this.calculateSmallWorldIndex();
    const modularity = this.calculateModularity();

    return {
      globalEfficiency: efficiency,
      clusteringCoefficient: clustering,
      averagePathLength: pathLength,
      smallWorldIndex: smallWorld,
      modularityScore: modularity
    };
  }

  private calculateGlobalEfficiency(): number {
    const size = this.adjacencyMatrix.size()[0];
    const distances = this.floydWarshall();
    let efficiency = 0;
    for (let i = 0; i < size; i++) {
      for (let j = i + 1; j < size; j++) {
        const dist = distances.get([i, j]);
        efficiency += dist === Infinity ? 0 : 1 / dist;
      }
    }
    return efficiency / (size * (size - 1) / 2) || 0;
  }

  private calculateClusteringCoefficient(): number {
    const size = this.adjacencyMatrix.size()[0];
    let totalClustering = 0;
    for (let i = 0; i < size; i++) {
      const neighbors = this.adjacencyMatrix.get([i]).map((val: number, idx: number) => val === 1 ? idx : -1)
        .filter((idx: number) => idx !== -1);
      let triangles = 0;
      const possibleTriangles = neighbors.length * (neighbors.length - 1) / 2;
      for (let j = 0; j < neighbors.length; j++) {
        for (let k = j + 1; k < neighbors.length; k++) {
          if (this.adjacencyMatrix.get([neighbors[j], neighbors[k]]) === 1) triangles++;
        }
      }
      totalClustering += possibleTriangles > 0 ? triangles / possibleTriangles : 0;
    }
    return totalClustering / size;
  }

  private calculateAveragePathLength(): number {
    const distances = this.floydWarshall();
    const size = this.adjacencyMatrix.size()[0];
    let totalLength = 0;
    let count = 0;
    for (let i = 0; i < size; i++) {
      for (let j = i + 1; j < size; j++) {
        const dist = distances.get([i, j]);
        if (dist !== Infinity) {
          totalLength += dist;
          count++;
        }
      }
    }
    return count > 0 ? totalLength / count : Infinity;
  }

  private calculateSmallWorldIndex(): number {
    const clustering = this.calculateClusteringCoefficient();
    const pathLength = this.calculateAveragePathLength();
    const randomClustering = this.adjacencyMatrix.toArray().flat().reduce((sum: number, val: number) => sum + val, 0) / (this.adjacencyMatrix.size()[0] * (this.adjacencyMatrix.size()[0] - 1));
    const randomPathLength = Math.log(this.adjacencyMatrix.size()[0]) / Math.log(2); // Approx for random graph
    return (clustering / randomClustering) / (pathLength / randomPathLength);
  }

  private calculateModularity(): number {
    const size = this.adjacencyMatrix.size()[0];
    const degrees = this.adjacencyMatrix.toArray().map((row: number[]) => row.reduce((sum: number, val: number) => sum + val, 0));
    const totalEdges = degrees.reduce((sum: number, val: number) => sum + val, 0) / 2;
    const clusters = this.identifyClusters();
    let modularity = 0;

    clusters.forEach(cluster => {
      let withinEdges = 0;
      let expectedEdges = 0;
      for (const i of cluster) {
        for (const j of cluster) {
          if (this.adjacencyMatrix.get([i, j]) === 1) withinEdges += 0.5; // Count each edge once
        }
        expectedEdges += degrees[i] * degrees[i] / (2 * totalEdges);
      }
      modularity += withinEdges - expectedEdges;
    });

    return modularity / (2 * totalEdges);
  }

  private floydWarshall(): Matrix {
    const size = this.adjacencyMatrix.size()[0];
    const distances = matrix(this.adjacencyMatrix.toArray()) as Matrix;
    for (let i = 0; i < size; i++) {
      for (let j = 0; j < size; j++) {
        if (i !== j && distances.get([i, j]) === 0) distances.set([i, j], Infinity);
      }
    }
    for (let k = 0; k < size; k++) {
      for (let i = 0; i < size; i++) {
        for (let j = 0; j < size; j++) {
          if (distances.get([i, k]) + distances.get([k, j]) < distances.get([i, j])) {
            distances.set([i, j], distances.get([i, k]) + distances.get([k, j]));
          }
        }
      }
    }
    return distances;
  }

  private rankStrategies(strategies: AdaptationStrategy[]): AdaptationStrategy[] {
    return strategies.sort((a, b) => (b.confidence * b.impact * b.quantumImpact) - (a.confidence * a.impact * a.quantumImpact));
  }

  private estimateStrategyImpact(type: AdaptationStrategy['type'], nodes: number[], performanceMetrics: Map<number, any>): number {
    switch (type) {
      case 'merge':
        return this.calculateClusterSimilarity(nodes.slice(0, nodes.length / 2), nodes.slice(nodes.length / 2), performanceMetrics) * 0.8;
      case 'split':
        return (1 - this.calculateIntraClusterConnectivity(nodes)) * 0.7;
      case 'rewire':
        return this.calculateNodeIsolation(nodes[0]) * 0.6;
      case 'strengthen':
        return nodes.reduce((sum, n) => sum + TopologyNode.centrality[n], 0) / nodes.length * 0.5;
      case 'weaken':
        return (1 - nodes.reduce((sum, n) => sum + TopologyNode.stability[n], 0) / nodes.length) * 0.5;
      default:
        return 0.5;
    }
  }

  private calculateQuantumImpact(nodes: number[]): number {
    return mean(nodes.map(n => TopologyNode.quantumEntanglement[n] || 0));
  }

  private async calculateClusterSimilarity(cluster1: number[], cluster2: number[], performanceMetrics: Map<number, any>): Promise<number> {
    const states1 = cluster1.map(n => performanceMetrics.get(n) || {});
    const states2 = cluster2.map(n => performanceMetrics.get(n) || {});
    const syncResults = await Promise.all(states1.map(s1 => 
      Promise.all(states2.map(s2 => this.neuralSync.synchronizeStates(s1, s2)))
    ));
    return mean(syncResults.flat().map(r => r.coherenceScore));
  }

  private calculateClusterVariance(cluster: number[], performanceMetrics: Map<number, any>): number {
    const influences = cluster.map(n => performanceMetrics.get(n)?.influence || TopologyNode.influence[n]);
    return std(influences) || 0;
  }

  private updateNodeStates(nodes: number[], performanceMetrics: Map<number, any>) {
    nodes.forEach(node => {
      const metrics = performanceMetrics.get(node) || {};
      TopologyNode.influence[node] = metrics.influence || TopologyNode.influence[node];
      TopologyNode.centrality[node] = this.calculateCentrality(node);
      TopologyNode.stability[node] = metrics.stability || TopologyNode.stability[node];
      TopologyNode.connectivityDegree[node] = this.adjacencyMatrix.get([node]).reduce((sum: number, val: number) => sum + val, 0);
      TopologyNode.clusterCoefficient[node] = this.calculateClusteringCoefficientForNode(node);
      TopologyNode.pathLength[node] = this.calculateAveragePathLengthForNode(node);
      TopologyNode.adaptationRate[node] = metrics.adaptationRate || TopologyNode.adaptationRate[node];
      TopologyNode.quantumEntanglement[node] = this.quantumEncoder.calculateEntanglementMetrics(
        this.quantumEncoder.encodeState(metrics, node.toString())
      ).score;
      TopologyNode.neuralCoherence[node] = (async () => {
        const sync = await this.neuralSync.synchronizeStates(metrics, metrics);
        return sync.coherenceScore;
      })();
    });
  }

  private calculateCentrality(node: number): number {
    const distances = this.floydWarshall();
    const totalCloseness = distances.get([node]).reduce((sum: number, dist: number) => 
      sum + (dist === Infinity ? 0 : 1 / dist), 0);
    return totalCloseness / (this.adjacencyMatrix.size()[0] - 1);
  }

  private calculateClusteringCoefficientForNode(node: number): number {
    const neighbors = this.adjacencyMatrix.get([node]).map((val: number, idx: number) => val === 1 ? idx : -1)
      .filter((idx: number) => idx !== -1);
    let triangles = 0;
    const possibleTriangles = neighbors.length * (neighbors.length - 1) / 2;
    for (let i = 0; i < neighbors.length; i++) {
      for (let j = i + 1; j < neighbors.length; j++) {
        if (this.adjacencyMatrix.get([neighbors[i], neighbors[j]]) === 1) triangles++;
      }
    }
    return possibleTriangles > 0 ? triangles / possibleTriangles : 0;
  }

  private calculateAveragePathLengthForNode(node: number): number {
    const distances = this.floydWarshall();
    const validDistances = distances.get([node]).filter((d: number) => d !== Infinity && d !== 0);
    return validDistances.length > 0 ? mean(validDistances) : Infinity;
  }

  private updateNetworkMetrics(strategies: AdaptationStrategy[]) {
    strategies.forEach(strategy => {
      strategy.nodes.forEach(node => {
        TopologyNode.centrality[node] = this.calculateCentrality(node);
        TopologyNode.clusterCoefficient[node] = this.calculateClusteringCoefficientForNode(node);
        TopologyNode.pathLength[node] = this.calculateAveragePathLengthForNode(node);
      });
    });
  }

  private visualizeTopology(): string {
    const size = this.adjacencyMatrix.size()[0];
    const lines = [];
    for (let i = 0; i < size; i++) {
      const connections = this.adjacencyMatrix.get([i]).map((val: number, idx: number) => val === 1 ? idx : -1)
        .filter((idx: number) => idx !== -1);
      lines.push(`Node ${i} [C:${TopologyNode.centrality[i].toFixed(2)}, S:${TopologyNode.stability[i].toFixed(2)}]: ${connections.join(', ')}`);
    }
    return lines.join('\n');
  }

  // New: Forecast topology evolution
  async forecastEvolution(nodes: number[], steps: number = this.PREDICTION_HORIZON): Promise<{ predictions: NetworkMetrics[]; confidence: number }> {
    const currentMetrics = await this.calculateNetworkMetrics();
    const predMetrics = nodes.map(n => ({
      cpuUsage: TopologyNode.centrality[n],
      memoryUsage: TopologyNode.stability[n],
      networkLatency: TopologyNode.pathLength[n] * 50,
      errorRate: 1 - TopologyNode.neuralCoherence[n],
      timestamp: Date.now()
    }));
    predMetrics.forEach(m => this.predictor.addDataPoint(nodes[0].toString(), m));
    const { predictions, confidence } = await this.predictor.predict(nodes[0].toString(), steps);
    const forecast = predictions.map(p => ({
      globalEfficiency: currentMetrics.globalEfficiency * (1 - p[0] * 0.1),
      clusteringCoefficient: currentMetrics.clusteringCoefficient * (1 + p[1] * 0.1),
      averagePathLength: currentMetrics.averagePathLength * (1 + p[2] * 0.1),
      smallWorldIndex: currentMetrics.smallWorldIndex,
      modularityScore: currentMetrics.modularityScore * (1 - p[3] * 0.1)
    }));
    this.wsServer.broadcastStateUpdate("topology_forecast", { predictions: forecast, confidence });
    return { predictions: forecast, confidence };
  }
}
