import { defineComponent, Types } from 'bitecs';
import { distance, mean, std, inv, svd, matrix, Matrix } from 'mathjs';

// Advanced Metrics Component (expanded for richer insights)
const AnomalyMetrics = defineComponent({
  anomalyScore: Types.f32,      // Overall anomaly score (0-1)
  isolationScore: Types.f32,    // Isolation Forest score
  dbscanLabel: Types.i8,        // DBSCAN cluster label (-1 = noise, 0+ = cluster)
  lastUpdateTime: Types.ui32,   // Timestamp of last update
  severity: Types.ui8,          // Severity level (0-3)
  confidence: Types.f32,        // Confidence in detection (0-1)
  forecastScore: Types.f32,     // Predicted anomaly likelihood (0-1)
  quantumWeight: Types.f32,     // Influence of quantum state (0-1)
  neuralCoherence: Types.f32    // Neural sync coherence (0-1)
});

// Mini LSTM-like predictor for anomaly forecasting
class AnomalyPredictor {
  private weights: number[][] = [[0.5, 0.3], [0.2, 0.4]]; // Simplified 2x2 weights
  private bias: number[] = [0.1, 0.1];
  private memory: number[] = [0, 0]; // Hidden state
  private horizon: number = 5; // Prediction steps

  update(data: number[]): void {
    const input = [data[0] || 0, data[1] || 0]; // Simplified to first two dimensions
    this.memory = input.map((x, i) => 
      this.sigmoid(this.weights[i].reduce((sum, w, j) => sum + w * this.memory[j], this.bias[i]) + x)
    );
  }

  predict(): number {
    return this.sigmoid(this.memory.reduce((sum, m) => sum + m, 0) / this.memory.length);
  }

  private sigmoid(x: number): number {
    return 1 / (1 + Math.exp(-x));
  }
}

class IsolationTree {
  private height: number;
  private splitValue: number;
  private splitAttribute: number;
  private left: IsolationTree | null = null;
  private right: IsolationTree | null = null;

  constructor(data: number[][], height: number = 0, maxHeight: number = 12) {
    this.height = height;
    if (data.length <= 1 || height >= maxHeight) return;

    const variances = data[0].map((_, i) => std(data.map(point => point[i])) || 0);
    this.splitAttribute = variances.indexOf(Math.max(...variances));
    const values = data.map(point => point[this.splitAttribute]);
    const minVal = Math.min(...values);
    const maxVal = Math.max(...values);
    this.splitValue = minVal + (maxVal - minVal) * Math.random();

    const leftData = data.filter(point => point[this.splitAttribute] < this.splitValue);
    const rightData = data.filter(point => point[this.splitAttribute] >= this.splitValue);

    if (leftData.length > 0) this.left = new IsolationTree(leftData, height + 1, maxHeight);
    if (rightData.length > 0) this.right = new IsolationTree(rightData, height + 1, maxHeight);
  }

  getPathLength(point: number[]): number {
    if (this.splitAttribute === undefined) return this.height;
    return point[this.splitAttribute] < this.splitValue && this.left
      ? this.left.getPathLength(point)
      : this.right
      ? this.right.getPathLength(point)
      : this.height;
  }
}

class IsolationForest {
  private trees: IsolationTree[] = [];
  private nTrees: number;
  private sampleSize: number;

  constructor(nTrees: number = 100, sampleSize: number = 256) {
    this.nTrees = nTrees;
    this.sampleSize = sampleSize;
  }

  fit(data: number[][]) {
    this.trees = [];
    const actualSampleSize = Math.min(this.sampleSize, data.length);
    for (let i = 0; i < this.nTrees; i++) {
      const sample = Array(actualSampleSize).fill(0).map(() => 
        data[Math.floor(Math.random() * data.length)]
      );
      this.trees.push(new IsolationTree(sample));
    }
  }

  predict(point: number[]): number {
    const avgPathLength = this.trees.reduce((sum, tree) => sum + tree.getPathLength(point), 0) / this.nTrees;
    const c = 2 * (Math.log(this.sampleSize) + 0.5772156649) - 2 * (this.sampleSize - 1) / this.sampleSize;
    return Math.pow(2, -avgPathLength / c);
  }

  adjustTrees(factor: number) {
    this.nTrees = Math.max(10, Math.min(500, Math.floor(this.nTrees * factor)));
  }
}

class DBSCAN {
  private eps: number;
  private minPts: number;

  constructor(eps: number = 0.5, minPts: number = 5) {
    this.eps = eps;
    this.minPts = minPts;
  }

  fit(data: number[][]): number[] {
    const labels = new Array(data.length).fill(-1);
    let cluster = 0;

    for (let i = 0; i < data.length; i++) {
      if (labels[i] !== -1) continue;

      const neighbors = this.getNeighbors(data, i);
      if (neighbors.length < this.minPts) {
        labels[i] = 0;
        continue;
      }

      cluster++;
      labels[i] = cluster;

      let j = 0;
      while (j < neighbors.length) {
        const currentPoint = neighbors[j];
        if (labels[currentPoint] === 0) labels[currentPoint] = cluster;
        if (labels[currentPoint] === -1) {
          labels[currentPoint] = cluster;
          const currentNeighbors = this.getNeighbors(data, currentPoint);
          if (currentNeighbors.length >= this.minPts) neighbors.push(...currentNeighbors);
        }
        j++;
      }
    }
    return labels;
  }

  private getNeighbors(data: number[][], pointIdx: number): number[] {
    const neighbors: number[] = [];
    const point = data[pointIdx];
    for (let i = 0; i < data.length; i++) {
      if (i === pointIdx) continue;
      if (this.calculateDistance(point, data[i]) <= this.eps) neighbors.push(i);
    }
    return neighbors;
  }

  private calculateDistance(point1: number[], point2: number[]): number {
    return distance(point1, point2) as number;
  }

  tuneEps(data: number[][]): void {
    const distances: number[] = [];
    for (let i = 0; i < Math.min(100, data.length); i++) {
      for (let j = i + 1; j < Math.min(100, data.length); j++) {
        distances.push(this.calculateDistance(data[i], data[j]));
      }
    }
    this.eps = mean(distances) + std(distances);
  }

  adjustMinPts(factor: number) {
    this.minPts = Math.max(2, Math.min(20, Math.floor(this.minPts * factor)));
  }
}

export class AdvancedAnomalyDetector {
  private isolationForest: IsolationForest;
  private dbscan: DBSCAN;
  private predictor: AnomalyPredictor;
  private dataHistory: Map<string, number[][]> = new Map();
  private anomalyHistory: Map<string, { score: number, correct: boolean }[]> = new Map();
  private readonly HISTORY_WINDOW = 1000;
  private readonly MIN_DATA_POINTS = 100;
  private featureMeans: Map<string, number[]> = new Map();
  private featureStds: Map<string, number[]> = new Map();
  private quantumInfluence: number = 0.5; // Placeholder for quantum state integration
  private neuralCoherence: number = 0.9;  // Placeholder for neural sync integration

  constructor(nTrees: number = 100, sampleSize: number = 256, eps: number = 0.5, minPts: number = 5) {
    this.isolationForest = new IsolationForest(nTrees, sampleSize);
    this.dbscan = new DBSCAN(eps, minPts);
    this.predictor = new AnomalyPredictor();
  }

  updateData(entityId: string, newData: number[], quantumState?: { coherence: number }, neuralState?: { coherence: number }) {
    if (!this.dataHistory.has(entityId)) {
      this.dataHistory.set(entityId, []);
      this.anomalyHistory.set(entityId, []);
      this.featureMeans.set(entityId, new Array(newData.length).fill(0));
      this.featureStds.set(entityId, new Array(newData.length).fill(1));
    }

    const history = this.dataHistory.get(entityId)!;
    const normalizedData = this.normalizeData(newData, entityId);
    history.push(normalizedData);
    this.predictor.update(normalizedData);

    if (history.length > this.HISTORY_WINDOW) history.shift();

    this.updateStatistics(entityId, history);
    if (quantumState) this.quantumInfluence = quantumState.coherence;
    if (neuralState) this.neuralCoherence = neuralState.coherence;

    if (history.length >= this.MIN_DATA_POINTS) {
      this.isolationForest.fit(history);
      this.dbscan.tuneEps(history);
    }

    const eid = parseInt(entityId);
    AnomalyMetrics.lastUpdateTime[eid] = Date.now();
  }

  detectAnomalies(entityId: string, currentData: number[]): {
    isAnomaly: boolean;
    score: number;
    cluster: number;
    severity: number;
    confidence: number;
    forecastScore: number;
    signature: { deviation: number[], trend: number };
  } {
    const history = this.dataHistory.get(entityId);
    if (!history || history.length < this.MIN_DATA_POINTS) {
      return { isAnomaly: false, score: 0, cluster: -1, severity: 0, confidence: 0, forecastScore: 0, signature: { deviation: [], trend: 0 } };
    }

    const normalizedData = this.normalizeData(currentData, entityId);
    const isolationScore = this.isolationForest.predict(normalizedData) * this.quantumInfluence;
    const dbscanLabels = this.dbscan.fit([...history, normalizedData]);
    const currentLabel = dbscanLabels[dbscanLabels.length - 1];
    const forecastScore = this.predictor.predict() * this.neuralCoherence;

    const isAnomaly = isolationScore > 0.6 || currentLabel === 0 || forecastScore > 0.7;
    const severity = this.calculateSeverity(isolationScore, currentLabel, forecastScore);
    const confidence = this.calculateConfidence(isolationScore, currentLabel, history, normalizedData);
    const signature = this.computeAnomalySignature(history, normalizedData);

    const eid = parseInt(entityId);
    AnomalyMetrics.anomalyScore[eid] = isolationScore;
    AnomalyMetrics.isolationScore[eid] = isolationScore;
    AnomalyMetrics.dbscanLabel[eid] = currentLabel;
    AnomalyMetrics.severity[eid] = severity;
    AnomalyMetrics.confidence[eid] = confidence;
    AnomalyMetrics.forecastScore[eid] = forecastScore;
    AnomalyMetrics.quantumWeight[eid] = this.quantumInfluence;
    AnomalyMetrics.neuralCoherence[eid] = this.neuralCoherence;

    // Self-tune based on this detection
    this.selfTune(entityId, isAnomaly, isolationScore);

    return { isAnomaly, score: isolationScore, cluster: currentLabel, severity, confidence, forecastScore, signature };
  }

  private selfTune(entityId: string, isAnomaly: boolean, score: number) {
    const anomalies = this.anomalyHistory.get(entityId)!;
    anomalies.push({ score, correct: isAnomaly }); // Assume correct until feedback provided
    if (anomalies.length > this.HISTORY_WINDOW) anomalies.shift();

    const accuracy = anomalies.reduce((sum, a) => sum + (a.correct ? 1 : 0), 0) / anomalies.length;
    if (anomalies.length > 100 && accuracy < 0.8) {
      this.isolationForest.adjustTrees(1.1); // Increase trees if accuracy drops
      this.dbscan.adjustMinPts(1.05); // Slightly increase minPts
    }
  }

  getAdaptiveThresholds(entityId: string): { [key: string]: number } {
    const history = this.dataHistory.get(entityId);
    if (!history || history.length < this.MIN_DATA_POINTS) return {};

    const thresholds: { [key: string]: number } = {};
    for (let i = 0; i < history[0].length; i++) {
      const values = history.map(point => point[i]);
      thresholds[`dim_${i}`] = this.computeAdaptiveThreshold(values);
    }
    return thresholds;
  }

  private normalizeData(data: number[], entityId: string): number[] {
    const means = this.featureMeans.get(entityId)!;
    const stds = this.featureStds.get(entityId)!;
    return data.map((val, i) => (val - means[i]) / (stds[i] || 1));
  }

  private updateStatistics(entityId: string, history: number[][]) {
    const means = history[0].map((_, i) => mean(history.map(p => p[i])));
    const stds = history[0].map((_, i) => std(history.map(p => p[i])) || 1);
    this.featureMeans.set(entityId, means);
    this.featureStds.set(entityId, stds);
  }

  private computeAdaptiveThreshold(values: number[], percentile: number = 95): number {
    const sorted = [...values].sort((a, b) => a - b);
    const q3 = sorted[Math.floor(0.75 * sorted.length)];
    const q1 = sorted[Math.floor(0.25 * sorted.length)];
    const iqr = q3 - q1;
    return q3 + 1.5 * iqr;
  }

  private calculateSeverity(isolationScore: number, dbscanLabel: number, forecastScore: number): number {
    if (isolationScore > 0.8 || dbscanLabel === 0 || forecastScore > 0.85) return 3;
    if (isolationScore > 0.6 || forecastScore > 0.7) return 2;
    if (isolationScore > 0.4) return 1;
    return 0;
  }

  private calculateConfidence(isolationScore: number, dbscanLabel: number, history: number[][], currentData: number[]): number {
    const mahalanobisDist = this.calculateMahalanobisDistance(history, currentData);
    const agreement = (isolationScore > 0.6 && dbscanLabel === 0) ? 1 : 0.5;
    return Math.min(1, agreement * (1 - 1 / (1 + mahalanobisDist)));
  }

  private calculateMahalanobisDistance(history: number[][], point: number[]): number {
    const means = history[0].map((_, i) => mean(history.map(p => p[i])));
    const covMatrix = this.computeCovarianceMatrix(history, means);
    const { U, S, V } = svd(matrix(covMatrix));
    const invCov = inv(covMatrix); // Proper inverse via SVD
    const diffs = point.map((val, i) => val - means[i]);
    return Math.sqrt(diffs.reduce((sum, d, i) => 
      sum + d * (invCov as Matrix).get([i, 0, history[0].length]) // Simplified indexing
        .reduce((s, v, j) => s + v * diffs[j], 0), 0));
  }

  private computeCovarianceMatrix(data: number[][], means: number[]): number[][] {
    const n = data.length;
    const cov = data[0].map(() => new Array(data[0].length).fill(0));
    for (let i = 0; i < data[0].length; i++) {
      for (let j = 0; j < data[0].length; j++) {
        cov[i][j] = data.reduce((sum, point) => 
          sum + (point[i] - means[i]) * (point[j] - means[j]), 0) / (n - 1);
      }
    }
    return cov;
  }

  private computeAnomalySignature(history: number[][], currentData: number[]): { deviation: number[], trend: number } {
    const lastTen = history.slice(-10);
    const deviation = currentData.map((val, i) => val - mean(lastTen.map(p => p[i])));
    const trend = mean(deviation.map((d, i) => d * (i + 1))); // Weighted trend
    return { deviation, trend };
  }
}
