import { defineComponent, Types } from 'bitecs';
import { distance, mean, std } from 'mathjs';

// Advanced Metrics Component (enhanced with severity and confidence)
const AnomalyMetrics = defineComponent({
  anomalyScore: Types.f32,      // Overall anomaly score (0-1)
  isolationScore: Types.f32,    // Isolation Forest score
  dbscanLabel: Types.i8,        // DBSCAN cluster label (-1 = noise, 0+ = cluster)
  lastUpdateTime: Types.ui32,   // Timestamp of last update
  severity: Types.ui8,          // Severity level (0 = none, 1 = low, 2 = medium, 3 = high)
  confidence: Types.f32         // Confidence in anomaly detection (0-1)
});

class IsolationTree {
  private height: number;
  private splitValue: number;
  private splitAttribute: number;
  private left: IsolationTree | null = null;
  private right: IsolationTree | null = null;

  constructor(data: number[][], height: number = 0, maxHeight: number = 12) {
    this.height = height;
    if (data.length <= 1 || height >= maxHeight) return;

    // Choose split attribute with highest variance for better isolation
    const variances = data[0].map((_, i) => std(data.map(point => point[i])) || 0);
    this.splitAttribute = variances.indexOf(Math.max(...variances));
    const values = data.map(point => point[this.splitAttribute]);
    const minVal = Math.min(...values);
    const maxVal = Math.max(...values);
    this.splitValue = minVal + (maxVal - minVal) * Math.random(); // Random split within range

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
  private readonly nTrees: number;
  private readonly sampleSize: number;

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
    return Math.pow(2, -avgPathLength / c); // Normalized anomaly score (0-1)
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
        labels[i] = 0; // Noise
        continue;
      }

      cluster++;
      labels[i] = cluster;

      let j = 0;
      while (j < neighbors.length) {
        const currentPoint = neighbors[j];
        if (labels[currentPoint] === 0) {
          labels[currentPoint] = cluster;
        }
        if (labels[currentPoint] === -1) {
          labels[currentPoint] = cluster;
          const currentNeighbors = this.getNeighbors(data, currentPoint);
          if (currentNeighbors.length >= this.minPts) {
            neighbors.push(...currentNeighbors);
          }
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
      if (this.calculateDistance(point, data[i]) <= this.eps) {
        neighbors.push(i);
      }
    }
    return neighbors;
  }

  private calculateDistance(point1: number[], point2: number[]): number {
    return distance(point1, point2) as number; // Euclidean distance
  }

  // New method to tune eps dynamically
  tuneEps(data: number[][]): void {
    const distances: number[] = [];
    for (let i = 0; i < Math.min(100, data.length); i++) {
      for (let j = i + 1; j < Math.min(100, data.length); j++) {
        distances.push(this.calculateDistance(data[i], data[j]));
      }
    }
    this.eps = mean(distances) + std(distances); // Mean + 1 SD for robustness
  }
}

export class AdvancedAnomalyDetector {
  private isolationForest: IsolationForest;
  private dbscan: DBSCAN;
  private dataHistory: Map<string, number[][]> = new Map();
  private readonly HISTORY_WINDOW = 1000;
  private readonly MIN_DATA_POINTS = 100;
  private featureMeans: Map<string, number[]> = new Map(); // For normalization
  private featureStds: Map<string, number[]> = new Map();  // For normalization

  constructor(nTrees: number = 100, sampleSize: number = 256, eps: number = 0.5, minPts: number = 5) {
    this.isolationForest = new IsolationForest(nTrees, sampleSize);
    this.dbscan = new DBSCAN(eps, minPts);
  }

  updateData(entityId: string, newData: number[]) {
    if (!this.dataHistory.has(entityId)) {
      this.dataHistory.set(entityId, []);
      this.featureMeans.set(entityId, new Array(newData.length).fill(0));
      this.featureStds.set(entityId, new Array(newData.length).fill(1));
    }

    const history = this.dataHistory.get(entityId)!;
    const normalizedData = this.normalizeData(newData, entityId);
    history.push(normalizedData);

    if (history.length > this.HISTORY_WINDOW) {
      history.shift();
    }

    // Update running statistics
    this.updateStatistics(entityId, history);

    if (history.length >= this.MIN_DATA_POINTS) {
      this.isolationForest.fit(history);
      this.dbscan.tuneEps(history); // Dynamically adjust eps
    }

    // Update BitECS component
    const eid = parseInt(entityId);
    AnomalyMetrics.lastUpdateTime[eid] = Date.now();
  }

  detectAnomalies(entityId: string, currentData: number[]): {
    isAnomaly: boolean;
    score: number;
    cluster: number;
    severity: number;
    confidence: number;
  } {
    const history = this.dataHistory.get(entityId);
    if (!history || history.length < this.MIN_DATA_POINTS) {
      return { isAnomaly: false, score: 0, cluster: -1, severity: 0, confidence: 0 };
    }

    const normalizedData = this.normalizeData(currentData, entityId);
    const isolationScore = this.isolationForest.predict(normalizedData);
    const dbscanLabels = this.dbscan.fit([...history, normalizedData]);
    const currentLabel = dbscanLabels[dbscanLabels.length - 1];

    const isAnomaly = isolationScore > 0.6 || currentLabel === 0;
    const severity = this.calculateSeverity(isolationScore, currentLabel);
    const confidence = this.calculateConfidence(isolationScore, currentLabel, history, normalizedData);

    const eid = parseInt(entityId);
    AnomalyMetrics.anomalyScore[eid] = isolationScore;
    AnomalyMetrics.isolationScore[eid] = isolationScore;
    AnomalyMetrics.dbscanLabel[eid] = currentLabel;
    AnomalyMetrics.severity[eid] = severity;
    AnomalyMetrics.confidence[eid] = confidence;

    return { isAnomaly, score: isolationScore, cluster: currentLabel, severity, confidence };
  }

  getAdaptiveThresholds(entityId: string): { [key: string]: number } {
    const history = this.dataHistory.get(entityId);
    if (!history || history.length < this.MIN_DATA_POINTS) {
      return {};
    }

    const thresholds: { [key: string]: number } = {};
    for (let i = 0; i < history[0].length; i++) {
      const values = history.map(point => point[i]);
      thresholds[`dim_${i}`] = this.computeAdaptiveThreshold(values);
    }
    return thresholds;
  }

  // New: Normalize data for consistency
  private normalizeData(data: number[], entityId: string): number[] {
    const means = this.featureMeans.get(entityId)!;
    const stds = this.featureStds.get(entityId)!;
    return data.map((val, i) => (val - means[i]) / (stds[i] || 1)); // Avoid division by zero
  }

  // New: Update running mean and std for normalization
  private updateStatistics(entityId: string, history: number[][]) {
    const means = history[0].map((_, i) => mean(history.map(p => p[i])));
    const stds = history[0].map((_, i) => std(history.map(p => p[i])) || 1); // Ensure non-zero std
    this.featureMeans.set(entityId, means);
    this.featureStds.set(entityId, stds);
  }

  // Enhanced: Compute adaptive threshold with statistical robustness
  private computeAdaptiveThreshold(values: number[], percentile: number = 95): number {
    const sorted = [...values].sort((a, b) => a - b);
    const index = Math.floor((percentile / 100) * sorted.length);
    const q3 = sorted[Math.floor(0.75 * sorted.length)];
    const q1 = sorted[Math.floor(0.25 * sorted.length)];
    const iqr = q3 - q1;
    return q3 + 1.5 * iqr; // Upper whisker of boxplot for anomaly threshold
  }

  // New: Calculate anomaly severity
  private calculateSeverity(isolationScore: number, dbscanLabel: number): number {
    if (isolationScore > 0.8 || dbscanLabel === 0) return 3; // High
    if (isolationScore > 0.6) return 2; // Medium
    if (isolationScore > 0.4) return 1; // Low
    return 0; // None
  }

  // New: Calculate confidence in anomaly detection
  private calculateConfidence(isolationScore: number, dbscanLabel: number, history: number[][], currentData: number[]): number {
    const mahalanobisDist = this.calculateMahalanobisDistance(history, currentData);
    const agreement = (isolationScore > 0.6 && dbscanLabel === 0) ? 1 : 0.5; // Higher if both agree
    return Math.min(1, agreement * (1 - 1 / (1 + mahalanobisDist))); // Confidence based on distance
  }

  // New: Mahalanobis distance for robust anomaly scoring
  private calculateMahalanobisDistance(history: number[][], point: number[]): number {
    const means = history[0].map((_, i) => mean(history.map(p => p[i])));
    const diffs = point.map((val, i) => val - means[i]);
    const covMatrix = this.computeCovarianceMatrix(history, means);
    const invCov = this.pseudoInverse(covMatrix); // Simplified inverse
    const mahalanobis = Math.sqrt(diffs.reduce((sum, d, i) => 
      sum + d * invCov[i].reduce((s, v, j) => s + v * diffs[j], 0), 0));
    return mahalanobis;
  }

  // New: Compute covariance matrix
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

  // New: Simplified pseudo-inverse for small matrices
  private pseudoInverse(matrix: number[][]): number[][] {
    // Placeholder: Real implementation would use SVD from a library like mathjs
    const size = matrix.length;
    const inv = matrix.map(row => [...row]);
    for (let i = 0; i < size; i++) inv[i][i] += 0.0001; // Regularization to avoid singularity
    return inv; // Simplified—replace with proper inverse in production
  }
}
