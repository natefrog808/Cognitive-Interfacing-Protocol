import { defineComponent, Types } from 'bitecs';
import { distance } from 'mathjs';

// Advanced Metrics Component
const AnomalyMetrics = defineComponent({
  anomalyScore: Types.f32,
  isolationScore: Types.f32,
  dbscanLabel: Types.i8,
  lastUpdateTime: Types.ui32
});

class IsolationTree {
  private height: number;
  private splitValue: number;
  private splitAttribute: number;
  private left: IsolationTree | null = null;
  private right: IsolationTree | null = null;

  constructor(data: number[][], height: number = 0) {
    this.height = height;
    if (data.length <= 1 || height >= 10) return; // Limit tree height

    // Choose random split attribute and value
    this.splitAttribute = Math.floor(Math.random() * data[0].length);
    const values = data.map(point => point[this.splitAttribute]);
    this.splitValue = values[Math.floor(Math.random() * values.length)];

    // Split data
    const leftData = data.filter(point => point[this.splitAttribute] < this.splitValue);
    const rightData = data.filter(point => point[this.splitAttribute] >= this.splitValue);

    if (leftData.length > 0) this.left = new IsolationTree(leftData, height + 1);
    if (rightData.length > 0) this.right = new IsolationTree(rightData, height + 1);
  }

  getPathLength(point: number[]): number {
    if (!this.splitAttribute) return this.height;

    if (point[this.splitAttribute] < this.splitValue) {
      return this.left ? this.left.getPathLength(point) : this.height;
    } else {
      return this.right ? this.right.getPathLength(point) : this.height;
    }
  }
}

class IsolationForest {
  private trees: IsolationTree[] = [];
  private readonly nTrees: number;

  constructor(nTrees: number = 100) {
    this.nTrees = nTrees;
  }

  fit(data: number[][]) {
    for (let i = 0; i < this.nTrees; i++) {
      // Sample data randomly with replacement
      const sampleSize = Math.min(256, data.length);
      const sample = Array(sampleSize).fill(0).map(() => 
        data[Math.floor(Math.random() * data.length)]
      );
      this.trees.push(new IsolationTree(sample));
    }
  }

  predict(point: number[]): number {
    const avgPathLength = this.trees.reduce(
      (sum, tree) => sum + tree.getPathLength(point), 
      0
    ) / this.nTrees;

    // Normalize score
    const c = 2 * (Math.log(256) + 0.5772156649) - 2 * (256 - 1) / 256;
    return Math.pow(2, -avgPathLength / c);
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
    for (let i = 0; i < data.length; i++) {
      if (i === pointIdx) continue;
      if (this.calculateDistance(data[pointIdx], data[i]) <= this.eps) {
        neighbors.push(i);
      }
    }
    return neighbors;
  }

  private calculateDistance(point1: number[], point2: number[]): number {
    return distance(point1, point2);
  }
}

export class AdvancedAnomalyDetector {
  private isolationForest: IsolationForest;
  private dbscan: DBSCAN;
  private dataHistory: Map<string, number[][]> = new Map();
  private readonly HISTORY_WINDOW = 1000;

  constructor() {
    this.isolationForest = new IsolationForest();
    this.dbscan = new DBSCAN();
  }

  updateData(entityId: string, newData: number[]) {
    if (!this.dataHistory.has(entityId)) {
      this.dataHistory.set(entityId, []);
    }

    const history = this.dataHistory.get(entityId)!;
    history.push(newData);

    if (history.length > this.HISTORY_WINDOW) {
      history.shift();
    }

    if (history.length >= 100) {  // Minimum data points for retraining
      this.isolationForest.fit(history);
    }
  }

  detectAnomalies(entityId: string, currentData: number[]): {
    isAnomaly: boolean;
    score: number;
    cluster: number;
  } {
    const history = this.dataHistory.get(entityId);
    if (!history || history.length < 100) {
      return { isAnomaly: false, score: 0, cluster: -1 };
    }

    const isolationScore = this.isolationForest.predict(currentData);
    const dbscanLabels = this.dbscan.fit([...history, currentData]);
    const currentLabel = dbscanLabels[dbscanLabels.length - 1];

    return {
      isAnomaly: isolationScore > 0.6 || currentLabel === 0,
      score: isolationScore,
      cluster: currentLabel
    };
  }

  getAdaptiveThresholds(entityId: string): {
    [key: string]: number;
  } {
    const history = this.dataHistory.get(entityId);
    if (!history || history.length < 100) {
      return {};
    }

    const thresholds: { [key: string]: number } = {};
    for (let i = 0; i < history[0].length; i++) {
      const values = history.map(point => point[i]);
      thresholds[`dim_${i}`] = this.computeAdaptiveThreshold(values);
    }

    return thresholds;
  }

  private computeAdaptiveThreshold(values: number[], percentile: number = 95): number {
    const sorted = [...values].sort((a, b) => a - b);
    const index = Math.floor((percentile / 100) * sorted.length);
    return sorted[index];
  }
}
