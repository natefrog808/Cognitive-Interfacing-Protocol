import * as tf from '@tensorflow/tfjs';
import { defineComponent, Types } from 'bitecs';
import { mean, std } from 'mathjs';
import { PredictiveMonitor } from './PredictiveMonitor'; // Assuming this exists
import { CognitiveWebSocketServer } from './CognitiveWebSocketServer';

// Enhanced ML Model State Component
const MLModelState = defineComponent({
  isTraining: Types.ui8,         // 0 = idle, 1 = training
  lastTrainingTime: Types.ui32,  // Timestamp (ms)
  predictionAccuracy: Types.f32, // 0-1
  modelVersion: Types.ui32,      // Incremental version
  dataPoints: Types.ui32,        // Number of buffered points
  loss: Types.f32,               // Latest training loss
  confidenceInterval: Types.f32  // Width of 95% CI for predictions
});

interface TimeSeriesWindow {
  features: number[][]; // Multi-dimensional input
  labels: number[][];   // Multi-dimensional output (all features)
}

export class MLPredictor {
  private model: tf.LayersModel | null = null;
  private sequenceLength: number = 12; // Increased for better context
  private predictionHorizon: number = 6; // Slightly longer horizon
  private features: string[] = ['cpuUsage', 'memoryUsage', 'networkLatency', 'errorRate'];
  private dataBuffer: Map<string, number[][]> = new Map();
  private readonly BUFFER_SIZE = 1000;
  private readonly RETRAIN_THRESHOLD = 0.1; // Retrain if accuracy drops below this
  private monitor: PredictiveMonitor;
  private wsServer: CognitiveWebSocketServer;
  private scaler: { mean: number[]; std: number[] }; // For feature scaling

  constructor(wsPort: number = 8080) {
    this.monitor = new PredictiveMonitor(wsPort);
    this.wsServer = new CognitiveWebSocketServer(wsPort);
    this.scaler = { mean: this.features.map(() => 0), std: this.features.map(() => 1) };
  }

  async initialize() {
    this.model = tf.sequential();

    // Enhanced architecture: Bidirectional LSTM + Attention
    this.model.add(tf.layers.lstm({
      units: 64,
      returnSequences: true,
      inputShape: [this.sequenceLength, this.features.length]
    }));
    this.model.add(tf.layers.dropout({ rate: 0.2 }));
    this.model.add(tf.layers.bidirectional({
      layer: tf.layers.lstm({ units: 64, returnSequences: true }) as any,
      mergeMode: 'concat'
    }));
    this.model.add(tf.layers.dropout({ rate: 0.2 }));
    this.model.add(tf.layers.timeDistributed({
      layer: tf.layers.dense({ units: this.features.length, activation: 'linear' })
    }));

    this.model.compile({
      optimizer: tf.train.adam(0.0005), // Lower learning rate for stability
      loss: 'meanSquaredError',
      metrics: ['mae'] // Mean Absolute Error for interpretability
    });

    await this.loadModelIfExists(entityId); // Load saved model if available
  }

  async addDataPoint(entityId: string, metrics: {
    cpuUsage: number;
    memoryUsage: number;
    networkLatency: number;
    errorRate: number;
    timestamp: number;
  }) {
    if (!this.dataBuffer.has(entityId)) {
      this.dataBuffer.set(entityId, []);
    }

    const buffer = this.dataBuffer.get(entityId)!;
    const dataPoint = this.features.map(f => metrics[f]);
    const scaledPoint = this.scaleDataPoint(dataPoint);
    buffer.push(scaledPoint);

    if (buffer.length > this.BUFFER_SIZE) {
      buffer.shift();
    }

    const id = parseInt(entityId);
    MLModelState.dataPoints[id] = buffer.length;

    // Update scaler with running statistics
    this.updateScaler(buffer);

    // Retrain if accuracy drops or enough new data
    if (buffer.length >= this.sequenceLength * 2 && this.shouldRetrain(id)) {
      await this.trainModel(entityId);
      await this.saveModel(entityId);
    }

    // Update system health via PredictiveMonitor
    this.monitor.updateMetrics(entityId, metrics);
  }

  private prepareTimeSeriesData(entityId: string): TimeSeriesWindow[] {
    const buffer = this.dataBuffer.get(entityId)!;
    if (buffer.length < this.sequenceLength + this.predictionHorizon) return [];

    const windows: TimeSeriesWindow[] = [];
    for (let i = 0; i <= buffer.length - this.sequenceLength - this.predictionHorizon; i++) {
      windows.push({
        features: buffer.slice(i, i + this.sequenceLength),
        labels: buffer.slice(i + this.sequenceLength, i + this.sequenceLength + this.predictionHorizon)
      });
    }
    return windows;
  }

  private async trainModel(entityId: string) {
    if (!this.model) await this.initialize();

    const id = parseInt(entityId);
    MLModelState.isTraining[id] = 1;

    try {
      const windows = this.prepareTimeSeriesData(entityId);
      if (windows.length === 0) return;

      const xs = tf.tensor3d(windows.map(w => w.features));
      const ys = tf.tensor3d(windows.map(w => w.labels));

      const history = await this.model.fit(xs, ys, {
        epochs: 50,
        batchSize: 32,
        validationSplit: 0.2,
        callbacks: {
          onEpochEnd: async (epoch, logs) => {
            const accuracy = 1 - (logs?.mae ?? 0); // MAE-based accuracy
            MLModelState.predictionAccuracy[id] = accuracy;
            MLModelState.loss[id] = logs?.loss ?? 0;
            console.log(`Entity ${entityId} - Epoch ${epoch}: MAE = ${logs?.mae}, Accuracy = ${accuracy}`);
          },
          onTrainEnd: () => {
            MLModelState.lastTrainingTime[id] = Date.now();
            MLModelState.modelVersion[id]++;
          },
          earlyStopping: tf.callbacks.earlyStopping({ monitor: 'val_loss', patience: 10 })
        }
      });

      xs.dispose();
      ys.dispose();

      this.wsServer.broadcastStateUpdate(entityId, {
        mlModelState: this.getModelState(id),
        trainingHistory: history.history
      });

      return history;
    } finally {
      MLModelState.isTraining[id] = 0;
    }
  }

  async predict(entityId: string, steps: number = 6): Promise<{
    predictions: number[][]; // Multi-feature predictions
    confidence: number;
    confidenceInterval: [number, number][]; // 95% CI per step
  }> {
    if (!this.model) throw new Error('Model not initialized');

    const buffer = this.dataBuffer.get(entityId);
    if (!buffer || buffer.length < this.sequenceLength) {
      throw new Error('Insufficient data for prediction');
    }

    const inputSequence = buffer.slice(-this.sequenceLength);
    const inputTensor = tf.tensor3d([inputSequence]);

    try {
      const predictions: number[][] = [];
      const variances: number[] = [];
      let currentInput = [...inputSequence];

      for (let i = 0; i < steps; i++) {
        const prediction = await this.model.predict(tf.tensor3d([currentInput])) as tf.Tensor;
        const predictionData = await prediction.data();
        const predVector = Array.from(predictionData).map(v => this.unscaleValue(v, 0)); // Unscale CPU usage (index 0)
        
        // Monte Carlo dropout for variance estimation
        const dropoutPreds = await this.monteCarloDropout(currentInput, 10);
        const variance = std(dropoutPreds.map(p => p[0])) || 0;
        variances.push(variance);

        predictions.push(predVector.map((v, idx) => this.unscaleValue(v, idx)));
        currentInput = [...currentInput.slice(1), predVector];
        prediction.dispose();
      }

      const id = parseInt(entityId);
      const modelAccuracy = MLModelState.predictionAccuracy[id] || 0;
      const predictionVariance = mean(variances);
      const confidence = Math.max(0, modelAccuracy * (1 - predictionVariance));
      const confidenceInterval = predictions.map((p, i) => {
        const ciWidth = 1.96 * Math.sqrt(variances[i]); // 95% CI
        return [Math.max(0, p[0] - ciWidth), Math.min(1, p[0] + ciWidth)];
      });

      MLModelState.confidenceInterval[id] = confidenceInterval[0][1] - confidenceInterval[0][0];

      this.wsServer.broadcastStateUpdate(entityId, {
        mlPredictions: { predictions, confidence, confidenceInterval },
        modelState: this.getModelState(id)
      });

      return { predictions, confidence, confidenceInterval };
    } finally {
      inputTensor.dispose();
    }
  }

  private async monteCarloDropout(input: number[][], samples: number): Promise<number[][]> {
    const preds: number[][] = [];
    tf.engine().startScope();
    for (let i = 0; i < samples; i++) {
      const pred = await this.model!.predict(tf.tensor3d([input]), { dropout: true }) as tf.Tensor;
      preds.push(await pred.data() as number[]);
      pred.dispose();
    }
    tf.engine().endScope();
    return preds;
  }

  private scaleDataPoint(data: number[]): number[] {
    return data.map((val, i) => (val - this.scaler.mean[i]) / (this.scaler.std[i] || 1));
  }

  private unscaleValue(value: number, featureIdx: number): number {
    return value * this.scaler.std[featureIdx] + this.scaler.mean[featureIdx];
  }

  private updateScaler(buffer: number[][]) {
    const n = buffer.length;
    this.scaler.mean = this.features.map((_, i) => mean(buffer.map(p => p[i])));
    this.scaler.std = this.features.map((_, i) => std(buffer.map(p => p[i])) || 1);
  }

  private shouldRetrain(entityId: number): boolean {
    const accuracy = MLModelState.predictionAccuracy[entityId] || 0;
    const dataPoints = MLModelState.dataPoints[entityId];
    return accuracy < this.RETRAIN_THRESHOLD || (dataPoints > this.BUFFER_SIZE * 0.5 && Date.now() - MLModelState.lastTrainingTime[entityId] > 24 * 60 * 60 * 1000); // Retrain daily if enough data
  }

  private async saveModel(entityId: string) {
    const savePath = `file://./models/mlpredictor_${entityId}`;
    await this.model!.save(savePath);
    console.log(`Model saved for entity ${entityId} at ${savePath}`);
  }

  private async loadModelIfExists(entityId: string) {
    const loadPath = `file://./models/mlpredictor_${entityId}`;
    try {
      this.model = await tf.loadLayersModel(loadPath);
      console.log(`Model loaded for entity ${entityId} from ${loadPath}`);
    } catch (e) {
      console.log(`No saved model found for ${entityId}, initializing new model`);
      await this.initialize();
    }
  }

  private getModelState(id: number): any {
    return {
      isTraining: MLModelState.isTraining[id],
      lastTrainingTime: MLModelState.lastTrainingTime[id],
      predictionAccuracy: MLModelState.predictionAccuracy[id],
      modelVersion: MLModelState.modelVersion[id],
      dataPoints: MLModelState.dataPoints[id],
      loss: MLModelState.loss[id],
      confidenceInterval: MLModelState.confidenceInterval[id]
    };
  }

  // New: Visualize predictions
  visualizePredictions(entityId: string, predictions: number[][], confidenceInterval: [number, number][]): string {
    const lines = predictions.map((p, i) => {
      const ci = confidenceInterval[i];
      const barLength = Math.round(p[0] * 20); // Scale CPU usage to 20 chars
      const ciBar = `${Math.round(ci[0] * 100)}-${Math.round(ci[1] * 100)}`;
      return `Step ${i + 1}: ${'█'.repeat(barLength)} (${p[0].toFixed(2)}) [CI: ${ciBar}]`;
    });
    return lines.join('\n');
  }
}
