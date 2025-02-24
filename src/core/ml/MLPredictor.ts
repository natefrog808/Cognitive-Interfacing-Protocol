import * as tf from '@tensorflow/tfjs';
import { defineComponent, Types } from 'bitecs';
import { mean, std } from 'mathjs';
import { PredictiveMonitor } from './PredictiveMonitor';
import { CognitiveWebSocketServer } from './CognitiveWebSocketServer';
import { QuantumStateEncoder } from './quantum/QuantumStateEncoder';
import { NeuralSynchronizer } from './neural/NeuralSynchronizer';
import { AdvancedAnomalyDetector } from './anomaly/AdvancedAnomalyDetector';

// Enhanced ML Model State Component
const MLModelState = defineComponent({
  isTraining: Types.ui8,
  lastTrainingTime: Types.ui32,
  predictionAccuracy: Types.f32,
  modelVersion: Types.ui32,
  dataPoints: Types.ui32,
  loss: Types.f32,
  confidenceInterval: Types.f32,
  quantumInfluence: Types.f32,   // New: Quantum coherence impact
  anomalySensitivity: Types.f32  // New: Anomaly-driven retraining trigger
});

interface TimeSeriesWindow {
  features: number[][]; // Multi-dimensional input
  labels: number[][];   // Multi-dimensional output
}

export class MLPredictor {
  private model: tf.LayersModel | null = null;
  private sequenceLength: number = 12;
  private predictionHorizon: number = 6;
  private features: string[] = ['cpuUsage', 'memoryUsage', 'networkLatency', 'errorRate', 'quantumCoherence', 'anomalyScore'];
  private dataBuffer: Map<string, number[][]> = new Map();
  private readonly BUFFER_SIZE = 1000;
  private RETRAIN_THRESHOLD = 0.1; // Dynamic now
  private monitor: PredictiveMonitor;
  private wsServer: CognitiveWebSocketServer;
  private quantumEncoder: QuantumStateEncoder;
  private neuralSync: NeuralSynchronizer;
  private anomalyDetector: AdvancedAnomalyDetector;
  private scaler: { mean: number[]; std: number[] };

  constructor(wsPort: number = 8080) {
    this.monitor = new PredictiveMonitor(wsPort);
    this.wsServer = new CognitiveWebSocketServer(wsPort);
    this.quantumEncoder = new QuantumStateEncoder();
    this.neuralSync = new NeuralSynchronizer();
    this.anomalyDetector = new AdvancedAnomalyDetector();
    this.scaler = { mean: this.features.map(() => 0), std: this.features.map(() => 1) };
  }

  async initialize() {
    await this.quantumEncoder.initialize();
    await this.neuralSync.initialize(0.5);
    this.model = tf.sequential();

    this.model.add(tf.layers.lstm({
      units: 128, // Boosted for richer patterns
      returnSequences: true,
      inputShape: [this.sequenceLength, this.features.length]
    }));
    this.model.add(tf.layers.dropout({ rate: 0.2 }));
    this.model.add(tf.layers.bidirectional({
      layer: tf.layers.lstm({ units: 64, returnSequences: true }) as any,
      mergeMode: 'concat'
    }));
    this.model.add(tf.layers.attention()); // New: Attention for key trends
    this.model.add(tf.layers.dropout({ rate: 0.2 }));
    this.model.add(tf.layers.timeDistributed({
      layer: tf.layers.dense({ units: this.features.length, activation: 'linear' })
    }));

    this.model.compile({
      optimizer: tf.train.adam(0.0003), // Finer tuning
      loss: 'meanSquaredError',
      metrics: ['mae']
    });

    console.log("MLPredictor initialized—ready to forecast the future!");
  }

  async addDataPoint(entityId: string, metrics: {
    cpuUsage: number;
    memoryUsage: number;
    networkLatency: number;
    errorRate: number;
    timestamp: number;
  }) {
    if (!this.dataBuffer.has(entityId)) this.dataBuffer.set(entityId, []);

    const buffer = this.dataBuffer.get(entityId)!;
    const quantumState = await this.quantumEncoder.createQuantumRegister(entityId);
    const quantumCoherence = this.quantumEncoder.measureState(quantumState).coherence;
    const baseData = this.features.slice(0, 4).map(f => metrics[f]);
    this.anomalyDetector.updateData(entityId, baseData, { coherence: quantumCoherence });
    const anomaly = this.anomalyDetector.detectAnomalies(entityId, baseData);
    const dataPoint = [...baseData, quantumCoherence, anomaly.score];
    const scaledPoint = this.scaleDataPoint(dataPoint);
    buffer.push(scaledPoint);

    if (buffer.length > this.BUFFER_SIZE) buffer.shift();

    const id = parseInt(entityId);
    MLModelState.dataPoints[id] = buffer.length;
    MLModelState.quantumInfluence[id] = quantumCoherence;
    MLModelState.anomalySensitivity[id] = anomaly.forecastScore;

    this.updateScaler(buffer);

    if (buffer.length >= this.sequenceLength * 2 && this.shouldRetrain(id)) {
      await this.trainModel(entityId);
      await this.saveModel(entityId);
    }

    this.monitor.updateMetrics(entityId, metrics);
    this.wsServer.broadcastStateUpdate(entityId, { metrics, anomaly, modelState: this.getModelState(id) });
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
            const accuracy = 1 - (logs?.mae ?? 0);
            MLModelState.predictionAccuracy[id] = accuracy;
            MLModelState.loss[id] = logs?.loss ?? 0;
            this.RETRAIN_THRESHOLD = Math.max(0.05, 0.2 - accuracy * 0.1); // Adaptive threshold
            console.log(`Entity ${entityId} - Epoch ${epoch}: MAE = ${logs?.mae}, Accuracy = ${accuracy}`);
          },
          onTrainEnd: () => {
            MLModelState.lastTrainingTime[id] = Date.now();
            MLModelState.modelVersion[id]++;
          },
          earlyStopping: tf.callbacks.earlyStopping({ monitor: 'val_loss', patience: 10 })
        }
      });

      tf.dispose([xs, ys]);
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
    predictions: number[][];        // Multi-feature predictions
    confidence: number;
    confidenceInterval: [number, number][]; // Per-step CI
    emergentState: any;             // New: Predicted emergent state
    trendVector: number[];          // New: Trend direction
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
        const predVector = Array.from(await prediction.data()).map((v, idx) => this.unscaleValue(v, idx));
        
        const dropoutPreds = await this.monteCarloDropout(currentInput, 10);
        const variance = mean(dropoutPreds.map(p => std(p) || 0));
        variances.push(variance);

        predictions.push(predVector);
        currentInput = [...currentInput.slice(1), predVector];
        prediction.dispose();
      }

      const id = parseInt(entityId);
      const modelAccuracy = MLModelState.predictionAccuracy[id] || 0;
      const predictionVariance = mean(variances);
      const confidence = Math.max(0, modelAccuracy * (1 - predictionVariance) * (1 + MLModelState.quantumInfluence[id]));
      const confidenceInterval = predictions.map((p, i) => {
        const ciWidth = 1.96 * Math.sqrt(variances[i]);
        return [Math.max(0, p[0] - ciWidth), Math.min(1, p[0] + ciWidth)];
      });

      MLModelState.confidenceInterval[id] = confidenceInterval[0][1] - confidenceInterval[0][0];

      // Generate emergent state
      const lastPred = predictions[predictions.length - 1];
      const emergentState = await this.neuralSync.generateNovelState({
        cognitive: { awareness: lastPred[0], coherence: lastPred[2] / 50 }, // Network latency scaled
        emotional: { stress: lastPred[3], motivation: MLModelState.quantumInfluence[id] }
      }, confidence);

      // Calculate trend vector
      const trendVector = this.features.map((_, idx) => {
        const series = predictions.map(p => p[idx]);
        return series[series.length - 1] - series[0];
      });

      this.wsServer.broadcastStateUpdate(entityId, {
        mlPredictions: { predictions, confidence, confidenceInterval, emergentState, trendVector },
        modelState: this.getModelState(id)
      });

      return { predictions, confidence, confidenceInterval, emergentState, trendVector };
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
    const anomalySens = MLModelState.anomalySensitivity[entityId] || 0;
    const dataPoints = MLModelState.dataPoints[entityId];
    return (accuracy < this.RETRAIN_THRESHOLD + anomalySens * 0.2) || 
           (dataPoints > this.BUFFER_SIZE * 0.5 && Date.now() - MLModelState.lastTrainingTime[entityId] > 12 * 60 * 60 * 1000);
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
      confidenceInterval: MLModelState.confidenceInterval[id],
      quantumInfluence: MLModelState.quantumInfluence[id],
      anomalySensitivity: MLModelState.anomalySensitivity[id]
    };
  }

  visualizePredictions(entityId: string, predictions: number[][], confidenceInterval: [number, number][]): string {
    const labels = ['CPU', 'Mem', 'Lat', 'Err', 'Qnt', 'Ano'];
    const lines = predictions.map((p, i) => {
      const ci = confidenceInterval[i];
      const bars = p.map((v, idx) => {
        const barLength = Math.round(v * (idx === 2 ? 0.4 : 20)); // Scale latency differently
        return `${labels[idx]}: ${'█'.repeat(barLength)} (${v.toFixed(2)})`;
      });
      return `Step ${i + 1}: ${bars.join(' | ')} [CI: ${ci[0].toFixed(2)}-${ci[1].toFixed(2)}]`;
    });
    return lines.join('\n');
  }
}
