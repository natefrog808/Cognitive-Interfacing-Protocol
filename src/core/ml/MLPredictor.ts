import * as tf from '@tensorflow/tfjs';
import { defineComponent, Types } from 'bitecs';

// ML model state component
const MLModelState = defineComponent({
  isTraining: Types.ui8,
  lastTrainingTime: Types.ui32,
  predictionAccuracy: Types.f32,
  modelVersion: Types.ui32,
  dataPoints: Types.ui32
});

interface TimeSeriesWindow {
  features: number[][];
  label: number[];
}

export class MLPredictor {
  private model: tf.LayersModel | null = null;
  private sequenceLength: number = 10;
  private predictionHorizon: number = 5;
  private features: string[] = ['cpuUsage', 'memoryUsage', 'networkLatency', 'errorRate'];
  private dataBuffer: Map<string, number[][]> = new Map();
  private readonly BUFFER_SIZE = 1000;

  async initialize() {
    this.model = tf.sequential();
    
    // Add LSTM layers
    this.model.add(tf.layers.lstm({
      units: 50,
      returnSequences: true,
      inputShape: [this.sequenceLength, this.features.length]
    }));
    
    this.model.add(tf.layers.dropout(0.2));
    
    this.model.add(tf.layers.lstm({
      units: 50,
      returnSequences: false
    }));
    
    this.model.add(tf.layers.dropout(0.2));
    
    this.model.add(tf.layers.dense({
      units: this.predictionHorizon,
      activation: 'linear'
    }));

    // Compile model
    this.model.compile({
      optimizer: tf.train.adam(0.001),
      loss: 'meanSquaredError',
      metrics: ['mse']
    });
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
    buffer.push(this.features.map(feature => metrics[feature]));

    if (buffer.length > this.BUFFER_SIZE) {
      buffer.shift();
    }

    // Update model state
    const id = parseInt(entityId);
    MLModelState.dataPoints[id] = buffer.length;
    
    // Retrain model if we have enough data
    if (buffer.length >= this.BUFFER_SIZE) {
      await this.trainModel(entityId);
    }
  }

  private prepareTimeSeriesData(entityId: string): TimeSeriesWindow[] {
    const buffer = this.dataBuffer.get(entityId)!;
    const windows: TimeSeriesWindow[] = [];

    for (let i = 0; i <= buffer.length - this.sequenceLength - this.predictionHorizon; i++) {
      const window: TimeSeriesWindow = {
        features: buffer.slice(i, i + this.sequenceLength),
        label: buffer.slice(i + this.sequenceLength, i + this.sequenceLength + this.predictionHorizon)
          .map(dataPoint => dataPoint[0]) // Predict CPU usage as primary metric
      };
      windows.push(window);
    }

    return windows;
  }

  private async trainModel(entityId: string) {
    if (!this.model) await this.initialize();

    const id = parseInt(entityId);
    MLModelState.isTraining[id] = 1;

    try {
      const windows = this.prepareTimeSeriesData(entityId);
      
      // Convert to tensors
      const xs = tf.tensor3d(windows.map(w => w.features));
      const ys = tf.tensor2d(windows.map(w => w.label));

      // Train model
      const history = await this.model!.fit(xs, ys, {
        epochs: 50,
        batchSize: 32,
        validationSplit: 0.2,
        callbacks: {
          onEpochEnd: async (epoch, logs) => {
            // Update accuracy metrics
            MLModelState.predictionAccuracy[id] = 1 - (logs?.loss ?? 0);
          }
        }
      });

      // Update model state
      MLModelState.lastTrainingTime[id] = Date.now();
      MLModelState.modelVersion[id]++;
      
      // Cleanup tensors
      xs.dispose();
      ys.dispose();

      return history;
    } finally {
      MLModelState.isTraining[id] = 0;
    }
  }

  async predict(entityId: string, steps: number = 5): Promise<{
    predictions: number[][];
    confidence: number;
  }> {
    if (!this.model) throw new Error('Model not initialized');

    const buffer = this.dataBuffer.get(entityId);
    if (!buffer || buffer.length < this.sequenceLength) {
      throw new Error('Insufficient data for prediction');
    }

    // Prepare input sequence
    const inputSequence = buffer.slice(-this.sequenceLength);
    const inputTensor = tf.tensor3d([inputSequence]);

    try {
      // Generate predictions
      const predictions: number[][] = [];
      let currentInput = inputSequence;

      for (let i = 0; i < steps; i++) {
        const prediction = await this.model.predict(tf.tensor3d([currentInput])) as tf.Tensor;
        const predictionData = await prediction.data();
        
        // Convert prediction to feature vector
        const predictionVector = Array.from(predictionData).map(value => 
          Math.max(0, Math.min(1, value)) // Clamp values between 0 and 1
        );
        
        predictions.push(predictionVector);

        // Update input sequence for next prediction
        currentInput = [...currentInput.slice(1), predictionVector];
        
        prediction.dispose();
      }

      // Calculate confidence based on model accuracy and prediction variance
      const id = parseInt(entityId);
      const modelAccuracy = MLModelState.predictionAccuracy[id];
      const predictionVariance = this.calculatePredictionVariance(predictions);
      const confidence = modelAccuracy * (1 - predictionVariance);

      return {
        predictions,
        confidence
      };
    } finally {
      inputTensor.dispose();
    }
  }

  private calculatePredictionVariance(predictions: number[][]): number {
    // Calculate variance of predictions to estimate uncertainty
    const mean = predictions.reduce((sum, pred) => sum + pred[0], 0) / predictions.length;
    const variance = predictions.reduce((sum, pred) => 
      sum + Math.pow(pred[0] - mean, 2), 0
    ) / predictions.length;
    
    return Math.min(variance, 1);
  }
}
