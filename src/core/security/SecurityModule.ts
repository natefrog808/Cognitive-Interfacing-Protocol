import CryptoJS from 'crypto-js';
import * as tf from '@tensorflow/tfjs';
import { defineComponent, Types } from 'bitecs';
import { mean, std } from 'mathjs';
import { MLPredictor } from './MLPredictor'; // Assuming this exists
import { QuantumStateEncoder } from './QuantumStateEncoder';
import { CognitiveWebSocketServer } from './CognitiveWebSocketServer';

// Enhanced Alert Thresholds Component
const AlertThresholds = defineComponent({
  moodSwing: Types.f32,
  stressPeak: Types.f32,
  cognitiveOverload: Types.f32,
  fearSpike: Types.f32,
  angerTrigger: Types.f32,
  dynamicAdjustment: Types.f32, // Adaptive threshold factor (0-1)
  lastThresholdUpdate: Types.ui32 // Timestamp of last adjustment
});

// Enhanced Historical State Component
const StateHistory = defineComponent({
  previousMood: Types.f32,
  previousStress: Types.f32,
  previousCognitiveLoad: Types.f32,
  previousFear: Types.f32,
  previousAnger: Types.f32,
  lastUpdateTime: Types.ui32,
  stateHash: Types.ui32 // Hash for integrity checking
});

export class SecurityManager {
  private static readonly KEY_SIZE = 256;
  private static readonly IV_SIZE = 16;
  private keyStore: Map<string, string> = new Map();
  private wsServer: CognitiveWebSocketServer;
  private quantumEncoder: QuantumStateEncoder;

  constructor(wsPort: number = 8080) {
    this.wsServer = new CognitiveWebSocketServer(wsPort);
    this.quantumEncoder = new QuantumStateEncoder(wsPort);
    this.initializeQuantum();
  }

  async initializeQuantum() {
    await this.quantumEncoder.initialize();
  }

  generateKey(): string {
    return CryptoJS.lib.WordArray.random(this.KEY_SIZE / 8).toString(CryptoJS.enc.Hex);
  }

  storeKey(entityId: string, key: string): void {
    // In production, use a secure key management service (e.g., AWS KMS)
    this.keyStore.set(entityId, key);
    console.log(`Key stored for entity ${entityId}`);
  }

  encryptState(state: any, entityId: string): string {
    const key = this.keyStore.get(entityId);
    if (!key) throw new Error('No encryption key found for entity');

    // Quantum-safe enhancement: Mix with quantum state encoding
    const register = this.quantumEncoder.encodeState(state, entityId);
    const quantumDigest = this.generateQuantumDigest(register);

    const iv = CryptoJS.lib.WordArray.random(this.IV_SIZE);
    const plaintext = JSON.stringify({ state, quantumDigest });
    const encrypted = CryptoJS.AES.encrypt(plaintext, key, {
      iv,
      mode: CryptoJS.mode.GCM,
      padding: CryptoJS.pad.Pkcs7
    });

    const result = `${iv.toString(CryptoJS.enc.Hex)}:${encrypted.toString()}`;
    this.wsServer.broadcastStateUpdate(entityId, { encryptedState: result });
    return result;
  }

  decryptState(encrypted: string, entityId: string): any {
    const key = this.keyStore.get(entityId);
    if (!key) throw new Error('No encryption key found for entity');

    const [ivString, ciphertext] = encrypted.split(':');
    const iv = CryptoJS.enc.Hex.parse(ivString);

    const decrypted = CryptoJS.AES.decrypt(ciphertext, key, {
      iv,
      mode: CryptoJS.mode.GCM,
      padding: CryptoJS.pad.Pkcs7
    });

    const decryptedData = JSON.parse(decrypted.toString(CryptoJS.enc.Utf8));
    const { state, quantumDigest } = decryptedData;

    // Verify quantum integrity
    const register = this.quantumEncoder.encodeState(state, entityId);
    const currentDigest = this.generateQuantumDigest(register);
    if (quantumDigest !== currentDigest) {
      this.wsServer.broadcastAlert(entityId, {
        type: 'security_violation',
        severity: 'error',
        message: 'Quantum state integrity compromised'
      });
      throw new Error('State integrity violation detected');
    }

    return state;
  }

  private generateQuantumDigest(register: QuantumRegister): string {
    const entanglementScore = this.quantumEncoder.calculateEntanglementMetrics(register).score;
    const stateVector = register.qubits.map(q => `${q.real.toFixed(3)},${q.imag.toFixed(3)}`).join('');
    return CryptoJS.SHA256(stateVector + entanglementScore).toString(CryptoJS.enc.Hex);
  }

  // New: Secure key rotation
  async rotateKey(entityId: string): Promise<string> {
    const newKey = this.generateKey();
    this.storeKey(entityId, newKey);
    this.wsServer.broadcastStateUpdate(entityId, { event: 'key_rotated', newKeyHash: CryptoJS.SHA256(newKey).toString() });
    return newKey;
  }
}

export class AlertManager {
  private thresholds: Map<number, any> = new Map();
  private history: Map<number, Array<{ state: any; timestamp: number }>> = new Map();
  private patternModel: tf.LayersModel | null = null;
  private predictor: MLPredictor;
  private wsServer: CognitiveWebSocketServer;

  constructor(wsPort: number = 8080) {
    this.predictor = new MLPredictor(wsPort);
    this.wsServer = new CognitiveWebSocketServer(wsPort);
    this.initializePatternModel();
  }

  async initializePatternModel() {
    await this.predictor.initialize();
    this.patternModel = tf.sequential();
    this.patternModel.add(tf.layers.dense({
      units: 64,
      activation: 'relu',
      inputShape: [5] // mood, stress, cognitiveLoad, fear, anger
    }));
    this.patternModel.add(tf.layers.dropout({ rate: 0.2 }));
    this.patternModel.add(tf.layers.dense({ units: 32, activation: 'relu' }));
    this.patternModel.add(tf.layers.dense({ units: 4, activation: 'softmax' })); // 4 emotional patterns
    this.patternModel.compile({
      optimizer: tf.train.adam(0.001),
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy']
    });
  }

  setThresholds(entityId: number, thresholds: {
    moodSwing?: number;
    stressPeak?: number;
    cognitiveOverload?: number;
    fearSpike?: number;
    angerTrigger?: number;
  }) {
    AlertThresholds.moodSwing[entityId] = thresholds.moodSwing ?? 0.3;
    AlertThresholds.stressPeak[entityId] = thresholds.stressPeak ?? 0.8;
    AlertThresholds.cognitiveOverload[entityId] = thresholds.cognitiveOverload ?? 0.9;
    AlertThresholds.fearSpike[entityId] = thresholds.fearSpike ?? 0.7;
    AlertThresholds.angerTrigger[entityId] = thresholds.angerTrigger ?? 0.7;
    this.adjustDynamicThresholds(entityId);
  }

  updateHistory(entityId: number, state: any) {
    StateHistory.previousMood[entityId] = state.mood;
    StateHistory.previousStress[entityId] = state.stress;
    StateHistory.previousCognitiveLoad[entityId] = state.cognitiveLoad;
    StateHistory.previousFear[entityId] = state.fear || 0;
    StateHistory.previousAnger[entityId] = state.anger || 0;
    StateHistory.lastUpdateTime[entityId] = Date.now();
    StateHistory.stateHash[entityId] = this.hashState(state);

    if (!this.history.has(entityId)) this.history.set(entityId, []);
    const history = this.history.get(entityId)!;
    history.push({ state, timestamp: Date.now() });
    if (history.length > 1000) history.shift(); // Limit history
  }

  checkAlerts(entityId: number, currentState: any): Array<{
    type: string;
    severity: 'info' | 'warning' | 'error';
    message: string;
    confidence: number;
  }> {
    const alerts: Array<{ type: string; severity: 'info' | 'warning' | 'error'; message: string; confidence: number }> = [];
    const id = entityId;

    // Check basic thresholds with dynamic adjustments
    const moodDelta = Math.abs(currentState.mood - StateHistory.previousMood[id]);
    if (moodDelta > AlertThresholds.moodSwing[id] * (1 + AlertThresholds.dynamicAdjustment[id])) {
      alerts.push({
        type: 'mood_swing',
        severity: moodDelta > 0.5 ? 'error' : 'warning',
        message: `Mood swing detected: ${moodDelta.toFixed(2)} units`,
        confidence: Math.min(1, moodDelta / AlertThresholds.moodSwing[id])
      });
    }

    if (currentState.stress > AlertThresholds.stressPeak[id] * (1 + AlertThresholds.dynamicAdjustment[id])) {
      alerts.push({
        type: 'high_stress',
        severity: 'error',
        message: `Critical stress: ${(currentState.stress * 100).toFixed(1)}%`,
        confidence: Math.min(1, currentState.stress / AlertThresholds.stressPeak[id])
      });
    }

    if (currentState.cognitiveLoad > AlertThresholds.cognitiveOverload[id] * (1 + AlertThresholds.dynamicAdjustment[id])) {
      alerts.push({
        type: 'cognitive_overload',
        severity: 'error',
        message: `Cognitive overload: ${(currentState.cognitiveLoad * 100).toFixed(1)}%`,
        confidence: Math.min(1, currentState.cognitiveLoad / AlertThresholds.cognitiveOverload[id])
      });
    }

    if (currentState.fear > AlertThresholds.fearSpike[id] * (1 + AlertThresholds.dynamicAdjustment[id])) {
      alerts.push({
        type: 'fear_spike',
        severity: 'warning',
        message: `Fear spike: ${(currentState.fear * 100).toFixed(1)}%`,
        confidence: Math.min(1, currentState.fear / AlertThresholds.fearSpike[id])
      });
    }

    if (currentState.anger > AlertThresholds.angerTrigger[id] * (1 + AlertThresholds.dynamicAdjustment[id])) {
      alerts.push({
        type: 'anger_trigger',
        severity: 'warning',
        message: `Anger spike: ${(currentState.anger * 100).toFixed(1)}%`,
        confidence: Math.min(1, currentState.anger / AlertThresholds.angerTrigger[id])
      });
    }

    this.updateHistory(entityId, currentState);
    this.wsServer.broadcastAlert(entityId.toString(), { alerts, timestamp: Date.now() });
    return alerts;
  }

  async analyzeEmotionalPatterns(entityId: number, timeWindow: number = 3600000): Promise<Array<{
    pattern: string;
    confidence: number;
    predictedImpact: number;
  }>> {
    const history = this.history.get(entityId);
    if (!history || history.length < 24) return [];

    const features = history.slice(-Math.min(history.length, timeWindow / 3600000)).map(h => [
      h.state.mood,
      h.state.stress,
      h.state.cognitiveLoad,
      h.state.fear || 0,
      h.state.anger || 0
    ]);

    const inputTensor = tf.tensor2d(features);
    const prediction = this.patternModel!.predict(inputTensor) as tf.Tensor;
    const patternProbs = await prediction.array() as number[][];
    const latestProbs = patternProbs[patternProbs.length - 1];

    const patterns = [
      { name: 'emotional_stability', desc: 'Stable emotional state' },
      { name: 'stress_spike', desc: 'Sudden stress increase' },
      { name: 'anger_outburst', desc: 'Rapid anger escalation' },
      { name: 'fear_response', desc: 'Fear-driven reaction' }
    ];

    // Train model incrementally if enough data
    if (features.length >= 24) {
      await this.trainPatternModel(features, this.generatePatternLabels(features));
    }

    // Predict future impact with MLPredictor
    const predInput = features[features.length - 1];
    const predMetrics = { cpuUsage: predInput[2], memoryUsage: predInput[1], networkLatency: 50, errorRate: predInput[0], timestamp: Date.now() };
    const { predictions, confidence } = await this.predictor.predict(entityId.toString(), 6);
    const predictedStress = mean(predictions.map(p => p[1])); // Stress index

    const results = patterns.map((p, i) => ({
      pattern: p.name,
      confidence: latestProbs[i],
      predictedImpact: latestProbs[i] > 0.5 ? predictedStress * latestProbs[i] : 0
    })).filter(p => p.confidence > 0.3);

    this.wsServer.broadcastStateUpdate(entityId.toString(), {
      emotionalPatterns: results,
      visualization: this.visualizePatterns(entityId, features, results)
    });

    return results;
  }

  private async trainPatternModel(features: number[][], labels: number[][]) {
    const xs = tf.tensor2d(features);
    const ys = tf.tensor2d(labels);
    await this.patternModel!.fit(xs, ys, {
      epochs: 10,
      batchSize: 8,
      validationSplit: 0.2,
      callbacks: tf.callbacks.earlyStopping({ monitor: 'val_loss', patience: 3 })
    });
    xs.dispose();
    ys.dispose();
  }

  private generatePatternLabels(features: number[][]): number[][] {
    return features.map(f => {
      const mood = f[0], stress = f[1], load = f[2], fear = f[3], anger = f[4];
      if (stress < 0.3 && Math.abs(mood) < 0.5) return [1, 0, 0, 0]; // Stability
      if (stress > 0.7) return [0, 1, 0, 0]; // Stress spike
      if (anger > 0.6) return [0, 0, 1, 0]; // Anger outburst
      if (fear > 0.6) return [0, 0, 0, 1]; // Fear response
      return [0.25, 0.25, 0.25, 0.25]; // Default balanced
    });
  }

  private adjustDynamicThresholds(entityId: number) {
    const history = this.history.get(entityId);
    if (!history || history.length < 24) return;

    const recent = history.slice(-24);
    const moodStd = std(recent.map(h => h.state.mood)) || 0;
    const stressStd = std(recent.map(h => h.state.stress)) || 0;
    const loadStd = std(recent.map(h => h.state.cognitiveLoad)) || 0;
    AlertThresholds.dynamicAdjustment[entityId] = Math.min(0.5, (moodStd + stressStd + loadStd) / 3);
    AlertThresholds.lastThresholdUpdate[entityId] = Date.now();
  }

  private hashState(state: any): number {
    const str = JSON.stringify(state);
    return parseInt(CryptoJS.SHA256(str).toString(CryptoJS.enc.Hex).slice(0, 8), 16);
  }

  private visualizePatterns(entityId: number, features: number[][], patterns: Array<{ pattern: string; confidence: number; predictedImpact: number }>): string {
    const lines = features.map((f, i) => {
      const moodBar = '█'.repeat(Math.round(Math.abs(f[0]) * 10));
      const stressBar = '█'.repeat(Math.round(f[1] * 10));
      return `${i}h ago: Mood[${f[0] > 0 ? '+' : ''}${moodBar}] Stress[${stressBar}]`;
    });
    const patternSummary = patterns.map(p => `${p.pattern}: ${(p.confidence * 100).toFixed(1)}% (Impact: ${p.predictedImpact.toFixed(2)})`).join('\n');
    return `${lines.slice(-5).join('\n')}\n\nPatterns:\n${patternSummary}`;
  }
}
