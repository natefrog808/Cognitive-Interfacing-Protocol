import CryptoJS from 'crypto-js';
import * as tf from '@tensorflow/tfjs';
import { defineComponent, Types } from 'bitecs';
import { mean, std } from 'mathjs';
import { MLPredictor } from './MLPredictor';
import { QuantumStateEncoder } from './quantum/QuantumStateEncoder';
import { CognitiveWebSocketServer } from './CognitiveWebSocketServer';
import { NeuralSynchronizer } from './neural/NeuralSynchronizer';
import { AdvancedAnomalyDetector } from './anomaly/AdvancedAnomalyDetector';

// Enhanced Alert Thresholds Component
const AlertThresholds = defineComponent({
  moodSwing: Types.f32,
  stressPeak: Types.f32,
  cognitiveOverload: Types.f32,
  fearSpike: Types.f32,
  angerTrigger: Types.f32,
  dynamicAdjustment: Types.f32,
  lastThresholdUpdate: Types.ui32,
  anomalySensitivity: Types.f32 // New: Anomaly-driven threshold boost (0-1)
});

// Enhanced Historical State Component
const StateHistory = defineComponent({
  previousMood: Types.f32,
  previousStress: Types.f32,
  previousCognitiveLoad: Types.f32,
  previousFear: Types.f32,
  previousAnger: Types.f32,
  lastUpdateTime: Types.ui32,
  stateHash: Types.ui32,
  quantumDigest: Types.ui32 // New: Quantum state integrity hash
});

export class SecurityManager {
  private static readonly KEY_SIZE = 256;
  private static readonly IV_SIZE = 16;
  private keyStore: Map<string, string> = new Map();
  private wsServer: CognitiveWebSocketServer;
  private quantumEncoder: QuantumStateEncoder;
  private neuralSync: NeuralSynchronizer;
  private anomalyDetector: AdvancedAnomalyDetector;

  constructor(wsPort: number = 8080) {
    this.wsServer = new CognitiveWebSocketServer(wsPort);
    this.quantumEncoder = new QuantumStateEncoder(wsPort);
    this.neuralSync = new NeuralSynchronizer();
    this.anomalyDetector = new AdvancedAnomalyDetector();
    this.initializeQuantum();
  }

  async initializeQuantum() {
    await Promise.all([this.quantumEncoder.initialize(), this.neuralSync.initialize()]);
    console.log("SecurityManager initialized—quantum defenses online!");
  }

  generateKey(): string {
    const key = CryptoJS.lib.WordArray.random(this.KEY_SIZE / 8).toString(CryptoJS.enc.Hex);
    const quantumReg = this.quantumEncoder.createQuantumRegister(key);
    const entanglement = this.quantumEncoder.calculateEntanglementMetrics(quantumReg).score;
    return `${key}:${entanglement.toFixed(3)}`; // Quantum-enhanced key
  }

  storeKey(entityId: string, key: string): void {
    this.keyStore.set(entityId, key);
    this.wsServer.broadcastStateUpdate(entityId, { event: 'key_stored', keyHash: CryptoJS.SHA256(key).toString() });
  }

  encryptState(state: any, entityId: string): string {
    const keyData = this.keyStore.get(entityId);
    if (!keyData) throw new Error('No encryption key found for entity');
    const [key] = keyData.split(':');

    const register = this.quantumEncoder.encodeState(state, entityId);
    const quantumDigest = this.generateQuantumDigest(register);
    const neuralDigest = this.generateNeuralDigest(state, entityId);

    const iv = CryptoJS.lib.WordArray.random(this.IV_SIZE);
    const plaintext = JSON.stringify({ state, quantumDigest, neuralDigest });
    const encrypted = CryptoJS.AES.encrypt(plaintext, key, {
      iv,
      mode: CryptoJS.mode.GCM,
      padding: CryptoJS.pad.Pkcs7
    });

    const result = `${iv.toString(CryptoJS.enc.Hex)}:${encrypted.toString()}`;
    this.wsServer.broadcastStateUpdate(entityId, { encryptedState: result, securityLevel: 'high' });
    return result;
  }

  decryptState(encrypted: string, entityId: string): any {
    const keyData = this.keyStore.get(entityId);
    if (!keyData) throw new Error('No encryption key found for entity');
    const [key] = keyData.split(':');

    const [ivString, ciphertext] = encrypted.split(':');
    const iv = CryptoJS.enc.Hex.parse(ivString);

    const decrypted = CryptoJS.AES.decrypt(ciphertext, key, {
      iv,
      mode: CryptoJS.mode.GCM,
      padding: CryptoJS.pad.Pkcs7
    });

    const decryptedData = JSON.parse(decrypted.toString(CryptoJS.enc.Utf8));
    const { state, quantumDigest, neuralDigest } = decryptedData;

    const register = this.quantumEncoder.encodeState(state, entityId);
    const currentQuantumDigest = this.generateQuantumDigest(register);
    const currentNeuralDigest = this.generateNeuralDigest(state, entityId);

    if (quantumDigest !== currentQuantumDigest || neuralDigest !== currentNeuralDigest) {
      const anomaly = this.anomalyDetector.detectAnomalies(entityId, [state.mood || 0, state.stress || 0, state.cognitiveLoad || 0]);
      this.wsServer.broadcastAlert(entityId, {
        type: 'security_violation',
        severity: anomaly.severity > 1 ? 'error' : 'warning',
        message: 'State integrity compromised (Quantum/Neural mismatch)',
        confidence: anomaly.confidence
      });
      throw new Error('State integrity violation detected');
    }

    return state;
  }

  private generateQuantumDigest(register: any): string {
    const entanglementScore = this.quantumEncoder.calculateEntanglementMetrics(register).score;
    const stateVector = register.qubits.map((q: any) => `${q.real.toFixed(3)},${q.imag.toFixed(3)}`).join('');
    return CryptoJS.SHA256(stateVector + entanglementScore).toString(CryptoJS.enc.Hex);
  }

  private generateNeuralDigest(state: any, entityId: string): string {
    return this.neuralSync.synchronizeStates(state, state)
      .then(sync => {
        const coherence = sync.coherenceScore;
        const neuralStr = `${state.mood || 0},${state.stress || 0},${state.cognitiveLoad || 0},${coherence}`;
        return CryptoJS.SHA256(neuralStr).toString(CryptoJS.enc.Hex);
      })
      .catch(() => CryptoJS.SHA256(JSON.stringify(state)).toString(CryptoJS.enc.Hex));
  }

  async rotateKey(entityId: string): Promise<string> {
    const newKey = this.generateKey();
    this.storeKey(entityId, newKey);
    const anomalyScore = this.anomalyDetector.detectAnomalies(entityId, [Math.random()]).score; // Placeholder for key rotation anomaly check
    this.wsServer.broadcastStateUpdate(entityId, {
      event: 'key_rotated',
      newKeyHash: CryptoJS.SHA256(newKey).toString(),
      anomalyScore
    });
    return newKey;
  }

  validateToken(token: string, secret: string): boolean {
    const hash = CryptoJS.HmacSHA256(token, secret).toString(CryptoJS.enc.Hex);
    return hash === token.split(':')[1]; // Assuming token format: "data:hash"
  }
}

export class AlertManager {
  private thresholds: Map<number, any> = new Map();
  private history: Map<number, Array<{ state: any; timestamp: number }>> = new Map();
  private patternModel: tf.LayersModel | null = null;
  private predictor: MLPredictor;
  private wsServer: CognitiveWebSocketServer;
  private anomalyDetector: AdvancedAnomalyDetector;
  private quantumEncoder: QuantumStateEncoder;

  constructor(wsPort: number = 8080) {
    this.predictor = new MLPredictor(wsPort);
    this.wsServer = new CognitiveWebSocketServer(wsPort);
    this.anomalyDetector = new AdvancedAnomalyDetector();
    this.quantumEncoder = new QuantumStateEncoder(wsPort);
    this.initializeComponents();
  }

  async initializeComponents() {
    await Promise.all([this.predictor.initialize(), this.quantumEncoder.initialize()]);
    this.patternModel = tf.sequential();
    this.patternModel.add(tf.layers.dense({
      units: 128, // Increased capacity
      activation: 'relu',
      inputShape: [5] // mood, stress, cognitiveLoad, fear, anger
    }));
    this.patternModel.add(tf.layers.dropout({ rate: 0.2 }));
    this.patternModel.add(tf.layers.dense({ units: 64, activation: 'relu' }));
    this.patternModel.add(tf.layers.dense({ units: 5, activation: 'softmax' })); // 5 patterns including 'threat'
    this.patternModel.compile({
      optimizer: tf.train.adam(0.0005),
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy']
    });
    console.log("AlertManager initialized—security radar active!");
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
    AlertThresholds.anomalySensitivity[entityId] = 0; // Reset, adjusted dynamically
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
    StateHistory.quantumDigest[entityId] = parseInt(this.generateQuantumDigest({ qubits: [] }), 16) % (2 ** 32); // Placeholder

    if (!this.history.has(entityId)) this.history.set(entityId, []);
    const history = this.history.get(entityId)!;
    history.push({ state, timestamp: Date.now() });
    if (history.length > 1000) history.shift();
  }

  checkAlerts(entityId: number, currentState: any): Array<{
    type: string;
    severity: 'info' | 'warning' | 'error';
    message: string;
    confidence: number;
  }> {
    const alerts: Array<{ type: string; severity: 'info' | 'warning' | 'error'; message: string; confidence: number }> = [];
    const id = entityId;

    const anomaly = this.anomalyDetector.detectAnomalies(entityId.toString(), [
      currentState.mood || 0,
      currentState.stress || 0,
      currentState.cognitiveLoad || 0,
      currentState.fear || 0,
      currentState.anger || 0
    ]);
    AlertThresholds.anomalySensitivity[id] = anomaly.score;

    const moodDelta = Math.abs(currentState.mood - StateHistory.previousMood[id]);
    if (moodDelta > AlertThresholds.moodSwing[id] * (1 + AlertThresholds.dynamicAdjustment[id] + anomaly.score * 0.2)) {
      alerts.push({
        type: 'mood_swing',
        severity: moodDelta > 0.5 || anomaly.severity === 3 ? 'error' : 'warning',
        message: `Mood swing detected: ${moodDelta.toFixed(2)} units`,
        confidence: Math.min(1, moodDelta / AlertThresholds.moodSwing[id] + anomaly.confidence * 0.1)
      });
    }

    if (currentState.stress > AlertThresholds.stressPeak[id] * (1 + AlertThresholds.dynamicAdjustment[id] + anomaly.score * 0.2)) {
      alerts.push({
        type: 'high_stress',
        severity: 'error',
        message: `Critical stress: ${(currentState.stress * 100).toFixed(1)}%`,
        confidence: Math.min(1, currentState.stress / AlertThresholds.stressPeak[id] + anomaly.confidence * 0.1)
      });
    }

    if (currentState.cognitiveLoad > AlertThresholds.cognitiveOverload[id] * (1 + AlertThresholds.dynamicAdjustment[id] + anomaly.score * 0.2)) {
      alerts.push({
        type: 'cognitive_overload',
        severity: 'error',
        message: `Cognitive overload: ${(currentState.cognitiveLoad * 100).toFixed(1)}%`,
        confidence: Math.min(1, currentState.cognitiveLoad / AlertThresholds.cognitiveOverload[id] + anomaly.confidence * 0.1)
      });
    }

    if (currentState.fear > AlertThresholds.fearSpike[id] * (1 + AlertThresholds.dynamicAdjustment[id] + anomaly.score * 0.2)) {
      alerts.push({
        type: 'fear_spike',
        severity: 'warning',
        message: `Fear spike: ${(currentState.fear * 100).toFixed(1)}%`,
        confidence: Math.min(1, currentState.fear / AlertThresholds.fearSpike[id] + anomaly.confidence * 0.1)
      });
    }

    if (currentState.anger > AlertThresholds.angerTrigger[id] * (1 + AlertThresholds.dynamicAdjustment[id] + anomaly.score * 0.2)) {
      alerts.push({
        type: 'anger_trigger',
        severity: 'warning',
        message: `Anger spike: ${(currentState.anger * 100).toFixed(1)}%`,
        confidence: Math.min(1, currentState.anger / AlertThresholds.angerTrigger[id] + anomaly.confidence * 0.1)
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

    const features = history.slice(-Math.min(history.length, Math.floor(timeWindow / 3600000))).map(h => [
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
      { name: 'fear_response', desc: 'Fear-driven reaction' },
      { name: 'security_threat', desc: 'Potential security breach' }
    ];

    if (features.length >= 24) {
      await this.trainPatternModel(features, this.generatePatternLabels(features, entityId));
    }

    const predMetrics = {
      cpuUsage: features[features.length - 1][2],
      memoryUsage: features[features.length - 1][1],
      networkLatency: 50,
      errorRate: features[features.length - 1][0],
      timestamp: Date.now()
    };
    this.predictor.addDataPoint(entityId.toString(), predMetrics);
    const { predictions, confidence } = await this.predictor.predict(entityId.toString(), 6);
    const predictedStress = mean(predictions.map((p: number[]) => p[1]));
    const quantumImpact = this.quantumEncoder.calculateEntanglementMetrics(this.quantumEncoder.createQuantumRegister(entityId.toString())).score;

    const results = patterns.map((p, i) => ({
      pattern: p.name,
      confidence: latestProbs[i] * confidence * (1 + quantumImpact * 0.1),
      predictedImpact: latestProbs[i] > 0.5 ? predictedStress * latestProbs[i] * (1 + quantumImpact * 0.2) : 0
    })).filter(p => p.confidence > 0.3);

    this.wsServer.broadcastStateUpdate(entityId.toString(), {
      emotionalPatterns: results,
      visualization: this.visualizePatterns(entityId, features, results)
    });

    return results;
  }

  private async trainPatternModel(features: number[][], labels: number[][], entityId: number) {
    const xs = tf.tensor2d(features);
    const ys = tf.tensor2d(labels);
    await this.patternModel!.fit(xs, ys, {
      epochs: 10,
      batchSize: 8,
      validationSplit: 0.2,
      callbacks: {
        onEpochEnd: async (epoch, logs) => {
          this.wsServer.broadcastStateUpdate(entityId.toString(), {
            event: 'pattern_model_training',
            epoch,
            accuracy: logs?.acc,
            loss: logs?.loss
          });
        },
        earlyStopping: tf.callbacks.earlyStopping({ monitor: 'val_loss', patience: 3 })
      }
    });
    xs.dispose();
    ys.dispose();
  }

  private generatePatternLabels(features: number[][], entityId: number): number[][] {
    return features.map(f => {
      const mood = f[0], stress = f[1], load = f[2], fear = f[3], anger = f[4];
      const anomaly = this.anomalyDetector.detectAnomalies(entityId.toString(), f);
      if (anomaly.score > 0.7) return [0, 0, 0, 0, 1]; // Security threat
      if (stress < 0.3 && Math.abs(mood) < 0.5) return [1, 0, 0, 0, 0]; // Stability
      if (stress > 0.7) return [0, 1, 0, 0, 0]; // Stress spike
      if (anger > 0.6) return [0, 0, 1, 0, 0]; // Anger outburst
      if (fear > 0.6) return [0, 0, 0, 1, 0]; // Fear response
      return [0.2, 0.2, 0.2, 0.2, 0.2]; // Balanced default
    });
  }

  private adjustDynamicThresholds(entityId: number) {
    const history = this.history.get(entityId);
    if (!history || history.length < 24) return;

    const recent = history.slice(-24);
    const moodStd = std(recent.map(h => h.state.mood)) || 0;
    const stressStd = std(recent.map(h => h.state.stress)) || 0;
    const loadStd = std(recent.map(h => h.state.cognitiveLoad)) || 0;
    const anomalyScore = this.anomalyDetector.detectAnomalies(entityId.toString(), recent.map(h => h.state.stress)).score;
    
    AlertThresholds.dynamicAdjustment[entityId] = Math.min(0.5, (moodStd + stressStd + loadStd) / 3);
    AlertThresholds.anomalySensitivity[entityId] = anomalyScore;
    AlertThresholds.lastThresholdUpdate[entityId] = Date.now();
  }

  private hashState(state: any): number {
    const str = JSON.stringify(state);
    return parseInt(CryptoJS.SHA256(str).toString(CryptoJS.enc.Hex).slice(0, 8), 16);
  }

  private generateQuantumDigest(register: any): string {
    const entanglementScore = this.quantumEncoder.calculateEntanglementMetrics(register).score;
    const stateVector = register.qubits.map((q: any) => `${q.real.toFixed(3)},${q.imag.toFixed(3)}`).join('');
    return CryptoJS.SHA256(stateVector + entanglementScore).toString(CryptoJS.enc.Hex);
  }

  private visualizePatterns(entityId: number, features: number[][], patterns: Array<{ pattern: string; confidence: number; predictedImpact: number }>): string {
    const lines = features.slice(-5).map((f, i) => {
      const moodBar = '█'.repeat(Math.round(Math.abs(f[0]) * 10));
      const stressBar = '█'.repeat(Math.round(f[1] * 10));
      const anomaly = this.anomalyDetector.detectAnomalies(entityId.toString(), f).score;
      return `${i}h ago: Mood[${f[0] > 0 ? '+' : ''}${moodBar}] Stress[${stressBar}] (A:${anomaly.toFixed(2)})`;
    });
    const patternSummary = patterns.map(p => `${p.pattern}: ${(p.confidence * 100).toFixed(1)}% (Impact: ${p.predictedImpact.toFixed(2)})`).join('\n');
    return `${lines.join('\n')}\n\nPatterns:\n${patternSummary}`;
  }
}
