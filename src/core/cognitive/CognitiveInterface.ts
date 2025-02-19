import { createWorld, defineComponent, defineQuery, Types } from 'bitecs';
import { createDreams } from '@daydreamsai/core/v1';
import { action, task, memory, extension } from '@daydreamsai/core/v1';
import { z } from 'zod';
import CryptoJS from 'crypto-js';
import { NeuralSynchronizer } from '../neural/NeuralSynchronizer';
import { QuantumStateEncoder } from '../quantum/QuantumStateEncoder';

// Enhanced Cognitive State Components
const CognitiveState = defineComponent({
  awareness: Types.f32,
  coherence: Types.f32,
  complexity: Types.f32,
  cognitiveLoad: Types.f32,
  shortTermPtr: Types.ui32,
  longTermPtr: Types.ui32,
  isProcessing: Types.ui8,
  isTransmitting: Types.ui8,
  isSynchronizing: Types.ui8,
  lastUpdateTime: Types.ui32,
  processingLatency: Types.f32,
  syncPriority: Types.ui8 // New: Priority for sync operations (0-255)
});

// Emotional State Component
const EmotionalState = defineComponent({
  mood: Types.f32,
  stress: Types.f32,
  motivation: Types.f32,
  empathy: Types.f32,
  curiosity: Types.f32,
  anger: Types.f32,
  fear: Types.f32,
  joy: Types.f32,
  disgust: Types.f32,
  emotionalStability: Types.f32 // New: Measure of emotional variance
});

// Enhanced Transfer Protocol
const CognitiveTransfer = defineComponent({
  sourceId: Types.ui32,
  targetId: Types.ui32,
  channelId: Types.ui32,
  bandwidth: Types.f32,
  signal: Types.f32,
  lastSyncTime: Types.ui32,
  syncType: Types.ui8,    // 0: full, 1: incremental, 2: adaptive
  errorCount: Types.ui8,
  retryAttempts: Types.ui8,
  compressionRatio: Types.f32 // New: Data compression efficiency
});

// Performance Metrics Component
const PerformanceMetrics = defineComponent({
  transferLatency: Types.f32,
  dataVolume: Types.ui32,
  successRate: Types.f32,
  errorRate: Types.f32,
  syncFrequency: Types.f32,
  throughput: Types.f32,      // New: Bytes per second
  packetLoss: Types.f32       // New: Percentage of lost data
});

// Security and Integrity
class StateIntegrityManager {
  private static readonly SECRET_KEY = 'your-secret-key-here'; // Replace with secure key management in prod

  static validateState(state: any): boolean {
    const validations = [
      state.awareness >= 0 && state.awareness <= 1,
      state.coherence >= 0 && state.coherence <= 1,
      state.complexity >= 0 && state.complexity <= 1,
      state.cognitiveLoad >= 0 && state.cognitiveLoad <= 1,
      state.emotional && state.emotional.mood >= -1 && state.emotional.mood <= 1,
      state.emotional && state.emotional.stress >= 0 && state.emotional.stress <= 1,
      state.emotional && state.emotional.motivation >= 0 && state.emotional.motivation <= 1,
      state.emotional && state.emotional.empathy >= 0 && state.emotional.empathy <= 1,
      state.emotional && state.emotional.curiosity >= 0 && state.emotional.curiosity <= 1,
      state.emotional && state.emotional.anger >= 0 && state.emotional.anger <= 1,
      state.emotional && state.emotional.fear >= 0 && state.emotional.fear <= 1,
      state.emotional && state.emotional.joy >= 0 && state.emotional.joy <= 1,
      state.emotional && state.emotional.disgust >= 0 && state.emotional.disgust <= 1
    ];
    return validations.every(v => v === true);
  }

  static encrypt(state: any): string {
    const iv = CryptoJS.lib.WordArray.random(16);
    const encrypted = CryptoJS.AES.encrypt(
      JSON.stringify(state),
      CryptoJS.enc.Utf8.parse(this.SECRET_KEY),
      { iv, mode: CryptoJS.mode.CBC, padding: CryptoJS.pad.Pkcs7 }
    );
    return `${iv.toString()}:${encrypted.toString()}`;
  }

  static decrypt(data: string): any {
    const [ivString, cipherText] = data.split(':');
    const iv = CryptoJS.enc.Hex.parse(ivString);
    const decrypted = CryptoJS.AES.decrypt(
      cipherText,
      CryptoJS.enc.Utf8.parse(this.SECRET_KEY),
      { iv, mode: CryptoJS.mode.CBC, padding: CryptoJS.pad.Pkcs7 }
    );
    return JSON.parse(decrypted.toString(CryptoJS.enc.Utf8));
  }

  static compressState(state: any): any {
    // Simple compression: Reduce precision to 2 decimals
    const compressValue = (val: number) => Math.round(val * 100) / 100;
    return {
      cognitive: Object.fromEntries(
        Object.entries(state.cognitive).map(([k, v]) => [k, typeof v === 'number' ? compressValue(v) : v])
      ),
      emotional: Object.fromEntries(
        Object.entries(state.emotional).map(([k, v]) => [k, typeof v === 'number' ? compressValue(v) : v])
      )
    };
  }
}

// Performance Monitoring
class PerformanceMonitor {
  private metrics: Map<number, { startTime: number; operations: number; bytesTransferred: number; errors: number }> = new Map();

  startOperation(entityId: number) {
    this.metrics.set(entityId, {
      startTime: Date.now(),
      operations: 0,
      bytesTransferred: 0,
      errors: 0
    });
  }

  endOperation(entityId: number, type: 'transfer' | 'sync' | 'process', bytesTransferred: number = 0, error: boolean = false) {
    const metric = this.metrics.get(entityId);
    if (!metric) return;

    const duration = Date.now() - metric.startTime;
    metric.operations++;
    metric.bytesTransferred += bytesTransferred;
    if (error) metric.errors++;

    PerformanceMetrics.transferLatency[entityId] = duration;
    PerformanceMetrics.syncFrequency[entityId] = metric.operations / ((Date.now() - metric.startTime) / 1000); // Ops per second
    PerformanceMetrics.throughput[entityId] = metric.bytesTransferred / (duration / 1000); // Bytes per second
    PerformanceMetrics.dataVolume[entityId] = metric.bytesTransferred;
    PerformanceMetrics.successRate[entityId] = 1 - metric.errors / metric.operations;
    PerformanceMetrics.errorRate[entityId] = metric.errors / metric.operations;
    PerformanceMetrics.packetLoss[entityId] = error ? 0.1 : 0; // Simplified packet loss simulation

    this.metrics.delete(entityId); // Clean up after operation
  }
}

// Enhanced Cognitive Channel System
class CognitiveChannel {
  private world: any;
  private stateQuery: any;
  private transferQuery: any;
  private performanceMonitor: PerformanceMonitor;
  private neuralSync: NeuralSynchronizer;
  private quantumEncoder: QuantumStateEncoder;
  private retryLimit: number = 3;
  private bandwidthBaseline: number = 1.0; // Initial bandwidth (e.g., Mbps)

  constructor() {
    this.world = createWorld();
    this.stateQuery = defineQuery([CognitiveState, EmotionalState]);
    this.transferQuery = defineQuery([CognitiveTransfer]);
    this.performanceMonitor = new PerformanceMonitor();
    this.neuralSync = new NeuralSynchronizer();
    this.quantumEncoder = new QuantumStateEncoder();
    this.initializeDependencies();
  }

  private async initializeDependencies() {
    await this.neuralSync.initialize();
  }

  createCognitiveEntity(config: {
    awareness: number,
    coherence: number,
    complexity: number,
    emotional?: {
      mood?: number,
      stress?: number,
      motivation?: number,
      empathy?: number,
      curiosity?: number,
      anger?: number,
      fear?: number,
      joy?: number,
      disgust?: number
    }
  }) {
    const entity = this.world.createEntity();
    CognitiveState.awareness[entity] = Math.min(1, Math.max(0, config.awareness));
    CognitiveState.coherence[entity] = Math.min(1, Math.max(0, config.coherence));
    CognitiveState.complexity[entity] = Math.min(1, Math.max(0, config.complexity));
    CognitiveState.cognitiveLoad[entity] = 0;
    CognitiveState.syncPriority[entity] = 100; // Default priority

    if (config.emotional) {
      EmotionalState.mood[entity] = Math.min(1, Math.max(-1, config.emotional.mood ?? 0));
      EmotionalState.stress[entity] = Math.min(1, Math.max(0, config.emotional.stress ?? 0));
      EmotionalState.motivation[entity] = Math.min(1, Math.max(0, config.emotional.motivation ?? 1));
      EmotionalState.empathy[entity] = Math.min(1, Math.max(0, config.emotional.empathy ?? 0));
      EmotionalState.curiosity[entity] = Math.min(1, Math.max(0, config.emotional.curiosity ?? 0));
      EmotionalState.anger[entity] = Math.min(1, Math.max(0, config.emotional.anger ?? 0));
      EmotionalState.fear[entity] = Math.min(1, Math.max(0, config.emotional.fear ?? 0));
      EmotionalState.joy[entity] = Math.min(1, Math.max(0, config.emotional.joy ?? 0));
      EmotionalState.disgust[entity] = Math.min(1, Math.max(0, config.emotional.disgust ?? 0));
      EmotionalState.emotionalStability[entity] = this.calculateEmotionalStability(config.emotional);
    }

    return entity;
  }

  async synchronize(source: number, target: number, type: 'full' | 'incremental' | 'adaptive' = 'full'): Promise<number> {
    const channelId = this.world.createEntity();
    this.performanceMonitor.startOperation(channelId);

    try {
      CognitiveTransfer.sourceId[channelId] = source;
      CognitiveTransfer.targetId[channelId] = target;
      CognitiveTransfer.syncType[channelId] = type === 'full' ? 0 : type === 'incremental' ? 1 : 2;
      CognitiveTransfer.bandwidth[channelId] = this.simulateBandwidth(source, target);
      CognitiveTransfer.signal[channelId] = this.simulateSignalStrength(source, target);
      CognitiveTransfer.lastSyncTime[channelId] = Date.now();
      CognitiveTransfer.compressionRatio[channelId] = 0; // Updated in transferState

      CognitiveState.isSynchronizing[source] = 1;
      CognitiveState.isSynchronizing[target] = 1;

      return channelId;
    } catch (error) {
      console.error('Synchronization failed:', error);
      throw error;
    } finally {
      this.performanceMonitor.endOperation(channelId, 'sync');
      CognitiveState.isSynchronizing[source] = 0;
      CognitiveState.isSynchronizing[target] = 0;
    }
  }

  async transferState(channelId: number): Promise<void> {
    this.performanceMonitor.startOperation(channelId);
    const sourceId = CognitiveTransfer.sourceId[channelId];
    const targetId = CognitiveTransfer.targetId[channelId];
    const syncType = CognitiveTransfer.syncType[channelId];

    try {
      const sourceState = this.getFullState(sourceId);
      if (!StateIntegrityManager.validateState(sourceState)) {
        throw new Error('Invalid source state detected');
      }

      // Compress and encrypt state
      const compressedState = StateIntegrityManager.compressState(sourceState);
      const encryptedState = StateIntegrityManager.encrypt(compressedState);
      const stateSize = new TextEncoder().encode(JSON.stringify(compressedState)).length;
      CognitiveTransfer.compressionRatio[channelId] = stateSize / new TextEncoder().encode(JSON.stringify(sourceState)).length;

      // Simulate network conditions
      const bandwidth = CognitiveTransfer.bandwidth[channelId] * this.adjustBandwidthForLoad(sourceId, targetId);
      const signal = CognitiveTransfer.signal[channelId] * this.adjustSignalForEmotion(sourceId);

      // Neural synchronization for enhanced state alignment
      const syncResult = await this.neuralSync.synchronizeStates(sourceState, this.getFullState(targetId), 0.5);
      const syncedState = syncResult.synchronizedState;

      // Quantum encoding for memory (if present)
      if (sourceState.memory) {
        const quantumRegister = this.quantumEncoder.encodeState(sourceState, sourceId.toString());
        sourceState.memory = this.quantumEncoder.measureState(quantumRegister).measuredState.memory;
      }

      // Apply sync based on type
      if (syncType === 0) { // Full sync
        this.applyFullSync(targetId, syncedState, bandwidth, signal);
      } else if (syncType === 1) { // Incremental sync
        this.applyIncrementalSync(targetId, syncedState, bandwidth, signal);
      } else { // Adaptive sync (new)
        this.applyAdaptiveSync(targetId, syncedState, bandwidth, signal, sourceId);
      }

      this.updatePerformanceMetrics(channelId, stateSize);
      this.updateEmotionalFeedback(targetId, sourceState.emotional);
    } catch (error) {
      CognitiveTransfer.errorCount[channelId]++;
      if (CognitiveTransfer.errorCount[channelId] < this.retryLimit) {
        await this.retryTransfer(channelId);
      } else {
        this.performanceMonitor.endOperation(channelId, 'transfer', 0, true);
        throw new Error(`Transfer failed after ${this.retryLimit} attempts: ${error}`);
      }
    } finally {
      this.performanceMonitor.endOperation(channelId, 'transfer');
    }
  }

  private getFullState(entityId: number): any {
    return {
      cognitive: {
        awareness: CognitiveState.awareness[entityId],
        coherence: CognitiveState.coherence[entityId],
        complexity: CognitiveState.complexity[entityId],
        cognitiveLoad: CognitiveState.cognitiveLoad[entityId]
      },
      emotional: {
        mood: EmotionalState.mood[entityId],
        stress: EmotionalState.stress[entityId],
        motivation: EmotionalState.motivation[entityId],
        empathy: EmotionalState.empathy[entityId],
        curiosity: EmotionalState.curiosity[entityId],
        anger: EmotionalState.anger[entityId],
        fear: EmotionalState.fear[entityId],
        joy: EmotionalState.joy[entityId],
        disgust: EmotionalState.disgust[entityId]
      }
    };
  }

  private applyFullSync(targetId: number, state: any, bandwidth: number, signal: number) {
    Object.entries(state.cognitive).forEach(([key, value]: [string, any]) => {
      CognitiveState[key][targetId] = value * bandwidth * signal;
    });
    Object.entries(state.emotional).forEach(([key, value]: [string, any]) => {
      EmotionalState[key][targetId] = value * bandwidth * signal;
    });
  }

  private applyIncrementalSync(targetId: number, state: any, bandwidth: number, signal: number) {
    const threshold = 0.1;
    Object.entries(state.cognitive).forEach(([key, value]: [string, any]) => {
      if (Math.abs(CognitiveState[key][targetId] - value) > threshold) {
        CognitiveState[key][targetId] = value * bandwidth * signal;
      }
    });
    Object.entries(state.emotional).forEach(([key, value]: [string, any]) => {
      if (Math.abs(EmotionalState[key][targetId] - value) > threshold) {
        EmotionalState[key][targetId] = value * bandwidth * signal;
      }
    });
  }

  private applyAdaptiveSync(targetId: number, state: any, bandwidth: number, signal: number, sourceId: number) {
    const priority = CognitiveState.syncPriority[sourceId] / 255; // Normalize 0-1
    const emotionalImpact = EmotionalState.stress[sourceId] + EmotionalState.anger[sourceId];
    const weight = priority * (1 - emotionalImpact); // High priority, low stress = more influence
    Object.entries(state.cognitive).forEach(([key, value]: [string, any]) => {
      CognitiveState[key][targetId] = CognitiveState[key][targetId] * (1 - weight) + value * weight * bandwidth * signal;
    });
    Object.entries(state.emotional).forEach(([key, value]: [string, any]) => {
      EmotionalState[key][targetId] = EmotionalState[key][targetId] * (1 - weight) + value * weight * bandwidth * signal;
    });
  }

  private async retryTransfer(channelId: number) {
    CognitiveTransfer.retryAttempts[channelId]++;
    const delay = Math.min(1000 * Math.pow(2, CognitiveTransfer.retryAttempts[channelId]), 10000); // Exponential backoff
    await new Promise(resolve => setTimeout(resolve, delay));
    return this.transferState(channelId);
  }

  private updatePerformanceMetrics(channelId: number, stateSize: number) {
    const successRate = 1 - (CognitiveTransfer.errorCount[channelId] / (CognitiveTransfer.retryAttempts[channelId] + 1));
    PerformanceMetrics.successRate[channelId] = successRate;
    PerformanceMetrics.errorRate[channelId] = 1 - successRate;
    PerformanceMetrics.dataVolume[channelId] = stateSize;
  }

  private simulateBandwidth(sourceId: number, targetId: number): number {
    const loadFactor = (CognitiveState.cognitiveLoad[sourceId] + CognitiveState.cognitiveLoad[targetId]) / 2;
    return this.bandwidthBaseline * (1 - loadFactor * 0.5) * (0.8 + Math.random() * 0.2); // 80-100% with noise
  }

  private simulateSignalStrength(entityId: number): number {
    const stressImpact = EmotionalState.stress[entityId];
    return 1 - stressImpact * 0.3 + Math.random() * 0.1; // Degrade with stress, add noise
  }

  private adjustBandwidthForLoad(sourceId: number, targetId: number): number {
    const totalLoad = CognitiveState.cognitiveLoad[sourceId] + CognitiveState.cognitiveLoad[targetId];
    return Math.max(0.1, 1 - totalLoad * 0.4); // Reduce bandwidth with load, min 10%
  }

  private adjustSignalForEmotion(sourceId: number): number {
    const emotionalNoise = (EmotionalState.anger[sourceId] + EmotionalState.fear[sourceId]) / 2;
    return Math.max(0.5, 1 - emotionalNoise * 0.3); // Emotional instability degrades signal
  }

  private calculateEmotionalStability(emotional: any): number {
    const values = [
      emotional.mood ?? 0,
      emotional.stress ?? 0,
      emotional.motivation ?? 1,
      emotional.empathy ?? 0,
      emotional.curiosity ?? 0,
      emotional.anger ?? 0,
      emotional.fear ?? 0,
      emotional.joy ?? 0,
      emotional.disgust ?? 0
    ];
    const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
    const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
    return 1 - Math.sqrt(variance); // Higher stability = lower variance
  }

  private updateEmotionalFeedback(targetId: number, sourceEmotional: any) {
    const influence = EmotionalState.empathy[targetId] * 0.5;
    EmotionalState.mood[targetId] += (sourceEmotional.mood - EmotionalState.mood[targetId]) * influence;
    EmotionalState.stress[targetId] += (sourceEmotional.stress - EmotionalState.stress[targetId]) * influence * 0.5;
    EmotionalState.emotionalStability[targetId] = this.calculateEmotionalStability(this.getFullState(targetId).emotional);
  }
}

// Daydreams Integration
const cognitiveMemory = memory<{
  channels: Map<string, number>,
  metrics: Map<string, { latency: number; success: number }>
}>({
  key: 'cognitive-memory',
  create() {
    return {
      channels: new Map(),
      metrics: new Map()
    };
  }
});

// Enhanced Actions
const initiateSyncAction = action({
  name: 'initiate-sync',
  schema: z.object({
    sourceId: z.string(),
    targetId: z.string(),
    syncType: z.enum(['full', 'incremental', 'adaptive']).default('full')
  }),
  memory: cognitiveMemory,
  async handler(call, ctx, agent) {
    const channel = new CognitiveChannel();
    const syncId = await channel.synchronize(
      parseInt(call.data.sourceId),
      parseInt(call.data.targetId),
      call.data.syncType
    );
    ctx.data.channels.set(`${call.data.sourceId}-${call.data.targetId}`, syncId);
    await channel.transferState(syncId); // Execute transfer immediately
    return `Synchronization completed with channel ID: ${syncId}`;
  }
});

export const cognitiveInterface = extension({
  name: 'cognitive-interface',
  actions: [initiateSyncAction],
  setup(agent) {
    const channel = new CognitiveChannel();
    agent.container.register('cognitiveChannel', channel);
  }
});

export function createCognitiveAgent(config: {
  model: any,
  awareness?: number,
  coherence?: number,
  complexity?: number,
  emotional?: {
    mood?: number,
    stress?: number,
    motivation?: number,
    empathy?: number,
    curiosity?: number,
    anger?: number,
    fear?: number,
    joy?: number,
    disgust?: number
  }
}) {
  const channel = new CognitiveChannel();
  const cognitiveEntity = channel.createCognitiveEntity({
    awareness: config.awareness ?? 1.0,
    coherence: config.coherence ?? 1.0,
    complexity: config.complexity ?? 1.0,
    emotional: config.emotional
  });

  return createDreams({
    model: config.model,
    extensions: [cognitiveInterface],
    context: { cognitiveEntity }
  });
}
