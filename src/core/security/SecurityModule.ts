import CryptoJS from 'crypto-js';
import { defineComponent, Types } from 'bitecs';

// Alert Thresholds Component
const AlertThresholds = defineComponent({
  moodSwing: Types.f32,
  stressPeak: Types.f32,
  cognitiveOverload: Types.f32,
  fearSpike: Types.f32,
  angerTrigger: Types.f32
});

// Historical State Component for Change Detection
const StateHistory = defineComponent({
  previousMood: Types.f32,
  previousStress: Types.f32,
  previousCognitiveLoad: Types.f32,
  lastUpdateTime: Types.ui32
});

export class SecurityManager {
  private static readonly KEY_SIZE = 256;
  private static readonly IV_SIZE = 16;
  private keyStore: Map<string, string> = new Map();

  generateKey(): string {
    return CryptoJS.lib.WordArray.random(this.KEY_SIZE / 8).toString();
  }

  storeKey(entityId: string, key: string): void {
    // In production, this should use a proper key management service
    this.keyStore.set(entityId, key);
  }

  encryptState(state: any, entityId: string): string {
    const key = this.keyStore.get(entityId);
    if (!key) throw new Error('No encryption key found for entity');

    const iv = CryptoJS.lib.WordArray.random(this.IV_SIZE);
    const encrypted = CryptoJS.AES.encrypt(JSON.stringify(state), key, {
      iv: iv,
      mode: CryptoJS.mode.GCM,
      padding: CryptoJS.pad.Pkcs7
    });

    return iv.toString() + ':' + encrypted.toString();
  }

  decryptState(encrypted: string, entityId: string): any {
    const key = this.keyStore.get(entityId);
    if (!key) throw new Error('No encryption key found for entity');

    const [ivString, ciphertext] = encrypted.split(':');
    const iv = CryptoJS.enc.Hex.parse(ivString);

    const decrypted = CryptoJS.AES.decrypt(ciphertext, key, {
      iv: iv,
      mode: CryptoJS.mode.GCM,
      padding: CryptoJS.pad.Pkcs7
    });

    return JSON.parse(decrypted.toString(CryptoJS.enc.Utf8));
  }
}

export class AlertManager {
  private thresholds: Map<number, any> = new Map();
  private history: Map<number, any> = new Map();

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
  }

  updateHistory(entityId: number, state: any) {
    StateHistory.previousMood[entityId] = state.mood;
    StateHistory.previousStress[entityId] = state.stress;
    StateHistory.previousCognitiveLoad[entityId] = state.cognitiveLoad;
    StateHistory.lastUpdateTime[entityId] = Date.now();
  }

  checkAlerts(entityId: number, currentState: any): Array<{
    type: string;
    severity: 'info' | 'warning' | 'error';
    message: string;
  }> {
    const alerts = [];

    // Check mood swings
    const moodDelta = Math.abs(currentState.mood - StateHistory.previousMood[entityId]);
    if (moodDelta > AlertThresholds.moodSwing[entityId]) {
      alerts.push({
        type: 'mood_swing',
        severity: 'warning',
        message: `Significant mood change detected: ${moodDelta.toFixed(2)} units`
      });
    }

    // Check stress levels
    if (currentState.stress > AlertThresholds.stressPeak[entityId]) {
      alerts.push({
        type: 'high_stress',
        severity: 'error',
        message: `Critical stress level: ${(currentState.stress * 100).toFixed(1)}%`
      });
    }

    // Check cognitive load
    if (currentState.cognitiveLoad > AlertThresholds.cognitiveOverload[entityId]) {
      alerts.push({
        type: 'cognitive_overload',
        severity: 'error',
        message: `Cognitive overload detected: ${(currentState.cognitiveLoad * 100).toFixed(1)}%`
      });
    }

    // Check emotional spikes
    if (currentState.fear > AlertThresholds.fearSpike[entityId]) {
      alerts.push({
        type: 'fear_spike',
        severity: 'warning',
        message: `Elevated fear levels detected: ${(currentState.fear * 100).toFixed(1)}%`
      });
    }

    if (currentState.anger > AlertThresholds.angerTrigger[entityId]) {
      alerts.push({
        type: 'anger_trigger',
        severity: 'warning',
        message: `Elevated anger levels detected: ${(currentState.anger * 100).toFixed(1)}%`
      });
    }

    // Update history after checks
    this.updateHistory(entityId, currentState);

    return alerts;
  }

  // Pattern recognition for complex emotional states
  analyzeEmotionalPatterns(entityId: number, timeWindow: number = 3600000): Array<{
    pattern: string;
    confidence: number;
  }> {
    // Implement pattern recognition logic here
    // This could use machine learning models in production
    return [];
  }
}
