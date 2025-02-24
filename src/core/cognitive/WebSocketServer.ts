import WebSocket from 'ws';
import { SecurityManager, AlertManager } from './cognitive-security';
import { PredictiveMonitor } from './PredictiveMonitor';
import { AdvancedAnomalyDetector } from './AdvancedAnomalyDetector';
import { CognitiveChannel } from './CognitiveChannel';
import { NeuralSynchronizer } from './neural/NeuralSynchronizer';

// Define monitoring system interface
interface MonitoringSystem {
  monitor: PredictiveMonitor;
  anomalyDetector: AdvancedAnomalyDetector;
}

export class CognitiveWebSocketServer {
  private wss: WebSocket.Server;
  private securityManager: SecurityManager;
  private alertManager: AlertManager;
  private clients: Map<string, Set<WebSocket>> = new Map();
  private monitor: PredictiveMonitor;
  private anomalyDetector: AdvancedAnomalyDetector;
  private channel: CognitiveChannel;
  private neuralSync: NeuralSynchronizer;
  private stateCache: Map<string, any> = new Map();
  private throttleLimit: number = 100; // Dynamic messages per second
  private throttleQueue: Map<string, { message: any; priority: number }[]> = new Map(); // Priority queue
  private heartbeatInterval: NodeJS.Timeout;
  private quantumPriorityThreshold: number = 0.8; // Quantum coherence for high-priority broadcasts

  constructor(port: number) {
    this.wss = new WebSocket.Server({ port });
    this.securityManager = new SecurityManager();
    this.alertManager = new AlertManager();
    const monitoringSystem = this.createMonitoringSystem();
    this.monitor = monitoringSystem.monitor;
    this.anomalyDetector = monitoringSystem.anomalyDetector;
    this.channel = new CognitiveChannel();
    this.neuralSync = new NeuralSynchronizer();

    this.setupWebSocketServer();
    this.startHeartbeat();
    console.log(`CognitiveWebSocketServer launched on port ${port}—ready to stream the cosmos!`);
  }

  private createMonitoringSystem(): MonitoringSystem {
    const monitor = new PredictiveMonitor();
    const anomalyDetector = new AdvancedAnomalyDetector(100, 256, 0.5, 5);
    return { monitor, anomalyDetector };
  }

  private async setupWebSocketServer() {
    await this.neuralSync.initialize(0.5); // Match README complexity factor

    this.wss.on('connection', (ws: WebSocket) => {
      let entityId: string | null = null;
      let authenticated = false;

      ws.on('message', async (message: string) => {
        try {
          const data = JSON.parse(message);

          if (data.type === 'handshake') {
            if (this.authenticateClient(data.token)) {
              authenticated = true;
              ws.send(JSON.stringify({ type: 'handshake_ack', status: 'success' }));
            } else {
              ws.send(JSON.stringify({ type: 'handshake_ack', status: 'failure' }));
              ws.close(1008, 'Authentication failed');
            }
            return;
          }

          if (!authenticated) {
            ws.send(JSON.stringify({ type: 'error', message: 'Not authenticated' }));
            return;
          }

          switch (data.type) {
            case 'subscribe':
              entityId = data.entityId;
              this.subscribeToEntity(entityId, ws);
              this.sendCachedState(entityId, ws);
              break;

            case 'unsubscribe':
              if (entityId) this.unsubscribeFromEntity(entityId, ws);
              break;

            case 'update_thresholds':
              if (entityId) {
                this.alertManager.setThresholds(parseInt(entityId), data.thresholds);
                ws.send(JSON.stringify({ type: 'thresholds_updated', entityId }));
              }
              break;

            case 'sync_request':
              if (entityId && data.targetId) {
                await this.handleSyncRequest(entityId, data.targetId, data.syncType || 'full', ws);
              }
              break;

            default:
              console.warn('Unknown message type:', data.type);
              ws.send(JSON.stringify({ type: 'error', message: `Unknown message type: ${data.type}` }));
          }
        } catch (error) {
          console.error('Error processing message:', error);
          ws.send(JSON.stringify({ type: 'error', message: 'Message processing failed' }));
        }
      });

      ws.on('close', () => {
        if (entityId) this.unsubscribeFromEntity(entityId, ws);
      });

      ws.on('error', (error) => {
        console.error('WebSocket error:', error);
      });
    });

    // Adaptive throttling based on anomaly forecasts and client load
    setInterval(() => this.processThrottleQueue(), 1000 / this.throttleLimit);
  }

  private authenticateClient(token: string): boolean {
    // Placeholder: Replace with JWT or quantum-safe auth
    return this.securityManager.validateToken(token, 'CogniVerse-Quantum-2025');
  }

  private subscribeToEntity(entityId: string, ws: WebSocket) {
    if (!this.clients.has(entityId)) this.clients.set(entityId, new Set());
    this.clients.get(entityId)!.add(ws);
    console.log(`Client subscribed to entity ${entityId}. Total clients: ${this.clients.get(entityId)!.size}`);
  }

  private unsubscribeFromEntity(entityId: string, ws: WebSocket) {
    const clientSet = this.clients.get(entityId);
    if (clientSet) {
      clientSet.delete(ws);
      if (clientSet.size === 0) this.clients.delete(entityId);
      console.log(`Client unsubscribed from entity ${entityId}. Remaining clients: ${clientSet?.size || 0}`);
    }
  }

  private sendCachedState(entityId: string, ws: WebSocket) {
    if (this.stateCache.has(entityId) && ws.readyState === WebSocket.OPEN) {
      const cachedState = this.stateCache.get(entityId);
      const encryptedState = this.securityManager.encryptState(cachedState, entityId);
      ws.send(JSON.stringify({
        type: 'state_update',
        entityId,
        data: encryptedState,
        timestamp: Date.now(),
        cached: true
      }));
    }
  }

  async broadcastStateUpdate(entityId: string, state: any, quantumCoherence: number = 0.9) {
    const clients = this.clients.get(entityId);
    if (!clients || clients.size === 0) return;

    // Monitoring and anomaly detection
    const metrics = this.extractMetricsFromState(state);
    this.monitor.updateMetrics(entityId, metrics);
    const anomalyData = [
      state.cognitive.awareness,
      state.cognitive.coherence,
      state.emotional.stress,
      state.emotional.mood,
      state.cognitive.complexity
    ];
    this.anomalyDetector.updateData(entityId, anomalyData, { coherence: quantumCoherence });
    const anomalyResult = this.anomalyDetector.detectAnomalies(entityId, anomalyData);

    // Cache and enrich state
    this.stateCache.set(entityId, state);
    const encryptedState = this.securityManager.encryptState(state, entityId);
    const alerts = this.alertManager.checkAlerts(parseInt(entityId), state);
    const update = {
      type: 'state_update',
      entityId,
      data: encryptedState,
      alerts,
      anomaly: {
        isAnomaly: anomalyResult.isAnomaly,
        score: anomalyResult.score,
        forecastScore: anomalyResult.forecastScore,
        signature: anomalyResult.signature
      },
      coherenceTrend: this.getCoherenceTrend(entityId),
      emotionalResonance: this.calculateEmotionalResonance(state),
      timestamp: Date.now()
    };

    // Adjust throttling based on anomaly forecast
    this.adjustThrottleLimit(anomalyResult.forecastScore, clients.size);
    this.throttleBroadcast(entityId, update, this.calculatePriority(quantumCoherence, anomalyResult.score));
  }

  broadcastAlert(entityId: string, alert: any, quantumCoherence: number = 0.9) {
    const clients = this.clients.get(entityId);
    if (!clients || clients.size === 0) return;

    const message = {
      type: 'alert',
      entityId,
      alert,
      timestamp: Date.now(),
      quantumCoherence
    };

    this.throttleBroadcast(entityId, message, this.calculatePriority(quantumCoherence, 0));
  }

  private throttleBroadcast(entityId: string, message: any, priority: number) {
    if (!this.throttleQueue.has(entityId)) this.throttleQueue.set(entityId, []);
    this.throttleQueue.get(entityId)!.push({ message, priority });
    this.throttleQueue.set(entityId, this.throttleQueue.get(entityId)!.sort((a, b) => b.priority - a.priority)); // Sort by priority
  }

  private processThrottleQueue() {
    this.throttleQueue.forEach((queue, entityId) => {
      const clients = this.clients.get(entityId);
      if (!clients || queue.length === 0) return;

      const batchSize = Math.min(clients.size, Math.floor(this.throttleLimit / this.clients.size) || 1);
      const batch = queue.splice(0, batchSize);

      clients.forEach(client => {
        if (client.readyState === WebSocket.OPEN) {
          batch.forEach(({ message }) => client.send(JSON.stringify(message)));
        } else {
          this.unsubscribeFromEntity(entityId, client);
        }
      });
    });
  }

  private async handleSyncRequest(sourceId: string, targetId: string, syncType: 'full' | 'incremental' | 'adaptive', ws: WebSocket) {
    try {
      const sourceEid = parseInt(sourceId);
      const targetEid = parseInt(targetId);
      const syncId = await this.channel.synchronize(sourceEid, targetEid, syncType);
      await this.channel.transferState(syncId);

      const syncedState = this.channel.getFullState(targetEid);
      const quantumCoherence = CognitiveState.quantumEntanglement[targetEid] || 0.9;

      // Generate emergent state if adaptive
      let finalState = syncedState;
      if (syncType === 'adaptive') {
        finalState = await this.neuralSync.generateNovelState(syncedState, 0.5);
      }

      await this.broadcastStateUpdate(targetId, finalState, quantumCoherence);
      ws.send(JSON.stringify({ type: 'sync_complete', sourceId, targetId, syncId, anomalyScore: CognitiveTransfer.anomalyScore[syncId] }));
    } catch (error) {
      ws.send(JSON.stringify({ type: 'sync_error', sourceId, targetId, message: error.message }));
    }
  }

  private extractMetricsFromState(state: any): any {
    return {
      cpuUsage: state.cognitive.cognitiveLoad * 0.5 + Math.random() * 0.1,
      memoryUsage: state.cognitive.complexity * 0.7,
      networkLatency: 5 + Math.random() * 10, // Simulated <5ms + noise
      messageQueueSize: this.throttleQueue.get(state.entityId)?.length || 0,
      errorRate: state.emotional.stress * 0.2 + (this.stateCache.get(state.entityId)?.errorRate || 0)
    };
  }

  private startHeartbeat() {
    this.heartbeatInterval = setInterval(() => {
      this.wss.clients.forEach((client: WebSocket) => {
        if (client.readyState === WebSocket.OPEN) {
          client.send(JSON.stringify({ type: 'heartbeat', timestamp: Date.now(), activeClients: this.clients.size }));
        }
      });
    }, 30000);
  }

  private adjustThrottleLimit(forecastScore: number, clientCount: number) {
    const baseLimit = 100;
    const anomalyAdjustment = forecastScore > 0.7 ? 0.5 : forecastScore > 0.5 ? 0.8 : 1; // Scale down with anomaly risk
    const loadAdjustment = Math.max(0.5, 1 - (clientCount / 1000)); // Scale with client load
    this.throttleLimit = Math.max(10, baseLimit * anomalyAdjustment * loadAdjustment);
  }

  private calculatePriority(quantumCoherence: number, anomalyScore: number): number {
    return quantumCoherence > this.quantumPriorityThreshold ? 1 + anomalyScore : 0.5 + anomalyScore * 0.5;
  }

  private getCoherenceTrend(entityId: string): number[] {
    const cachedState = this.stateCache.get(entityId);
    if (!cachedState || !cachedState.coherenceTrend) return [cachedState?.cognitive.coherence || 0];
    return cachedState.coherenceTrend; // Populated by NeuralSynchronizer
  }

  private calculateEmotionalResonance(state: any): number {
    const emo = state.emotional || {};
    const keys = ['mood', 'stress', 'motivation', 'empathy', 'curiosity', 'anger', 'fear', 'joy', 'disgust'];
    const values = keys.map(k => emo[k] || 0);
    const mean = values.reduce((sum, v) => sum + v, 0) / values.length;
    const variance = values.reduce((sum, v) => sum + (v - mean) ** 2, 0) / values.length;
    return Math.min(1, 1 - Math.sqrt(variance)); // High resonance = low variance
  }

  shutdown() {
    clearInterval(this.heartbeatInterval);
    this.wss.close();
    console.log('CognitiveWebSocketServer shut down—cosmic stream offline');
  }
}
