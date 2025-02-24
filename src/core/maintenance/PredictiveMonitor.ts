import { defineComponent, Types } from 'bitecs';
import { mean, std, variance } from 'mathjs';
import { MaintenanceScheduler } from './MaintenanceScheduler';
import { CognitiveWebSocketServer } from './CognitiveWebSocketServer';
import { QuantumStateEncoder } from '../quantum/QuantumStateEncoder';
import { NeuralSynchronizer } from '../neural/NeuralSynchronizer';
import { AdvancedAnomalyDetector } from '../anomaly/AdvancedAnomalyDetector';
import { MLPredictor } from '../MLPredictor';

// Enhanced System Health Metrics Component
const SystemHealth = defineComponent({
  cpuUsage: Types.f32,
  memoryUsage: Types.f32,
  networkLatency: Types.f32,
  messageQueueSize: Types.ui32,
  errorRate: Types.f32,
  lastMaintenanceTime: Types.ui32,
  uptime: Types.ui32,
  quantumCoherence: Types.f32,    // New: Quantum state coherence (0-1)
  trendVector: Types.f32Array(3)  // New: [CPU, Memory, Latency] trend slopes
});

// Enhanced Predictive Metrics Component
const PredictiveMetrics = defineComponent({
  failureProbability: Types.f32,
  expectedUptime: Types.f32,
  maintenanceUrgency: Types.f32,
  healthScore: Types.f32,
  performanceScore: Types.f32,
  trendSlope: Types.f32,
  anomalyImpact: Types.f32        // New: Anomaly-driven urgency boost (0-1)
});

class HoltWintersSmoothing {
  private alpha: number;
  private beta: number;
  private gamma: number;
  private period: number;
  private level: number;
  private trend: number;
  private seasonal: number[];
  private data: number[];

  constructor(alpha: number = 0.3, beta: number = 0.1, gamma: number = 0.2, period: number = 24) {
    this.alpha = alpha;
    this.beta = beta;
    this.gamma = gamma;
    this.period = period;
    this.level = 0;
    this.trend = 0;
    this.seasonal = new Array(period).fill(0);
    this.data = [];
  }

  update(value: number): number {
    if (this.data.length < this.period) {
      this.data.push(value);
      if (this.data.length === this.period) this.initialize();
      return value;
    }

    const prevLevel = this.level;
    const prevTrend = this.trend;
    const prevSeasonal = this.seasonal.shift()!;
    this.level = this.alpha * (value - prevSeasonal) + (1 - this.alpha) * (prevLevel + prevTrend);
    this.trend = this.beta * (this.level - prevLevel) + (1 - this.beta) * prevTrend;
    this.seasonal.push(this.gamma * (value - this.level) + (1 - this.gamma) * prevSeasonal);

    this.data.push(value);
    if (this.data.length > this.period * 2) this.data.shift();
    return this.level + this.trend + this.seasonal[0];
  }

  predict(steps: number): number[] {
    if (this.data.length < this.period) return new Array(steps).fill(0);
    const forecast: number[] = [];
    let level = this.level;
    let trend = this.trend;
    const seasonal = [...this.seasonal];

    for (let i = 0; i < steps; i++) {
      const seasonalIdx = i % this.period;
      forecast.push(level + trend * (i + 1) + seasonal[seasonalIdx]);
    }
    return forecast;
  }

  private initialize() {
    this.level = mean(this.data);
    this.trend = (this.data[this.data.length - 1] - this.data[0]) / (this.period - 1);
    const seasonalMeans = this.data.map((val, idx) => val - this.level);
    this.seasonal = seasonalMeans.slice(0, this.period);
  }
}

class BottleneckDetector {
  private metrics: Map<string, number[]> = new Map();
  private thresholds: Map<string, { static: number; dynamic: number }> = new Map();
  private historyWindow: number = 100;
  private anomalyDetector: AdvancedAnomalyDetector;

  constructor(anomalyDetector: AdvancedAnomalyDetector) {
    this.anomalyDetector = anomalyDetector;
  }

  addMetric(name: string, value: number, staticThreshold: number) {
    if (!this.metrics.has(name)) {
      this.metrics.set(name, []);
      this.thresholds.set(name, { static: staticThreshold, dynamic: staticThreshold });
    }
    const values = this.metrics.get(name)!;
    values.push(value);
    if (values.length > this.historyWindow) values.shift();
    this.updateDynamicThreshold(name);
  }

  detectBottlenecks(): Array<{
    metric: string;
    severity: 'low' | 'medium' | 'high';
    value: number;
    threshold: number;
    confidence: number;
  }> {
    const bottlenecks: Array<{ metric: string; severity: 'low' | 'medium' | 'high'; value: number; threshold: number; confidence: number }> = [];

    for (const [name, values] of this.metrics.entries()) {
      const currentValue = values[values.length - 1];
      const thresholdData = this.thresholds.get(name)!;
      const effectiveThreshold = Math.max(thresholdData.static, thresholdData.dynamic);
      const anomaly = this.anomalyDetector.detectAnomalies(name, values.slice(-5));

      if (currentValue > effectiveThreshold || anomaly.isAnomaly) {
        const severity = this.calculateSeverity(currentValue, effectiveThreshold, anomaly);
        const confidence = Math.max(this.calculateConfidence(values, currentValue, effectiveThreshold), anomaly.confidence);
        bottlenecks.push({
          metric: name,
          severity,
          value: currentValue,
          threshold: effectiveThreshold,
          confidence
        });
      }
    }
    return bottlenecks;
  }

  private calculateSeverity(value: number, threshold: number, anomaly: any): 'low' | 'medium' | 'high' {
    const ratio = value / threshold;
    if (anomaly.isAnomaly && anomaly.severity > 1) return anomaly.severity === 3 ? 'high' : 'medium';
    return ratio > 2 ? 'high' : ratio > 1.5 ? 'medium' : 'low';
  }

  private calculateConfidence(values: number[], value: number, threshold: number): number {
    const meanVal = mean(values);
    const stdVal = std(values) || 1;
    const zScore = Math.abs(value - meanVal) / stdVal;
    return Math.min(1, zScore / 3);
  }

  private updateDynamicThreshold(name: string) {
    const values = this.metrics.get(name)!;
    if (values.length < this.historyWindow) return;
    const meanVal = mean(values);
    const stdVal = std(values);
    const thresholdData = this.thresholds.get(name)!;
    thresholdData.dynamic = meanVal + 2 * stdVal;
  }
}

export class PredictiveMonitor {
  private smoothing: Map<string, HoltWintersSmoothing> = new Map();
  private bottleneckDetector: BottleneckDetector;
  private maintenanceSchedule: Map<string, { time: number; urgency: number }> = new Map();
  private scheduler: MaintenanceScheduler;
  private wsServer: CognitiveWebSocketServer;
  private quantumEncoder: QuantumStateEncoder;
  private neuralSync: NeuralSynchronizer;
  private anomalyDetector: AdvancedAnomalyDetector;
  private predictor: MLPredictor;
  private history: Map<string, Array<{ timestamp: number; metrics: any }>> = new Map();

  constructor(wsPort: number = 8080) {
    this.scheduler = new MaintenanceScheduler(wsPort);
    this.wsServer = new CognitiveWebSocketServer(wsPort);
    this.quantumEncoder = new QuantumStateEncoder(wsPort);
    this.neuralSync = new NeuralSynchronizer();
    this.anomalyDetector = new AdvancedAnomalyDetector();
    this.predictor = new MLPredictor(wsPort);
    this.bottleneckDetector = new BottleneckDetector(this.anomalyDetector);
    this.initializeComponents();
  }

  private async initializeComponents() {
    await Promise.all([
      this.quantumEncoder.initialize(),
      this.neuralSync.initialize(),
      this.predictor.initialize()
    ]);
    console.log("PredictiveMonitor initialized—watching the galaxy’s pulse!");
  }

  async updateMetrics(entityId: string, metrics: {
    cpuUsage: number;
    memoryUsage: number;
    networkLatency: number;
    messageQueueSize: number;
    errorRate: number;
  }) {
    const id = parseInt(entityId);
    const timestamp = Date.now();

    // Quantum enhancement
    const quantumReg = await this.quantumEncoder.encodeState(metrics, entityId);
    const quantumCoherence = this.quantumEncoder.calculateEntanglementMetrics(quantumReg).score;

    // Update SystemHealth
    SystemHealth.cpuUsage[id] = metrics.cpuUsage;
    SystemHealth.memoryUsage[id] = metrics.memoryUsage;
    SystemHealth.networkLatency[id] = metrics.networkLatency;
    SystemHealth.messageQueueSize[id] = metrics.messageQueueSize;
    SystemHealth.errorRate[id] = metrics.errorRate;
    SystemHealth.lastMaintenanceTime[id] = SystemHealth.lastMaintenanceTime[id] || 0;
    SystemHealth.uptime[id] = this.calculateUptime(id, timestamp);
    SystemHealth.quantumCoherence[id] = quantumCoherence;

    // Update history and trend vector
    if (!this.history.has(entityId)) this.history.set(entityId, []);
    this.history.get(entityId)!.push({ timestamp, metrics });
    if (this.history.get(entityId)!.length > 1000) this.history.get(entityId)!.shift();
    this.updateTrendVector(id);

    // Update smoothing predictors
    Object.entries(metrics).forEach(([metric, value]) => {
      const key = `${entityId}-${metric}`;
      if (!this.smoothing.has(key)) this.smoothing.set(key, new HoltWintersSmoothing(0.3, 0.1, 0.2, 24));
      this.smoothing.get(key)!.update(value);
    });

    // Update bottleneck detection
    this.bottleneckDetector.addMetric('cpu', metrics.cpuUsage, 0.8);
    this.bottleneckDetector.addMetric('memory', metrics.memoryUsage, 0.9);
    this.bottleneckDetector.addMetric('latency', metrics.networkLatency, 100);
    this.bottleneckDetector.addMetric('queue', metrics.messageQueueSize, 1000);
    this.bottleneckDetector.addMetric('errors', metrics.errorRate, 0.1);

    // Update predictive metrics and schedule
    await this.updatePredictiveMetrics(entityId, metrics, timestamp);
    await this.scheduleMaintenanceIfNeeded(entityId, timestamp);
  }

  private async updatePredictiveMetrics(entityId: string, currentMetrics: any, timestamp: number) {
    const id = parseInt(entityId);
    const failureProbability = await this.calculateFailureProbability(entityId, currentMetrics);
    const expectedUptime = await this.predictUptime(entityId, currentMetrics);
    const maintenanceUrgency = this.calculateMaintenanceUrgency(failureProbability, expectedUptime, entityId);
    const healthScore = this.calculateHealthScore(currentMetrics);
    const performanceScore = this.calculatePerformanceScore(currentMetrics);
    const trendSlope = this.calculateTrendSlope(entityId);
    const anomalyImpact = this.anomalyDetector.detectAnomalies(entityId, [
      currentMetrics.cpuUsage,
      currentMetrics.memoryUsage,
      currentMetrics.networkLatency
    ]).score;

    PredictiveMetrics.failureProbability[id] = failureProbability;
    PredictiveMetrics.expectedUptime[id] = expectedUptime;
    PredictiveMetrics.maintenanceUrgency[id] = maintenanceUrgency;
    PredictiveMetrics.healthScore[id] = healthScore;
    PredictiveMetrics.performanceScore[id] = performanceScore;
    PredictiveMetrics.trendSlope[id] = trendSlope;
    PredictiveMetrics.anomalyImpact[id] = anomalyImpact;

    this.wsServer.broadcastStateUpdate(entityId, {
      systemHealth: this.getSystemHealth(id),
      predictiveMetrics: {
        failureProbability,
        expectedUptime,
        maintenanceUrgency,
        healthScore,
        performanceScore,
        trendSlope,
        anomalyImpact
      }
    });
  }

  getMaintenanceRecommendations(entityId: string): Array<{
    component: string;
    urgency: 'low' | 'medium' | 'high';
    recommendation: string;
    estimatedDowntime: number;
    confidence: number;
  }> {
    const id = parseInt(entityId);
    const bottlenecks = this.bottleneckDetector.detectBottlenecks();
    const recommendations: Array<{ component: string; urgency: 'low' | 'medium' | 'high'; recommendation: string; estimatedDowntime: number; confidence: number }> = [];

    const failureProb = PredictiveMetrics.failureProbability[id];
    if (failureProb > 0.7) {
      recommendations.push({
        component: 'system',
        urgency: 'high',
        recommendation: 'Immediate full system maintenance due to high failure risk',
        estimatedDowntime: 180,
        confidence: Math.min(1, failureProb * 1.5)
      });
    } else if (failureProb > 0.4) {
      recommendations.push({
        component: 'system',
        urgency: 'medium',
        recommendation: 'Schedule system check due to moderate failure risk',
        estimatedDowntime: 60,
        confidence: Math.min(1, failureProb * 1.2)
      });
    }

    bottlenecks.forEach(bottleneck => {
      recommendations.push({
        component: bottleneck.metric,
        urgency: bottleneck.severity,
        recommendation: this.generateRecommendation(bottleneck),
        estimatedDowntime: this.estimateDowntime(bottleneck),
        confidence: bottleneck.confidence
      });
    });

    this.wsServer.broadcastAlert(entityId, { type: 'maintenance_recommendations', recommendations });
    return recommendations;
  }

  private async calculateFailureProbability(entityId: string, metrics: any): Promise<number> {
    const weights = {
      cpu: 0.3,
      memory: 0.25,
      latency: 0.2,
      queue: 0.15,
      errors: 0.1
    };
    const normalizedLatency = metrics.networkLatency / 200;
    const normalizedQueue = metrics.messageQueueSize / 2000;
    const anomaly = this.anomalyDetector.detectAnomalies(entityId, [
      metrics.cpuUsage,
      metrics.memoryUsage,
      metrics.networkLatency,
      metrics.messageQueueSize,
      metrics.errorRate
    ]);
    const quantumReg = this.quantumEncoder.createQuantumRegister(entityId);
    const quantumFactor = 1 + this.quantumEncoder.calculateEntanglementMetrics(quantumReg).score * 0.1;

    const baseProb = (
      weights.cpu * metrics.cpuUsage +
      weights.memory * metrics.memoryUsage +
      weights.latency * Math.min(1, normalizedLatency) +
      weights.queue * Math.min(1, normalizedQueue) +
      weights.errors * metrics.errorRate
    );

    return Math.min(1, baseProb * (1 + anomaly.score * 0.2) * quantumFactor);
  }

  private async predictUptime(entityId: string, metrics: any): Promise<number> {
    const baseUptime = 168;
    const failureProb = PredictiveMetrics.failureProbability[parseInt(entityId)];
    const cpuForecast = this.smoothing.get(`${entityId}-cpuUsage`)!.predict(24)[23];
    const trendImpact = this.calculateTrendSlope(entityId);
    const pred = await this.predictor.predict(entityId, 24);
    const predictedLoad = mean(pred.predictions.map(p => p[0] + p[1])); // CPU + Memory
    return baseUptime * (1 - failureProb) * (1 - cpuForecast * 0.2) * (1 + trendImpact * 0.1) * (1 - predictedLoad * 0.15);
  }

  private calculateMaintenanceUrgency(failureProbability: number, uptime: number, entityId: string): number {
    const id = parseInt(entityId);
    const lastMaintenance = SystemHealth.lastMaintenanceTime[id] || 0;
    const timeSinceMaintenance = (Date.now() - lastMaintenance) / (1000 * 60 * 60);
    const agingFactor = Math.min(1, timeSinceMaintenance / 168);
    const anomalyImpact = PredictiveMetrics.anomalyImpact[id];
    return Math.min(1, (failureProbability * 0.5) + ((168 - uptime) / 168 * 0.3) + (agingFactor * 0.1) + (anomalyImpact * 0.1));
  }

  private calculateHealthScore(metrics: any): number {
    const weights = { cpu: 0.25, memory: 0.25, latency: 0.2, queue: 0.15, errors: 0.15 };
    const normalizedLatency = Math.min(1, metrics.networkLatency / 200);
    const normalizedQueue = Math.min(1, metrics.messageQueueSize / 2000);
    const quantumBoost = SystemHealth.quantumCoherence[parseInt(Object.keys(metrics)[0])] || 1;
    return Math.max(0, 1 - (
      weights.cpu * metrics.cpuUsage +
      weights.memory * metrics.memoryUsage +
      weights.latency * normalizedLatency +
      weights.queue * normalizedQueue +
      weights.errors * metrics.errorRate
    ) * (1 / quantumBoost));
  }

  private calculatePerformanceScore(metrics: any): number {
    const latencyScore = Math.max(0, 1 - (metrics.networkLatency / 200));
    const queueScore = Math.max(0, 1 - (metrics.messageQueueSize / 2000));
    const errorScore = Math.max(0, 1 - metrics.errorRate * 2);
    const neuralBoost = this.neuralSync.synchronizeStates(metrics, metrics).then(r => r.coherenceScore * 0.1).catch(() => 0);
    return (latencyScore * 0.5) + (queueScore * 0.3) + (errorScore * 0.2) + (await neuralBoost);
  }

  private calculateTrendSlope(entityId: string): number {
    const history = this.history.get(entityId);
    if (!history || history.length < 24) return 0;
    const recent = history.slice(-24).map(h => this.calculateHealthScore(h.metrics));
    const x = Array(24).fill(0).map((_, i) => i);
    const meanX = 11.5;
    const meanY = mean(recent);
    const numerator = x.reduce((sum, xi, i) => sum + (xi - meanX) * (recent[i] - meanY), 0);
    const denominator = x.reduce((sum, xi) => sum + Math.pow(xi - meanX, 2), 0);
    return numerator / denominator;
  }

  private updateTrendVector(entityId: number) {
    const history = this.history.get(entityId.toString());
    if (!history || history.length < 24) return;
    const recent = history.slice(-24);
    const slopes = ['cpuUsage', 'memoryUsage', 'networkLatency'].map(metric => {
      const values = recent.map(h => h.metrics[metric]);
      const x = Array(24).fill(0).map((_, i) => i);
      const meanX = 11.5;
      const meanY = mean(values);
      const num = x.reduce((sum, xi, i) => sum + (xi - meanX) * (values[i] - meanY), 0);
      const den = x.reduce((sum, xi) => sum + Math.pow(xi - meanX, 2), 0);
      return num / den;
    });
    SystemHealth.trendVector[entityId] = slopes as any;
  }

  private estimateDowntime(bottleneck: { metric: string; severity: 'low' | 'medium' | 'high'; value: number; threshold: number }): number {
    const baseDowntime = { low: 30, medium: 60, high: 120 };
    const severityMultiplier = { low: 1, medium: 1.5, high: 2 };
    const impactFactor = Math.min(2, bottleneck.value / bottleneck.threshold);
    return Math.round(baseDowntime[bottleneck.severity] * severityMultiplier[bottleneck.severity] * impactFactor);
  }

  private generateRecommendation(bottleneck: { metric: string; severity: 'low' | 'medium' | 'high' }): string {
    const recs = {
      cpu: `Optimize CPU-intensive processes or scale resources`,
      memory: `Increase memory allocation or reduce memory leaks`,
      latency: `Investigate network bottlenecks or optimize data transfer`,
      queue: `Scale message queue capacity or optimize processing`,
      errors: `Debug error sources or increase fault tolerance`
    };
    return `${recs[bottleneck.metric]} (Severity: ${bottleneck.severity})`;
  }

  private calculateUptime(entityId: number, currentTime: number): number {
    const lastMaintenance = SystemHealth.lastMaintenanceTime[entityId] || 0;
    return Math.floor((currentTime - lastMaintenance) / (1000 * 60 * 60));
  }

  private async scheduleMaintenanceIfNeeded(entityId: string, timestamp: number) {
    const id = parseInt(entityId);
    const urgency = PredictiveMetrics.maintenanceUrgency[id];
    if (urgency > 0.7 && !this.maintenanceSchedule.has(entityId)) {
      const tasks = this.getMaintenanceRecommendations(entityId).map(rec => ({
        id: `${entityId}_${rec.component}`,
        duration: rec.estimatedDowntime * 60 * 1000,
        priority: rec.urgency === 'high' ? 200 : rec.urgency === 'medium' ? 150 : 100,
        dependencies: rec.component === 'system' ? [] : [`${entityId}_system`],
        resourceRequirements: {
          cpu: rec.component === 'cpu' ? 0.5 : 0.2,
          memory: rec.component === 'memory' ? 0.5 : 0.2,
          network: rec.component === 'latency' ? 0.5 : 0.2
        }
      }));

      const schedule = await this.scheduler.optimizeSchedule(tasks, {
        cpuUsage: SystemHealth.cpuUsage[id],
        memoryUsage: SystemHealth.memoryUsage[id],
        networkLatency: SystemHealth.networkLatency[id]
      }, {
        maxDowntime: 7200000,
        maxConcurrentTasks: 2,
        availableTimeSlots: [{ start: timestamp + 3600000, end: timestamp + 86400000 }]
      });

      this.maintenanceSchedule.set(entityId, { time: schedule.tasks[0]?.startTime || timestamp + 3600000, urgency });
      SystemHealth.lastMaintenanceTime[id] = schedule.tasks[0]?.startTime || timestamp;
    }
  }

  private getSystemHealth(id: number): any {
    return {
      cpuUsage: SystemHealth.cpuUsage[id],
      memoryUsage: SystemHealth.memoryUsage[id],
      networkLatency: SystemHealth.networkLatency[id],
      messageQueueSize: SystemHealth.messageQueueSize[id],
      errorRate: SystemHealth.errorRate[id],
      lastMaintenanceTime: SystemHealth.lastMaintenanceTime[id],
      uptime: SystemHealth.uptime[id],
      quantumCoherence: SystemHealth.quantumCoherence[id],
      trendVector: SystemHealth.trendVector[id]
    };
  }

  visualizeHealthTrends(entityId: string, hours: number = 24): string {
    const history = this.history.get(entityId);
    if (!history || history.length < 2) return "Insufficient data";

    const recent = history.slice(-hours).map(h => ({
      time: new Date(h.timestamp).toISOString().slice(11, 16),
      health: this.calculateHealthScore(h.metrics),
      quantum: SystemHealth.quantumCoherence[parseInt(entityId)]
    }));
    return recent.map(r => `${r.time}: ${'█'.repeat(Math.round(r.health * 10))} (Q:${r.quantum.toFixed(2)})`).join('\n');
  }
}
