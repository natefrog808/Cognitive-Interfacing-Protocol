import { defineComponent, Types } from 'bitecs';
import { mean, std, variance } from 'mathjs';
import { MaintenanceScheduler } from './MaintenanceScheduler'; // Assuming this exists
import { CognitiveWebSocketServer } from './CognitiveWebSocketServer';

// Enhanced System Health Metrics Component
const SystemHealth = defineComponent({
  cpuUsage: Types.f32,          // 0-1
  memoryUsage: Types.f32,       // 0-1
  networkLatency: Types.f32,    // ms
  messageQueueSize: Types.ui32, // Count
  errorRate: Types.f32,         // 0-1
  lastMaintenanceTime: Types.ui32, // Timestamp (ms)
  uptime: Types.ui32            // Current uptime in hours
});

// Enhanced Predictive Metrics Component
const PredictiveMetrics = defineComponent({
  failureProbability: Types.f32,  // 0-1
  expectedUptime: Types.f32,      // Hours
  maintenanceUrgency: Types.f32,  // 0-1
  healthScore: Types.f32,         // 0-1
  performanceScore: Types.f32,    // 0-1
  trendSlope: Types.f32           // Rate of change in health (positive = improving)
});

// Holt-Winters Smoothing for Seasonality and Trends
class HoltWintersSmoothing {
  private alpha: number; // Level smoothing
  private beta: number;  // Trend smoothing
  private gamma: number; // Seasonal smoothing
  private period: number; // Seasonal period (e.g., 24 hours)
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
      if (this.data.length === this.period) {
        this.initialize();
      }
      return value;
    }

    const prevLevel = this.level;
    const prevTrend = this.trend;
    const prevSeasonal = this.seasonal.shift()!;

    this.level = this.alpha * (value - prevSeasonal) + (1 - this.alpha) * (prevLevel + prevTrend);
    this.trend = this.beta * (this.level - prevLevel) + (1 - this.beta) * prevTrend;
    this.seasonal.push(this.gamma * (value - this.level) + (1 - this.gamma) * prevSeasonal);

    this.data.push(value);
    if (this.data.length > this.period * 2) this.data.shift(); // Keep 2 periods for stability
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

      if (currentValue > effectiveThreshold) {
        const severity = this.calculateSeverity(currentValue, effectiveThreshold);
        const confidence = this.calculateConfidence(values, currentValue, effectiveThreshold);
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

  private calculateSeverity(value: number, threshold: number): 'low' | 'medium' | 'high' {
    const ratio = value / threshold;
    return ratio > 2 ? 'high' : ratio > 1.5 ? 'medium' : 'low';
  }

  private calculateConfidence(values: number[], value: number, threshold: number): number {
    const meanVal = mean(values);
    const stdVal = std(values) || 1;
    const zScore = Math.abs(value - meanVal) / stdVal;
    return Math.min(1, zScore / 3); // Normalize to 0-1, cap at 3 SDs
  }

  private updateDynamicThreshold(name: string) {
    const values = this.metrics.get(name)!;
    if (values.length < this.historyWindow) return;
    const meanVal = mean(values);
    const stdVal = std(values);
    const thresholdData = this.thresholds.get(name)!;
    thresholdData.dynamic = meanVal + 2 * stdVal; // 2 SDs above mean
  }
}

export class PredictiveMonitor {
  private smoothing: Map<string, HoltWintersSmoothing> = new Map();
  private bottleneckDetector: BottleneckDetector;
  private maintenanceSchedule: Map<string, { time: number; urgency: number }> = new Map();
  private scheduler: MaintenanceScheduler;
  private wsServer: CognitiveWebSocketServer;
  private history: Map<string, Array<{ timestamp: number; metrics: any }>> = new Map();

  constructor(wsPort: number = 8080) {
    this.bottleneckDetector = new BottleneckDetector();
    this.scheduler = new MaintenanceScheduler(wsPort);
    this.wsServer = new CognitiveWebSocketServer(wsPort);
  }

  updateMetrics(entityId: string, metrics: {
    cpuUsage: number;
    memoryUsage: number;
    networkLatency: number;
    messageQueueSize: number;
    errorRate: number;
  }) {
    const id = parseInt(entityId);
    const timestamp = Date.now();

    // Update SystemHealth
    SystemHealth.cpuUsage[id] = metrics.cpuUsage;
    SystemHealth.memoryUsage[id] = metrics.memoryUsage;
    SystemHealth.networkLatency[id] = metrics.networkLatency;
    SystemHealth.messageQueueSize[id] = metrics.messageQueueSize;
    SystemHealth.errorRate[id] = metrics.errorRate;
    SystemHealth.uptime[id] = this.calculateUptime(id, timestamp);

    // Store history
    if (!this.history.has(entityId)) this.history.set(entityId, []);
    this.history.get(entityId)!.push({ timestamp, metrics });
    if (this.history.get(entityId)!.length > 1000) this.history.get(entityId)!.shift();

    // Update smoothing predictors
    Object.entries(metrics).forEach(([metric, value]) => {
      const key = `${entityId}-${metric}`;
      if (!this.smoothing.has(key)) {
        this.smoothing.set(key, new HoltWintersSmoothing(0.3, 0.1, 0.2, 24)); // 24-hour seasonality
      }
      this.smoothing.get(key)!.update(value);
    });

    // Update bottleneck detection with dynamic thresholds
    this.bottleneckDetector.addMetric('cpu', metrics.cpuUsage, 0.8);
    this.bottleneckDetector.addMetric('memory', metrics.memoryUsage, 0.9);
    this.bottleneckDetector.addMetric('latency', metrics.networkLatency, 100);
    this.bottleneckDetector.addMetric('queue', metrics.messageQueueSize, 1000);
    this.bottleneckDetector.addMetric('errors', metrics.errorRate, 0.1);

    // Update predictive metrics and schedule maintenance if needed
    this.updatePredictiveMetrics(entityId, metrics, timestamp);
    this.scheduleMaintenanceIfNeeded(entityId, timestamp);
  }

  private updatePredictiveMetrics(entityId: string, currentMetrics: any, timestamp: number) {
    const id = parseInt(entityId);
    const failureProbability = this.calculateFailureProbability(currentMetrics);
    const expectedUptime = this.predictUptime(entityId, currentMetrics);
    const maintenanceUrgency = this.calculateMaintenanceUrgency(failureProbability, expectedUptime, entityId);
    const healthScore = this.calculateHealthScore(currentMetrics);
    const performanceScore = this.calculatePerformanceScore(currentMetrics);
    const trendSlope = this.calculateTrendSlope(entityId);

    PredictiveMetrics.failureProbability[id] = failureProbability;
    PredictiveMetrics.expectedUptime[id] = expectedUptime;
    PredictiveMetrics.maintenanceUrgency[id] = maintenanceUrgency;
    PredictiveMetrics.healthScore[id] = healthScore;
    PredictiveMetrics.performanceScore[id] = performanceScore;
    PredictiveMetrics.trendSlope[id] = trendSlope;

    // Broadcast updates
    this.wsServer.broadcastStateUpdate(entityId, {
      systemHealth: this.getSystemHealth(id),
      predictiveMetrics: {
        failureProbability,
        expectedUptime,
        maintenanceUrgency,
        healthScore,
        performanceScore,
        trendSlope
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
        recommendation: 'Immediate full system maintenance required due to high failure risk',
        estimatedDowntime: 180, // 3 hours
        confidence: Math.min(1, failureProb * 1.5)
      });
    } else if (failureProb > 0.4) {
      recommendations.push({
        component: 'system',
        urgency: 'medium',
        recommendation: 'Schedule system check due to moderate failure risk',
        estimatedDowntime: 60, // 1 hour
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

  private calculateFailureProbability(metrics: any): number {
    const weights = {
      cpu: 0.3,
      memory: 0.25,
      latency: 0.2,
      queue: 0.15,
      errors: 0.1
    };
    const normalizedLatency = metrics.networkLatency / 200; // Scale to 0-1 based on 200ms max
    const normalizedQueue = metrics.messageQueueSize / 2000; // Scale to 0-1 based on 2000 max
    return Math.min(1, (
      weights.cpu * metrics.cpuUsage +
      weights.memory * metrics.memoryUsage +
      weights.latency * Math.min(1, normalizedLatency) +
      weights.queue * Math.min(1, normalizedQueue) +
      weights.errors * metrics.errorRate
    ));
  }

  private predictUptime(entityId: string, metrics: any): number {
    const baseUptime = 168; // 1 week in hours
    const failureProb = this.calculateFailureProbability(metrics);
    const cpuForecast = this.smoothing.get(`${entityId}-cpuUsage`)!.predict(24)[23]; // 24-hour forecast
    const trendImpact = this.calculateTrendSlope(entityId);
    return baseUptime * (1 - failureProb) * (1 - cpuForecast * 0.2) * (1 + trendImpact * 0.1);
  }

  private calculateMaintenanceUrgency(failureProbability: number, uptime: number, entityId: string): number {
    const lastMaintenance = SystemHealth.lastMaintenanceTime[parseInt(entityId)] || 0;
    const timeSinceMaintenance = (Date.now() - lastMaintenance) / (1000 * 60 * 60); // Hours
    const agingFactor = Math.min(1, timeSinceMaintenance / 168); // Cap at 1 week
    return Math.min(1, (failureProbability * 0.6) + ((168 - uptime) / 168 * 0.3) + (agingFactor * 0.1));
  }

  private calculateHealthScore(metrics: any): number {
    const weights = { cpu: 0.25, memory: 0.25, latency: 0.2, queue: 0.15, errors: 0.15 };
    const normalizedLatency = Math.min(1, metrics.networkLatency / 200);
    const normalizedQueue = Math.min(1, metrics.messageQueueSize / 2000);
    return Math.max(0, 1 - (
      weights.cpu * metrics.cpuUsage +
      weights.memory * metrics.memoryUsage +
      weights.latency * normalizedLatency +
      weights.queue * normalizedQueue +
      weights.errors * metrics.errorRate
    ));
  }

  private calculatePerformanceScore(metrics: any): number {
    const latencyScore = Math.max(0, 1 - (metrics.networkLatency / 200));
    const queueScore = Math.max(0, 1 - (metrics.messageQueueSize / 2000));
    const errorScore = Math.max(0, 1 - metrics.errorRate * 2);
    return (latencyScore * 0.5) + (queueScore * 0.3) + (errorScore * 0.2);
  }

  private calculateTrendSlope(entityId: string): number {
    const history = this.history.get(entityId);
    if (!history || history.length < 24) return 0;
    const recent = history.slice(-24).map(h => this.calculateHealthScore(h.metrics));
    const x = Array(24).fill(0).map((_, i) => i);
    const meanX = 11.5; // Mean of 0-23
    const meanY = mean(recent);
    const numerator = x.reduce((sum, xi, i) => sum + (xi - meanX) * (recent[i] - meanY), 0);
    const denominator = x.reduce((sum, xi) => sum + Math.pow(xi - meanX, 2), 0);
    return numerator / denominator; // Linear regression slope
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
    return Math.floor((currentTime - lastMaintenance) / (1000 * 60 * 60)); // Hours
  }

  private async scheduleMaintenanceIfNeeded(entityId: string, timestamp: number) {
    const id = parseInt(entityId);
    const urgency = PredictiveMetrics.maintenanceUrgency[id];
    if (urgency > 0.7 && !this.maintenanceSchedule.has(entityId)) {
      const tasks = this.getMaintenanceRecommendations(entityId).map(rec => ({
        id: `${entityId}_${rec.component}`,
        duration: rec.estimatedDowntime * 60 * 1000, // Convert to ms
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
        maxDowntime: 7200000, // 2 hours
        maxConcurrentTasks: 2,
        availableTimeSlots: [{ start: timestamp + 3600000, end: timestamp + 86400000 }] // Next 24 hours
      });

      this.maintenanceSchedule.set(entityId, { time: schedule.tasks[0]?.startTime || timestamp + 3600000, urgency });
    }
  }

  private getSystemHealth(id: number): any {
    return {
      cpuUsage: SystemHealth.cpuUsage[id],
      memoryUsage: SystemHealth.memoryUsage[id],
      networkLatency: SystemHealth.networkLatency[id],
      messageQueueSize: SystemHealth.messageQueueSize[id],
      errorRate: SystemHealth.errorRate[id],
      uptime: SystemHealth.uptime[id]
    };
  }

  // New: Visualize system health trends
  visualizeHealthTrends(entityId: string, hours: number = 24): string {
    const history = this.history.get(entityId);
    if (!history || history.length < 2) return "Insufficient data";

    const recent = history.slice(-hours).map(h => ({
      time: new Date(h.timestamp).toISOString().slice(11, 16),
      health: this.calculateHealthScore(h.metrics)
    }));
    return recent.map(r => `${r.time}: ${'█'.repeat(Math.round(r.health * 10))}`).join('\n');
  }
}
