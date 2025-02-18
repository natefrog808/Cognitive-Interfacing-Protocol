import { defineComponent, Types } from 'bitecs';

// System health metrics component
const SystemHealth = defineComponent({
  cpuUsage: Types.f32,
  memoryUsage: Types.f32,
  networkLatency: Types.f32,
  messageQueueSize: Types.ui32,
  errorRate: Types.f32,
  lastMaintenanceTime: Types.ui32
});

// Predictive metrics component
const PredictiveMetrics = defineComponent({
  failureProbability: Types.f32,
  expectedUptime: Types.f32,
  maintenanceUrgency: Types.f32,
  healthScore: Types.f32,
  performanceScore: Types.f32
});

class ExponentialSmoothing {
  private alpha: number;
  private smoothedData: number[];

  constructor(alpha: number = 0.2) {
    this.alpha = alpha;
    this.smoothedData = [];
  }

  update(newValue: number): number {
    if (this.smoothedData.length === 0) {
      this.smoothedData.push(newValue);
    } else {
      const smoothed = this.alpha * newValue + 
                      (1 - this.alpha) * this.smoothedData[this.smoothedData.length - 1];
      this.smoothedData.push(smoothed);
    }
    return this.smoothedData[this.smoothedData.length - 1];
  }

  predict(steps: number = 1): number {
    if (this.smoothedData.length === 0) return 0;
    return this.smoothedData[this.smoothedData.length - 1];
  }
}

class BottleneckDetector {
  private metrics: Map<string, number[]> = new Map();
  private thresholds: Map<string, number> = new Map();

  addMetric(name: string, value: number, threshold: number) {
    if (!this.metrics.has(name)) {
      this.metrics.set(name, []);
      this.thresholds.set(name, threshold);
    }
    const values = this.metrics.get(name)!;
    values.push(value);
    if (values.length > 100) values.shift();
  }

  detectBottlenecks(): Array<{
    metric: string;
    severity: 'low' | 'medium' | 'high';
    value: number;
    threshold: number;
  }> {
    const bottlenecks = [];
    
    for (const [name, values] of this.metrics.entries()) {
      const currentValue = values[values.length - 1];
      const threshold = this.thresholds.get(name)!;
      
      if (currentValue > threshold) {
        const severity = this.calculateSeverity(currentValue, threshold);
        bottlenecks.push({
          metric: name,
          severity,
          value: currentValue,
          threshold
        });
      }
    }
    
    return bottlenecks;
  }

  private calculateSeverity(value: number, threshold: number): 'low' | 'medium' | 'high' {
    const ratio = value / threshold;
    if (ratio > 2) return 'high';
    if (ratio > 1.5) return 'medium';
    return 'low';
  }
}

export class PredictiveMonitor {
  private smoothing: Map<string, ExponentialSmoothing> = new Map();
  private bottleneckDetector: BottleneckDetector;
  private maintenanceSchedule: Map<string, number> = new Map();
  
  constructor() {
    this.bottleneckDetector = new BottleneckDetector();
  }

  updateMetrics(entityId: string, metrics: {
    cpuUsage: number;
    memoryUsage: number;
    networkLatency: number;
    messageQueueSize: number;
    errorRate: number;
  }) {
    // Update system health metrics
    SystemHealth.cpuUsage[parseInt(entityId)] = metrics.cpuUsage;
    SystemHealth.memoryUsage[parseInt(entityId)] = metrics.memoryUsage;
    SystemHealth.networkLatency[parseInt(entityId)] = metrics.networkLatency;
    SystemHealth.messageQueueSize[parseInt(entityId)] = metrics.messageQueueSize;
    SystemHealth.errorRate[parseInt(entityId)] = metrics.errorRate;

    // Update smoothing predictors
    Object.entries(metrics).forEach(([metric, value]) => {
      const key = `${entityId}-${metric}`;
      if (!this.smoothing.has(key)) {
        this.smoothing.set(key, new ExponentialSmoothing());
      }
      this.smoothing.get(key)!.update(value);
    });

    // Update bottleneck detection
    this.bottleneckDetector.addMetric('cpu', metrics.cpuUsage, 0.8);
    this.bottleneckDetector.addMetric('memory', metrics.memoryUsage, 0.9);
    this.bottleneckDetector.addMetric('latency', metrics.networkLatency, 100);
    this.bottleneckDetector.addMetric('queue', metrics.messageQueueSize, 1000);
    this.bottleneckDetector.addMetric('errors', metrics.errorRate, 0.1);

    // Calculate predictive metrics
    this.updatePredictiveMetrics(entityId, metrics);
  }

  private updatePredictiveMetrics(entityId: string, currentMetrics: any) {
    const id = parseInt(entityId);
    
    // Calculate failure probability based on multiple factors
    const failureProbability = this.calculateFailureProbability(currentMetrics);
    PredictiveMetrics.failureProbability[id] = failureProbability;

    // Estimate expected uptime
    const uptime = this.estimateUptime(currentMetrics);
    PredictiveMetrics.expectedUptime[id] = uptime;

    // Calculate maintenance urgency
    const urgency = this.calculateMaintenanceUrgency(failureProbability, uptime);
    PredictiveMetrics.maintenanceUrgency[id] = urgency;

    // Update health and performance scores
    PredictiveMetrics.healthScore[id] = this.calculateHealthScore(currentMetrics);
    PredictiveMetrics.performanceScore[id] = this.calculatePerformanceScore(currentMetrics);
  }

  getMaintenanceRecommendations(entityId: string): Array<{
    component: string;
    urgency: 'low' | 'medium' | 'high';
    recommendation: string;
    estimatedDowntime: number;
  }> {
    const bottlenecks = this.bottleneckDetector.detectBottlenecks();
    const id = parseInt(entityId);
    const recommendations = [];

    if (PredictiveMetrics.failureProbability[id] > 0.7) {
      recommendations.push({
        component: 'system',
        urgency: 'high',
        recommendation: 'Immediate system maintenance required',
        estimatedDowntime: 120 // minutes
      });
    }

    for (const bottleneck of bottlenecks) {
      recommendations.push({
        component: bottleneck.metric,
        urgency: bottleneck.severity,
        recommendation: `Optimize ${bottleneck.metric} usage`,
        estimatedDowntime: this.estimateDowntime(bottleneck)
      });
    }

    return recommendations;
  }

  private calculateFailureProbability(metrics: any): number {
    // Implement failure probability calculation based on current metrics
    const cpuWeight = 0.3;
    const memoryWeight = 0.25;
    const latencyWeight = 0.2;
    const queueWeight = 0.15;
    const errorWeight = 0.1;

    return (
      cpuWeight * metrics.cpuUsage +
      memoryWeight * metrics.memoryUsage +
      latencyWeight * (metrics.networkLatency / 100) +
      queueWeight * (metrics.messageQueueSize / 1000) +
      errorWeight * metrics.errorRate
    );
  }

  private estimateUptime(metrics: any): number {
    // Calculate expected uptime in hours based on current metrics
    const baseUptime = 168; // One week in hours
    const failureProbability = this.calculateFailureProbability(metrics);
    
    return baseUptime * (1 - failureProbability);
  }

  private calculateMaintenanceUrgency(failureProbability: number, uptime: number): number {
    // Higher urgency when failure probability is high or expected uptime is low
    return (failureProbability * 0.7) + ((168 - uptime) / 168 * 0.3);
  }

  private calculateHealthScore(metrics: any): number {
    // Calculate overall system health score (0-1)
    const weights = {
      cpu: 0.25,
      memory: 0.25,
      latency: 0.2,
      queue: 0.15,
      errors: 0.15
    };

    return 1 - (
      weights.cpu * metrics.cpuUsage +
      weights.memory * metrics.memoryUsage +
      weights.latency * (metrics.networkLatency / 100) +
      weights.queue * (metrics.messageQueueSize / 1000) +
      weights.errors * metrics.errorRate
    );
  }

  private calculatePerformanceScore(metrics: any): number {
    // Calculate performance score based on latency and queue size
    const latencyScore = Math.max(0, 1 - (metrics.networkLatency / 100));
    const queueScore = Math.max(0, 1 - (metrics.messageQueueSize / 1000));
    
    return (latencyScore * 0.6) + (queueScore * 0.4);
  }

  private estimateDowntime(bottleneck: {
    metric: string;
    severity: 'low' | 'medium' | 'high';
    value: number;
    threshold: number;
  }): number {
    // Estimate maintenance downtime in minutes based on bottleneck severity
    const baseDowntime = {
      low: 30,
      medium: 60,
      high: 120
    };

    const severityMultiplier = {
      low: 1,
      medium: 1.5,
      high: 2
    };

    return baseDowntime[bottleneck.severity] * 
           severityMultiplier[bottleneck.severity] * 
           (bottleneck.value / bottleneck.threshold);
  }
