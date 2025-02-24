import { defineComponent, Types } from 'bitecs';
import { Matrix, eigen, multiply, mean, std } from 'mathjs';
import * as tf from '@tensorflow/tfjs';
import { NeuralSynchronizer } from '../neural/NeuralSynchronizer';
import { QuantumStateEncoder } from '../quantum/QuantumStateEncoder';
import { AdvancedAnomalyDetector } from '../anomaly/AdvancedAnomalyDetector';

// Enhanced Emergent Behavior Components
const EmergentPatterns = defineComponent({
  complexityScore: Types.f32,
  stabilityMetric: Types.f32,
  noveltyIndex: Types.f32,
  duration: Types.ui32,
  frequency: Types.f32,
  phaseShift: Types.f32,
  influenceScore: Types.f32,
  propagationSpeed: Types.f32,
  cascadeDepth: Types.ui32,
  quantumCoherence: Types.f32,   // New: Quantum influence
  neuralCoherence: Types.f32,    // New: Neural sync alignment
  anomalyImpact: Types.f32       // New: Anomaly-driven disruption
});

interface BehaviorPattern {
  id: string;
  type: 'cyclic' | 'emergent' | 'cascade' | 'stable';
  participants: string[];
  startTime: number;
  duration: number;
  strength: number;
  causalChain: Map<string, string[]>;
  phaseMap: number[];           // New: Temporal phase signature
  influenceVector: number[];    // New: Spatial influence spread
}

export class EmergentBehaviorAnalyzer {
  private patterns: Map<string, BehaviorPattern> = new Map();
  private timeSeriesData: Map<string, number[][]> = new Map();
  private readonly HISTORY_WINDOW = 1000;
  private MIN_PATTERN_STRENGTH = 0.3; // Dynamic now
  private patternRecognizer: tf.LayersModel;
  private neuralSync: NeuralSynchronizer;
  private quantumEncoder: QuantumStateEncoder;
  private anomalyDetector: AdvancedAnomalyDetector;

  constructor() {
    this.neuralSync = new NeuralSynchronizer();
    this.quantumEncoder = new QuantumStateEncoder();
    this.anomalyDetector = new AdvancedAnomalyDetector();
  }

  async initialize() {
    await this.neuralSync.initialize(0.5);
    await this.quantumEncoder.initialize();
    this.patternRecognizer = await this.createPatternRecognizer();
    console.log("EmergentBehaviorAnalyzer initialized—ready to unravel the cosmos!");
  }

  private async createPatternRecognizer(): Promise<tf.LayersModel> {
    const model = tf.sequential();
    model.add(tf.layers.conv1d({
      filters: 64, // Doubled for richer feature extraction
      kernelSize: 5,
      activation: 'relu',
      inputShape: [this.HISTORY_WINDOW, 10] // Expanded for quantum/anomaly inputs
    }));
    model.add(tf.layers.lstm({
      units: 128, // Increased capacity
      returnSequences: true
    }));
    model.add(tf.layers.dropout({ rate: 0.2 }));
    model.add(tf.layers.lstm({ units: 64 }));
    model.add(tf.layers.dense({ units: 32, activation: 'relu' }));
    model.add(tf.layers.dense({ units: 4, activation: 'softmax' }));
    model.compile({
      optimizer: tf.train.adam(0.0005), // Finer learning rate
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy']
    });
    return model;
  }

  async train(trainingData: { features: number[][][], labels: number[][] }, epochs: number = 50) {
    const xs = tf.tensor3d(trainingData.features);
    const ys = tf.tensor2d(trainingData.labels);

    console.log("Training pattern recognizer...");
    await this.patternRecognizer.fit(xs, ys, {
      epochs,
      batchSize: 32,
      validationSplit: 0.2,
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          console.log(`Epoch ${epoch}: Accuracy = ${logs?.acc}, Loss = ${logs?.loss}`);
          this.MIN_PATTERN_STRENGTH = Math.max(0.2, 0.5 - (logs?.acc || 0) * 0.2); // Adaptive threshold
        }
      }
    });

    tf.dispose([xs, ys]);
    console.log(`Training complete—MIN_PATTERN_STRENGTH tuned to ${this.MIN_PATTERN_STRENGTH}`);
  }

  async analyzeSystemState(
    entities: Map<string, any>,
    relationships: Map<string, any[]>
  ): Promise<{
    patterns: BehaviorPattern[];
    metrics: EmergentMetrics;
    predictions: EmergencePrediction[];
  }> {
    this.updateTimeSeriesData(entities);
    const currentPatterns = await this.detectPatterns(entities, relationships);
    const causalGraph = this.analyzeCausality(currentPatterns);
    const predictions = await this.predictEmergence(currentPatterns, causalGraph);
    const metrics = this.calculateSystemMetrics(currentPatterns, causalGraph);

    // Update BitECS components
    currentPatterns.forEach((pattern, idx) => {
      const eid = idx; // Simplified entity ID for demo
      EmergentPatterns.complexityScore[eid] = metrics.systemComplexity;
      EmergentPatterns.stabilityMetric[eid] = metrics.stabilityScore;
      EmergentPatterns.noveltyIndex[eid] = pattern.strength * 0.5;
      EmergentPatterns.duration[eid] = pattern.duration;
      EmergentPatterns.frequency[eid] = pattern.type === 'cyclic' ? 1 / pattern.duration : 0;
      EmergentPatterns.phaseShift[eid] = pattern.phaseMap[0] || 0;
      EmergentPatterns.influenceScore[eid] = pattern.influenceVector.reduce((sum, v) => sum + v, 0);
      EmergentPatterns.propagationSpeed[eid] = pattern.type === 'cascade' ? pattern.strength / pattern.duration : 0;
      EmergentPatterns.cascadeDepth[eid] = pattern.causalChain.size;
      EmergentPatterns.quantumCoherence[eid] = entities.get(pattern.participants[0])?.quantumCoherence || 0;
      EmergentPatterns.neuralCoherence[eid] = entities.get(pattern.participants[0])?.neuralCoherence || 0;
      EmergentPatterns.anomalyImpact[eid] = pattern.strength * (entities.get(pattern.participants[0])?.anomalyScore || 0);
    });

    return { patterns: currentPatterns, metrics, predictions };
  }

  private updateTimeSeriesData(entities: Map<string, any>) {
    entities.forEach((entity, id) => {
      if (!this.timeSeriesData.has(id)) this.timeSeriesData.set(id, []);
      const timeSeries = this.timeSeriesData.get(id)!;
      const vector = this.extractFeatureVector(entity);
      this.anomalyDetector.updateData(id, vector.slice(0, 5)); // Update anomaly detector
      timeSeries.push(vector);
      if (timeSeries.length > this.HISTORY_WINDOW) timeSeries.shift();
    });
  }

  private extractFeatureVector(entity: any): number[] {
    return [
      entity.cognitive.awareness || 0,
      entity.cognitive.coherence || 0,
      entity.cognitive.complexity || 0,
      entity.emotional.mood || 0,
      entity.emotional.stress || 0,
      entity.emotional.motivation || 0,
      entity.performance?.efficiency || 0,
      entity.performance?.stability || 0,
      entity.quantumCoherence || 0, // New: Quantum influence
      entity.anomalyScore || 0      // New: Anomaly impact
    ];
  }

  private async detectPatterns(
    entities: Map<string, any>,
    relationships: Map<string, any[]>
  ): Promise<BehaviorPattern[]> {
    const tensorData = await this.prepareTensorData();
    const predictions = this.patternRecognizer.predict(tensorData) as tf.Tensor;
    const patternTypes = await predictions.array() as number[][];

    const patterns: BehaviorPattern[] = [];
    let idx = 0;
    entities.forEach((entity, id) => {
      const entityPatterns = this.analyzeEntityPatterns(id, entity, patternTypes[idx], relationships);
      patterns.push(...entityPatterns);
      idx++;
    });

    tf.dispose([tensorData, predictions]);
    return this.mergePatterns(patterns);
  }

  private async prepareTensorData(): Promise<tf.Tensor> {
    const tensorData: number[][][] = [];
    this.timeSeriesData.forEach(timeSeries => {
      tensorData.push(timeSeries.length === this.HISTORY_WINDOW ? timeSeries : this.padTimeSeries(timeSeries));
    });
    return tf.tensor3d(tensorData);
  }

  private padTimeSeries(series: number[][]): number[][] {
    const padded = Array(this.HISTORY_WINDOW).fill(0).map(() => new Array(10).fill(0));
    series.forEach((point, i) => padded[i] = point);
    return padded;
  }

  private analyzeEntityPatterns(
    entityId: string,
    entity: any,
    patternProbabilities: number[],
    relationships: Map<string, any[]>
  ): BehaviorPattern[] {
    const patterns: BehaviorPattern[] = [];
    const anomaly = this.anomalyDetector.detectAnomalies(entityId, this.timeSeriesData.get(entityId)!.slice(-1)[0]);
    
    if (patternProbabilities[0] > this.MIN_PATTERN_STRENGTH) {
      patterns.push(this.detectCyclicPattern(entityId, entity, relationships));
    }
    if (patternProbabilities[1] > this.MIN_PATTERN_STRENGTH || anomaly.isAnomaly) {
      patterns.push(this.detectEmergentPattern(entityId, entity, relationships, anomaly));
    }
    if (patternProbabilities[2] > this.MIN_PATTERN_STRENGTH) {
      patterns.push(this.detectCascadePattern(entityId, entity, relationships));
    }
    if (!patterns.length) {
      patterns.push(this.detectStablePattern(entityId, entity, relationships));
    }
    return patterns;
  }

  private detectCyclicPattern(entityId: string, entity: any, relationships: Map<string, any[]>): BehaviorPattern {
    const timeSeries = this.timeSeriesData.get(entityId)!;
    const frequency = this.detectFrequency(timeSeries);
    const phaseMap = this.calculatePhaseMap(timeSeries);
    return {
      id: `cyclic_${entityId}_${Date.now()}`,
      type: 'cyclic',
      participants: [entityId],
      startTime: Date.now(),
      duration: frequency,
      strength: this.calculatePatternStrength(timeSeries) * (1 + entity.quantumCoherence || 0),
      causalChain: new Map(),
      phaseMap,
      influenceVector: [1] // Single entity influence
    };
  }

  private detectEmergentPattern(entityId: string, entity: any, relationships: Map<string, any[]>, anomaly: any): BehaviorPattern {
    const relatedEntities = relationships.get(entityId) || [];
    const participants = this.findCoordinatedEntities(entityId, relatedEntities);
    const influenceVector = this.calculateInfluenceVector(participants, relationships);
    return {
      id: `emergent_${entityId}_${Date.now()}`,
      type: 'emergent',
      participants,
      startTime: Date.now(),
      duration: 0,
      strength: this.calculateCollectiveStrength(participants) * (1 + anomaly.forecastScore || 0),
      causalChain: this.buildCausalChain(participants),
      phaseMap: Array(participants.length).fill(0), // Placeholder
      influenceVector
    };
  }

  private detectCascadePattern(entityId: string, entity: any, relationships: Map<string, any[]>): BehaviorPattern {
    const cascade = this.analyzeCascadeEffect(entityId, relationships);
    const influenceVector = this.calculateInfluenceVector(cascade.participants, relationships);
    return {
      id: `cascade_${entityId}_${Date.now()}`,
      type: 'cascade',
      participants: cascade.participants,
      startTime: Date.now(),
      duration: cascade.duration,
      strength: cascade.strength,
      causalChain: cascade.causalChain,
      phaseMap: Array(cascade.participants.length).fill(0), // Placeholder
      influenceVector
    };
  }

  private detectStablePattern(entityId: string, entity: any, relationships: Map<string, any[]>): BehaviorPattern {
    return {
      id: `stable_${entityId}_${Date.now()}`,
      type: 'stable',
      participants: [entityId],
      startTime: Date.now(),
      duration: this.HISTORY_WINDOW,
      strength: entity.cognitive.coherence || 0.5,
      causalChain: new Map(),
      phaseMap: [0],
      influenceVector: [1]
    };
  }

  private findCoordinatedEntities(entityId: string, relatedEntities: any[]): string[] {
    const coordinated = new Set<string>([entityId]);
    relatedEntities.forEach(related => {
      const correlation = this.calculateCorrelation(
        this.timeSeriesData.get(entityId)!,
        this.timeSeriesData.get(related.id)!
      );
      if (correlation > 0.7) coordinated.add(related.id);
    });
    return Array.from(coordinated);
  }

  private calculateCorrelation(series1: number[][], series2: number[][]): number {
    const n = Math.min(series1.length, series2.length);
    let sum1 = 0, sum2 = 0, sum1Sq = 0, sum2Sq = 0, pSum = 0;
    for (let i = 0; i < n; i++) {
      const x = series1[i][0];
      const y = series2[i][0];
      sum1 += x;
      sum2 += y;
      sum1Sq += x * x;
      sum2Sq += y * y;
      pSum += x * y;
    }
    const num = pSum - (sum1 * sum2 / n);
    const den = Math.sqrt((sum1Sq - sum1 * sum1 / n) * (sum2Sq - sum2 * sum2 / n));
    return den === 0 ? 0 : num / den;
  }

  private analyzeCascadeEffect(entityId: string, relationships: Map<string, any[]>): {
    participants: string[];
    duration: number;
    strength: number;
    causalChain: Map<string, string[]>;
  } {
    const visited = new Set<string>();
    const cascade = {
      participants: [] as string[],
      duration: 0,
      strength: 0,
      causalChain: new Map<string, string[]>()
    };
    this.traverseCascade(entityId, relationships, visited, cascade, 0);
    return cascade;
  }

  private traverseCascade(
    entityId: string,
    relationships: Map<string, any[]>,
    visited: Set<string>,
    cascade: any,
    depth: number
  ) {
    if (visited.has(entityId)) return;
    visited.add(entityId);
    cascade.participants.push(entityId);
    const related = relationships.get(entityId) || [];
    const influencedEntities = related.filter(r => this.detectStateInfluence(entityId, r.id));
    cascade.causalChain.set(entityId, influencedEntities.map(e => e.id));
    influencedEntities.forEach(influenced => {
      this.traverseCascade(influenced.id, relationships, visited, cascade, depth + 1);
    });
    cascade.duration = Math.max(cascade.duration, depth * 100);
    cascade.strength = this.calculateCascadeStrength(cascade);
  }

  private detectStateInfluence(sourceId: string, targetId: string): boolean {
    const sourceSeries = this.timeSeriesData.get(sourceId)!;
    const targetSeries = this.timeSeriesData.get(targetId)!;
    const lag = 5;
    let influenceScore = 0;
    for (let i = lag; i < sourceSeries.length; i++) {
      const sourceState = sourceSeries[i - lag][0];
      const targetState = targetSeries[i][0];
      influenceScore += Math.abs(targetState - sourceState);
    }
    return influenceScore > 0.5;
  }

  private calculateCascadeStrength(cascade: any): number {
    return cascade.participants.length * Math.log(cascade.duration + 1) * 10;
  }

  private calculatePatternStrength(timeSeries: number[][]): number {
    const values = timeSeries.map(point => point[0]);
    return std(values) || 0;
  }

  private calculateCollectiveStrength(participants: string[]): number {
    return participants.reduce((sum, id) => {
      const series = this.timeSeriesData.get(id)!;
      return sum + this.calculatePatternStrength(series);
    }, 0) / participants.length;
  }

  private calculateInfluenceVector(participants: string[], relationships: Map<string, any[]>): number[] {
    const vector = participants.map(id => {
      const related = relationships.get(id) || [];
      return related.length / participants.length;
    });
    return vector.length ? vector : [1];
  }

  private calculatePhaseMap(timeSeries: number[][]): number[] {
    const signal = timeSeries.map(point => point[0]);
    const peaks = this.findPeaks(signal);
    return peaks.map(p => p / this.HISTORY_WINDOW); // Normalized phase positions
  }

  private mergePatterns(patterns: BehaviorPattern[]): BehaviorPattern[] {
    const merged: BehaviorPattern[] = [];
    const groups = new Map<string, BehaviorPattern[]>();
    patterns.forEach(pattern => {
      let found = false;
      for (const [key, group] of groups) {
        if (this.patternsOverlap(pattern, group[0])) {
          group.push(pattern);
          found = true;
          break;
        }
      }
      if (!found) groups.set(pattern.id, [pattern]);
    });
    groups.forEach(group => {
      merged.push(group.length === 1 ? group[0] : this.mergePatternGroup(group));
    });
    return merged;
  }

  private patternsOverlap(p1: BehaviorPattern, p2: BehaviorPattern): boolean {
    const p1Set = new Set(p1.participants);
    const p2Set = new Set(p2.participants);
    let overlap = 0;
    p1Set.forEach(p => { if (p2Set.has(p)) overlap++; });
    return overlap / Math.min(p1Set.size, p2Set.size) > 0.5;
  }

  private mergePatternGroup(patterns: BehaviorPattern[]): BehaviorPattern {
    const allParticipants = new Set<string>();
    let totalStrength = 0;
    const mergedChain = new Map<string, string[]>();
    const influenceVector: number[] = [];
    patterns.forEach(pattern => {
      pattern.participants.forEach(p => allParticipants.add(p));
      totalStrength += pattern.strength;
      pattern.causalChain.forEach((targets, source) => {
        if (!mergedChain.has(source)) mergedChain.set(source, [...targets]);
        else mergedChain.get(source)!.push(...targets.filter(t => !mergedChain.get(source)!.includes(t)));
      });
      influenceVector.push(...pattern.influenceVector);
    });
    return {
      id: `merged_${Date.now()}`,
      type: patterns[0].type,
      participants: Array.from(allParticipants),
      startTime: Math.min(...patterns.map(p => p.startTime)),
      duration: Math.max(...patterns.map(p => p.duration)),
      strength: totalStrength / patterns.length,
      causalChain: mergedChain,
      phaseMap: patterns[0].phaseMap, // Simplified merge
      influenceVector: influenceVector.length ? influenceVector : [1]
    };
  }

  private detectFrequency(timeSeries: number[][]): number {
    const signal = timeSeries.map(point => point[0]);
    const peaks = this.findPeaks(signal);
    if (peaks.length < 2) return 0;
    const intervals = peaks.slice(1).map((p, i) => p - peaks[i]);
    return mean(intervals) || 0;
  }

  private findPeaks(signal: number[]): number[] {
    const peaks = [];
    for (let i = 1; i < signal.length - 1; i++) {
      if (signal[i] > signal[i - 1] && signal[i] > signal[i + 1]) peaks.push(i);
    }
    return peaks;
  }

  private calculateSystemMetrics(patterns: BehaviorPattern[], causalGraph: Map<string, string[]>): EmergentMetrics {
    return {
      totalPatterns: patterns.length,
      averageStrength: mean(patterns.map(p => p.strength)) || 0,
      dominantType: this.findDominantPatternType(patterns),
      systemComplexity: this.calculateComplexity(patterns, causalGraph),
      stabilityScore: this.calculateStabilityScore(patterns),
      quantumInfluence: mean(patterns.map(p => p.strength * (this.timeSeriesData.get(p.participants[0])?.slice(-1)[0][8] || 0))) || 0
    };
  }

  private findDominantPatternType(patterns: BehaviorPattern[]): 'cyclic' | 'emergent' | 'cascade' | 'stable' {
    const typeCounts = new Map<string, number>();
    patterns.forEach(p => typeCounts.set(p.type, (typeCounts.get(p.type) || 0) + 1));
    let maxCount = 0;
    let dominant = 'stable';
    typeCounts.forEach((count, type) => {
      if (count > maxCount) { maxCount = count; dominant = type; }
    });
    return dominant as any;
  }

  private calculateComplexity(patterns: BehaviorPattern[], causalGraph: Map<string, string[]>): number {
    const interactionCount = Array.from(causalGraph.values()).reduce((sum, t) => sum + t.length, 0);
    const patternDiversity = new Set(patterns.map(p => p.type)).size;
    return (interactionCount * patternDiversity) / (patterns.length || 1);
  }

  private calculateStabilityScore(patterns: BehaviorPattern[]): number {
    const cyclicRatio = patterns.filter(p => p.type === 'cyclic').length / patterns.length || 0;
    const emergentRatio = patterns.filter(p => p.type === 'emergent').length / patterns.length || 0;
    const cascadeRatio = patterns.filter(p => p.type === 'cascade').length / patterns.length || 0;
    return Math.min(1, Math.max(0, cyclicRatio * 0.8 + emergentRatio * 0.4 - cascadeRatio * 0.5));
  }

  async predictEmergence(currentPatterns: BehaviorPattern[], causalGraph: Map<string, string[]>): Promise<EmergencePrediction[]> {
    const predictions: EmergencePrediction[] = [];
    const evolutionTrends = this.analyzePatternEvolution(currentPatterns);
    for (const trend of evolutionTrends) {
      const prediction = await this.generatePrediction(trend, causalGraph);
      if (prediction.confidence > 0.6) {
        const emergentState = await this.neuralSync.generateNovelState(
          { cognitive: { coherence: prediction.probability }, emotional: { motivation: trend.stabilityTrend } },
          prediction.confidence
        );
        prediction.emergentState = emergentState;
        predictions.push(prediction);
      }
    }
    return predictions;
  }

  private analyzePatternEvolution(patterns: BehaviorPattern[]): EvolutionTrend[] {
    const trends: EvolutionTrend[] = [];
    const typeGroups = new Map<string, BehaviorPattern[]>();
    patterns.forEach(p => {
      if (!typeGroups.has(p.type)) typeGroups.set(p.type, []);
      typeGroups.get(p.type)!.push(p);
    });
    typeGroups.forEach((group, type) => {
      const trend = this.calculateTrend(group);
      trends.push({
        patternType: type as any,
        growthRate: trend.growthRate,
        stabilityTrend: trend.stability,
        participants: trend.participants,
        anomalyInfluence: mean(group.map(p => this.timeSeriesData.get(p.participants[0])?.slice(-1)[0][9] || 0)) || 0
      });
    });
    return trends;
  }

  private calculateTrend(patterns: BehaviorPattern[]): { growthRate: number; stability: number; participants: Set<string> } {
    const participants = new Set<string>();
    patterns.forEach(p => p.participants.forEach(part => participants.add(part)));
    return {
      growthRate: patterns.length / this.HISTORY_WINDOW,
      stability: this.calculatePatternStability(patterns),
      participants
    };
  }

  private calculatePatternStability(patterns: BehaviorPattern[]): number {
    if (patterns.length < 2) return 1;
    const overlaps = patterns.slice(1).map((p, i) => this.calculateParticipantOverlap(patterns[i], p));
    return mean(overlaps) || 1;
  }

  private calculateParticipantOverlap(p1: BehaviorPattern, p2: BehaviorPattern): number {
    const set1 = new Set(p1.participants);
    const set2 = new Set(p2.participants);
    let overlap = 0;
    set1.forEach(p => { if (set2.has(p)) overlap++; });
    return overlap / Math.max(set1.size, set2.size);
  }

  private async generatePrediction(trend: EvolutionTrend, causalGraph: Map<string, string[]>): Promise<EmergencePrediction> {
    const tensorData = tf.tensor2d([[trend.growthRate, trend.stabilityTrend, trend.participants.size, trend.anomalyInfluence]]);
    const prediction = this.patternRecognizer.predict(tensorData) as tf.Tensor;
    const predictionArray = await prediction.array() as number[][];
    const quantumBoost = mean(Array.from(trend.participants).map(p => this.timeSeriesData.get(p)?.slice(-1)[0][8] || 0)) || 0;
    tf.dispose([tensorData, prediction]);
    return {
      type: trend.patternType,
      probability: predictionArray[0][0] * (1 + quantumBoost),
      timeframe: this.estimateTimeframe(trend),
      confidence: this.calculatePredictionConfidence(trend, predictionArray[0]),
      potentialParticipants: Array.from(trend.participants),
      expectedDuration: this.estimateDuration(trend),
      emergentState: null // Populated in predictEmergence
    };
  }

  private estimateTimeframe(trend: EvolutionTrend): number {
    return Date.now() + (1000 * 60 * (1 / Math.max(0.01, trend.growthRate)));
  }

  private estimateDuration(trend: EvolutionTrend): number {
    return 1000 * 60 * trend.stabilityTrend * (1 + trend.anomalyInfluence);
  }

  private calculatePredictionConfidence(trend: EvolutionTrend, prediction: number[]): number {
    return Math.min(1, trend.stabilityTrend * 0.5 + prediction[0] * 0.4 + trend.anomalyInfluence * 0.1);
  }
}

interface EmergentMetrics {
  totalPatterns: number;
  averageStrength: number;
  dominantType: 'cyclic' | 'emergent' | 'cascade' | 'stable';
  systemComplexity: number;
  stabilityScore: number;
  quantumInfluence: number; // New: Quantum impact on complexity
}

interface EmergencePrediction {
  type: 'cyclic' | 'emergent' | 'cascade' | 'stable';
  probability: number;
  timeframe: number;
  confidence: number;
  potentialParticipants: string[];
  expectedDuration: number;
  emergentState: any; // New: Predicted novel state
}

interface EvolutionTrend {
  patternType: 'cyclic' | 'emergent' | 'cascade' | 'stable';
  growthRate: number;
  stabilityTrend: number;
  participants: Set<string>;
  anomalyInfluence: number; // New: Anomaly-driven evolution
}
