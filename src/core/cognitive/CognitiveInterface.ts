import { createWorld, defineComponent, defineQuery, Types } from 'bitecs'
import { createDreams } from '@daydreamsai/core/v1'
import { action, task, memory, extension } from '@daydreamsai/core/v1'
import { z } from 'zod'

// Enhanced Cognitive State Components
const CognitiveState = defineComponent({
  // Core cognitive parameters
  awareness: Types.f32,
  coherence: Types.f32,
  complexity: Types.f32,
  cognitiveLoad: Types.f32,
  
  // Memory indices
  shortTermPtr: Types.ui32,
  longTermPtr: Types.ui32,
  
  // State flags
  isProcessing: Types.ui8,
  isTransmitting: Types.ui8,
  isSynchronizing: Types.ui8,
  
  // Performance metrics
  lastUpdateTime: Types.ui32,
  processingLatency: Types.f32
})

// Emotional State Component
const EmotionalState = defineComponent({
  mood: Types.f32,        // Range: -1.0 to 1.0
  stress: Types.f32,      // Range: 0.0 to 1.0
  motivation: Types.f32,  // Range: 0.0 to 1.0
  empathy: Types.f32,     // Range: 0.0 to 1.0
  curiosity: Types.f32,   // Range: 0.0 to 1.0
  anger: Types.f32,       // Range: 0.0 to 1.0
  fear: Types.f32,        // Range: 0.0 to 1.0
  joy: Types.f32,         // Range: 0.0 to 1.0
  disgust: Types.f32      // Range: 0.0 to 1.0
})

// Enhanced Transfer Protocol
const CognitiveTransfer = defineComponent({
  sourceId: Types.ui32,
  targetId: Types.ui32,
  channelId: Types.ui32,
  bandwidth: Types.f32,
  signal: Types.f32,
  lastSyncTime: Types.ui32,
  syncType: Types.ui8,    // 0: full, 1: incremental
  errorCount: Types.ui8,
  retryAttempts: Types.ui8
})

// Performance Metrics Component
const PerformanceMetrics = defineComponent({
  transferLatency: Types.f32,
  dataVolume: Types.ui32,
  successRate: Types.f32,
  errorRate: Types.f32,
  syncFrequency: Types.f32
})

// Security and Integrity
class StateIntegrityManager {
  static validateState(state: any): boolean {
    const validations = [
      state.awareness >= 0 && state.awareness <= 1,
      state.coherence >= 0 && state.coherence <= 1,
      state.complexity >= 0,
      state.mood >= -1 && state.mood <= 1,
      state.stress >= 0 && state.stress <= 1
    ]
    return validations.every(v => v === true)
  }

  static encrypt(state: any): string {
    // Simple encryption for demo - replace with proper encryption in production
    return Buffer.from(JSON.stringify(state)).toString('base64')
  }

  static decrypt(data: string): any {
    return JSON.parse(Buffer.from(data, 'base64').toString())
  }
}

// Performance Monitoring
class PerformanceMonitor {
  private metrics: Map<number, any> = new Map()
  
  startOperation(entityId: number) {
    this.metrics.set(entityId, {
      startTime: Date.now(),
      operations: 0
    })
  }

  endOperation(entityId: number, type: 'transfer' | 'sync' | 'process') {
    const metric = this.metrics.get(entityId)
    if (metric) {
      const duration = Date.now() - metric.startTime
      metric.operations++
      
      // Update entity performance metrics
      PerformanceMetrics.transferLatency[entityId] = duration
      PerformanceMetrics.syncFrequency[entityId] = metric.operations
    }
  }
}

// Enhanced Cognitive Channel System
class CognitiveChannel {
  private world: any
  private stateQuery: any
  private transferQuery: any
  private performanceMonitor: PerformanceMonitor
  private retryLimit: number = 3
  
  constructor() {
    this.world = createWorld()
    this.stateQuery = defineQuery([CognitiveState, EmotionalState])
    this.transferQuery = defineQuery([CognitiveTransfer])
    this.performanceMonitor = new PerformanceMonitor()
  }

  createCognitiveEntity(config: {
    awareness: number,
    coherence: number,
    complexity: number,
    emotional?: {
      mood?: number,
      stress?: number,
      motivation?: number
    }
  }) {
    const entity = this.world.createEntity()
    
    // Initialize cognitive state
    CognitiveState.awareness[entity] = config.awareness
    CognitiveState.coherence[entity] = config.coherence
    CognitiveState.complexity[entity] = config.complexity
    CognitiveState.cognitiveLoad[entity] = 0
    
    // Initialize emotional state
    if (config.emotional) {
      EmotionalState.mood[entity] = config.emotional.mood ?? 0
      EmotionalState.stress[entity] = config.emotional.stress ?? 0
      EmotionalState.motivation[entity] = config.emotional.motivation ?? 1
    }
    
    return entity
  }

  async synchronize(source: number, target: number, type: 'full' | 'incremental' = 'full') {
    const channelId = this.world.createEntity()
    this.performanceMonitor.startOperation(channelId)
    
    try {
      CognitiveTransfer.sourceId[channelId] = source
      CognitiveTransfer.targetId[channelId] = target
      CognitiveTransfer.syncType[channelId] = type === 'full' ? 0 : 1
      
      // Initialize transfer parameters
      CognitiveTransfer.bandwidth[channelId] = 1.0
      CognitiveTransfer.signal[channelId] = 1.0
      CognitiveTransfer.lastSyncTime[channelId] = Date.now()
      
      return channelId
    } catch (error) {
      console.error('Synchronization failed:', error)
      throw error
    } finally {
      this.performanceMonitor.endOperation(channelId, 'sync')
    }
  }

  async transferState(channelId: number) {
    this.performanceMonitor.startOperation(channelId)
    
    try {
      const sourceId = CognitiveTransfer.sourceId[channelId]
      const targetId = CognitiveTransfer.targetId[channelId]
      const syncType = CognitiveTransfer.syncType[channelId]
      
      // Get complete source state
      const sourceState = {
        cognitive: {
          awareness: CognitiveState.awareness[sourceId],
          coherence: CognitiveState.coherence[sourceId],
          complexity: CognitiveState.complexity[sourceId],
          cognitiveLoad: CognitiveState.cognitiveLoad[sourceId]
        },
        emotional: {
          mood: EmotionalState.mood[sourceId],
          stress: EmotionalState.stress[sourceId],
          motivation: EmotionalState.motivation[sourceId]
        }
      }
      
      // Validate state integrity
      if (!StateIntegrityManager.validateState(sourceState)) {
        throw new Error('Invalid state detected')
      }
      
      // Encrypt state for transfer
      const encryptedState = StateIntegrityManager.encrypt(sourceState)
      
      // Simulate network transfer
      const bandwidth = CognitiveTransfer.bandwidth[channelId]
      const signal = CognitiveTransfer.signal[channelId]
      
      // Apply transfer based on sync type
      if (syncType === 0) { // Full sync
        this.applyFullSync(targetId, sourceState, bandwidth, signal)
      } else { // Incremental sync
        this.applyIncrementalSync(targetId, sourceState, bandwidth, signal)
      }
      
      // Update metrics
      this.updatePerformanceMetrics(channelId)
      
    } catch (error) {
      CognitiveTransfer.errorCount[channelId]++
      if (CognitiveTransfer.errorCount[channelId] < this.retryLimit) {
        await this.retryTransfer(channelId)
      } else {
        throw new Error(`Transfer failed after ${this.retryLimit} attempts`)
      }
    } finally {
      this.performanceMonitor.endOperation(channelId, 'transfer')
    }
  }

  private applyFullSync(targetId: number, state: any, bandwidth: number, signal: number) {
    // Apply cognitive state
    Object.entries(state.cognitive).forEach(([key, value]: [string, any]) => {
      CognitiveState[key][targetId] = value * bandwidth * signal
    })
    
    // Apply emotional state
    Object.entries(state.emotional).forEach(([key, value]: [string, any]) => {
      EmotionalState[key][targetId] = value * bandwidth * signal
    })
  }

  private applyIncrementalSync(targetId: number, state: any, bandwidth: number, signal: number) {
    // Only update values that have changed significantly
    const threshold = 0.1
    
    Object.entries(state.cognitive).forEach(([key, value]: [string, any]) => {
      if (Math.abs(CognitiveState[key][targetId] - value) > threshold) {
        CognitiveState[key][targetId] = value * bandwidth * signal
      }
    })
    
    Object.entries(state.emotional).forEach(([key, value]: [string, any]) => {
      if (Math.abs(EmotionalState[key][targetId] - value) > threshold) {
        EmotionalState[key][targetId] = value * bandwidth * signal
      }
    })
  }

  private async retryTransfer(channelId: number) {
    CognitiveTransfer.retryAttempts[channelId]++
    await new Promise(resolve => setTimeout(resolve, 1000))
    return this.transferState(channelId)
  }

  private updatePerformanceMetrics(channelId: number) {
    const successRate = 1 - (CognitiveTransfer.errorCount[channelId] / (CognitiveTransfer.retryAttempts[channelId] + 1))
    PerformanceMetrics.successRate[channelId] = successRate
    PerformanceMetrics.errorRate[channelId] = 1 - successRate
  }
}

// Daydreams Integration
const cognitiveMemory = memory<{
  channels: Map<string, number>,
  metrics: Map<string, any>
}>({
  key: 'cognitive-memory',
  create() {
    return {
      channels: new Map(),
      metrics: new Map()
    }
  }
})

// Enhanced Actions
const initiateSyncAction = action({
  name: 'initiate-sync',
  schema: z.object({
    sourceId: z.string(),
    targetId: z.string(),
    syncType: z.enum(['full', 'incremental'])
  }),
  memory: cognitiveMemory,
  async handler(call, ctx, agent) {
    const channel = new CognitiveChannel()
    const syncId = await channel.synchronize(
      parseInt(call.data.sourceId),
      parseInt(call.data.targetId),
      call.data.syncType
    )
    
    ctx.data.channels.set(
      `${call.data.sourceId}-${call.data.targetId}`,
      syncId
    )
    
    return `Synchronization initiated with channel ID: ${syncId}`
  }
})

// Cognitive Interface Extension
export const cognitiveInterface = extension({
  name: 'cognitive-interface',
  actions: [initiateSyncAction],
  setup(agent) {
    const channel = new CognitiveChannel()
    agent.container.register('cognitiveChannel', channel)
  }
})

// Enhanced Agent Creation
export function createCognitiveAgent(config: {
  model: any,
  awareness?: number,
  coherence?: number,
  complexity?: number,
  emotional?: {
    mood?: number,
    stress?: number,
    motivation?: number
  }
}) {
  const channel = new CognitiveChannel()
  const cognitiveEntity = channel.createCognitiveEntity({
    awareness: config.awareness ?? 1.0,
    coherence: config.coherence ?? 1.0,
    complexity: config.complexity ?? 1.0,
    emotional: config.emotional
  })
  
  return createDreams({
    model: config.model,
    extensions: [cognitiveInterface],
    context: {
      cognitiveEntity
    }
  })
}
