import { defineComponent, Types } from 'bitecs';
import { mean, std } from 'mathjs';
import { PredictiveMonitor } from './PredictiveMonitor'; // Assuming this exists
import { CognitiveWebSocketServer } from './CognitiveWebSocketServer';

// Enhanced Maintenance Schedule Component
const MaintenanceSchedule = defineComponent({
  nextMaintenanceTime: Types.ui32, // Timestamp in ms
  estimatedDuration: Types.ui32,   // Duration in ms
  priority: Types.ui8,             // 0-255
  optimizationScore: Types.f32,    // 0-1
  resourceImpact: Types.f32,       // Estimated resource load (0-1)
  status: Types.ui8               // 0 = pending, 1 = scheduled, 2 = running, 3 = completed
});

interface MaintenanceTask {
  id: string;
  duration: number; // ms
  priority: number; // 0-255
  dependencies: string[]; // Task IDs that must complete first
  resourceRequirements: {
    cpu?: number;    // 0-1
    memory?: number; // 0-1
    network?: number;// 0-1
  };
}

interface Schedule {
  tasks: Array<{ taskId: string; startTime: number }>;
  fitness: number;
  resourceProfile: { cpu: number[]; memory: number[]; network: number[] }; // Resource usage over time
}

export class MaintenanceScheduler {
  private readonly POPULATION_SIZE = 100;
  private readonly MAX_GENERATIONS = 50;
  private readonly MUTATION_RATE = 0.15; // Increased for diversity
  private readonly ELITE_SIZE = 10;
  private readonly MIN_FITNESS_THRESHOLD = 0.9; // Early stopping if good enough
  private monitor: PredictiveMonitor;
  private wsServer: CognitiveWebSocketServer;

  constructor(wsPort: number = 8080) {
    this.monitor = new PredictiveMonitor();
    this.wsServer = new CognitiveWebSocketServer(wsPort);
  }

  async optimizeSchedule(
    tasks: MaintenanceTask[],
    systemMetrics: { cpuUsage: number; memoryUsage: number; networkLatency: number },
    constraints: {
      maxDowntime: number; // ms
      maxConcurrentTasks: number;
      availableTimeSlots: Array<{ start: number; end: number }>;
    }
  ): Promise<Schedule> {
    // Validate dependencies
    if (!this.validateDependencies(tasks)) {
      throw new Error('Invalid task dependencies detected');
    }

    let population = this.initializePopulation(tasks, constraints);
    population = await this.evaluatePopulation(population, tasks, systemMetrics, constraints);

    for (let generation = 0; generation < this.MAX_GENERATIONS; generation++) {
      const parents = this.selectParents(population);
      const offspring = this.createOffspring(parents, tasks, constraints);
      const evaluatedOffspring = await this.evaluatePopulation(offspring, tasks, systemMetrics, constraints);
      population = this.elitism(population, evaluatedOffspring);

      const bestFitness = Math.max(...population.map(p => p.fitness));
      console.log(`Generation ${generation}: Best Fitness = ${bestFitness}`);
      if (bestFitness >= this.MIN_FITNESS_THRESHOLD) break; // Early stopping
      if (this.hasConverged(population)) break;
    }

    const bestSchedule = this.getBestSchedule(population);
    this.updateBitECS(bestSchedule, tasks);
    this.broadcastSchedule(bestSchedule, tasks);
    return bestSchedule;
  }

  private initializePopulation(tasks: MaintenanceTask[], constraints: any): Schedule[] {
    const population: Schedule[] = [];
    for (let i = 0; i < this.POPULATION_SIZE; i++) {
      const schedule = this.generateSmartSchedule(tasks, constraints);
      population.push(schedule);
    }
    return population;
  }

  private generateSmartSchedule(tasks: MaintenanceTask[], constraints: any): Schedule {
    const schedule: Schedule = {
      tasks: [],
      fitness: 0,
      resourceProfile: { cpu: [], memory: [], network: [] }
    };

    const taskMap = new Map(tasks.map(t => [t.id, t]));
    const scheduled = new Set<string>();
    const queue = [...tasks].sort((a, b) => b.priority - a.priority);

    while (queue.length > 0) {
      const task = queue.shift()!;
      if (task.dependencies.every(dep => scheduled.has(dep))) {
        const availableSlots = this.findAvailableTimeSlots(schedule.tasks, task, constraints);
        if (availableSlots.length > 0) {
          // Prioritize slots with lower resource contention
          const bestSlot = availableSlots.reduce((best, slot) => {
            const overlap = this.calculateResourceOverlap(schedule, task, slot.start, tasks);
            return overlap < best.overlap ? { slot, overlap } : best;
          }, { slot: availableSlots[0], overlap: Infinity }).slot;

          schedule.tasks.push({ taskId: task.id, startTime: bestSlot.start });
          scheduled.add(task.id);
        }
      } else {
        queue.push(task); // Re-queue if dependencies unmet
      }
    }

    this.updateResourceProfile(schedule, tasks);
    return schedule;
  }

  private async evaluatePopulation(
    population: Schedule[],
    tasks: MaintenanceTask[],
    systemMetrics: any,
    constraints: any
  ): Promise<Schedule[]> {
    return Promise.all(population.map(async schedule => ({
      ...schedule,
      fitness: await this.calculateFitness(schedule, tasks, systemMetrics, constraints)
    })));
  }

  private async calculateFitness(
    schedule: Schedule,
    tasks: MaintenanceTask[],
    systemMetrics: any,
    constraints: any
  ): Promise<number> {
    const taskMap = new Map(tasks.map(t => [t.id, t]));
    const downtime = this.calculateTotalDowntime(schedule, tasks);
    const resourceUtilization = this.calculateResourceUtilization(schedule, tasks, systemMetrics);
    const priorityScore = this.calculatePriorityScore(schedule, tasks);
    const dependencyScore = this.calculateDependencyScore(schedule, taskMap);
    const constraintViolations = this.checkConstraintViolations(schedule, tasks, constraints);

    // Dynamic weights based on system state
    const systemLoad = (systemMetrics.cpuUsage + systemMetrics.memoryUsage) / 2;
    const weights = {
      downtime: 0.3 + systemLoad * 0.1,      // More weight if system is loaded
      resource: 0.3 - systemLoad * 0.05,     // Less if loaded
      priority: 0.25,
      dependency: 0.15,
      violations: -0.2
    };

    const fitness = (
      weights.downtime * (1 - downtime / constraints.maxDowntime) +
      weights.resource * resourceUtilization +
      weights.priority * priorityScore +
      weights.dependency * dependencyScore +
      weights.violations * constraintViolations
    );

    return Math.max(0, Math.min(1, fitness)); // Clamp to 0-1
  }

  private selectParents(population: Schedule[]): Schedule[] {
    const parents: Schedule[] = [];
    const tournamentSize = 5;
    while (parents.length < this.POPULATION_SIZE - this.ELITE_SIZE) {
      const tournament = Array(tournamentSize).fill(null).map(() =>
        population[Math.floor(Math.random() * population.length)]
      );
      parents.push(tournament.reduce((best, current) => 
        current.fitness > best.fitness ? current : best
      ));
    }
    return parents;
  }

  private createOffspring(parents: Schedule[], tasks: MaintenanceTask[], constraints: any): Schedule[] {
    const offspring: Schedule[] = [];
    const taskMap = new Map(tasks.map(t => [t.id, t]));

    while (offspring.length < this.POPULATION_SIZE - this.ELITE_SIZE) {
      const parent1 = parents[Math.floor(Math.random() * parents.length)];
      const parent2 = parents[Math.floor(Math.random() * parents.length)];
      const child = this.dependencyAwareCrossover(parent1, parent2, taskMap);
      if (Math.random() < this.MUTATION_RATE) {
        this.smartMutate(child, tasks, constraints);
      }
      this.updateResourceProfile(child, tasks);
      offspring.push(child);
    }
    return offspring;
  }

  private dependencyAwareCrossover(parent1: Schedule, parent2: Schedule, taskMap: Map<string, MaintenanceTask>): Schedule {
    const childTasks: { taskId: string; startTime: number }[] = [];
    const crossoverPoint = Math.floor(Math.random() * parent1.tasks.length);
    const scheduled = new Set<string>();

    // Take from parent1 up to crossover, respecting dependencies
    for (const task of parent1.tasks.slice(0, crossoverPoint)) {
      const taskData = taskMap.get(task.taskId)!;
      if (taskData.dependencies.every(dep => scheduled.has(dep))) {
        childTasks.push({ ...task });
        scheduled.add(task.taskId);
      }
    }

    // Fill from parent2, ensuring no duplicates and dependency order
    for (const task of parent2.tasks) {
      if (!scheduled.has(task.taskId) && taskMap.get(task.taskId)!.dependencies.every(dep => scheduled.has(dep))) {
        childTasks.push({ ...task });
        scheduled.add(task.taskId);
      }
    }

    return { tasks: childTasks, fitness: 0, resourceProfile: { cpu: [], memory: [], network: [] } };
  }

  private smartMutate(schedule: Schedule, tasks: MaintenanceTask[], constraints: any) {
    const taskIndex = Math.floor(Math.random() * schedule.tasks.length);
    const task = tasks.find(t => t.id === schedule.tasks[taskIndex].taskId)!;
    const availableSlots = this.findAvailableTimeSlots(
      schedule.tasks.filter((_, i) => i !== taskIndex),
      task,
      constraints
    );

    if (availableSlots.length > 0) {
      // Choose slot with lowest resource contention
      const bestSlot = availableSlots.reduce((best, slot) => {
        const overlap = this.calculateResourceOverlap(schedule, task, slot.start, tasks);
        return overlap < best.overlap ? { slot, overlap } : best;
      }, { slot: availableSlots[0], overlap: Infinity }).slot;
      schedule.tasks[taskIndex].startTime = bestSlot.start;
    }
  }

  private elitism(currentPopulation: Schedule[], newPopulation: Schedule[]): Schedule[] {
    const sortedCurrent = [...currentPopulation].sort((a, b) => b.fitness - a.fitness);
    const sortedNew = [...newPopulation].sort((a, b) => b.fitness - a.fitness);
    return [...sortedCurrent.slice(0, this.ELITE_SIZE), ...sortedNew.slice(0, this.POPULATION_SIZE - this.ELITE_SIZE)];
  }

  private hasConverged(population: Schedule[]): boolean {
    const bestFitness = Math.max(...population.map(p => p.fitness));
    const avgFitness = mean(population.map(p => p.fitness));
    const stdFitness = std(population.map(p => p.fitness));
    return Math.abs(bestFitness - avgFitness) < 0.01 && stdFitness < 0.05; // Tight convergence
  }

  private getBestSchedule(population: Schedule[]): Schedule {
    return population.reduce((best, current) => current.fitness > best.fitness ? current : best);
  }

  private calculateTotalDowntime(schedule: Schedule, tasks: MaintenanceTask[]): number {
    const taskMap = new Map(tasks.map(t => [t.id, t]));
    let maxEndTime = 0;
    schedule.tasks.forEach(({ taskId, startTime }) => {
      const task = taskMap.get(taskId)!;
      const endTime = startTime + task.duration;
      maxEndTime = Math.max(maxEndTime, endTime);
    });
    return maxEndTime - Math.min(...schedule.tasks.map(t => t.startTime));
  }

  private calculateResourceUtilization(schedule: Schedule, tasks: MaintenanceTask[], systemMetrics: any): number {
    const taskMap = new Map(tasks.map(t => [t.id, t]));
    const timePoints = schedule.resourceProfile.cpu.map((_, i) => i * 1000); // 1-second granularity
    let totalUtil = 0;
    let maxUtil = 0;

    timePoints.forEach(time => {
      const activeTasks = schedule.tasks.filter(t => 
        t.startTime <= time && t.startTime + taskMap.get(t.taskId)!.duration > time
      );
      const cpu = activeTasks.reduce((sum, t) => sum + (taskMap.get(t.taskId)!.resourceRequirements.cpu || 0), 0);
      const mem = activeTasks.reduce((sum, t) => sum + (taskMap.get(t.taskId)!.resourceRequirements.memory || 0), 0);
      const net = activeTasks.reduce((sum, t) => sum + (taskMap.get(t.taskId)!.resourceRequirements.network || 0), 0);
      const util = (cpu + mem + net) / 3;
      totalUtil += Math.min(1, util); // Cap at 100%
      maxUtil += 1;
    });

    const systemFactor = 1 - (systemMetrics.cpuUsage + systemMetrics.memoryUsage) / 2; // Favor lower utilization if system is loaded
    return totalUtil / maxUtil * systemFactor;
  }

  private calculatePriorityScore(schedule: Schedule, tasks: MaintenanceTask[]): number {
    const taskMap = new Map(tasks.map(t => [t.id, t]));
    let totalScore = 0;
    let maxScore = tasks.reduce((sum, t) => sum + t.priority, 0);
    schedule.tasks.forEach((task, idx) => {
      const priority = taskMap.get(task.taskId)!.priority;
      totalScore += priority * (1 - idx / schedule.tasks.length); // Earlier = better
    });
    return totalScore / (maxScore || 1);
  }

  private calculateDependencyScore(schedule: Schedule, taskMap: Map<string, MaintenanceTask>): number {
    const scheduledTimes = new Map(schedule.tasks.map(t => [t.taskId, t.startTime]));
    let score = 0;
    let totalDeps = 0;

    schedule.tasks.forEach(task => {
      const taskData = taskMap.get(task.taskId)!;
      taskData.dependencies.forEach(dep => {
        totalDeps++;
        const depTime = scheduledTimes.get(dep);
        if (depTime && depTime + taskMap.get(dep)!.duration <= task.startTime) {
          score += 1; // Dependency satisfied
        }
      });
    });

    return totalDeps > 0 ? score / totalDeps : 1; // Perfect if no dependencies
  }

  private checkConstraintViolations(schedule: Schedule, tasks: MaintenanceTask[], constraints: any): number {
    const taskMap = new Map(tasks.map(t => [t.id, t]));
    let violations = 0;

    // Time slot violations
    schedule.tasks.forEach(task => {
      const taskData = taskMap.get(task.taskId)!;
      const inSlot = constraints.availableTimeSlots.some(slot => 
        task.startTime >= slot.start && task.startTime + taskData.duration <= slot.end
      );
      if (!inSlot) violations += 2; // Higher penalty for slot violations
    });

    // Concurrent tasks
    const timePoints = schedule.tasks.flatMap(t => [t.startTime, t.startTime + taskMap.get(t.taskId)!.duration]);
    timePoints.sort((a, b) => a - b);
    timePoints.forEach(time => {
      const concurrent = schedule.tasks.filter(t => 
        t.startTime <= time && t.startTime + taskMap.get(t.taskId)!.duration > time
      );
      if (concurrent.length > constraints.maxConcurrentTasks) {
        violations += concurrent.length - constraints.maxConcurrentTasks;
      }
    });

    return violations;
  }

  private findAvailableTimeSlots(
    existingTasks: Array<{ taskId: string; startTime: number }>,
    task: MaintenanceTask,
    constraints: any
  ): Array<{ start: number; end: number }> {
    const availableSlots: Array<{ start: number; end: number }> = [];
    const taskMap = new Map(existingTasks.map(t => [t.taskId, t]));

    constraints.availableTimeSlots.forEach(slot => {
      if (slot.end - slot.start < task.duration) return;

      const conflicts = existingTasks.some(t => {
        const existing = taskMap.get(t.taskId)!;
        return t.startTime < slot.end && t.startTime + taskMap.get(t.taskId)!.duration > slot.start;
      });

      if (!conflicts && task.dependencies.every(dep => {
        const depTask = existingTasks.find(t => t.taskId === dep);
        return depTask && depTask.startTime + taskMap.get(dep)!.duration <= slot.start;
      })) {
        for (let start = slot.start; start <= slot.end - task.duration; start += 300000) { // 5-min increments
          availableSlots.push({ start, end: start + task.duration });
        }
      }
    });

    return availableSlots;
  }

  private calculateResourceOverlap(schedule: Schedule, task: MaintenanceTask, startTime: number, tasks: MaintenanceTask[]): number {
    const taskMap = new Map(tasks.map(t => [t.id, t]));
    const overlapping = schedule.tasks.filter(t => {
      const tData = taskMap.get(t.taskId)!;
      return t.startTime < startTime + task.duration && t.startTime + tData.duration > startTime;
    });
    return overlapping.reduce((sum, t) => {
      const tData = taskMap.get(t.taskId)!;
      return sum + (tData.resourceRequirements.cpu || 0) + (tData.resourceRequirements.memory || 0) + (tData.resourceRequirements.network || 0);
    }, (task.resourceRequirements.cpu || 0) + (task.resourceRequirements.memory || 0) + (task.resourceRequirements.network || 0)) / 3;
  }

  private updateResourceProfile(schedule: Schedule, tasks: MaintenanceTask[]) {
    const taskMap = new Map(tasks.map(t => [t.id, t]));
    const maxTime = Math.max(...schedule.tasks.map(t => t.startTime + taskMap.get(t.taskId)!.duration));
    const timePoints = Math.ceil(maxTime / 1000); // 1-second granularity
    schedule.resourceProfile = {
      cpu: new Array(timePoints).fill(0),
      memory: new Array(timePoints).fill(0),
      network: new Array(timePoints).fill(0)
    };

    schedule.tasks.forEach(task => {
      const tData = taskMap.get(task.taskId)!;
      const startIdx = Math.floor(task.startTime / 1000);
      const endIdx = Math.floor((task.startTime + tData.duration) / 1000);
      for (let i = startIdx; i < endIdx && i < timePoints; i++) {
        schedule.resourceProfile.cpu[i] += tData.resourceRequirements.cpu || 0;
        schedule.resourceProfile.memory[i] += tData.resourceRequirements.memory || 0;
        schedule.resourceProfile.network[i] += tData.resourceRequirements.network || 0;
      }
    });
  }

  private validateDependencies(tasks: MaintenanceTask[]): boolean {
    const taskMap = new Map(tasks.map(t => [t.id, t]));
    const visited = new Set<string>();
    const recStack = new Set<string>();

    const dfs = (taskId: string): boolean => {
      if (recStack.has(taskId)) return false; // Cycle detected
      if (visited.has(taskId)) return true;
      visited.add(taskId);
      recStack.add(taskId);
      const task = taskMap.get(taskId);
      if (!task) return false;
      const valid = task.dependencies.every(dep => dfs(dep));
      recStack.delete(taskId);
      return valid;
    };

    return tasks.every(task => dfs(task.id));
  }

  private updateBitECS(schedule: Schedule, tasks: MaintenanceTask[]) {
    const taskMap = new Map(tasks.map(t => [t.id, t]));
    schedule.tasks.forEach(({ taskId, startTime }) => {
      const task = taskMap.get(taskId)!;
      const eid = parseInt(taskId.split('_')[1] || taskId); // Assuming ID format like "task_1"
      MaintenanceSchedule.nextMaintenanceTime[eid] = startTime;
      MaintenanceSchedule.estimatedDuration[eid] = task.duration;
      MaintenanceSchedule.priority[eid] = task.priority;
      MaintenanceSchedule.optimizationScore[eid] = schedule.fitness;
      MaintenanceSchedule.resourceImpact[eid] = (task.resourceRequirements.cpu || 0) + (task.resourceRequirements.memory || 0) + (task.resourceRequirements.network || 0) / 3;
      MaintenanceSchedule.status[eid] = 1; // Scheduled
    });
  }

  private broadcastSchedule(schedule: Schedule, tasks: MaintenanceTask[]) {
    this.wsServer.broadcastStateUpdate("scheduler", {
      schedule: schedule.tasks,
      fitness: schedule.fitness,
      resourceProfile: schedule.resourceProfile,
      tasks: tasks.map(t => ({ id: t.id, priority: t.priority, duration: t.duration }))
    });
  }

  // New: Visualize schedule as a simple timeline
  visualizeSchedule(schedule: Schedule, tasks: MaintenanceTask[]): string {
    const taskMap = new Map(tasks.map(t => [t.id, t]));
    const maxTime = Math.max(...schedule.tasks.map(t => t.startTime + taskMap.get(t.taskId)!.duration));
    const timeline: string[] = [];
    for (let t = 0; t < maxTime; t += 3600000) { // 1-hour steps
      const active = schedule.tasks.filter(task => 
        task.startTime <= t && task.startTime + taskMap.get(task.taskId)!.duration > t
      );
      timeline.push(`${new Date(t).toISOString().slice(11, 16)}: ${active.map(task => task.taskId).join(', ') || 'None'}`);
    }
    return timeline.join('\n');
  }
}
