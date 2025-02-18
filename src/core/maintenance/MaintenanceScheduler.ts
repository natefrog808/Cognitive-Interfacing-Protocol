import { defineComponent, Types } from 'bitecs';

// Maintenance schedule component
const MaintenanceSchedule = defineComponent({
  nextMaintenanceTime: Types.ui32,
  estimatedDuration: Types.ui32,
  priority: Types.ui8,
  optimizationScore: Types.f32
});

interface MaintenanceTask {
  id: string;
  duration: number;
  priority: number;
  dependencies: string[];
  resourceRequirements: {
    cpu?: number;
    memory?: number;
    network?: number;
  };
}

interface Schedule {
  tasks: Array<{
    taskId: string;
    startTime: number;
  }>;
  fitness: number;
}

export class MaintenanceScheduler {
  private readonly POPULATION_SIZE = 100;
  private readonly MAX_GENERATIONS = 50;
  private readonly MUTATION_RATE = 0.1;
  private readonly ELITE_SIZE = 10;

  async optimizeSchedule(
    tasks: MaintenanceTask[],
    systemMetrics: any,
    constraints: {
      maxDowntime: number;
      maxConcurrentTasks: number;
      availableTimeSlots: Array<{ start: number; end: number; }>;
    }
  ): Promise<Schedule> {
    // Initialize population
    let population = this.initializePopulation(tasks, constraints);
    
    // Evaluate initial population
    population = await this.evaluatePopulation(population, tasks, systemMetrics, constraints);
    
    // Evolution loop
    for (let generation = 0; generation < this.MAX_GENERATIONS; generation++) {
      // Select parents for next generation
      const parents = this.selectParents(population);
      
      // Create next generation through crossover and mutation
      const offspring = this.createOffspring(parents, tasks, constraints);
      
      // Evaluate new population
      const evaluatedOffspring = await this.evaluatePopulation(
        offspring, 
        tasks, 
        systemMetrics, 
        constraints
      );
      
      // Elitism: Keep best solutions from previous generation
      population = this.elitism(population, evaluatedOffspring);
      
      // Check for convergence
      if (this.hasConverged(population)) break;
    }

    // Return best schedule
    return this.getBestSchedule(population);
  }

  private initializePopulation(
    tasks: MaintenanceTask[], 
    constraints: any
  ): Schedule[] {
    const population: Schedule[] = [];
    
    for (let i = 0; i < this.POPULATION_SIZE; i++) {
      const schedule = this.generateRandomSchedule(tasks, constraints);
      population.push(schedule);
    }
    
    return population;
  }

  private generateRandomSchedule(
    tasks: MaintenanceTask[], 
    constraints: any
  ): Schedule {
    const schedule: Schedule = {
      tasks: [],
      fitness: 0
    };

    // Sort tasks by priority
    const sortedTasks = [...tasks].sort((a, b) => b.priority - a.priority);
    
    // Assign random start times within constraints
    for (const task of sortedTasks) {
      const availableSlots = this.findAvailableTimeSlots(
        schedule.tasks,
        task,
        constraints
      );
      
      if (availableSlots.length > 0) {
        const randomSlot = availableSlots[
          Math.floor(Math.random() * availableSlots.length)
        ];
        
        schedule.tasks.push({
          taskId: task.id,
          startTime: randomSlot.start
        });
      }
    }

    return schedule;
  }

  private async evaluatePopulation(
    population: Schedule[],
    tasks: MaintenanceTask[],
    systemMetrics: any,
    constraints: any
  ): Promise<Schedule[]> {
    return Promise.all(
      population.map(async schedule => ({
        ...schedule,
        fitness: await this.calculateFitness(schedule, tasks, systemMetrics, constraints)
      }))
    );
  }

  private async calculateFitness(
    schedule: Schedule,
    tasks: MaintenanceTask[],
    systemMetrics: any,
    constraints: any
  ): Promise<number> {
    let fitness = 0;
    
    // Evaluate various objectives
    const downtime = this.calculateTotalDowntime(schedule, tasks);
    const resourceUtilization = this.calculateResourceUtilization(schedule, tasks);
    const priorityScore = this.calculatePriorityScore(schedule, tasks);
    const constraintViolations = this.checkConstraintViolations(schedule, constraints);
    
    // Weighted sum of objectives
    fitness = (
      0.4 * (1 - downtime / constraints.maxDowntime) +
      0.3 * resourceUtilization +
      0.2 * priorityScore -
      0.1 * constraintViolations
    );

    return Math.max(0, fitness);
  }

  private selectParents(population: Schedule[]): Schedule[] {
    // Tournament selection
    const parents: Schedule[] = [];
    const tournamentSize = 5;

    while (parents.length < this.POPULATION_SIZE - this.ELITE_SIZE) {
      const tournament = Array(tournamentSize).fill(null).map(() =>
        population[Math.floor(Math.random() * population.length)]
      );
      
      parents.push(
        tournament.reduce((best, current) =>
          current.fitness > best.fitness ? current : best
        )
      );
    }

    return parents;
  }

  private createOffspring(
    parents: Schedule[],
    tasks: MaintenanceTask[],
    constraints: any
  ): Schedule[] {
    const offspring: Schedule[] = [];
    
    while (offspring.length < this.POPULATION_SIZE - this.ELITE_SIZE) {
      // Select two parents
      const parent1 = parents[Math.floor(Math.random() * parents.length)];
      const parent2 = parents[Math.floor(Math.random() * parents.length)];
      
      // Perform crossover
      const child = this.crossover(parent1, parent2);
      
      // Perform mutation
      if (Math.random() < this.MUTATION_RATE) {
        this.mutate(child, tasks, constraints);
      }
      
      offspring.push(child);
    }

    return offspring;
  }

  private crossover(parent1: Schedule, parent2: Schedule): Schedule {
    const childTasks = [];
    const crossoverPoint = Math.floor(Math.random() * parent1.tasks.length);
    
    // Take tasks from both parents
    childTasks.push(...parent1.tasks.slice(0, crossoverPoint));
    childTasks.push(...parent2.tasks.slice(crossoverPoint));

    return {
      tasks: childTasks,
      fitness: 0
    };
  }

  private mutate(schedule: Schedule, tasks: MaintenanceTask[], constraints: any) {
    // Randomly select a task to mutate
    const taskIndex = Math.floor(Math.random() * schedule.tasks.length);
    const task = tasks.find(t => t.id === schedule.tasks[taskIndex].taskId)!;
    
    // Find new available time slot
    const availableSlots = this.findAvailableTimeSlots(
      schedule.tasks.filter((_, i) => i !== taskIndex),
      task,
      constraints
    );
    
    if (availableSlots.length > 0) {
      const newSlot = availableSlots[
        Math.floor(Math.random() * availableSlots.length)
      ];
      
      schedule.tasks[taskIndex].startTime = newSlot.start;
    }
  }

  private elitism(currentPopulation: Schedule[], newPopulation: Schedule[]): Schedule[] {
    // Sort both populations by fitness
    const sortedCurrent = [...currentPopulation].sort((a, b) => b.fitness - a.fitness);
    const sortedNew = [...newPopulation].sort((a, b) => b.fitness - a.fitness);
    
    // Take best solutions from both populations
    return [
      ...sortedCurrent.slice(0, this.ELITE_SIZE),
      ...sortedNew.slice(0, this.POPULATION_SIZE - this.ELITE_SIZE)
    ];
  }

  private hasConverged(population: Schedule[]): boolean {
    // Check if population has converged (best solutions are similar)
    const bestFitness = Math.max(...population.map(p => p.fitness));
    const averageFitness = population.reduce((sum, p) => sum + p.fitness, 0) / population.length;
    
    return Math.abs(bestFitness - averageFitness) < 0.01;
  }

  private getBestSchedule(population: Schedule[]): Schedule {
    return population.reduce((best, current) => 
      current.fitness > best.fitness ? current : best
    );
  }

  private calculateTotalDowntime(schedule: Schedule, tasks: MaintenanceTask[]): number {
    let totalDowntime = 0;
    
    for (const scheduledTask of schedule.tasks) {
      const task = tasks.find(t => t.id === scheduledTask.taskId)!;
      totalDowntime += task.duration;
    }
    
    return totalDowntime;
  }

  private calculateResourceUtilization(schedule: Schedule, tasks: MaintenanceTask[]): number {
    let totalUtilization = 0;
    let maxUtilization = 0;
    
    schedule.tasks.forEach(scheduledTask => {
      const task = tasks.find(t => t.id === scheduledTask.taskId)!;
      
      // Sum up resource requirements
      const taskUtilization = (
        (task.resourceRequirements.cpu ?? 0) +
        (task.resourceRequirements.memory ?? 0) +
        (task.resourceRequirements.network ?? 0)
      ) / 3;
      
      totalUtilization += taskUtilization;
      maxUtilization += 1;
    });
    
    return totalUtilization / maxUtilization;
  }

  private calculatePriorityScore(schedule: Schedule, tasks: MaintenanceTask[]): number {
    let totalPriorityScore = 0;
    let maxPriorityScore = 0;
    
    schedule.tasks.forEach((scheduledTask, index) => {
      const task = tasks.find(t => t.id === scheduledTask.taskId)!;
      
      // Higher priority tasks should be scheduled earlier
      totalPriorityScore += task.priority * (1 - (index / schedule.tasks.length));
      maxPriorityScore += task.priority;
    });
    
    return totalPriorityScore / maxPriorityScore;
  }

  private checkConstraintViolations(schedule: Schedule, constraints: any): number {
    let violations = 0;
    
    // Check time slot violations
    schedule.tasks.forEach(task => {
      const isInValidTimeSlot = constraints.availableTimeSlots.some(slot =>
        task.startTime >= slot.start && task.startTime <= slot.end
      );
      
      if (!isInValidTimeSlot) violations++;
    });
    
    // Check concurrent task violations
    const timePoints = schedule.tasks.flatMap(task => [task.startTime]);
    timePoints.sort((a, b) => a - b);
    
    timePoints.forEach(time => {
      const concurrentTasks = schedule.tasks.filter(task =>
        task.startTime <= time && 
        task.startTime + tasks.find(t => t.id === task.taskId)!.duration > time
      );
      
      if (concurrentTasks.length > constraints.maxConcurrentTasks) {
        violations += concurrentTasks.length - constraints.maxConcurrentTasks;
      }
    });
    
    return violations;
  }

  private findAvailableTimeSlots(
    existingTasks: Array<{ taskId: string; startTime: number; }>,
    task: MaintenanceTask,
    constraints: any
  ): Array<{ start: number; end: number; }> {
    const availableSlots: Array<{ start: number; end: number; }> = [];
    
    constraints.availableTimeSlots.forEach(slot => {
      // Check if slot is large enough for task
      if (slot.end - slot.start < task.duration) return;
      
      // Check for conflicts with existing tasks
      const conflicts = existingTasks.some(existingTask => {
        const existingTaskDuration = tasks.find(
          t => t.id === existingTask.taskId
        )!.duration;
        
        return (
          existingTask.startTime < slot.end &&
          existingTask.startTime + existingTaskDuration > slot.start
        );
      });
      
      if (!conflicts) {
        // Add multiple possible start times within the slot
        for (
          let startTime = slot.start; 
          startTime <= slot.end - task.duration; 
          startTime += 60 // 1-hour increments
        ) {
          availableSlots.push({
            start: startTime,
            end: startTime + task.duration
          });
        }
      }
    });
    
    return availableSlots;
  }
}
