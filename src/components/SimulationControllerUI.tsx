import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar } from 'recharts';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Play, Pause, Zap, Cog, AlertTriangle, Brain } from 'lucide-react';
import { motion } from 'framer-motion';
import * as THREE from 'three';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';

// WebSocket connection
const ws = new WebSocket('ws://localhost:8080');

// Simulation Metrics Chart
const SimulationMetricsChart = React.memo(({ data }) => (
  <Card className="w-full h-96 bg-gray-900 text-white rounded-lg shadow-lg overflow-hidden">
    <CardHeader>
      <CardTitle className="flex items-center gap-2">
        <Cog className="w-5 h-5 text-blue-400 animate-spin-slow" />
        Simulation Metrics
      </CardTitle>
    </CardHeader>
    <CardContent>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#444" />
          <XAxis dataKey="timestamp" stroke="#ccc" tick={{ fontSize: 12 }} />
          <YAxis stroke="#ccc" tick={{ fontSize: 12 }} domain={[0, 1]} />
          <Tooltip contentStyle={{ backgroundColor: '#333', border: 'none', borderRadius: '4px' }} />
          <Legend verticalAlign="top" height={36} />
          <Line type="monotone" dataKey="successRate" stroke="#52c41a" strokeWidth={2} name="Success Rate" />
          <Line type="monotone" dataKey="errorRate" stroke="#ff4d4f" strokeWidth={2} name="Error Rate" />
          <Line type="monotone" dataKey="complexityIndex" stroke="#1890ff" strokeWidth={2} name="Complexity Index" />
          <Line type="monotone" dataKey="agentSyncRate" stroke="#ffeb3b" strokeWidth={2} name="Sync Rate" />
        </LineChart>
      </ResponsiveContainer>
    </CardContent>
  </Card>
));

// 3D Agent Visualization
const AgentVisualization = React.memo(({ agents }) => {
  const nodes = useMemo(() => agents.map((agent: any, idx: number) => ({
    id: idx,
    x: Math.cos(idx * 0.5) * 50,
    y: Math.sin(idx * 0.5) * 50,
    z: (agent.quantumEntanglement || 0) * 50,
    size: (agent.cognitiveComplexity || 0.5) * 5,
    color: agent.emotionalStress > 0.7 ? '#ff4d4f' : agent.emotionalMotivation > 0.8 ? '#52c41a' : '#8884d8',
    stress: agent.emotionalStress || 0
  })), [agents]);

  const NodeMesh = ({ node }: { node: any }) => {
    const ref = React.useRef<THREE.Mesh>(null!);
    useFrame((state) => {
      if (ref.current) {
        ref.current.rotation.y += 0.01;
        ref.current.scale.setScalar(1 + Math.sin(state.clock.elapsedTime + node.id) * node.stress * 0.2);
      }
    });

    return (
      <mesh ref={ref} position={[node.x, node.y, node.z]}>
        <sphereGeometry args={[node.size, 32, 32]} />
        <meshStandardMaterial color={node.color} emissive={node.color} emissiveIntensity={node.stress * 0.5} />
      </mesh>
    );
  };

  return (
    <Card className="w-full h-96 bg-gray-900 text-white rounded-lg shadow-lg overflow-hidden">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Brain className="w-5 h-5 text-purple-400 animate-pulse" />
          Agent Galaxy
        </CardTitle>
      </CardHeader>
      <CardContent>
        <Canvas camera={{ position: [0, 0, 150], fov: 60 }}>
          <ambientLight intensity={0.5} />
          <pointLight position={[10, 10, 10]} intensity={1} />
          {nodes.map(node => <NodeMesh key={node.id} node={node} />)}
          <OrbitControls />
        </Canvas>
      </CardContent>
    </Card>
  );
});

// Pattern Analysis Bar Chart
const PatternAnalysisChart = React.memo(({ data }) => (
  <Card className="w-full h-96 bg-gray-900 text-white rounded-lg shadow-lg overflow-hidden">
    <CardHeader>
      <CardTitle className="flex items-center gap-2">
        <Zap className="w-5 h-5 text-yellow-400 animate-bounce" />
        Emergent Patterns
      </CardTitle>
    </CardHeader>
    <CardContent>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#444" />
          <XAxis dataKey="type" stroke="#ccc" tick={{ fontSize: 12 }} />
          <YAxis stroke="#ccc" tick={{ fontSize: 12 }} domain={[0, 1]} />
          <Tooltip contentStyle={{ backgroundColor: '#333', border: 'none', borderRadius: '4px' }} />
          <Legend verticalAlign="top" height={36} />
          <Bar dataKey="confidence" fill="#8884d8" barSize={20} name="Confidence" />
        </BarChart>
      </ResponsiveContainer>
    </CardContent>
  </Card>
));

// Animated Alerts
const SimulationAlerts = React.memo(({ alerts }) => (
  <div className="space-y-4">
    {alerts.map((alert, index) => (
      <motion.div
        key={index}
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.5, delay: index * 0.1 }}
      >
        <Alert
          variant={alert.severity === 'error' ? 'destructive' : alert.severity === 'warning' ? 'warning' : 'default'}
          className={`bg-gray-800 text-white border-${alert.severity === 'error' ? 'red' : alert.severity === 'warning' ? 'yellow' : 'gray'}-700 ${alert.severity === 'error' ? 'animate-pulse' : ''}`}
        >
          <AlertTriangle className={`w-4 h-4 ${alert.severity === 'error' ? 'text-red-500' : 'text-yellow-500'}`} />
          <AlertDescription>{alert.message}</AlertDescription>
        </Alert>
      </motion.div>
    ))}
  </div>
));

const SimulationControllerUI = ({ entityId }) => {
  const [metricsData, setMetricsData] = useState([]);
  const [agentData, setAgentData] = useState([]);
  const [patternData, setPatternData] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [isRunning, setIsRunning] = useState(false);
  const [scenarioId, setScenarioId] = useState("");
  const [scenarioList, setScenarioList] = useState<string[]>(["test_1", "test_2"]); // Example scenarios

  useEffect(() => {
    ws.onopen = () => {
      ws.send(JSON.stringify({ type: 'handshake', token: 'valid-token' }));
      ws.send(JSON.stringify({ type: 'subscribe', entityId }));
    };

    ws.onmessage = (event) => {
      const message = JSON.parse(event.data);
      handleWebSocketMessage(message);
    };

    ws.onclose = () => console.log('WebSocket disconnected');
    ws.onerror = (error) => console.error('WebSocket error:', error);

    return () => ws.close();
  }, [entityId]);

  const handleWebSocketMessage = useCallback((message: any) => {
    const timestamp = new Date().toISOString().slice(11, 19);
    switch (message.type) {
      case 'state_update':
        const state = message;
        if (state.step !== undefined) {
          setMetricsData(prev => [...prev, {
            timestamp: `${timestamp} (Step ${state.step})`,
            successRate: state.metrics.successRate || 0,
            errorRate: state.metrics.errorRate || 0,
            complexityIndex: state.metrics.complexityIndex || 0,
            agentSyncRate: state.metrics.agentSyncRate || 0
          }].slice(-20));
          setAgentData(state.events.reduce((acc: any, e: any) => {
            e.participants.forEach((id: string) => {
              const agent = state.patterns.patterns.find((p: any) => p.participants.includes(id)) || {};
              acc[id] = {
                cognitiveComplexity: agent.complexity || Math.random(),
                emotionalStress: agent.stress || Math.random(),
                emotionalMotivation: agent.motivation || Math.random(),
                quantumEntanglement: agent.quantumEntanglement || Math.random()
              };
            });
            return acc;
          }, {}));
          setPatternData(state.patterns.patterns.map((p: any) => ({
            type: p.type,
            confidence: p.confidence
          })));
        }
        break;
      case 'alert':
        setAlerts(prev => [...prev, {
          severity: message.severity,
          message: message.message || 'Simulation alert'
        }].slice(-5));
        break;
      case 'simulation_complete':
        setIsRunning(false);
        setMetricsData(prev => [...prev, {
          timestamp: `${timestamp} (Complete)`,
          successRate: message.result.metrics.successRate,
          errorRate: message.result.metrics.errorRate,
          complexityIndex: message.result.metrics.complexityIndex,
          agentSyncRate: message.result.metrics.agentSyncRate
        }].slice(-20));
        break;
    }
  }, []);

  const startSimulation = () => {
    if (scenarioId && !isRunning) {
      setIsRunning(true);
      ws.send(JSON.stringify({ type: 'run_simulation', scenarioId }));
    }
  };

  const stopSimulation = () => {
    if (isRunning) {
      setIsRunning(false);
      ws.send(JSON.stringify({ type: 'stop_simulation', scenarioId }));
    }
  };

  const handleScenarioChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setScenarioId(e.target.value);
    ws.send(JSON.stringify({ type: 'load_scenario', scenarioId: e.target.value }));
  };

  return (
    <div className="space-y-6 p-6 bg-gray-800 min-h-screen">
      <motion.h1
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 1 }}
        className="text-3xl font-bold text-white flex items-center gap-2"
      >
        <Zap className="w-8 h-8 text-yellow-400 animate-spin" />
        Simulation Controller - Entity {entityId}
      </motion.h1>

      <div className="flex items-center gap-4">
        <select
          value={scenarioId}
          onChange={handleScenarioChange}
          className="px-4 py-2 bg-gray-700 text-white rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
        >
          <option value="">Select Scenario</option>
          {scenarioList.map(id => (
            <option key={id} value={id}>{id}</option>
          ))}
        </select>
        <button
          onClick={startSimulation}
          disabled={isRunning || !scenarioId}
          className={`px-4 py-2 ${isRunning ? 'bg-gray-500' : 'bg-green-500'} text-white rounded hover:${isRunning ? 'bg-gray-600' : 'bg-green-600'} transition duration-200 flex items-center gap-2`}
        >
          <Play className="w-4 h-4" /> Start
        </button>
        <button
          onClick={stopSimulation}
          disabled={!isRunning}
          className={`px-4 py-2 ${!isRunning ? 'bg-gray-500' : 'bg-red-500'} text-white rounded hover:${!isRunning ? 'bg-gray-600' : 'bg-red-600'} transition duration-200 flex items-center gap-2`}
        >
          <Pause className="w-4 h-4" /> Stop
        </button>
      </div>

      <SimulationAlerts alerts={alerts} />

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <SimulationMetricsChart data={metricsData} />
        <AgentVisualization agents={Object.entries(agentData).map(([id, state]) => ({ id, ...state }))} />
        <PatternAnalysisChart data={patternData} />
      </div>
    </div>
  );
};

export default SimulationControllerUI;
