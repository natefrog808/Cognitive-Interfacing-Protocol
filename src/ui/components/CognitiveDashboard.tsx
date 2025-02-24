import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, AreaChart, Area, BarChart, Bar } from 'recharts';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Activity, Brain, Heart, Zap, AlertTriangle } from 'lucide-react';
import { motion } from 'framer-motion';
import * as THREE from 'three';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Sphere } from '@react-three/drei';

// WebSocket connection
const ws = new WebSocket('ws://localhost:8080');

// Emotional State Chart with Stacked Areas
const EmotionalStateChart = React.memo(({ data }) => (
  <Card className="w-full h-96 bg-gray-900 text-white rounded-lg shadow-lg overflow-hidden">
    <CardHeader>
      <CardTitle className="flex items-center gap-2">
        <Heart className="w-5 h-5 text-red-400 animate-pulse" />
        Emotional State Dynamics
      </CardTitle>
    </CardHeader>
    <CardContent>
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#444" />
          <XAxis dataKey="timestamp" stroke="#ccc" tick={{ fontSize: 12 }} />
          <YAxis stroke="#ccc" tick={{ fontSize: 12 }} domain={[0, 1]} />
          <Tooltip contentStyle={{ backgroundColor: '#333', border: 'none', borderRadius: '4px' }} />
          <Legend verticalAlign="top" height={36} />
          <Area type="monotone" dataKey="joy" stackId="1" stroke="#8884d8" fill="#8884d8" fillOpacity={0.6} />
          <Area type="monotone" dataKey="anger" stackId="1" stroke="#ff4d4f" fill="#ff4d4f" fillOpacity={0.6} />
          <Area type="monotone" dataKey="fear" stackId="1" stroke="#722ed1" fill="#722ed1" fillOpacity={0.6} />
          <Area type="monotone" dataKey="disgust" stackId="1" stroke="#52c41a" fill="#52c41a" fillOpacity={0.6} />
        </AreaChart>
      </ResponsiveContainer>
    </CardContent>
  </Card>
));

// Cognitive Metrics Chart with Predictions
const CognitiveMetricsChart = React.memo(({ data, predictions }) => (
  <Card className="w-full h-96 bg-gray-900 text-white rounded-lg shadow-lg overflow-hidden">
    <CardHeader>
      <CardTitle className="flex items-center gap-2">
        <Brain className="w-5 h-5 text-blue-400 animate-spin-slow" />
        Cognitive Metrics & Forecasts
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
          <Line type="monotone" dataKey="awareness" stroke="#1890ff" strokeWidth={2} dot={false} />
          <Line type="monotone" dataKey="coherence" stroke="#faad14" strokeWidth={2} dot={false} />
          <Line type="monotone" dataKey="complexity" stroke="#13c2c2" strokeWidth={2} dot={false} />
          {predictions.map((p: any[], i: number) => (
            <Line
              key={i}
              type="monotone"
              dataKey="awareness"
              data={p}
              stroke="#1890ff"
              strokeDasharray="5 5"
              strokeWidth={1}
              dot={false}
              opacity={0.5}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </CardContent>
  </Card>
));

// Performance Metrics Chart with Gradient Fill
const PerformanceMetrics = React.memo(({ data }) => (
  <Card className="w-full h-96 bg-gray-900 text-white rounded-lg shadow-lg overflow-hidden">
    <CardHeader>
      <CardTitle className="flex items-center gap-2">
        <Activity className="w-5 h-5 text-green-400 animate-bounce" />
        System Performance Pulse
      </CardTitle>
    </CardHeader>
    <CardContent>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data}>
          <defs>
            <linearGradient id="latencyGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#eb2f96" stopOpacity={0.8} />
              <stop offset="95%" stopColor="#eb2f96" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#444" />
          <XAxis dataKey="timestamp" stroke="#ccc" tick={{ fontSize: 12 }} />
          <YAxis stroke="#ccc" tick={{ fontSize: 12 }} domain={[0, 'auto']} />
          <Tooltip contentStyle={{ backgroundColor: '#333', border: 'none', borderRadius: '4px' }} />
          <Legend verticalAlign="top" height={36} />
          <Line type="monotone" dataKey="transferLatency" stroke="#eb2f96" fill="url(#latencyGradient)" strokeWidth={2} dot={false} />
          <Line type="monotone" dataKey="successRate" stroke="#52c41a" strokeWidth={2} dot={false} />
          <Line type="monotone" dataKey="errorRate" stroke="#f5222d" strokeWidth={2} dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </CardContent>
  </Card>
));

// Quantum Metrics Chart with Bar Visualization
const QuantumMetricsChart = React.memo(({ data }) => (
  <Card className="w-full h-96 bg-gray-900 text-white rounded-lg shadow-lg overflow-hidden">
    <CardHeader>
      <CardTitle className="flex items-center gap-2">
        <Zap className="w-5 h-5 text-yellow-400 animate-pulse" />
        Quantum State Insights
      </CardTitle>
    </CardHeader>
    <CardContent>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#444" />
          <XAxis dataKey="timestamp" stroke="#ccc" tick={{ fontSize: 12 }} />
          <YAxis stroke="#ccc" tick={{ fontSize: 12 }} domain={[0, 1]} />
          <Tooltip contentStyle={{ backgroundColor: '#333', border: 'none', borderRadius: '4px' }} />
          <Legend verticalAlign="top" height={36} />
          <Bar dataKey="entanglementScore" fill="#ffeb3b" barSize={20} />
          <Bar dataKey="coherenceMetric" fill="#40c4ff" barSize={20} />
        </BarChart>
      </ResponsiveContainer>
    </CardContent>
  </Card>
));

// Animated State Alerts
const StateAlerts = React.memo(({ alerts }) => (
  <div className="space-y-4">
    {alerts.map((alert, index) => (
      <motion.div
        key={index}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: index * 0.1 }}
      >
        <Alert
          variant={alert.severity === 'high' ? 'destructive' : alert.severity === 'medium' ? 'warning' : 'default'}
          className={`bg-gray-800 text-white border-gray-700 ${alert.severity === 'high' ? 'animate-pulse' : ''}`}
        >
          <AlertTriangle className={`w-4 h-4 ${alert.severity === 'high' ? 'text-red-500' : 'text-yellow-500'}`} />
          <AlertDescription>
            {alert.message} (Confidence: {(alert.confidence * 100).toFixed(1)}%)
          </AlertDescription>
        </Alert>
      </motion.div>
    ))}
  </div>
));

// 3D Topology Visualization with Animation
const TopologyVisualization = ({ topologyData }) => {
  const nodes = useMemo(() => topologyData.nodes.map((node: any, idx: number) => ({
    id: idx,
    x: node.spatialVector ? node.spatialVector[0] * 100 : Math.cos(idx * 0.5) * 100,
    y: node.spatialVector ? node.spatialVector[1] * 100 : Math.sin(idx * 0.5) * 100,
    z: node.spatialVector ? node.spatialVector[2] * 100 : node.centrality * 50,
    size: (node.influence || Math.random()) * 10,
    color: node.quantumEntanglement ? `hsl(${node.quantumEntanglement * 360}, 70%, 50%)` : `hsl(${node.stability * 360}, 70%, 50%)`,
    entanglementFlow: node.entanglementFlow || 0
  })), [topologyData.nodes]);

  const edges = useMemo(() => topologyData.edges.map((edge: any) => ({
    source: edge[0],
    target: edge[1]
  })), [topologyData.edges]);

  const NodeMesh = ({ node }: { node: any }) => {
    const ref = React.useRef<THREE.Mesh>(null!);
    useFrame((state) => {
      if (ref.current) {
        ref.current.rotation.y += 0.01;
        ref.current.scale.setScalar(1 + Math.sin(state.clock.elapsedTime + node.id) * node.entanglementFlow * 0.2);
      }
    });

    return (
      <mesh ref={ref} position={[node.x, node.y, node.z]}>
        <sphereGeometry args={[node.size, 32, 32]} />
        <meshStandardMaterial color={node.color} emissive={node.color} emissiveIntensity={node.entanglementFlow * 0.5} />
      </mesh>
    );
  };

  return (
    <Card className="w-full h-96 bg-gray-900 text-white rounded-lg shadow-lg overflow-hidden">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Zap className="w-5 h-5 text-yellow-400 animate-spin" />
          Quantum Topology Network
        </CardTitle>
      </CardHeader>
      <CardContent>
        <Canvas camera={{ position: [0, 0, 200], fov: 60 }}>
          <ambientLight intensity={0.5} />
          <pointLight position={[10, 10, 10]} intensity={1} />
          {nodes.map(node => <NodeMesh key={node.id} node={node} />)}
          {edges.map((edge, idx) => (
            <line key={idx}>
              <bufferGeometry>
                <bufferAttribute
                  attach="attributes-position"
                  array={new Float32Array([
                    nodes[edge.source].x, nodes[edge.source].y, nodes[edge.source].z,
                    nodes[edge.target].x, nodes[edge.target].y, nodes[edge.target].z
                  ])}
                  itemSize={3}
                  count={2}
                />
              </bufferGeometry>
              <lineBasicMaterial color="#888" linewidth={2} transparent opacity={0.5 + nodes[edge.source].entanglementFlow * 0.5} />
            </line>
          ))}
          <OrbitControls enableZoom={true} enablePan={true} />
        </Canvas>
      </CardContent>
    </Card>
  );
};

const CognitiveDashboard = ({ entityId }) => {
  const [emotionalData, setEmotionalData] = useState([]);
  const [cognitiveData, setCognitiveData] = useState([]);
  const [performanceData, setPerformanceData] = useState([]);
  const [quantumData, setQuantumData] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [topologyData, setTopologyData] = useState({ nodes: [], edges: [] });
  const [predictions, setPredictions] = useState([]);

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
        const state = message.data; // Assuming proper decryption elsewhere
        setEmotionalData(prev => [...prev, {
          timestamp,
          joy: state.emotional?.joy || 0,
          anger: state.emotional?.anger || 0,
          fear: state.emotional?.fear || 0,
          disgust: state.emotional?.disgust || 0
        }].slice(-20));
        setCognitiveData(prev => [...prev, {
          timestamp,
          awareness: state.cognitive?.awareness || 0,
          coherence: state.cognitive?.coherence || 0,
          complexity: state.cognitive?.complexity || 0
        }].slice(-20));
        setPerformanceData(prev => [...prev, {
          timestamp,
          transferLatency: state.mlPredictions?.transferLatency || state.transferLatency || 0,
          successRate: state.mlPredictions?.successRate || state.successRate || 0,
          errorRate: state.mlPredictions?.errorRate || state.errorRate || 0
        }].slice(-20));
        setQuantumData(prev => [...prev, {
          timestamp,
          entanglementScore: state.quantumState?.entanglementScore || 0,
          coherenceMetric: state.quantumState?.coherenceMetric || 0
        }].slice(-20));
        break;
      case 'alert':
        const alertData = message.alert.alerts || [message.alert];
        setAlerts(prev => [...prev, ...alertData.map((a: any) => ({
          severity: a.severity || (a.alert?.severity > 1 ? (a.alert.severity === 3 ? 'high' : 'medium') : 'low'),
          message: a.recommendation || a.message || a.alert?.message,
          confidence: a.confidence || a.alert?.confidence || 0
        }))].slice(-5));
        break;
      case 'topology_update':
        setTopologyData({
          nodes: message.adaptations.flatMap((a: any) => a.nodes.map((n: any) => ({
            spatialVector: n.spatialVector || [Math.random(), Math.random(), Math.random()],
            influence: n.influence || Math.random(),
            centrality: n.centrality || Math.random(),
            stability: n.stability || Math.random(),
            quantumEntanglement: n.quantumEntanglement || Math.random(),
            entanglementFlow: n.entanglementFlow || 0
          }))),
          edges: message.adaptations.reduce((acc: any[], a: any) => {
            for (let i = 0; i < a.nodes.length; i++) {
              for (let j = i + 1; j < a.nodes.length; j++) {
                acc.push([i, j]);
              }
            }
            return acc;
          }, [])
        });
        break;
      case 'mlPredictions':
        setPredictions(message.mlPredictions.predictions.map((p: number[], i: number) => ({
          timestamp: `${timestamp}+${i+1}`,
          awareness: p[0] // Assuming awareness is first
        })));
        break;
      case 'topology_forecast':
        setPredictions(message.predictions.map((p: any, i: number) => ({
          timestamp: `${timestamp}+${i+1}`,
          awareness: p.globalEfficiency // Mapping to cognitive for demo
        })));
        break;
      case 'scheduler':
        setPerformanceData(prev => [...prev, {
          timestamp,
          transferLatency: message.resourceProfile.network.reduce((sum: number, val: number) => sum + val, 0) / message.resourceProfile.network.length,
          successRate: message.fitness,
          errorRate: 1 - message.quantumFitness
        }].slice(-20));
        break;
    }
  }, []);

  return (
    <div className="space-y-6 p-6 bg-gray-800 min-h-screen">
      <motion.h1
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 1 }}
        className="text-3xl font-bold text-white flex items-center gap-2"
      >
        <Brain className="w-8 h-8 text-blue-400 animate-pulse" />
        CogniVerse Dashboard - Entity {entityId}
      </motion.h1>

      <StateAlerts alerts={alerts} />

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <EmotionalStateChart data={emotionalData} />
        <CognitiveMetricsChart data={cognitiveData} predictions={predictions} />
        <PerformanceMetrics data={performanceData} />
        <QuantumMetricsChart data={quantumData} />
        <TopologyVisualization topologyData={topologyData} />
      </div>
    </div>
  );
};

export default CognitiveDashboard;
