import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, AreaChart, Area } from 'recharts';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Activity, Brain, Heart, Zap } from 'lucide-react';
import { motion } from 'framer-motion'; // For animations
import * as THREE from 'three'; // For 3D topology
import { Canvas } from '@react-three/fiber'; // React Three Fiber for 3D rendering
import { OrbitControls } from '@react-three/drei'; // Camera controls

// WebSocket connection
const ws = new WebSocket('ws://localhost:8080');

// Emotional State Chart with Stacked Areas
const EmotionalStateChart = React.memo(({ data }) => (
  <Card className="w-full h-96 bg-gray-900 text-white rounded-lg shadow-lg">
    <CardHeader>
      <CardTitle className="flex items-center gap-2">
        <Heart className="w-5 h-5 text-red-400" />
        Emotional State Trends
      </CardTitle>
    </CardHeader>
    <CardContent>
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#444" />
          <XAxis dataKey="timestamp" stroke="#ccc" />
          <YAxis stroke="#ccc" />
          <Tooltip contentStyle={{ backgroundColor: '#333', border: 'none' }} />
          <Legend />
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
  <Card className="w-full h-96 bg-gray-900 text-white rounded-lg shadow-lg">
    <CardHeader>
      <CardTitle className="flex items-center gap-2">
        <Brain className="w-5 h-5 text-blue-400" />
        Cognitive Metrics & Predictions
      </CardTitle>
    </CardHeader>
    <CardContent>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#444" />
          <XAxis dataKey="timestamp" stroke="#ccc" />
          <YAxis stroke="#ccc" />
          <Tooltip contentStyle={{ backgroundColor: '#333', border: 'none' }} />
          <Legend />
          <Line type="monotone" dataKey="awareness" stroke="#1890ff" />
          <Line type="monotone" dataKey="coherence" stroke="#faad14" />
          <Line type="monotone" dataKey="complexity" stroke="#13c2c2" />
          {predictions && predictions.map((p, i) => (
            <Line key={i} type="monotone" dataKey={`pred_awareness_${i}`} stroke="#1890ff" strokeDasharray="5 5" data={p} />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </CardContent>
  </Card>
));

// Performance Metrics Chart with Gradient Fill
const PerformanceMetrics = React.memo(({ data }) => (
  <Card className="w-full h-96 bg-gray-900 text-white rounded-lg shadow-lg">
    <CardHeader>
      <CardTitle className="flex items-center gap-2">
        <Activity className="w-5 h-5 text-green-400" />
        System Performance
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
          <XAxis dataKey="timestamp" stroke="#ccc" />
          <YAxis stroke="#ccc" />
          <Tooltip contentStyle={{ backgroundColor: '#333', border: 'none' }} />
          <Legend />
          <Line type="monotone" dataKey="transferLatency" stroke="#eb2f96" fill="url(#latencyGradient)" />
          <Line type="monotone" dataKey="successRate" stroke="#52c41a" />
          <Line type="monotone" dataKey="errorRate" stroke="#f5222d" />
        </LineChart>
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
        transition={{ duration: 0.5 }}
      >
        <Alert variant={alert.severity === 'error' ? 'destructive' : alert.severity === 'warning' ? 'warning' : 'default'} className="bg-gray-800 text-white border-gray-700">
          <AlertDescription>{alert.message} (Confidence: {(alert.confidence * 100).toFixed(1)}%)</AlertDescription>
        </Alert>
      </motion.div>
    ))}
  </div>
));

// 3D Topology Visualization
const TopologyVisualization = ({ topologyData }) => {
  const nodes = topologyData.nodes.map((node: any, idx: number) => ({
    id: idx,
    x: Math.cos(idx * 0.5) * 100,
    y: Math.sin(idx * 0.5) * 100,
    z: node.centrality * 50,
    size: node.influence * 10,
    color: `hsl(${node.stability * 360}, 70%, 50%)`
  }));
  const edges = topologyData.edges.map((edge: any) => ({
    source: edge[0],
    target: edge[1]
  }));

  return (
    <Card className="w-full h-96 bg-gray-900 text-white rounded-lg shadow-lg">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Zap className="w-5 h-5 text-yellow-400" />
          Network Topology
        </CardTitle>
      </CardHeader>
      <CardContent>
        <Canvas camera={{ position: [0, 0, 200], fov: 60 }}>
          <ambientLight intensity={0.5} />
          <pointLight position={[10, 10, 10]} />
          {nodes.map(node => (
            <mesh key={node.id} position={[node.x, node.y, node.z]}>
              <sphereGeometry args={[node.size, 32, 32]} />
              <meshStandardMaterial color={node.color} />
            </mesh>
          ))}
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
              <lineBasicMaterial color="#888" linewidth={2} />
            </line>
          ))}
          <OrbitControls />
        </Canvas>
      </CardContent>
    </Card>
  );
};

const CognitiveDashboard = ({ entityId }) => {
  const [emotionalData, setEmotionalData] = useState([]);
  const [cognitiveData, setCognitiveData] = useState([]);
  const [performanceData, setPerformanceData] = useState([]);
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
        const state = JSON.parse(atob(message.data.split(':')[1])); // Simplified decryption for demo
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
        break;
      case 'alert':
        setAlerts(prev => [...prev, ...message.alert.alerts || [message.alert]].slice(-5));
        break;
      case 'topology_update':
        setTopologyData({
          nodes: message.adaptations.flatMap((a: any) => a.nodes.map((n: number) => ({
            id: n,
            influence: TopologyNode.influence[n] || Math.random(),
            centrality: TopologyNode.centrality[n] || Math.random(),
            stability: TopologyNode.stability[n] || Math.random()
          }))),
          edges: message.adaptations.reduce((acc: any[], a: any) => {
            for (let i = 0; i < a.nodes.length; i++) {
              for (let j = i + 1; j < a.nodes.length; j++) {
                acc.push([a.nodes[i], a.nodes[j]]);
              }
            }
            return acc;
          }, [])
        });
        break;
      case 'mlPredictions':
        setPredictions(message.mlPredictions.predictions.map((p: number[], i: number) => ({
          timestamp: `${timestamp}+${i+1}`,
          [`pred_awareness_${i}`]: p[0] // Assuming awareness is first
        })));
        break;
    }
  }, []);

  return (
    <div className="space-y-6 p-6 bg-gray-800 min-h-screen">
      <motion.h1
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 1 }}
        className="text-2xl font-bold text-white flex items-center gap-2"
      >
        <Brain className="w-6 h-6 text-blue-400" />
        Cognitive Entity Dashboard - Entity {entityId}
      </motion.h1>

      <StateAlerts alerts={alerts} />

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <EmotionalStateChart data={emotionalData} />
        <CognitiveMetricsChart data={cognitiveData} predictions={predictions} />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <PerformanceMetrics data={performanceData} />
        <TopologyVisualization topologyData={topologyData} />
      </div>
    </div>
  );
};

// Placeholder for TopologyNode (assuming it’s defined elsewhere)
const TopologyNode = { influence: {}, centrality: {}, stability: {} };

export default CognitiveDashboard;
