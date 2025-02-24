import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ScatterChart, Scatter, AreaChart, Area } from 'recharts';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Activity, Brain, AlertTriangle, Zap } from 'lucide-react';
import { motion } from 'framer-motion';
import * as THREE from 'three';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';

// WebSocket connection
const ws = new WebSocket('ws://localhost:8080');

// Anomaly Scatter Plot with Enhanced 3D Visualization
const AnomalyScatterPlot = React.memo(({ data, is3D }) => {
  const nodes = useMemo(() => data.map((d: any, idx: number) => ({
    id: idx,
    x: d.x,
    y: d.y,
    z: d.z || 0,
    size: d.isAnomaly ? 2 + d.severity * 2 : 1,
    color: d.isAnomaly ? (d.severity === 3 ? '#ff4d4f' : d.severity === 2 ? '#ff9800' : '#ffeb3b') : '#8884d8',
    anomalyScore: d.anomalyScore || 0
  })), [data]);

  const NodeMesh = ({ node }: { node: any }) => {
    const ref = React.useRef<THREE.Mesh>(null!);
    useFrame((state) => {
      if (ref.current && node.anomalyScore > 0) {
        ref.current.scale.setScalar(1 + Math.sin(state.clock.elapsedTime * 2 + node.id) * node.anomalyScore * 0.5);
      }
    });

    return (
      <mesh ref={ref} position={[node.x * 10, node.y * 10, node.z * 10]}>
        <sphereGeometry args={[node.size, 32, 32]} />
        <meshStandardMaterial
          color={node.color}
          emissive={node.anomalyScore > 0 ? node.color : '#000'}
          emissiveIntensity={node.anomalyScore * 0.5}
        />
      </mesh>
    );
  };

  return (
    <Card className="w-full h-96 bg-gray-900 text-white rounded-lg shadow-lg overflow-hidden">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <AlertTriangle className="w-5 h-5 text-orange-400 animate-pulse" />
          Anomaly Detection {is3D ? '(3D)' : '(2D)'}
        </CardTitle>
      </CardHeader>
      <CardContent>
        {is3D ? (
          <Canvas camera={{ position: [0, 0, 100], fov: 60 }}>
            <ambientLight intensity={0.5} />
            <pointLight position={[10, 10, 10]} intensity={1} />
            {nodes.map(node => <NodeMesh key={node.id} node={node} />)}
            <OrbitControls />
          </Canvas>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart>
              <CartesianGrid strokeDasharray="3 3" stroke="#444" />
              <XAxis dataKey="x" stroke="#ccc" tick={{ fontSize: 12 }} domain={[-1, 1]} />
              <YAxis dataKey="y" stroke="#ccc" tick={{ fontSize: 12 }} domain={[-1, 1]} />
              <Tooltip contentStyle={{ backgroundColor: '#333', border: 'none', borderRadius: '4px' }} />
              <Legend verticalAlign="top" height={36} />
              <Scatter name="Normal" data={data.filter((d: any) => !d.isAnomaly)} fill="#8884d8" shape="circle" />
              <Scatter name="Anomalies" data={data.filter((d: any) => d.isAnomaly)} fill="#ff4d4f" shape="star" />
            </ScatterChart>
          </ResponsiveContainer>
        )}
      </CardContent>
    </Card>
  );
});

// Metrics Timeline with Gradient Areas
const MetricsTimeline = React.memo(({ data }) => (
  <Card className="w-full h-96 bg-gray-900 text-white rounded-lg shadow-lg overflow-hidden">
    <CardHeader>
      <CardTitle className="flex items-center gap-2">
        <Activity className="w-5 h-5 text-green-400 animate-bounce" />
        System Metrics Timeline
      </CardTitle>
    </CardHeader>
    <CardContent>
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data}>
          <defs>
            <linearGradient id="latencyGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#1890ff" stopOpacity={0.8} />
              <stop offset="95%" stopColor="#1890ff" stopOpacity={0} />
            </linearGradient>
            <linearGradient id="messageGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#52c41a" stopOpacity={0.8} />
              <stop offset="95%" stopColor="#52c41a" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#444" />
          <XAxis dataKey="timestamp" stroke="#ccc" tick={{ fontSize: 12 }} />
          <YAxis stroke="#ccc" tick={{ fontSize: 12 }} domain={[0, 'auto']} />
          <Tooltip contentStyle={{ backgroundColor: '#333', border: 'none', borderRadius: '4px' }} />
          <Legend verticalAlign="top" height={36} />
          <Area type="monotone" dataKey="latency" stroke="#1890ff" fill="url(#latencyGradient)" name="Latency (ms)" />
          <Area type="monotone" dataKey="messageRate" stroke="#52c41a" fill="url(#messageGradient)" name="Message Rate" />
          <Line type="monotone" dataKey="errorRate" stroke="#ff4d4f" name="Error Rate" strokeWidth={2} dot={false} />
        </AreaChart>
      </ResponsiveContainer>
    </CardContent>
  </Card>
));

// Animated Anomaly Alerts
const AnomalyAlerts = React.memo(({ alerts }) => (
  <div className="space-y-4">
    {alerts.map((alert, index) => (
      <motion.div
        key={index}
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.5, delay: index * 0.1 }}
      >
        <Alert
          variant={alert.severity === 'high' ? 'destructive' : alert.severity === 'medium' ? 'warning' : 'default'}
          className={`bg-gray-800 text-white border-${alert.severity === 'high' ? 'red' : alert.severity === 'medium' ? 'yellow' : 'gray'}-700 ${alert.severity === 'high' ? 'animate-pulse' : ''}`}
        >
          <AlertTriangle className={`w-4 h-4 ${alert.severity === 'high' ? 'text-red-500' : alert.severity === 'medium' ? 'text-yellow-500' : 'text-gray-400'}`} />
          <AlertDescription>
            {alert.message} (Confidence: {(alert.confidence * 100).toFixed(1)}%, Severity: {alert.severity})
          </AlertDescription>
        </Alert>
      </motion.div>
    ))}
  </div>
));

// Enhanced Cluster Visualization with Predictive Overlay
const ClusterVisualization = React.memo(({ data, predictions }) => (
  <Card className="w-full h-96 bg-gray-900 text-white rounded-lg shadow-lg overflow-hidden">
    <CardHeader>
      <CardTitle className="flex items-center gap-2">
        <Brain className="w-5 h-5 text-purple-400 animate-spin-slow" />
        Behavioral Clusters & Predictions
      </CardTitle>
    </CardHeader>
    <CardContent>
      <ResponsiveContainer width="100%" height="100%">
        <ScatterChart>
          <CartesianGrid strokeDasharray="3 3" stroke="#444" />
          <XAxis dataKey="x" stroke="#ccc" tick={{ fontSize: 12 }} domain={[-1, 1]} />
          <YAxis dataKey="y" stroke="#ccc" tick={{ fontSize: 12 }} domain={[-1, 1]} />
          <Tooltip contentStyle={{ backgroundColor: '#333', border: 'none', borderRadius: '4px' }} />
          <Legend verticalAlign="top" height={36} />
          {Array.from(new Set(data.map((d: any) => d.cluster))).map((cluster, index) => (
            <Scatter
              key={cluster}
              name={`Cluster ${cluster}`}
              data={data.filter((d: any) => d.cluster === cluster)}
              fill={`hsl(${index * 45}, 70%, 50%)`}
              shape="circle"
            />
          ))}
          {predictions && (
            <Scatter
              name="Predicted Anomalies"
              data={predictions}
              fill="#ffeb3b"
              shape="star"
              opacity={0.7}
            />
          )}
        </ScatterChart>
      </ResponsiveContainer>
    </CardContent>
  </Card>
));

const MonitoringDashboard = ({ entityId }) => {
  const [metricsData, setMetricsData] = useState([]);
  const [anomalyData, setAnomalyData] = useState([]);
  const [clusterData, setClusterData] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [predictions, setPredictions] = useState([]);
  const [is3D, setIs3D] = useState(false);

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
        const state = message.data; // Assuming proper formatting from server
        setMetricsData(prev => [...prev, {
          timestamp,
          latency: state.systemHealth?.networkLatency || state.performance?.transferLatency || Math.random() * 100,
          messageRate: (state.systemHealth?.messageQueueSize || 0) / 10 || Math.random() * 1000,
          errorRate: state.systemHealth?.errorRate || state.performance?.errorRate || Math.random() * 0.05
        }].slice(-100));
        break;
      case 'anomaly':
        const anomaly = message.anomaly;
        setAnomalyData(prev => [...prev, {
          x: anomaly.score || Math.random(),
          y: anomaly.cluster || Math.random(),
          z: anomaly.signature?.reduce((sum: number, val: number) => sum + val, 0) / 10 || Math.random(),
          isAnomaly: anomaly.isAnomaly,
          anomalyScore: anomaly.score || 0,
          severity: anomaly.severity || (anomaly.isAnomaly ? 2 : 1)
        }].slice(-1000));
        setAlerts(prev => [...prev, {
          severity: anomaly.severity === 3 ? 'high' : anomaly.severity === 2 ? 'medium' : 'low',
          message: anomaly.message || 'Anomaly detected',
          confidence: anomaly.confidence || 0.9
        }].slice(-5));
        break;
      case 'topology_update':
        setClusterData(message.adaptations.flatMap((a: any) => a.nodes.map((n: any) => ({
          x: n.spatialVector ? n.spatialVector[0] : Math.random(),
          y: n.spatialVector ? n.spatialVector[1] : Math.random(),
          cluster: n.cluster || Math.floor(Math.random() * 5)
        })));
        break;
      case 'mlPredictions':
        setPredictions(message.mlPredictions.predictions.map((p: number[], i: number) => ({
          x: p[0], // Assuming score-like metric
          y: p[1] // Assuming cluster-like metric
        })).slice(-10));
        break;
      case 'scheduler':
        setMetricsData(prev => [...prev, {
          timestamp,
          latency: message.resourceProfile.network.reduce((sum: number, val: number) => sum + val, 0) / message.resourceProfile.network.length,
          messageRate: message.fitness * 1000, // Scale for visibility
          errorRate: 1 - message.quantumFitness
        }].slice(-100));
        break;
      case 'topology_forecast':
        setPredictions(message.predictions.map((p: any, i: number) => ({
          x: p.globalEfficiency,
          y: p.clusteringCoefficient
        })).slice(-10));
        break;
    }
  }, []);

  const toggle3D = () => setIs3D(prev => !prev);

  return (
    <div className="space-y-6 p-6 bg-gray-800 min-h-screen">
      <motion.div
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 1 }}
        className="flex justify-between items-center"
      >
        <h1 className="text-3xl font-bold text-white flex items-center gap-2">
          <Zap className="w-8 h-8 text-yellow-400 animate-spin" />
          CogniVerse Monitoring - Entity {entityId}
        </h1>
        <button
          onClick={toggle3D}
          className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 transition duration-200"
        >
          Toggle {is3D ? '2D' : '3D'} Anomaly View
        </button>
      </motion.div>

      <AnomalyAlerts alerts={alerts} />

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <AnomalyScatterPlot data={anomalyData} is3D={is3D} />
        <ClusterVisualization data={clusterData} predictions={predictions} />
        <MetricsTimeline data={metricsData} />
      </div>
    </div>
  );
};

export default MonitoringDashboard;
