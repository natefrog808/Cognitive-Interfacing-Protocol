import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ScatterChart, Scatter, AreaChart, Area } from 'recharts';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Activity, Brain, AlertTriangle, Zap } from 'lucide-react';
import { motion } from 'framer-motion'; // For animations
import * as THREE from 'three'; // For 3D visualization
import { Canvas } from '@react-three/fiber'; // React Three Fiber
import { OrbitControls } from '@react-three/drei'; // Camera controls

// WebSocket connection
const ws = new WebSocket('ws://localhost:8080');

// Anomaly Scatter Plot with 3D Option
const AnomalyScatterPlot = React.memo(({ data, is3D }) => (
  <Card className="w-full h-96 bg-gray-900 text-white rounded-lg shadow-lg">
    <CardHeader>
      <CardTitle className="flex items-center gap-2">
        <AlertTriangle className="w-5 h-5 text-orange-400" />
        Anomaly Detection {is3D ? '(3D)' : '(2D)'}
      </CardTitle>
    </CardHeader>
    <CardContent>
      {is3D ? (
        <Canvas camera={{ position: [0, 0, 100], fov: 60 }}>
          <ambientLight intensity={0.5} />
          <pointLight position={[10, 10, 10]} />
          {data.map((d, idx) => (
            <mesh key={idx} position={[d.x * 10, d.y * 10, d.z || 0]}>
              <sphereGeometry args={[d.isAnomaly ? 2 : 1, 32, 32]} />
              <meshStandardMaterial color={d.isAnomaly ? '#ff4d4f' : '#8884d8'} />
            </mesh>
          ))}
          <OrbitControls />
        </Canvas>
      ) : (
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart>
            <CartesianGrid strokeDasharray="3 3" stroke="#444" />
            <XAxis dataKey="x" stroke="#ccc" />
            <YAxis dataKey="y" stroke="#ccc" />
            <Tooltip contentStyle={{ backgroundColor: '#333', border: 'none' }} />
            <Legend />
            <Scatter name="Normal Data" data={data.filter(d => !d.isAnomaly)} fill="#8884d8" />
            <Scatter name="Anomalies" data={data.filter(d => d.isAnomaly)} fill="#ff4d4f" />
          </ScatterChart>
        </ResponsiveContainer>
      )}
    </CardContent>
  </Card>
));

// Metrics Timeline with Gradient Areas
const MetricsTimeline = React.memo(({ data }) => (
  <Card className="w-full h-96 bg-gray-900 text-white rounded-lg shadow-lg">
    <CardHeader>
      <CardTitle className="flex items-center gap-2">
        <Activity className="w-5 h-5 text-green-400" />
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
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#444" />
          <XAxis dataKey="timestamp" stroke="#ccc" />
          <YAxis stroke="#ccc" />
          <Tooltip contentStyle={{ backgroundColor: '#333', border: 'none' }} />
          <Legend />
          <Area type="monotone" dataKey="latency" stroke="#1890ff" fill="url(#latencyGradient)" name="Latency" />
          <Line type="monotone" dataKey="messageRate" stroke="#52c41a" name="Message Rate" />
          <Line type="monotone" dataKey="errorRate" stroke="#ff4d4f" name="Error Rate" />
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
          variant={alert.severity === 'high' ? 'destructive' : 'warning'}
          className={`bg-gray-800 text-white border-${alert.severity === 'high' ? 'red' : 'yellow'}-700`}
        >
          <AlertTriangle className="w-4 h-4" />
          <AlertDescription>{alert.message} (Confidence: {(alert.confidence * 100).toFixed(1)}%)</AlertDescription>
        </Alert>
      </motion.div>
    ))}
  </div>
));

// Enhanced Cluster Visualization with Predictive Overlay
const ClusterVisualization = React.memo(({ data, predictions }) => (
  <Card className="w-full h-96 bg-gray-900 text-white rounded-lg shadow-lg">
    <CardHeader>
      <CardTitle className="flex items-center gap-2">
        <Brain className="w-5 h-5 text-purple-400" />
        Behavioral Clusters & Predictions
      </CardTitle>
    </CardHeader>
    <CardContent>
      <ResponsiveContainer width="100%" height="100%">
        <ScatterChart>
          <CartesianGrid strokeDasharray="3 3" stroke="#444" />
          <XAxis dataKey="x" stroke="#ccc" />
          <YAxis dataKey="y" stroke="#ccc" />
          <Tooltip contentStyle={{ backgroundColor: '#333', border: 'none' }} />
          <Legend />
          {Array.from(new Set(data.map((d: any) => d.cluster))).map((cluster, index) => (
            <Scatter
              key={cluster}
              name={`Cluster ${cluster}`}
              data={data.filter((d: any) => d.cluster === cluster)}
              fill={`hsl(${index * 45}, 70%, 50%)`}
            />
          ))}
          {predictions && (
            <Scatter
              name="Predicted Anomalies"
              data={predictions}
              fill="#ffeb3b"
              shape="star"
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
        const state = JSON.parse(atob(message.data.split(':')[1])); // Simplified decryption
        setMetricsData(prev => [...prev, {
          timestamp,
          latency: state.performance?.transferLatency || Math.random() * 100,
          messageRate: state.performance?.messageRate || Math.random() * 1000,
          errorRate: state.performance?.errorRate || Math.random() * 0.05
        }].slice(-100));
        break;
      case 'anomaly':
        setAnomalyData(prev => [...prev, {
          x: message.anomaly.score || Math.random(),
          y: message.anomaly.cluster || Math.random(),
          z: message.anomaly.severity === 'high' ? 1 : 0.5, // For 3D
          isAnomaly: message.anomaly.isAnomaly
        }].slice(-1000));
        setAlerts(prev => [...prev, {
          severity: message.anomaly.severity,
          message: message.anomaly.message || 'Anomaly detected',
          confidence: message.anomaly.confidence || 0.9
        }].slice(0, 5));
        break;
      case 'cluster':
        setClusterData(message.clusters.map((c: any, i: number) => ({
          x: c[0] || Math.random(),
          y: c[1] || Math.random(),
          cluster: i
        })));
        break;
      case 'mlPredictions':
        setPredictions(message.mlPredictions.predictions.map((p: number[], i: number) => ({
          x: p[0], // Assuming score-like metric
          y: p[1], // Assuming cluster-like metric
        })));
        break;
    }
  }, []);

  const toggle3D = () => setIs3D(prev => !prev);

  return (
    <div className="space-y-6 p-6 bg-gray-800 min-h-screen">
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 1 }}
        className="flex justify-between items-center"
      >
        <h1 className="text-2xl font-bold text-white flex items-center gap-2">
          <Zap className="w-6 h-6 text-yellow-400" />
          Entity Monitoring Dashboard - Entity {entityId}
        </h1>
        <button
          onClick={toggle3D}
          className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 transition"
        >
          Toggle {is3D ? '2D' : '3D'} Anomaly View
        </button>
      </motion.div>

      <AnomalyAlerts alerts={alerts} />

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <AnomalyScatterPlot data={anomalyData} is3D={is3D} />
        <ClusterVisualization data={clusterData} predictions={predictions} />
      </div>

      <MetricsTimeline data={metricsData} />
    </div>
  );
};

export default MonitoringDashboard;
