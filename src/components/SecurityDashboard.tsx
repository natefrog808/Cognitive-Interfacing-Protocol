import React, { useState, useEffect, useCallback, useMemo } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar, PieChart, Pie, Cell } from 'recharts';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Shield, Lock, AlertTriangle, Zap } from 'lucide-react';
import { motion } from 'framer-motion';
import * as THREE from 'three';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';

// WebSocket connection
const ws = new WebSocket('ws://localhost:8080');

// Encryption Strength Visualization (3D)
const EncryptionStrengthVisualization = React.memo(({ data }) => {
  const nodes = useMemo(() => data.map((entry: any, idx: number) => ({
    id: idx,
    x: Math.cos(idx * 0.5) * 50,
    y: Math.sin(idx * 0.5) * 50,
    z: (entry.quantumImpact || 0) * 50,
    size: (entry.encryptionStrength || 0.5) * 10,
    color: entry.isRotated ? '#ffeb3b' : '#40c4ff'
  })), [data]);

  const NodeMesh = ({ node }: { node: any }) => {
    const ref = React.useRef<THREE.Mesh>(null!);
    useFrame((state) => {
      if (ref.current) {
        ref.current.rotation.y += 0.02;
        ref.current.scale.setScalar(1 + Math.sin(state.clock.elapsedTime + node.id) * 0.1);
      }
    });

    return (
      <mesh ref={ref} position={[node.x, node.y, node.z]}>
        <sphereGeometry args={[node.size, 32, 32]} />
        <meshStandardMaterial color={node.color} emissive={node.color} emissiveIntensity={0.5} />
      </mesh>
    );
  };

  return (
    <Card className="w-full h-96 bg-gray-900 text-white rounded-lg shadow-lg overflow-hidden">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Lock className="w-5 h-5 text-blue-400 animate-pulse" />
          Encryption Strength
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

// Alert Trends Chart
const AlertTrendsChart = React.memo(({ data }) => (
  <Card className="w-full h-96 bg-gray-900 text-white rounded-lg shadow-lg overflow-hidden">
    <CardHeader>
      <CardTitle className="flex items-center gap-2">
        <AlertTriangle className="w-5 h-5 text-orange-400 animate-bounce" />
        Alert Trends
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
          <Line type="monotone" dataKey="moodSwing" stroke="#ff4d4f" strokeWidth={2} dot={false} name="Mood Swing" />
          <Line type="monotone" dataKey="highStress" stroke="#ff9800" strokeWidth={2} dot={false} name="High Stress" />
          <Line type="monotone" dataKey="cognitiveOverload" stroke="#ffeb3b" strokeWidth={2} dot={false} name="Cognitive Overload" />
          <Line type="monotone" dataKey="fearSpike" stroke="#722ed1" strokeWidth={2} dot={false} name="Fear Spike" />
          <Line type="monotone" dataKey="angerTrigger" stroke="#52c41a" strokeWidth={2} dot={false} name="Anger Trigger" />
        </LineChart>
      </ResponsiveContainer>
    </CardContent>
  </Card>
));

// Emotional Patterns Pie Chart
const EmotionalPatternsChart = React.memo(({ data }) => {
  const COLORS = ['#8884d8', '#ff4d4f', '#ff9800', '#722ed1', '#52c41a'];

  return (
    <Card className="w-full h-96 bg-gray-900 text-white rounded-lg shadow-lg overflow-hidden">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Brain className="w-5 h-5 text-purple-400 animate-spin-slow" />
          Emotional Patterns
        </CardTitle>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={data}
              cx="50%"
              cy="50%"
              labelLine={false}
              outerRadius={80}
              fill="#8884d8"
              dataKey="confidence"
              label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
            >
              {data.map((entry: any, index: number) => (
                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip contentStyle={{ backgroundColor: '#333', border: 'none', borderRadius: '4px' }} />
            <Legend verticalAlign="top" height={36} />
          </PieChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
});

// Animated Security Alerts
const SecurityAlerts = React.memo(({ alerts }) => (
  <div className="space-y-4">
    {alerts.map((alert, index) => (
      <motion.div
        key={index}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: index * 0.1 }}
      >
        <Alert
          variant={alert.severity === 'error' ? 'destructive' : alert.severity === 'warning' ? 'warning' : 'default'}
          className={`bg-gray-800 text-white border-${alert.severity === 'error' ? 'red' : alert.severity === 'warning' ? 'yellow' : 'gray'}-700 ${alert.severity === 'error' ? 'animate-pulse' : ''}`}
        >
          <AlertTriangle className={`w-4 h-4 ${alert.severity === 'error' ? 'text-red-500' : 'text-yellow-500'}`} />
          <AlertDescription>
            {alert.message} (Confidence: {(alert.confidence * 100).toFixed(1)}%)
          </AlertDescription>
        </Alert>
      </motion.div>
    ))}
  </div>
));

const SecurityDashboard = ({ entityId }) => {
  const [encryptionData, setEncryptionData] = useState([]);
  const [alertData, setAlertData] = useState([]);
  const [patternData, setPatternData] = useState([]);
  const [alerts, setAlerts] = useState([]);

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
        const state = message.data || message;
        if (state.encryptedState) {
          setEncryptionData(prev => [...prev, {
            timestamp,
            encryptionStrength: state.securityLevel === 'high' ? 1 : 0.5,
            quantumImpact: state.quantumState?.entanglementScore || 0,
            isRotated: state.event === 'key_rotated'
          }].slice(-20));
        }
        if (state.emotionalPatterns) {
          setPatternData(state.emotionalPatterns.map((p: any) => ({
            name: p.pattern,
            confidence: p.confidence,
            impact: p.predictedImpact
          })));
        }
        break;
      case 'alert':
        const alertArray = message.alerts || [message];
        setAlerts(prev => [...prev, ...alertArray.map((a: any) => ({
          severity: a.severity,
          message: a.message,
          confidence: a.confidence
        }))].slice(-5));
        setAlertData(prev => {
          const newData = { ...prev.slice(-20).reduce((acc: any, d: any) => ({
            ...acc,
            timestamp: d.timestamp,
            moodSwing: 0,
            highStress: 0,
            cognitiveOverload: 0,
            fearSpike: 0,
            angerTrigger: 0
          }), { timestamp })};
          alertArray.forEach((a: any) => {
            if (a.type === 'mood_swing') newData.moodSwing = a.confidence;
            if (a.type === 'high_stress') newData.highStress = a.confidence;
            if (a.type === 'cognitive_overload') newData.cognitiveOverload = a.confidence;
            if (a.type === 'fear_spike') newData.fearSpike = a.confidence;
            if (a.type === 'anger_trigger') newData.angerTrigger = a.confidence;
          });
          return [...prev, newData].slice(-20);
        });
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
        <Shield className="w-8 h-8 text-blue-400 animate-pulse" />
        Security Dashboard - Entity {entityId}
      </motion.h1>

      <SecurityAlerts alerts={alerts} />

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <EncryptionStrengthVisualization data={encryptionData} />
        <AlertTrendsChart data={alertData} />
        <EmotionalPatternsChart data={patternData} />
      </div>
    </div>
  );
};

export default SecurityDashboard;
