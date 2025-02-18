import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Activity, Brain, Heart } from 'lucide-react';

const EmotionalStateChart = ({ data }) => {
  return (
    <Card className="w-full h-96">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Heart className="w-5 h-5" />
          Emotional State Trends
        </CardTitle>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="timestamp" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="joy" stroke="#8884d8" />
            <Line type="monotone" dataKey="anger" stroke="#ff4d4f" />
            <Line type="monotone" dataKey="fear" stroke="#722ed1" />
            <Line type="monotone" dataKey="disgust" stroke="#52c41a" />
          </LineChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
};

const CognitiveMetricsChart = ({ data }) => {
  return (
    <Card className="w-full h-96">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Brain className="w-5 h-5" />
          Cognitive Metrics
        </CardTitle>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="timestamp" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="awareness" stroke="#1890ff" />
            <Line type="monotone" dataKey="coherence" stroke="#faad14" />
            <Line type="monotone" dataKey="complexity" stroke="#13c2c2" />
          </LineChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
};

const PerformanceMetrics = ({ data }) => {
  return (
    <Card className="w-full h-96">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Activity className="w-5 h-5" />
          System Performance
        </CardTitle>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="timestamp" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="transferLatency" stroke="#eb2f96" />
            <Line type="monotone" dataKey="successRate" stroke="#52c41a" />
            <Line type="monotone" dataKey="errorRate" stroke="#f5222d" />
          </LineChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
};

const StateAlerts = ({ alerts }) => {
  return (
    <div className="space-y-4">
      {alerts.map((alert, index) => (
        <Alert key={index} variant={alert.severity}>
          <AlertDescription>
            {alert.message}
          </AlertDescription>
        </Alert>
      ))}
    </div>
  );
};

const CognitiveDashboard = ({ entityId }) => {
  const [emotionalData, setEmotionalData] = useState([]);
  const [cognitiveData, setCognitiveData] = useState([]);
  const [performanceData, setPerformanceData] = useState([]);
  const [alerts, setAlerts] = useState([]);

  useEffect(() => {
    // Simulated data updates - replace with actual WebSocket connection
    const interval = setInterval(() => {
      // Fetch latest data for the entity
      fetchEntityData(entityId);
    }, 1000);

    return () => clearInterval(interval);
  }, [entityId]);

  const fetchEntityData = async (id) => {
    // This would be replaced with actual API calls to your cognitive system
    const timestamp = new Date().toISOString();
    
    // Simulate emotional state data
    setEmotionalData(prev => [...prev, {
      timestamp,
      joy: Math.random(),
      anger: Math.random(),
      fear: Math.random(),
      disgust: Math.random()
    }].slice(-20));

    // Simulate cognitive metrics
    setCognitiveData(prev => [...prev, {
      timestamp,
      awareness: Math.random(),
      coherence: Math.random(),
      complexity: Math.random()
    }].slice(-20));

    // Simulate performance metrics
    setPerformanceData(prev => [...prev, {
      timestamp,
      transferLatency: Math.random() * 100,
      successRate: 0.95 + Math.random() * 0.05,
      errorRate: Math.random() * 0.05
    }].slice(-20));

    // Check for alerts
    checkAlerts();
  };

  const checkAlerts = () => {
    // Example alert logic
    const newAlerts = [];
    
    if (Math.random() > 0.9) {
      newAlerts.push({
        severity: 'warning',
        message: 'High cognitive load detected'
      });
    }

    setAlerts(newAlerts);
  };

  return (
    <div className="space-y-6 p-6">
      <h1 className="text-2xl font-bold">Cognitive Entity Dashboard</h1>
      
      <StateAlerts alerts={alerts} />
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <EmotionalStateChart data={emotionalData} />
        <CognitiveMetricsChart data={cognitiveData} />
      </div>
      
      <PerformanceMetrics data={performanceData} />
    </div>
  );
};

export default CognitiveDashboard;
