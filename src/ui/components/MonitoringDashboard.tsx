import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ScatterChart, Scatter } from 'recharts';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Activity, Brain, Heart, AlertTriangle } from 'lucide-react';

const AnomalyScatterPlot = ({ data }) => {
  return (
    <Card className="w-full h-96">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <AlertTriangle className="w-5 h-5" />
          Anomaly Detection
        </CardTitle>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="x" />
            <YAxis dataKey="y" />
            <Tooltip />
            <Legend />
            <Scatter 
              name="Normal Data" 
              data={data.filter(d => !d.isAnomaly)}
              fill="#8884d8" 
            />
            <Scatter 
              name="Anomalies" 
              data={data.filter(d => d.isAnomaly)}
              fill="#ff4d4f" 
            />
          </ScatterChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
};

const MetricsTimeline = ({ data }) => {
  return (
    <Card className="w-full h-96">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Activity className="w-5 h-5" />
          System Metrics Timeline
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
            <Line type="monotone" dataKey="latency" stroke="#1890ff" name="Latency" />
            <Line type="monotone" dataKey="messageRate" stroke="#52c41a" name="Message Rate" />
            <Line type="monotone" dataKey="errorRate" stroke="#ff4d4f" name="Error Rate" />
          </LineChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
};

const AnomalyAlerts = ({ alerts }) => {
  return (
    <div className="space-y-4">
      {alerts.map((alert, index) => (
        <Alert 
          key={index} 
          variant={alert.severity === 'high' ? 'destructive' : 'warning'}
        >
          <AlertTriangle className="w-4 h-4" />
          <AlertDescription>
            {alert.message}
          </AlertDescription>
        </Alert>
      ))}
    </div>
  );
};

const ClusterVisualization = ({ data }) => {
  return (
    <Card className="w-full h-96">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Brain className="w-5 h-5" />
          Behavioral Clusters
        </CardTitle>
      </CardHeader>
      <CardContent>
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="x" />
            <YAxis dataKey="y" />
            <Tooltip />
            <Legend />
            {Array.from(new Set(data.map(d => d.cluster))).map((cluster, index) => (
              <Scatter
                key={cluster}
                name={`Cluster ${cluster}`}
                data={data.filter(d => d.cluster === cluster)}
                fill={`hsl(${index * 45}, 70%, 50%)`}
              />
            ))}
          </ScatterChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
};

const MonitoringDashboard = ({ entityId }) => {
  const [metricsData, setMetricsData] = useState([]);
  const [anomalyData, setAnomalyData] = useState([]);
  const [clusterData, setClusterData] = useState([]);
  const [alerts, setAlerts] = useState([]);

  useEffect(() => {
    const ws = new WebSocket(`ws://localhost:8080/monitor/${entityId}`);
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      switch (data.type) {
        case 'metrics':
          setMetricsData(prev => [...prev, data.metrics].slice(-100));
          break;
        case 'anomaly':
          setAnomalyData(prev => [...prev, data.anomaly].slice(-1000));
          break;
        case 'cluster':
          setClusterData(data.clusters);
          break;
        case 'alert':
          setAlerts(prev => [data.alert, ...prev].slice(0, 5));
          break;
      }
    };

    return () => ws.close();
  }, [entityId]);

  return (
    <div className="space-y-6 p-6">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold">Entity Monitoring Dashboard</h1>
        <span className="text-sm text-gray-500">Entity ID: {entityId}</span>
      </div>

      <AnomalyAlerts alerts={alerts} />

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <AnomalyScatterPlot data={anomalyData} />
        <ClusterVisualization data={clusterData} />
      </div>

      <MetricsTimeline data={metricsData} />
    </div>
  );
};

export default MonitoringDashboard;
