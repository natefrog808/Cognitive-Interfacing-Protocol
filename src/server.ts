import { CognitiveWebSocketServer } from './core/CognitiveWebSocketServer';
import { SecurityManager } from './core/SecurityManager';
import { AlertManager } from './core/AlertManager';
import { SimulationEngine } from './core/SimulationEngine';
import dotenv from 'dotenv';

dotenv.config();

const PORT = process.env.WS_PORT ? parseInt(process.env.WS_PORT) : 8080;
const SECRET = process.env.SECRET_KEY || 'your-secret-here';

async function startServer() {
  const wsServer = new CognitiveWebSocketServer(PORT);
  const security = new SecurityManager(PORT);
  const alerts = new AlertManager(PORT);
  const simulation = new SimulationEngine(PORT);

  await Promise.all([
    security.initializeQuantum(),
    alerts.initializePatternModel(),
    simulation.initializeComponents()
  ]);

  wsServer.onConnection((socket) => {
    socket.on('message', (message) => {
      const data = JSON.parse(message.toString());
      if (data.type === 'handshake') {
        if (security.validateToken(data.token, SECRET)) {
          socket.send(JSON.stringify({ type: 'handshake_success' }));
        } else {
          socket.close(1008, 'Invalid token');
        }
      } else if (data.type === 'subscribe') {
        socket.send(JSON.stringify({ type: 'subscribed', entityId: data.entityId }));
      } else if (data.type === 'run_simulation') {
        simulation.runSimulation(data.scenarioId).catch(err => console.error(err));
      } else if (data.type === 'stop_simulation') {
        simulation.stopSimulation();
      } else if (data.type === 'load_scenario') {
        simulation.loadScenario(data.scenarioId).catch(err => console.error(err));
      }
    });
  });

  console.log(`CogniVerse Server rocking on port ${PORT}`);
}

startServer().catch(err => console.error('Server failed to start:', err));
