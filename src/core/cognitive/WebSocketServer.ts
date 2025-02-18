import WebSocket from 'ws';
import { SecurityManager, AlertManager } from './cognitive-security';

export class CognitiveWebSocketServer {
  private wss: WebSocket.Server;
  private securityManager: SecurityManager;
  private alertManager: AlertManager;
  private clients: Map<string, Set<WebSocket>> = new Map();

  constructor(port: number) {
    this.wss = new WebSocket.Server({ port });
    this.securityManager = new SecurityManager();
    this.alertManager = new AlertManager();
    const { monitor, anomalyDetector } = createMonitoringSystem();
    this.monitor = monitor;
    this.anomalyDetector = anomalyDetector;

    this.setupWebSocketServer();
  }

  private setupWebSocketServer() {
    this.wss.on('connection', (ws: WebSocket) => {
      let entityId: string | null = null;

      ws.on('message', (message: string) => {
        try {
          const data = JSON.parse(message);
          
          switch (data.type) {
            case 'subscribe':
              entityId = data.entityId;
              this.subscribeToEntity(entityId, ws);
              break;

            case 'unsubscribe':
              if (entityId) {
                this.unsubscribeFromEntity(entityId, ws);
              }
              break;

            case 'update_thresholds':
              if (entityId) {
                this.alertManager.setThresholds(parseInt(entityId), data.thresholds);
              }
              break;

            default:
              console.warn('Unknown message type:', data.type);
          }
        } catch (error) {
          console.error('Error processing message:', error);
        }
      });

      ws.on('close', () => {
        if (entityId) {
          this.unsubscribeFromEntity(entityId, ws);
        }
      });
    });
  }

  private subscribeToEntity(entityId: string, ws: WebSocket) {
    if (!this.clients.has(entityId)) {
      this.clients.set(entityId, new Set());
    }
    this.clients.get(entityId)?.add(ws);
  }

  private unsubscribeFromEntity(entityId: string, ws: WebSocket) {
    this.clients.get(entityId)?.delete(ws);
    if (this.clients.get(entityId)?.size === 0) {
      this.clients.delete(entityId);
    }
  }

  broadcastStateUpdate(entityId: string, state: any) {
    const clients = this.clients.get(entityId);
    if (!clients) return;

    // Encrypt state before broadcasting
    const encryptedState = this.securityManager.encryptState(state, entityId);
    
    // Check for alerts
    const alerts = this.alertManager.checkAlerts(parseInt(entityId), state);

    const update = {
      type: 'state_update',
      entityId,
      data: encryptedState,
      alerts,
      timestamp: Date.now()
    };

    clients.forEach(client => {
      if (client.readyState === WebSocket.OPEN) {
        client.send(JSON.stringify(update));
      }
    });
  }

  broadcastAlert(entityId: string, alert: any) {
    const clients = this.clients.get(entityId);
    if (!clients) return;

    const message = {
      type: 'alert',
      entityId,
      alert,
      timestamp: Date.now()
    };

    clients.forEach(client => {
      if (client.readyState === WebSocket.OPEN) {
        client.send(JSON.stringify(message));
      }
    });
  }
}
