# server.py
import threading, time
from flask import Flask, send_from_directory
from flask_socketio import SocketIO
from world_sim import SimulationWorld, SimulationConfig

app = Flask(__name__, static_folder='.', static_url_path='')
socketio = SocketIO(app, cors_allowed_origins="*")

# ← configure your sim however you like:
config = SimulationConfig(world_size=(30,30), initial_agents=15, resource_spawn_rate=0.05, time_per_tick=0.1)
world = SimulationWorld(config)

def tick_loop():
    while True:
        world.tick()
        state = world.get_world_state()
        socketio.emit('state', state)
        time.sleep(config.time_per_tick)

@socketio.on('connect')
def on_connect():
    socketio.emit('state', world.get_world_state())

@app.route('/')
def index():
    # serves index.html and all its assets
    return send_from_directory('.', 'static/index.html')

if __name__ == '__main__':
    # start sim thread
    t = threading.Thread(target=tick_loop, daemon=True)
    t.start()
    socketio.run(app, host='0.0.0.0', port=5000)
