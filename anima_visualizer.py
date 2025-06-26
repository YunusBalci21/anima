import pygame
import numpy as np
import json
import asyncio
import threading
from collections import deque
import math
import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time

# Import from main engine
from agent_arch import Agent, Emotion, ResourceType, Action
from world_sim import SimulationWorld, SimulationConfig

# Visualization Configuration
@dataclass 
class VisualizerConfig:
    window_width: int = 1400
    window_height: int = 900
    cell_size: int = 20
    sidebar_width: int = 400
    fps: int = 30
    
    # Colors
    bg_color: Tuple[int, int, int] = (10, 10, 20)
    grid_color: Tuple[int, int, int] = (30, 30, 40)
    text_color: Tuple[int, int, int] = (200, 200, 220)
    
    # Agent colors by emotion
    emotion_colors: Dict[Emotion, Tuple[int, int, int]] = None
    
    # Resource colors
    resource_colors: Dict[ResourceType, Tuple[int, int, int]] = None
    
    def __post_init__(self):
        if self.emotion_colors is None:
            self.emotion_colors = {
                Emotion.NEUTRAL: (100, 100, 100),
                Emotion.HAPPY: (100, 200, 100),
                Emotion.ANGRY: (200, 50, 50),
                Emotion.FEARFUL: (150, 100, 200),
                Emotion.CURIOUS: (100, 150, 200),
                Emotion.LONELY: (80, 80, 150),
                Emotion.LOVING: (255, 150, 200),
                Emotion.CONFUSED: (150, 150, 100),
                Emotion.HOPEFUL: (200, 200, 100),
                Emotion.DESPERATE: (100, 50, 50)
            }
            
        if self.resource_colors is None:
            self.resource_colors = {
                ResourceType.FOOD: (100, 200, 100),
                ResourceType.WATER: (100, 150, 255),
                ResourceType.SHELTER: (150, 100, 50),
                ResourceType.LIGHT: (255, 255, 150),
                ResourceType.KNOWLEDGE: (200, 100, 255)
            }

class ANIMAVisualizer:
    def __init__(self, world: SimulationWorld, config: VisualizerConfig = None):
        self.world = world
        self.config = config or VisualizerConfig()
        
        # Pygame setup
        pygame.init()
        self.screen = pygame.display.set_mode((self.config.window_width, self.config.window_height))
        pygame.display.set_caption("ANIMA - The Emergent Self Engine")
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, 16)
        self.font_medium = pygame.font.Font(None, 20)
        self.font_large = pygame.font.Font(None, 28)
        
        # UI state
        self.selected_agent = None
        self.show_relationships = False
        self.show_language = False
        self.show_beliefs = False
        self.pause = False
        self.speed_multiplier = 1.0
        
        # History tracking
        self.population_history = deque(maxlen=200)
        self.culture_history = deque(maxlen=200)
        self.event_log = deque(maxlen=20)
        self.communication_trails = []  # Visual trails for communications
        
        # Camera
        self.camera_x = 0
        self.camera_y = 0
        self.zoom = 1.0
        
        # Calculate grid display area
        self.grid_width = self.config.window_width - self.config.sidebar_width
        self.grid_height = self.config.window_height
        
    def run(self):
        """Main visualization loop"""
        running = True
        
        # Start simulation in separate thread
        sim_thread = threading.Thread(target=self._run_simulation)
        sim_thread.daemon = True
        sim_thread.start()
        
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    self._handle_key(event.key)
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self._handle_mouse_click(event.pos)
                elif event.type == pygame.MOUSEWHEEL:
                    self._handle_zoom(event.y)
                    
            # Update
            self._update()
            
            # Draw
            self._draw()
            
            # Cap framerate
            self.clock.tick(self.config.fps)
            
        pygame.quit()
        
    def _run_simulation(self):
        """Run simulation in background thread"""
        while True:
            if not self.pause:
                self.world.tick()
                
                # Track history
                self.population_history.append(len(self.world.agents))
                self.culture_history.append(len(self.world.cultures))
                
                # Log events
                for event in self.world.events.events:
                    self.event_log.append({
                        "time": self.world.time,
                        "type": event.type,
                        "summary": self._summarize_event(event)
                    })
                    
                time.sleep(0.1 / self.speed_multiplier)
            else:
                time.sleep(0.1)
                
    def _handle_key(self, key):
        """Handle keyboard input"""
        if key == pygame.K_SPACE:
            self.pause = not self.pause
        elif key == pygame.K_r:
            self.show_relationships = not self.show_relationships
        elif key == pygame.K_l:
            self.show_language = not self.show_language
        elif key == pygame.K_b:
            self.show_beliefs = not self.show_beliefs
        elif key == pygame.K_EQUALS:
            self.speed_multiplier = min(5.0, self.speed_multiplier * 1.5)
        elif key == pygame.K_MINUS:
            self.speed_multiplier = max(0.1, self.speed_multiplier / 1.5)
        elif key == pygame.K_UP:
            self.camera_y -= 10
        elif key == pygame.K_DOWN:
            self.camera_y += 10
        elif key == pygame.K_LEFT:
            self.camera_x -= 10
        elif key == pygame.K_RIGHT:
            self.camera_x += 10
            
    def _handle_mouse_click(self, pos):
        """Handle mouse clicks"""
        x, y = pos
        
        # Check if click is in grid area
        if x < self.grid_width:
            # Convert to world coordinates
            world_x = int((x - self.camera_x) / (self.config.cell_size * self.zoom))
            world_y = int((y - self.camera_y) / (self.config.cell_size * self.zoom))
            
            # Find agent at position
            for agent in self.world.agents.values():
                ax, ay = agent.physical_state.position
                if ax == world_x and ay == world_y:
                    self.selected_agent = agent
                    break
                    
    def _handle_zoom(self, direction):
        """Handle zoom"""
        if direction > 0:
            self.zoom = min(2.0, self.zoom * 1.1)
        else:
            self.zoom = max(0.5, self.zoom / 1.1)
            
    def _update(self):
        """Update visualization state"""
        # Update communication trails
        self.communication_trails = [
            trail for trail in self.communication_trails 
            if trail["alpha"] > 0
        ]
        
        for trail in self.communication_trails:
            trail["alpha"] -= 5
            
    def _draw(self):
        """Main draw function"""
        self.screen.fill(self.config.bg_color)
        
        # Draw main grid area
        grid_surface = pygame.Surface((self.grid_width, self.grid_height))
        grid_surface.fill(self.config.bg_color)
        
        # Draw world
        self._draw_grid(grid_surface)
        self._draw_resources(grid_surface)
        self._draw_agents(grid_surface)
        
        if self.show_relationships:
            self._draw_relationships(grid_surface)
            
        if self.show_language:
            self._draw_language_connections(grid_surface)
            
        self._draw_communication_trails(grid_surface)
        
        self.screen.blit(grid_surface, (0, 0))
        
        # Draw sidebar
        self._draw_sidebar()
        
        # Draw overlays
        if self.pause:
            self._draw_pause_overlay()
            
        pygame.display.flip()
        
    def _draw_grid(self, surface):
        """Draw world grid"""
        cell_size = int(self.config.cell_size * self.zoom)
        
        for x in range(self.world.size[0]):
            for y in range(self.world.size[1]):
                rect = pygame.Rect(
                    x * cell_size + self.camera_x,
                    y * cell_size + self.camera_y,
                    cell_size,
                    cell_size
                )
                pygame.draw.rect(surface, self.config.grid_color, rect, 1)
                
    def _draw_resources(self, surface):
        """Draw resources on grid"""
        cell_size = int(self.config.cell_size * self.zoom)
        
        for pos, resource in self.world.resources.resources.items():
            x, y = pos
            color = self.config.resource_colors.get(resource["type"], (100, 100, 100))
            
            # Fade color based on amount
            color = tuple(int(c * resource["amount"]) for c in color)
            
            rect = pygame.Rect(
                x * cell_size + self.camera_x + cell_size // 4,
                y * cell_size + self.camera_y + cell_size // 4,
                cell_size // 2,
                cell_size // 2
            )
            pygame.draw.rect(surface, color, rect)
            
    def _draw_agents(self, surface):
        """Draw agents on grid"""
        cell_size = int(self.config.cell_size * self.zoom)
        
        for agent in self.world.agents.values():
            x, y = agent.physical_state.position
            
            # Get color based on emotion
            color = self.config.emotion_colors.get(
                agent.emotional_state.current_emotion,
                (100, 100, 100)
            )
            
            # Fade based on energy
            energy_mult = max(0.3, agent.physical_state.energy)
            color = tuple(int(c * energy_mult) for c in color)
            
            # Draw agent
            center = (
                x * cell_size + self.camera_x + cell_size // 2,
                y * cell_size + self.camera_y + cell_size // 2
            )
            
            radius = int(cell_size * 0.4 * self.zoom)
            pygame.draw.circle(surface, color, center, radius)
            
            # Highlight selected agent
            if agent == self.selected_agent:
                pygame.draw.circle(surface, (255, 255, 100), center, radius + 3, 2)
                
            # Draw name if zoomed in
            if self.zoom > 1.2:
                name_surface = self.font_small.render(agent.name, True, self.config.text_color)
                surface.blit(name_surface, (center[0] - 20, center[1] + radius + 2))
                
    def _draw_relationships(self, surface):
        """Draw relationship connections"""
        if not self.selected_agent:
            return
            
        cell_size = int(self.config.cell_size * self.zoom)
        
        ax, ay = self.selected_agent.physical_state.position
        a_center = (
            ax * cell_size + self.camera_x + cell_size // 2,
            ay * cell_size + self.camera_y + cell_size // 2
        )
        
        for other_id, relationship_strength in self.selected_agent.relationships.items():
            if other_id in self.world.agents:
                other = self.world.agents[other_id]
                bx, by = other.physical_state.position
                b_center = (
                    bx * cell_size + self.camera_x + cell_size // 2,
                    by * cell_size + self.camera_y + cell_size // 2
                )
                
                # Color based on relationship strength
                strength = min(1.0, relationship_strength)
                color = (
                    int(100 + 155 * strength),
                    int(100 + 155 * strength),
                    100
                )
                
                pygame.draw.line(surface, color, a_center, b_center, max(1, int(strength * 3)))
                
    def _draw_language_connections(self, surface):
        """Draw shared language symbols between agents"""
        cell_size = int(self.config.cell_size * self.zoom)
        drawn_pairs = set()
        
        for agent1 in self.world.agents.values():
            if not agent1.language.symbols:
                continue
                
            for agent2 in self.world.agents.values():
                if agent1.id >= agent2.id or not agent2.language.symbols:
                    continue
                    
                # Find shared symbols
                shared = set(agent1.language.symbols.keys()) & set(agent2.language.symbols.keys())
                
                if shared and (agent1.id, agent2.id) not in drawn_pairs:
                    drawn_pairs.add((agent1.id, agent2.id))
                    
                    ax, ay = agent1.physical_state.position
                    bx, by = agent2.physical_state.position
                    
                    a_center = (
                        ax * cell_size + self.camera_x + cell_size // 2,
                        ay * cell_size + self.camera_y + cell_size // 2
                    )
                    b_center = (
                        bx * cell_size + self.camera_x + cell_size // 2,
                        by * cell_size + self.camera_y + cell_size // 2
                    )
                    
                    # Draw dotted line
                    self._draw_dotted_line(surface, a_center, b_center, (150, 200, 255), 2)
                    
    def _draw_dotted_line(self, surface, start, end, color, width):
        """Draw a dotted line between two points"""
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        distance = math.sqrt(dx**2 + dy**2)
        
        if distance == 0:
            return
            
        dots = int(distance / 10)
        for i in range(0, dots, 2):
            t = i / dots
            x = int(start[0] + dx * t)
            y = int(start[1] + dy * t)
            pygame.draw.circle(surface, color, (x, y), width)
            
    def _draw_communication_trails(self, surface):
        """Draw visual trails for recent communications"""
        for trail in self.communication_trails:
            if trail["alpha"] > 0:
                color = (*trail["color"], trail["alpha"])
                pygame.draw.line(surface, color[:3], trail["start"], trail["end"], 2)
                
    def _draw_sidebar(self):
        """Draw information sidebar"""
        sidebar_x = self.grid_width
        sidebar = pygame.Surface((self.config.sidebar_width, self.config.window_height))
        sidebar.fill((20, 20, 30))
        
        y_offset = 20
        
        # Title
        title = self.font_large.render("ANIMA", True, (255, 255, 255))
        sidebar.blit(title, (20, y_offset))
        y_offset += 40
        
        # World stats
        self._draw_text(sidebar, f"Time: {self.world.time}", 20, y_offset)
        y_offset += 25
        self._draw_text(sidebar, f"Population: {len(self.world.agents)}", 20, y_offset)
        y_offset += 25
        self._draw_text(sidebar, f"Cultures: {len(self.world.cultures)}", 20, y_offset)
        y_offset += 25
        self._draw_text(sidebar, f"Language Symbols: {len(self.world.languages)}", 20, y_offset)
        y_offset += 25
        self._draw_text(sidebar, f"Myths: {len(self.world.myths)}", 20, y_offset)
        y_offset += 40
        
        # Controls
        self._draw_text(sidebar, "Controls:", 20, y_offset, self.font_medium)
        y_offset += 25
        self._draw_text(sidebar, "SPACE - Pause/Resume", 30, y_offset, self.font_small)
        y_offset += 20
        self._draw_text(sidebar, "R - Show Relationships", 30, y_offset, self.font_small)
        y_offset += 20
        self._draw_text(sidebar, "L - Show Language Links", 30, y_offset, self.font_small)
        y_offset += 20
        self._draw_text(sidebar, "B - Show Beliefs", 30, y_offset, self.font_small)
        y_offset += 20
        self._draw_text(sidebar, "+/- Speed Up/Down", 30, y_offset, self.font_small)
        y_offset += 20
        self._draw_text(sidebar, f"Speed: {self.speed_multiplier:.1f}x", 30, y_offset, self.font_small)
        y_offset += 40
        
        # Selected agent info
        if self.selected_agent and self.selected_agent.physical_state.is_alive():
            self._draw_text(sidebar, f"Selected: {self.selected_agent.name}", 20, y_offset, self.font_medium)
            y_offset += 25
            
            agent = self.selected_agent
            self._draw_text(sidebar, f"Age: {agent.physical_state.age}", 30, y_offset, self.font_small)
            y_offset += 20
            self._draw_text(sidebar, f"Energy: {agent.physical_state.energy:.2f}", 30, y_offset, self.font_small)
            y_offset += 20
            self._draw_text(sidebar, f"Emotion: {agent.emotional_state.current_emotion.value}", 30, y_offset, self.font_small)
            y_offset += 20
            self._draw_text(sidebar, f"Symbols Known: {len(agent.language.symbols)}", 30, y_offset, self.font_small)
            y_offset += 20
            
            if self.show_beliefs and agent.beliefs.beliefs:
                self._draw_text(sidebar, "Beliefs:", 30, y_offset, self.font_small)
                y_offset += 20
                for belief in list(agent.beliefs.beliefs.keys())[:3]:
                    self._draw_text(sidebar, f"  - {belief}", 40, y_offset, self.font_small)
                    y_offset += 18
                y_offset += 10
                    
        # Event log
        if y_offset < self.config.window_height - 200:
            y_offset = self.config.window_height - 200
            
        self._draw_text(sidebar, "Recent Events:", 20, y_offset, self.font_medium)
        y_offset += 25
        
        for event in list(self.event_log)[-5:]:
            text = f"[{event['time']}] {event['summary']}"
            # Wrap text
            if len(text) > 45:
                text = text[:45] + "..."
            self._draw_text(sidebar, text, 30, y_offset, self.font_small)
            y_offset += 20
            
        self.screen.blit(sidebar, (sidebar_x, 0))
        
    def _draw_text(self, surface, text, x, y, font=None):
        """Helper to draw text"""
        if font is None:
            font = self.font_small
        text_surface = font.render(text, True, self.config.text_color)
        surface.blit(text_surface, (x, y))
        
    def _draw_pause_overlay(self):
        """Draw pause indicator"""
        overlay = pygame.Surface((200, 60))
        overlay.set_alpha(200)
        overlay.fill((50, 50, 60))
        
        pause_text = self.font_large.render("PAUSED", True, (255, 255, 100))
        overlay.blit(pause_text, (50, 15))
        
        self.screen.blit(overlay, (self.grid_width // 2 - 100, 20))
        
    def _summarize_event(self, event):
        """Create short summary of event"""
        if event.type == "communication":
            return f"{event.data.get('from', '?')} → {event.data.get('to', '?')}"
        elif event.type == "birth":
            return f"{event.data.get('child', '?')} born"
        elif event.type == "death":
            return f"{event.data.get('agent', '?')} died"
        elif event.type == "symbol_created":
            return f"'{event.data.get('symbol', '?')}' created"
        elif event.type == "myth_created":
            return f"Myth by {event.data.get('creator', '?')}"
        elif event.type == "culture_emerged":
            return f"Culture: {event.data.get('culture', '?')}"
        else:
            return event.type
            
    def add_communication_trail(self, from_agent, to_agent):
        """Add visual trail for communication"""
        cell_size = int(self.config.cell_size * self.zoom)
        
        ax, ay = from_agent.physical_state.position
        bx, by = to_agent.physical_state.position
        
        trail = {
            "start": (
                ax * cell_size + self.camera_x + cell_size // 2,
                ay * cell_size + self.camera_y + cell_size // 2
            ),
            "end": (
                bx * cell_size + self.camera_x + cell_size // 2,
                by * cell_size + self.camera_y + cell_size // 2
            ),
            "color": (100, 200, 255),
            "alpha": 255
        }
        
        self.communication_trails.append(trail)


# Main runner
def run_anima_with_visualization():
    """Run ANIMA with visualization"""
    # Create world
    config = SimulationConfig(
        world_size=(30, 30),
        initial_agents=15,
        resource_spawn_rate=0.05,
        time_per_tick=0.1,
        language_mutation_rate=0.05,
        death_threshold=0.0,
        reproduction_threshold=0.7
    )
    
    world = SimulationWorld(config)
    
    # Create and run visualizer
    viz_config = VisualizerConfig(
        cell_size=20,
        window_width=1400,
        window_height=900,
        sidebar_width=400
    )
    
    visualizer = ANIMAVisualizer(world, viz_config)
    visualizer.run()


if __name__ == "__main__":
    run_anima_with_visualization()