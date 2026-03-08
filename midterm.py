import numpy as np
import matplotlib.pyplot as plt
import random
import math
from shapely.geometry import Polygon, Point, LineString, box
from shapely.ops import unary_union
from shapely import affinity
from original_eco import OriginalECO as solver
from optimizer import FloatVar

class GeometryGenerator:
    @staticmethod
    def generate_polyomino_room(num_tiles=15):
        occupied = {(0,0)}
        candidates = [(0,1), (0,-1), (1,0), (-1,0)]

        count = 1
        while count < num_tiles:
            if not candidates: break

            idx = random.randrange(len(candidates))
            cx, cy = candidates.pop(idx)
            
            if (cx, cy) in occupied:
                continue
                
            occupied.add((cx, cy))
            count += 1

            for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                nx, ny = cx + dx, cy + dy
                if (nx, ny) not in occupied:
                    candidates.append((nx, ny))
                    
        tiles = [box(x, y, x+1, y+1) for x, y in occupied]
        merged = unary_union(tiles)
        
        if merged.geom_type == 'MultiPolygon':
            merged = max(merged.geoms, key=lambda p: p.area)
            
        final_poly = Polygon(merged.exterior)
        return final_poly.simplify(0.05, preserve_topology=True)

    @staticmethod
    def generate_walls(polygon, num_walls=1):
        walls = []
        minx, miny, maxx, maxy = polygon.bounds
        
        for _ in range(num_walls):
            for attempt in range(50):
                x1 = random.uniform(minx, maxx)
                y1 = random.uniform(miny, maxy)
                if not polygon.contains(Point(x1, y1)): continue
                
                angle = random.uniform(0, 2 * math.pi)
                length = random.uniform(0.1, 0.3) * (maxx - minx)
                
                x2 = x1 + length * math.cos(angle)
                y2 = y1 + length * math.sin(angle)
                
                wall = LineString([(x1, y1), (x2, y2)])

                if polygon.contains(wall):
                    walls.append(wall)
                    break
        return walls

    @staticmethod
    def polygon_from_list(points):
        return Polygon(points)
    
    @staticmethod
    def walls_from_list(wall_segments):
        return [LineString(seg) for seg in wall_segments]

class ObstacleAGP:
    def __init__(self,polygon,walls, n_guards=3, grid_res=20):
        self.n_guards = n_guards
        self.grid_res = grid_res
        
        raw_room = polygon
        raw_walls = walls
        
        # print("Normalizing Coordinates...")
        self.room, self.walls = self._normalize(raw_room, raw_walls)
        
        self.sensors = []
        xs = np.linspace(0, 1, grid_res)
        ys = np.linspace(0, 1, grid_res)
        
        for x in xs:
            for y in ys:
                p = Point(x, y)
                if not self.room.contains(p):
                    continue
                
                too_close = False
                for w in self.walls:
                    if p.distance(w) < 0.01:
                        too_close = True
                        break
                
                if not too_close:
                    self.sensors.append(p)
                    
        print(f"Map Ready: {len(self.sensors)} sensors.")

    def _normalize(self, polygon, walls):
        minx, miny, maxx, maxy = polygon.bounds
        max_dim = max(maxx - minx, maxy - miny)
        scale = 0.9 / max_dim 
        
        def transform(geom):
            t = affinity.translate(geom, -minx, -miny)
            s = affinity.scale(t, xfact=scale, yfact=scale, origin=(0,0))
            return s

        poly_s = transform(polygon)
        
        walls_s = [transform(w) for w in walls]
        
        cx, cy = poly_s.centroid.x, poly_s.centroid.y
        off_x, off_y = 0.5 - cx, 0.5 - cy
        
        poly_final = affinity.translate(poly_s, off_x, off_y)
        walls_final = [affinity.translate(w, off_x, off_y) for w in walls_s]
        
        return poly_final, walls_final

    def objective_function(self, solution):
        guards = []
        penalty = 0.0
        for i in range(0, len(solution), 2):
            p = Point(solution[i], solution[i+1])
            if not self.room.contains(p): 
                penalty += -1000.0
            else :
                guards.append(p)
            
        visible_count = 0
        for s in self.sensors:
            seen = False
            for g in guards:
                ray = LineString([g, s])
                
                blocked_by_wall = False
                for w in self.walls:
                    if ray.intersects(w):
                        blocked_by_wall = True
                        break
                if blocked_by_wall: continue
                
                if not self.room.contains(ray):
                    if ray.crosses(self.room.boundary): continue
                    
                seen = True
                break
            if seen: visible_count += 1
            
        return (visible_count / len(self.sensors)) + penalty

    def render(self, solution, filename="solution.jpg"):
        guards = [Point(solution[i], solution[i+1]) for i in range(0, len(solution), 2)]
        
        colors = []
        for s in self.sensors:
            seen = False
            for g in guards:
                ray = LineString([g, s])
                
                blocked = False
                for w in self.walls:
                    if ray.intersects(w):
                        blocked = True
                        break
                if blocked: continue

                if self.room.contains(ray) or not ray.crosses(self.room.boundary):
                    seen = True
                    break
            colors.append('gold' if seen else 'black')

        fig, ax = plt.subplots(figsize=(8, 8))
    
        x, y = self.room.exterior.xy
        ax.fill(x, y, fc='#cccccc', ec='black', linewidth=2, label="Walls")
        
        for i, w in enumerate(self.walls):
            wx, wy = w.xy
            label = "Obstacles" if i == 0 else None
            ax.plot(wx, wy, 'r-', linewidth=3, label=label)
        
        xs = [s.x for s in self.sensors]
        ys = [s.y for s in self.sensors]
        ax.scatter(xs, ys, c=colors, s=25, marker='s', alpha=1)
        
        gx = [g.x for g in guards]
        gy = [g.y for g in guards]
        ax.scatter(gx, gy, c='blue', s=250, marker='*', edgecolors='white', zorder=10, label="Guards")
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"Generated Map (Tiles={len(self.room.exterior.coords)//4}, Walls={len(self.walls)})")
        ax.legend(loc='upper right')
        plt.savefig(filename)
        print(f"Visualization saved to {filename}")

if __name__ == "__main__":
    N_GUARDS = 2
    MAP_COMPLEXITY = 50
    NUM_WALLS = 6
    
    polygon_points = [
    (0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (2.0, 0.0), (2.0, 1.0),
    (3.0, 1.0), (3.0, 2.0), (2.0, 2.0), (2.0, 3.0), (1.0, 3.0),
    (1.0, 2.0), (0.0, 2.0), (0.0, 1.0)
]
    wall_segments = [((0.2, 0.2), (0.2, 2.8))]

    # polygon = GeometryGenerator.generate_polyomino_room(MAP_COMPLEXITY)
    # walls = GeometryGenerator.generate_walls(polygon,NUM_WALLS)
    
    polygon = GeometryGenerator.polygon_from_list(polygon_points)
    walls = GeometryGenerator.walls_from_list(wall_segments)

    problem = ObstacleAGP(
        n_guards=N_GUARDS, 
        grid_res=40, 
        polygon=polygon,
        walls=walls
    )
    
    lb = [0.0] * (N_GUARDS * 2)
    ub = [1.0] * (N_GUARDS * 2)
    
    problem_dict = {
        "obj_func": problem.objective_function,
        "bounds": FloatVar(lb=lb, ub=ub),
        "minmax": "max",
        "log_to": "console"
    }
    
    print("Running Optimizer...")
    optimizer = solver(epoch=200, pop_size=100)
    best_agent = optimizer.solve(problem_dict)
    
    print(f"Final Cost: {best_agent.target.fitness:.4f}")
    problem.render(best_agent.solution)
    print(best_agent.solution)