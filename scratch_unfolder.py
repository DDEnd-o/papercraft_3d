import sys
import numpy as np
from shapely.geometry import Polygon

# Override overlap check to print
import modules.unfolder as unfolder

original_unfold = unfolder._unfold_island

def _debug_unfold(mesh, start_face, neighbors, adj, adj_edges, island_id, unvisited):
    print(f"Unfolding island {island_id} from face {start_face}")
    panels = original_unfold(mesh, start_face, neighbors, adj, adj_edges, island_id, unvisited)
    print(f"Island {island_id} finished with {len(panels)} panels")
    
    # Check for self-intersections in the final panels
    placed = [Polygon(p.verts_2d) for p in panels]
    overlap_count = 0
    for i, p1 in enumerate(placed):
        for j, p2 in enumerate(placed):
            if i >= j: continue
            area = p1.intersection(p2).area
            if area > 1e-7 and (area > p1.area * 0.01 or area > p2.area * 0.01):
                print(f"OVERLAP DETECTED IN FINAL RESULTS: {panels[i].face_idx} and {panels[j].face_idx}. Area: {area}")
                overlap_count += 1
    print(f"Total overlaps in island {island_id}: {overlap_count}")
    return panels

unfolder._unfold_island = _debug_unfold

sys.argv = ["main.py", "--file", r"D:\MyProject\PapercraftApp\image_demo\Flexi-Rex-improved.stl"]
import main
main.main()
