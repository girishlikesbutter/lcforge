# poc_scripts/poc_phase1_geometry_definition.py

import numpy as np
import quaternion  # For component orientation
from typing import List, Tuple, Optional  # For type hinting
from dataclasses import dataclass, field
from pathlib import Path

import trimesh # For 3D geometry and visualization
from trimesh.ray.ray_triangle import RayMeshIntersector

# --- Data structures (simplified from model_definitions.py for this script) ---
@dataclass
class BRDFMaterialProperties:
    diffuse_reflectivity: float = 0.18
    specular_reflectivity: float = 0.05
    phong_exponent: float = 10.0

@dataclass
class Facet:  # This will represent a TRIANGULAR facet
    vertices: np.ndarray  # Shape (3, 3) - 3 points, 3D coords for a triangle
    normal: np.ndarray  # Shape (3,)
    area: float
    material_properties: Optional[BRDFMaterialProperties] = None

@dataclass
class Component:
    name: str
    facets: List[Facet]  # List of triangular Facet objects
    relative_position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    relative_orientation: quaternion.quaternion = field(default_factory=lambda: np.quaternion(1, 0, 0, 0))
    default_material: Optional[BRDFMaterialProperties] = None

@dataclass
class Satellite:
    name: str
    components: List[Component]
    spice_id: Optional[int] = None

# --- Helper function for Triangulation and Facet Creation ---
def create_triangular_facets_from_conceptual_face(
        conceptual_face_vertices: List[List[float]],
        material: Optional[BRDFMaterialProperties] = None
) -> List[Facet]:
    n_vertices = len(conceptual_face_vertices)
    if n_vertices < 3:
        raise ValueError("A conceptual face must have at least 3 vertices.")

    face_vertices_np = np.array(conceptual_face_vertices, dtype=float)
    triangular_facets: List[Facet] = []

    v0 = face_vertices_np[0]
    for i in range(1, n_vertices - 1):
        v1 = face_vertices_np[i]
        v2 = face_vertices_np[i + 1]
        triangle_vertices = np.array([v0, v1, v2])
        edge1 = v1 - v0
        edge2 = v2 - v0
        normal = np.cross(edge1, edge2)
        norm_magnitude = np.linalg.norm(normal)
        if norm_magnitude < 1e-9:
            continue
        normal /= norm_magnitude
        area = 0.5 * norm_magnitude
        triangular_facets.append(
            Facet(vertices=triangle_vertices, normal=normal, area=area, material_properties=material)
        )
    return triangular_facets

# --- Function to create a Component from conceptual face definitions ---
def create_component_from_conceptual_faces(
        name: str,
        conceptual_faces_definitions: List[List[List[float]]],
        relative_position: Optional[List[float]] = None,
        relative_orientation_euler_deg: Optional[Tuple[float, float, float]] = None,
        default_material: Optional[BRDFMaterialProperties] = None
) -> Component:
    all_triangular_facets: List[Facet] = []
    for conceptual_face_verts in conceptual_faces_definitions:
        tri_facets = create_triangular_facets_from_conceptual_face(
            conceptual_face_verts,
            material=default_material
        )
        all_triangular_facets.extend(tri_facets)
    pos = np.array(relative_position) if relative_position else np.array([0.0, 0.0, 0.0])
    if relative_orientation_euler_deg:
        roll_rad = np.deg2rad(relative_orientation_euler_deg[0])
        pitch_rad = np.deg2rad(relative_orientation_euler_deg[1])
        yaw_rad = np.deg2rad(relative_orientation_euler_deg[2])
        orient = quaternion.from_euler_angles(roll_rad, pitch_rad, yaw_rad)
    else:
        orient = np.quaternion(1, 0, 0, 0)
    return Component(
        name=name,
        facets=all_triangular_facets,
        relative_position=pos,
        relative_orientation=orient,
        default_material=default_material
    )

# --- Convert Component to Trimesh object (Corrected Coloring) ---
def component_to_trimesh(component: Component) -> Optional[trimesh.Trimesh]:
    if not component.facets:
        print(f"Warning: Component '{component.name}' has no facets to convert to Trimesh.")
        return None
    component_vertices = []
    component_faces = []
    current_vertex_index = 0
    for facet_obj in component.facets:
        component_vertices.extend(facet_obj.vertices.tolist())
        component_faces.append([current_vertex_index,
                                current_vertex_index + 1,
                                current_vertex_index + 2])
        current_vertex_index += 3
    if not component_faces:
        print(f"Warning: No faces generated for Trimesh object of component '{component.name}'.")
        return None
    mesh = trimesh.Trimesh(vertices=np.array(component_vertices),
                           faces=np.array(component_faces))

    # --- MODIFICATION FOR DIAGNOSTICS: Conditionally Subdivide ---
    if "SolarPanel" not in component.name:  # <<-- ADDED THIS CONDITION
        if mesh.faces.shape[0] > 0:
            mesh = mesh.subdivide()
            print(f"    Subdivided component '{component.name}', new face count: {len(mesh.faces)}")
    else:
        print(f"    Skipped subdivision for component '{component.name}'.")
    # --- END MODIFICATION ---

    color_to_set = None
    if component.default_material:
        blue_val = int(min(1.0, component.default_material.diffuse_reflectivity) * 200) + 55
        color_to_set = [100, 150, blue_val, 200]  # RGBA
    else:
        color_to_set = [150, 150, 150, 200]  # Default grey
    mesh.visual.face_colors = color_to_set
    return mesh


# (The dataclass definitions and helper functions:
#  BRDFMaterialProperties, Facet, Component, Satellite,
#  create_triangular_facets_from_conceptual_face,
#  create_component_from_conceptual_faces,
#  and component_to_trimesh
#  should be defined above this block, as in your current script)

# (The dataclass definitions and helper functions:
#  BRDFMaterialProperties, Facet, Component, Satellite,
#  create_triangular_facets_from_conceptual_face,
#  create_component_from_conceptual_faces,
#  and component_to_trimesh
#  should be defined above this block, as in your current script)

# --- Main part of the script ---
if __name__ == "__main__":
    print("Phase 1: Defining Conceptual Faces, Triangulating, and Visualizing...")

    # Define materials
    mat_grey = BRDFMaterialProperties(diffuse_reflectivity=0.6, specular_reflectivity=0.1, phong_exponent=5)
    mat_solar_panel = BRDFMaterialProperties(diffuse_reflectivity=0.1, specular_reflectivity=0.8, phong_exponent=50)

    # Satellite Bus Dimensions (IS901-like, half-dimensions for vertex definition from center)
    H_bus = 2.8 / 2  # Half-height (along local Z of bus)
    W_bus = 3.5 / 2  # Half-width (along local Y of bus)
    D_bus = 5.6 / 2  # Half-depth (along local X of bus)
    bus_conceptual_faces = [
        [[D_bus, -W_bus, -H_bus], [D_bus, W_bus, -H_bus], [D_bus, W_bus, H_bus], [D_bus, -W_bus, H_bus]],
        [[-D_bus, -W_bus, H_bus], [-D_bus, W_bus, H_bus], [-D_bus, W_bus, -H_bus], [-D_bus, -W_bus, -H_bus]],
        [[D_bus, W_bus, -H_bus], [-D_bus, W_bus, -H_bus], [-D_bus, W_bus, H_bus], [D_bus, W_bus, H_bus]],
        [[D_bus, -W_bus, H_bus], [-D_bus, -W_bus, H_bus], [-D_bus, -W_bus, -H_bus], [D_bus, -W_bus, -H_bus]],
        [[D_bus, -W_bus, H_bus], [D_bus, W_bus, H_bus], [-D_bus, W_bus, H_bus], [-D_bus, -W_bus, H_bus]],
        [[D_bus, W_bus, -H_bus], [D_bus, -W_bus, -H_bus], [-D_bus, -W_bus, -H_bus], [-D_bus, W_bus, -H_bus]],
    ]
    bus_component = create_component_from_conceptual_faces(
        name="IS901_Bus_Manual",
        conceptual_faces_definitions=bus_conceptual_faces, default_material=mat_grey)
    print(f"Created component: {bus_component.name} with {len(bus_component.facets)} triangular facets.")

    total_span_along_z = 31.0
    bus_dimension_along_z = H_bus * 2
    strut_length = 1.5
    total_panel_assembly_length = total_span_along_z - bus_dimension_along_z
    single_panel_assembly_length = total_panel_assembly_length / 2
    panel_actual_length = single_panel_assembly_length - strut_length
    L_panel_half = panel_actual_length / 2
    W_panel_half = W_bus
    panel_conceptual_faces = [
        [[0, -W_panel_half, -L_panel_half], [0, W_panel_half, -L_panel_half],
         [0, W_panel_half, L_panel_half], [0, -W_panel_half, L_panel_half]]
    ]
    panel_center_offset_from_bus_origin_z = H_bus + strut_length + L_panel_half
    solar_panel_1 = create_component_from_conceptual_faces(
        name="IS901_SolarPanel_1_Manual", conceptual_faces_definitions=panel_conceptual_faces,
        relative_position=[0.0, 0.0, panel_center_offset_from_bus_origin_z],
        relative_orientation_euler_deg=(0, 0, 0), default_material=mat_solar_panel)
    print(f"Created component: {solar_panel_1.name} with {len(solar_panel_1.facets)} triangular facets.")
    solar_panel_2 = create_component_from_conceptual_faces(
        name="IS901_SolarPanel_2_Manual", conceptual_faces_definitions=panel_conceptual_faces,
        relative_position=[0.0, 0.0, -panel_center_offset_from_bus_origin_z],
        relative_orientation_euler_deg=(0, 0, 0), default_material=mat_solar_panel)
    print(f"Created component: {solar_panel_2.name} with {len(solar_panel_2.facets)} triangular facets.")

    is901_satellite_manual = Satellite(
        name="Intelsat_901_Manual_POC",
        components=[bus_component, solar_panel_1, solar_panel_2], spice_id=-126824)
    print(f"Created satellite: {is901_satellite_manual.name}")
    for comp in is901_satellite_manual.components:
        print(f"  Component '{comp.name}': {len(comp.facets)} triangular facets.")

    satellite_scene = trimesh.Scene()
    component_trimesh_objects = {}
    for comp_obj in is901_satellite_manual.components:
        local_mesh = component_to_trimesh(comp_obj)
        if local_mesh:
            transform_matrix = np.eye(4)
            transform_matrix[:3, :3] = quaternion.as_rotation_matrix(comp_obj.relative_orientation)
            transform_matrix[:3, 3] = comp_obj.relative_position
            transformed_mesh = local_mesh.copy()
            transformed_mesh.apply_transform(transform_matrix)
            satellite_scene.add_geometry(transformed_mesh, geom_name=comp_obj.name)
            component_trimesh_objects[comp_obj.name] = transformed_mesh
            print(f"Added {comp_obj.name} to the scene.")

            # --- MODIFICATION: Add mesh edges for visibility in initial scene ---
            if transformed_mesh.vertices.shape[0] > 0 and transformed_mesh.edges.shape[0] > 0:
                edges_path = trimesh.load_path(transformed_mesh.vertices[transformed_mesh.edges_unique])
                # You can set edge color if desired, e.g., by modifying edges_path.colors
                # For now, they will use a default path color (often black or grey)
                satellite_scene.add_geometry(edges_path, geom_name=comp_obj.name + "_edges")
            # --- END MODIFICATION ---


        else:
            print(f"Could not create Trimesh object for component {comp_obj.name}.")

    # --- ADD XYZ AXES TO THE FIRST SCENE (satellite_scene) ---
    if not satellite_scene.is_empty:
        try:
            axis_length = satellite_scene.scale * 0.75
            if axis_length < 1e-6:
                max_bound_abs = np.max(np.abs(satellite_scene.bounds)) if satellite_scene.bounds is not None else 1.0
                axis_length = max(max_bound_abs, 1.0) * 0.75
                if axis_length < 1e-6: axis_length = 10.0
        except Exception:
            axis_length = 10.0
    else:
        axis_length = 10.0
    x_axis_vertices_initial = np.array([[0, 0, 0], [axis_length, 0, 0]])
    x_axis_line_initial = trimesh.path.entities.Line(points=[0, 1])
    x_axis_path_initial = trimesh.path.Path3D(entities=[x_axis_line_initial], vertices=x_axis_vertices_initial,
                                              colors=[[255, 0, 0, 255]])
    satellite_scene.add_geometry(x_axis_path_initial, geom_name="X_Axis_Initial")
    y_axis_vertices_initial = np.array([[0, 0, 0], [0, axis_length, 0]])
    y_axis_line_initial = trimesh.path.entities.Line(points=[0, 1])
    y_axis_path_initial = trimesh.path.Path3D(entities=[y_axis_line_initial], vertices=y_axis_vertices_initial,
                                              colors=[[0, 255, 0, 255]])
    satellite_scene.add_geometry(y_axis_path_initial, geom_name="Y_Axis_Initial")
    z_axis_vertices_initial = np.array([[0, 0, 0], [0, 0, axis_length]])
    z_axis_line_initial = trimesh.path.entities.Line(points=[0, 1])
    z_axis_path_initial = trimesh.path.Path3D(entities=[z_axis_line_initial], vertices=z_axis_vertices_initial,
                                              colors=[[0, 0, 255, 255]])
    satellite_scene.add_geometry(z_axis_path_initial, geom_name="Z_Axis_Initial")
    print("Added XYZ axes to the initial scene.")

    if not satellite_scene.is_empty:
        print("\nDisplaying initial satellite model with XYZ axes...")
        print("Please close the Trimesh window to continue the script.")
        satellite_scene.show()
        print("Trimesh window closed.")
    else:
        print("\nInitial satellite scene is empty. Nothing to display.")

    print("\nPhase 1 (Geometry Definition and Visualization) is complete.")
    print("-------------------------------------------------------------")
    print("\nStarting Phase 2: SPICE Kinematics...")

    try:
        from src.spice.spice_handler import SpiceHandler
        from trimesh.ray.ray_triangle import RayMeshIntersector
    except ImportError as e:
        print(f"\nERROR: Could not import required module: {e}")
        exit()

    try:
        script_file_path = Path(__file__).resolve()
        project_root = script_file_path.parent.parent
        relative_metakernel_location = Path("data/spice_kernels/missions/dst-is901/INTELSAT_901-metakernel.tm")
        metakernel_full_path = project_root / relative_metakernel_location
        if not metakernel_full_path.exists():
            alt_path_from_script_dir = script_file_path.parent / ".." / "data" / "spice_kernels" / "missions" / "dst-is901" / "INTELSAT_901-metakernel.tm"
            if alt_path_from_script_dir.resolve().exists():
                metakernel_full_path = alt_path_from_script_dir.resolve()
            else:
                raise FileNotFoundError(f"Metakernel not found. Checked: {metakernel_full_path}")
        METAKERNEL_PATH = str(metakernel_full_path)
        print(f"Using dynamically determined metakernel path: {METAKERNEL_PATH}")
    except NameError:
        print("Warning: __file__ not defined. Using direct relative path from CWD for metakernel.")
        METAKERNEL_PATH = "data/spice_kernels/missions/dst-is901/INTELSAT_901-metakernel.tm"
        if not Path(METAKERNEL_PATH).exists(): raise FileNotFoundError(f"Metakernel not found: {METAKERNEL_PATH}")
        print(f"Using metakernel path (relative to CWD): {METAKERNEL_PATH}")

    utc_time_str = "2020-02-05T12:00:00"
    INERTIAL_FRAME = 'J2000'
    SATELLITE_BODY_FIXED_FRAME = 'IS901_BUS_FRAME'
    sun_direction_in_body_frame = None
    spice_handler_instance = None
    try:
        print(f"\nCreating SpiceHandler instance...")
        spice_handler_instance = SpiceHandler()
        print(f"Loading metakernel: {METAKERNEL_PATH}...")
        spice_handler_instance.load_metakernel(METAKERNEL_PATH)
        print("SpiceHandler ready.")
        print(f"Converting UTC '{utc_time_str}' to Ephemeris Time (ET)...")
        et = spice_handler_instance.utc_to_et(utc_time_str)
        print(f"ET: {et}")
        satellite_spice_id_str = str(is901_satellite_manual.spice_id)
        if not is901_satellite_manual.spice_id: raise ValueError("Satellite SPICE ID not set.")
        print(f"Getting Sun position relative to satellite '{satellite_spice_id_str}' in '{INERTIAL_FRAME}' frame...")
        sun_vector_inertial, _ = spice_handler_instance.get_body_position(
            target='SUN', et=et, observer=satellite_spice_id_str, frame=INERTIAL_FRAME)
        print(f"Sun vector in '{INERTIAL_FRAME}': {sun_vector_inertial}")
        print(f"Getting satellite orientation: '{INERTIAL_FRAME}' to '{SATELLITE_BODY_FIXED_FRAME}'...")
        inertial_to_body_matrix = spice_handler_instance.get_target_orientation(
            from_frame=INERTIAL_FRAME, to_frame=SATELLITE_BODY_FIXED_FRAME, et=et)
        sun_vector_body_unnormalized = np.dot(inertial_to_body_matrix, sun_vector_inertial)
        norm_sun_vector_body = np.linalg.norm(sun_vector_body_unnormalized)
        if norm_sun_vector_body < 1e-9: raise ValueError("Sun vector in body frame too small.")
        sun_direction_in_body_frame = sun_vector_body_unnormalized / norm_sun_vector_body
        print(
            f"Sun direction in satellite body-fixed frame ('{SATELLITE_BODY_FIXED_FRAME}'): {sun_direction_in_body_frame}")
    except Exception as e:
        print(f"\nAn error occurred during SPICE processing: {e}")
        sun_direction_in_body_frame = None
    finally:
        if spice_handler_instance is not None and hasattr(spice_handler_instance, 'unload_all_kernels'):
            print("Unloading all SPICE kernels via SpiceHandler...")
            spice_handler_instance.unload_all_kernels()
            print("SPICE kernels unloaded.")

    if sun_direction_in_body_frame is not None:
        print("\nPhase 2 (SPICE Kinematics) complete. Sun vector obtained.")
    else:
        print("\nPhase 2 (SPICE Kinematics) encountered an error. Sun vector not obtained.")
    print("-------------------------------------------------------------")

    # --- PHASE 3: SHADOWING ANALYSIS ---
    if sun_direction_in_body_frame is not None and component_trimesh_objects:
        print("\nStarting Phase 3: Shadowing Analysis...")

        print("  Concatenating component meshes for ray intersector...")
        all_satellite_meshes = list(component_trimesh_objects.values())
        if not all_satellite_meshes:
            print("  Error: No component meshes found to build the full satellite mesh.")
            full_satellite_trimesh = None
        else:
            full_satellite_trimesh = trimesh.util.concatenate(all_satellite_meshes)
            if not isinstance(full_satellite_trimesh, trimesh.Trimesh) or not full_satellite_trimesh.faces.shape[0] > 0:
                print(
                    f"  Error: Failed to create a valid concatenated mesh. Object type: {type(full_satellite_trimesh)}")
                full_satellite_trimesh = None

        intersector = None
        if full_satellite_trimesh:
            print(
                f"  Full satellite mesh created with {len(full_satellite_trimesh.vertices)} vertices and {len(full_satellite_trimesh.faces)} faces.")
            print("  Creating RayMeshIntersector (native triangle intersector)...")
            intersector = RayMeshIntersector(full_satellite_trimesh)
            print("  RayMeshIntersector (native) created.")
        else:
            print("  Skipping RayMeshIntersector creation as full_satellite_trimesh is invalid.")

        triangle_shadow_status = {}
        ray_direction_towards_sun = np.array(sun_direction_in_body_frame, dtype=float)

        if intersector:
            print("\n  Calculating shadow status for each triangle...")
            for comp_name, comp_mesh in component_trimesh_objects.items():
                print(f"    Processing component: {comp_name} ({len(comp_mesh.faces)} triangles)")
                for tri_idx in range(len(comp_mesh.faces)):
                    tri_centroid = comp_mesh.triangles_center[tri_idx]
                    tri_normal = comp_mesh.face_normals[tri_idx]
                    dot_product = np.dot(tri_normal, ray_direction_towards_sun)
                    if dot_product <= 1e-6:
                        triangle_shadow_status[(comp_name, tri_idx)] = 1.0
                        continue
                    # --- MODIFICATION: Adjust ray origin offset based on satellite scale ---
                    if full_satellite_trimesh is not None and hasattr(full_satellite_trimesh,
                                                                      'scale') and full_satellite_trimesh.scale > 1e-9:
                        epsilon_offset = full_satellite_trimesh.scale * 1e-4  # e.g., 0.01% of satellite scale
                    else:
                        epsilon_offset = 1e-5  # Fallback to previous fixed small offset

                    ray_origin = tri_centroid + tri_normal * epsilon_offset
                    # --- END MODIFICATION ---
                    # --- MODIFICATION: Use intersects_location and check hit distance ---
                    # Check if this ray (from surface towards Sun) hits any part of the satellite
                    locations, index_ray, index_tri_hit = intersector.intersects_location(
                        ray_origins=np.array([ray_origin]),
                        ray_directions=np.array([ray_direction_towards_sun]),
                        multiple_hits=False  # We only care about the first hit for occlusion
                    )

                    is_truly_occluded = False
                    if len(locations) > 0:
                        first_hit_location = locations[0]
                        distance_to_hit = np.linalg.norm(first_hit_location - ray_origin)

                        # Define a minimum distance for an occlusion to be considered "real"
                        # This helps ignore self-hits or hits on immediately adjacent co-planar faces.
                        # This threshold should be larger than the epsilon_offset used for ray_origin,
                        # but small relative to component sizes.
                        min_occlusion_distance = epsilon_offset * 5.0  # Example: 5x the ray origin offset
                        # Alternatively, a small absolute value like 1e-4 (0.1mm if units are meters)
                        # min_occlusion_distance = max(1e-4, epsilon_offset * 2.0)

                        if distance_to_hit > min_occlusion_distance:
                            is_truly_occluded = True
                        # else: The hit is too close, likely a self-intersection artifact on the same surface.
                        #      Treat as not occluded by a *distinct* part.

                    triangle_shadow_status[(comp_name, tri_idx)] = 1.0 if is_truly_occluded else 0.0
                    # --- END MODIFICATION ---
            print("  Shadow status calculation complete.")
        else:
            print("  Skipping shadow calculation as intersector is not available.")

        # --- Task 3.3: Visualize Shadowed Satellite ---
        if triangle_shadow_status:
            print("\n  Preparing scene for shadowed satellite visualization...")
            shadow_visualization_scene = trimesh.Scene()

            # This dictionary will store the approximate centers of the solar panels
            solar_panel_centers = {}

            for comp_name, original_comp_mesh in component_trimesh_objects.items():
                mesh_for_shadow_viz = original_comp_mesh.copy()
                if not isinstance(mesh_for_shadow_viz.visual, trimesh.visual.ColorVisuals):
                    mesh_for_shadow_viz.visual = trimesh.visual.ColorVisuals(mesh_for_shadow_viz)

                # --- MODIFICATION: Direct per-face color assignment for shadowing ---
                num_faces_in_viz_mesh = len(mesh_for_shadow_viz.faces)
                # Create an array to hold the new colors for each face
                explicit_face_colors = np.zeros((num_faces_in_viz_mesh, 4), dtype=np.uint8)

                # --- MODIFICATION: Define clear, opaque colors for shadowed/illuminated ---
                # Unshadowed: Grey, 100% opacity
                illuminated_color_rgba = [128, 128, 128, 255]  # Medium Grey, fully opaque
                # Shadowed: Red, 100% opacity
                shadowed_color_rgba = [255, 0, 0, 255]  # Bright Red, fully opaque
                # --- END MODIFICATION ---

                # The found_shadowed flag and its print statement can be removed if you no longer need that specific debug output.
                # If you keep it, it's fine.
                # found_shadowed = False

                for tri_idx in range(num_faces_in_viz_mesh):
                    status = triangle_shadow_status.get((comp_name, tri_idx), 0.0)

                    if status == 1.0:  # Shadowed
                        explicit_face_colors[tri_idx] = shadowed_color_rgba
                        # if not found_shadowed: found_shadowed = True # If keeping the flag
                    else:  # Illuminated
                        explicit_face_colors[tri_idx] = illuminated_color_rgba

                mesh_for_shadow_viz.visual.face_colors = explicit_face_colors
                shadow_visualization_scene.add_geometry(mesh_for_shadow_viz, geom_name=comp_name + "_shadowed")

                # --- MODIFICATION: Add mesh edges for visibility in shadow scene ---
                if mesh_for_shadow_viz.vertices.shape[0] > 0 and mesh_for_shadow_viz.edges.shape[0] > 0:
                    edges_path_shadow = trimesh.load_path(
                        mesh_for_shadow_viz.vertices[mesh_for_shadow_viz.edges_unique])
                    # Optional: set edge color, e.g., slightly brighter than the mesh if it's dark
                    # edges_path_shadow.colors = [[100, 100, 100, 150]] # Example: semi-transparent grey
                    shadow_visualization_scene.add_geometry(edges_path_shadow, geom_name=comp_name + "_shadowed_edges")
                # --- END MODIFICATION ---

                # Store the center of solar panels for drawing rays
                if "SolarPanel" in comp_name:  # Assuming solar panel names contain "SolarPanel"
                    solar_panel_centers[comp_name] = mesh_for_shadow_viz.bounds.mean(axis=0)

            # --- ADD XYZ AXES TO THE SHADOW SCENE ---
            if not shadow_visualization_scene.is_empty:
                try:
                    axis_length_shadow = shadow_visualization_scene.scale * 0.75
                    if axis_length_shadow < 1e-6:
                        max_bound_abs_shadow = np.max(np.abs(
                            shadow_visualization_scene.bounds)) if shadow_visualization_scene.bounds is not None else 1.0
                        axis_length_shadow = max(max_bound_abs_shadow, 1.0) * 0.75
                        if axis_length_shadow < 1e-6: axis_length_shadow = 10.0
                except Exception:
                    axis_length_shadow = 10.0

                x_axis_vertices_shadow = np.array([[0, 0, 0], [axis_length_shadow, 0, 0]])
                x_axis_line_shadow = trimesh.path.entities.Line(points=[0, 1])
                x_axis_path_shadow = trimesh.path.Path3D(entities=[x_axis_line_shadow],
                                                         vertices=x_axis_vertices_shadow, colors=[[255, 0, 0, 255]])
                shadow_visualization_scene.add_geometry(x_axis_path_shadow, geom_name="X_Axis_Shadow")

                y_axis_vertices_shadow = np.array([[0, 0, 0], [0, axis_length_shadow, 0]])
                y_axis_line_shadow = trimesh.path.entities.Line(points=[0, 1])
                y_axis_path_shadow = trimesh.path.Path3D(entities=[y_axis_line_shadow],
                                                         vertices=y_axis_vertices_shadow, colors=[[0, 255, 0, 255]])
                shadow_visualization_scene.add_geometry(y_axis_path_shadow, geom_name="Y_Axis_Shadow")

                z_axis_vertices_shadow = np.array([[0, 0, 0], [0, 0, axis_length_shadow]])
                z_axis_line_shadow = trimesh.path.entities.Line(points=[0, 1])
                z_axis_path_shadow = trimesh.path.Path3D(entities=[z_axis_line_shadow],
                                                         vertices=z_axis_vertices_shadow, colors=[[0, 0, 255, 255]])
                shadow_visualization_scene.add_geometry(z_axis_path_shadow, geom_name="Z_Axis_Shadow")
                print("  Added XYZ axes to the shadow scene.")
            # --- END OF ADDING XYZ AXES TO SHADOW SCENE ---

            # --- ADD SUN RAYS POINTING TO SOLAR PANEL CENTERS ---
            if not shadow_visualization_scene.is_empty and sun_direction_in_body_frame is not None:
                light_direction_vector = -np.array(sun_direction_in_body_frame,
                                                   dtype=float)  # Direction of incoming light

                ray_length_factor = 1.5  # How long the ray should appear (factor of scene scale)
                sun_ray_length = shadow_visualization_scene.scale * ray_length_factor
                if sun_ray_length < 1e-6: sun_ray_length = 15.0  # Fallback length

                for panel_name, panel_center in solar_panel_centers.items():
                    # Start the ray/arrow far away from the panel, along the light vector
                    ray_origin_solar_panel = panel_center - light_direction_vector * sun_ray_length

                    try:
                        sun_ray_arrow = trimesh.creation.arrow(
                            origin_point=ray_origin_solar_panel,
                            vector=light_direction_vector * sun_ray_length,  # Vector points towards panel center
                            sections=6
                        )
                        sun_ray_arrow.visual.face_colors = [255, 255, 0, 200]  # Yellow
                        shadow_visualization_scene.add_geometry(sun_ray_arrow, geom_name=f"SunRay_to_{panel_name}")
                    except AttributeError:
                        print(
                            f"    Warning: 'trimesh.creation.arrow' not found. Using a Path3D line for Sun ray to {panel_name}.")
                        ray_end_solar_panel = panel_center  # Ray terminates at panel center
                        ray_vertices = np.array([ray_origin_solar_panel, ray_end_solar_panel])
                        ray_line_entity = trimesh.path.entities.Line(points=[0, 1])
                        ray_path = trimesh.path.Path3D(entities=[ray_line_entity],
                                                       vertices=ray_vertices,
                                                       colors=[[255, 255, 0, 255]])  # Yellow
                        shadow_visualization_scene.add_geometry(ray_path, geom_name=f"SunRayPath_to_{panel_name}")
                print("  Added Sun rays pointing to solar panel centers.")

            # --- DISPLAY THE SHADOW SCENE ---
            if not shadow_visualization_scene.is_empty:
                print("\n  Displaying shadowed satellite model with axes and sun rays...")
                print("  Please close the Trimesh window to continue.")
                shadow_visualization_scene.show()
                print("  Shadow visualization window closed.")
            else:
                print("  Shadow visualization scene is empty (even after attempting to add components).")

        else:  # This else belongs to "if triangle_shadow_status:"
            print("\n  No shadow status data to visualize (triangle_shadow_status is empty).")

        # --- Task 3.4: Output Shadow Data ---
        if triangle_shadow_status:
            print("\n  Shadow Status per Triangle (Component, Triangle Index):")
            for (comp_name, tri_idx), status in triangle_shadow_status.items():
                status_str = "Shadowed" if status == 1.0 else "Illuminated"
                print(f"    Component '{comp_name}', Triangle {tri_idx}: {status_str} ({status})")

        print("\nPhase 3 (Shadowing Analysis) complete.")

    else:
        print(
            "\nCannot proceed to Phase 3 as Sun vector or component meshes were not properly determined in earlier phases.")

    print("-------------------------------------------------------------")
    print("End of POC script.")