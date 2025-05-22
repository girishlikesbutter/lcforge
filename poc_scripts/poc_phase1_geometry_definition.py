# poc_scripts/poc_phase1_geometry_definition.py

import numpy as np
import quaternion  # For component orientation
from typing import List, Tuple, Optional  # For type hinting
from dataclasses import dataclass, field
from pathlib import Path

import trimesh  # For 3D geometry and visualization
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


# --- Convert Component to Trimesh object ---
def component_to_trimesh(component: Component, subdivide: bool = True) -> Optional[trimesh.Trimesh]:
    if not component.facets:
        # print(f"Warning: Component '{component.name}' has no facets to convert to Trimesh.") # Kept for critical warnings
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
        # print(f"Warning: No faces generated for Trimesh object of component '{component.name}'.") # Kept
        return None

    try:
        mesh = trimesh.Trimesh(vertices=np.array(component_vertices),
                               faces=np.array(component_faces),
                               process=False)

        # print(f"    Explicitly calling mesh.process() for '{component.name}'...") # Can be removed
        mesh.process()

        # if not mesh.is_watertight: # Informational, can be removed for cleaner output
        # print(f"    INFO: Initial mesh for '{component.name}' is not watertight before merge/fix (this is common).")

        mesh.merge_vertices()
        mesh.fix_normals(multibody=True)

        # if mesh.is_watertight: # Informational
        # print(f"    INFO: Mesh for '{component.name}' IS watertight after processing and fixes.")
        # else:
        # print(f"    WARNING: Mesh for '{component.name}' is STILL NOT watertight after processing and fixes.") # Keep warning

    except Exception as e:
        print(f"Error creating/processing Trimesh object for component '{component.name}': {e}")  # Keep error
        return None

    if subdivide:
        if mesh.faces.shape[0] > 0:
            # print(f"    Attempting to subdivide component '{component.name}' (current faces: {len(mesh.faces)})") # Can remove
            try:
                mesh = mesh.subdivide()
                # print(f"    Subdivided component '{component.name}', new face count: {len(mesh.faces)}") # Can remove
            except Exception as e:
                print(
                    f"    Error during subdivision of '{component.name}': {e}. Using unsubdivided mesh.")  # Keep error
    # else:  # Informational, can remove
    # print(f"    Subdivision explicitly disabled for component '{component.name}'.")

    color_to_set = None
    if component.default_material:
        blue_val = int(min(1.0, component.default_material.diffuse_reflectivity) * 200) + 55
        color_to_set = [100, 150, blue_val, 200]
    else:
        color_to_set = [150, 150, 150, 200]

    if not isinstance(mesh.visual, trimesh.visual.ColorVisuals):
        mesh.visual = trimesh.visual.ColorVisuals(mesh)
    mesh.visual.face_colors = color_to_set

    return mesh


# --- Main part of the script ---
if __name__ == "__main__":
    print("--- Phase 1: Geometry Definition & Initial Visualization ---")

    SUBDIVIDE_BUS_GEOMETRY = True

    mat_grey = BRDFMaterialProperties(diffuse_reflectivity=0.6, specular_reflectivity=0.1, phong_exponent=5)
    mat_solar_panel = BRDFMaterialProperties(diffuse_reflectivity=0.1, specular_reflectivity=0.8, phong_exponent=50)

    H_bus = 2.8 / 2;
    W_bus = 3.5 / 2;
    D_bus = 5.6 / 2
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
    # print(f"Created component: {bus_component.name} with {len(bus_component.facets)} triangular facets (from conceptual).") # Simplified

    total_span_along_z = 31.0;
    bus_dimension_along_z = H_bus * 2;
    strut_length = 1.5
    total_panel_assembly_length = total_span_along_z - bus_dimension_along_z
    single_panel_assembly_length = total_panel_assembly_length / 2
    panel_actual_length = single_panel_assembly_length - strut_length
    L_panel_half = panel_actual_length / 2;
    W_panel_half = W_bus
    panel_conceptual_faces = [
        [[0, -W_panel_half, -L_panel_half], [0, W_panel_half, -L_panel_half],
         [0, W_panel_half, L_panel_half], [0, -W_panel_half, L_panel_half]]
    ]
    panel_center_offset_from_bus_origin_z = H_bus + strut_length + L_panel_half
    solar_panel_1 = create_component_from_conceptual_faces(
        name="IS901_SolarPanel_1_Manual", conceptual_faces_definitions=panel_conceptual_faces,
        relative_position=[0.0, 0.0, panel_center_offset_from_bus_origin_z],
        default_material=mat_solar_panel)
    solar_panel_2 = create_component_from_conceptual_faces(
        name="IS901_SolarPanel_2_Manual", conceptual_faces_definitions=panel_conceptual_faces,
        relative_position=[0.0, 0.0, -panel_center_offset_from_bus_origin_z],
        default_material=mat_solar_panel)

    is901_satellite_manual = Satellite(
        name="Intelsat_901_Manual_POC",
        components=[bus_component, solar_panel_1, solar_panel_2], spice_id=-126824)
    print(
        f"Satellite model '{is901_satellite_manual.name}' defined with {len(is901_satellite_manual.components)} components.")

    satellite_scene = trimesh.Scene()
    component_trimesh_objects = {}

    for comp_obj in is901_satellite_manual.components:
        do_subdivide_this_one = False
        if "Bus" in comp_obj.name and SUBDIVIDE_BUS_GEOMETRY:
            do_subdivide_this_one = True
        elif "SolarPanel" in comp_obj.name:
            do_subdivide_this_one = False

        local_mesh = component_to_trimesh(comp_obj, subdivide=do_subdivide_this_one)
        if local_mesh:
            transform_matrix = np.eye(4)
            transform_matrix[:3, :3] = quaternion.as_rotation_matrix(comp_obj.relative_orientation)
            transform_matrix[:3, 3] = comp_obj.relative_position

            local_mesh.apply_transform(transform_matrix)
            component_trimesh_objects[comp_obj.name] = local_mesh

            mesh_for_initial_scene = local_mesh.copy()
            satellite_scene.add_geometry(mesh_for_initial_scene, geom_name=comp_obj.name)
            # print(f"Added {comp_obj.name} to initial scene (faces: {len(mesh_for_initial_scene.faces)}).") # Can remove

            if mesh_for_initial_scene.vertices.shape[0] > 0 and mesh_for_initial_scene.edges.shape[0] > 0:
                edges_path = trimesh.load_path(mesh_for_initial_scene.vertices[mesh_for_initial_scene.edges_unique])
                satellite_scene.add_geometry(edges_path, geom_name=comp_obj.name + "_edges")
        else:
            print(f"Could not create Trimesh object for component {comp_obj.name}.")  # Keep error

    if not satellite_scene.is_empty:
        try:
            axis_length = satellite_scene.scale * 0.75
            if axis_length < 1e-6: max_bound_abs = np.max(
                np.abs(satellite_scene.bounds)) if satellite_scene.bounds is not None else 1.0; axis_length = max(
                max_bound_abs, 1.0) * 0.75;
            if axis_length < 1e-6: axis_length = 10.0
        except Exception:
            axis_length = 10.0
        x_axis_path_initial = trimesh.path.Path3D(entities=[trimesh.path.entities.Line(points=[0, 1])],
                                                  vertices=np.array([[0, 0, 0], [axis_length, 0, 0]]),
                                                  colors=[[255, 0, 0, 255]]);
        satellite_scene.add_geometry(x_axis_path_initial, geom_name="X_Axis_Initial")
        y_axis_path_initial = trimesh.path.Path3D(entities=[trimesh.path.entities.Line(points=[0, 1])],
                                                  vertices=np.array([[0, 0, 0], [0, axis_length, 0]]),
                                                  colors=[[0, 255, 0, 255]]);
        satellite_scene.add_geometry(y_axis_path_initial, geom_name="Y_Axis_Initial")
        z_axis_path_initial = trimesh.path.Path3D(entities=[trimesh.path.entities.Line(points=[0, 1])],
                                                  vertices=np.array([[0, 0, 0], [0, 0, axis_length]]),
                                                  colors=[[0, 0, 255, 255]]);
        satellite_scene.add_geometry(z_axis_path_initial, geom_name="Z_Axis_Initial")

    if not satellite_scene.is_empty:
        print("Displaying initial satellite model (flat shading). Close window to continue.")
        satellite_scene.show(smooth=False)
    else:
        print("Initial satellite scene is empty.")

    print("--- Phase 1 Complete ---")
    print("\n--- Phase 2: SPICE Kinematics ---")
    try:
        from src.spice.spice_handler import SpiceHandler
    except ImportError as e:
        print(f"ERROR: SpiceHandler import: {e}"); exit()
    script_file_path = Path(__file__).resolve();
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
    utc_time_str = "2020-02-05T15:30:00";
    INERTIAL_FRAME = 'J2000';
    SATELLITE_BODY_FIXED_FRAME = 'IS901_BUS_FRAME'
    sun_direction_in_body_frame = None;
    spice_handler_instance = None
    try:
        # print(f"Loading metakernel: {METAKERNEL_PATH}...") # Can remove
        spice_handler_instance = SpiceHandler();
        spice_handler_instance.load_metakernel(METAKERNEL_PATH)
        et = spice_handler_instance.utc_to_et(utc_time_str)
        if not is901_satellite_manual.spice_id: raise ValueError("Satellite SPICE ID not set.")
        satellite_spice_id_str = str(is901_satellite_manual.spice_id)
        sun_vector_inertial, _ = spice_handler_instance.get_body_position(target='SUN', et=et,
                                                                          observer=satellite_spice_id_str,
                                                                          frame=INERTIAL_FRAME)
        inertial_to_body_matrix = spice_handler_instance.get_target_orientation(from_frame=INERTIAL_FRAME,
                                                                                to_frame=SATELLITE_BODY_FIXED_FRAME,
                                                                                et=et)
        sun_vector_body_unnormalized = np.dot(inertial_to_body_matrix, sun_vector_inertial)
        norm_sun_vector_body = np.linalg.norm(sun_vector_body_unnormalized)
        if norm_sun_vector_body < 1e-9: raise ValueError("Sun vector in body frame too small.")
        sun_direction_in_body_frame = sun_vector_body_unnormalized / norm_sun_vector_body
        print(
            f"Sun direction in satellite body-fixed frame ('{SATELLITE_BODY_FIXED_FRAME}'): {sun_direction_in_body_frame}")
        # print(f"  (This vector points FROM the satellite TOWARDS the Sun)") # Already clear from above
    except Exception as e:
        print(f"SPICE error: {e}")  # Keep error
    finally:
        if spice_handler_instance is not None and hasattr(spice_handler_instance, 'unload_all_kernels'):
            spice_handler_instance.unload_all_kernels()
            # print("SPICE kernels unloaded.") # Can remove
    if sun_direction_in_body_frame is None: print("Sun vector not obtained. Exiting."); exit()
    print("--- Phase 2 Complete ---")
    print("\n--- Phase 3: Shadowing Analysis ---")
    if sun_direction_in_body_frame is not None and component_trimesh_objects:

        all_satellite_meshes_for_intersector = [
            mesh.copy() for mesh in component_trimesh_objects.values() if mesh is not None
        ]
        # print(f"    DEBUG: Intersector mesh now uses ALL components.") # Can remove

        full_satellite_trimesh_for_intersector = None
        if all_satellite_meshes_for_intersector:
            full_satellite_trimesh_for_intersector = trimesh.util.concatenate(all_satellite_meshes_for_intersector)
            if not isinstance(full_satellite_trimesh_for_intersector, trimesh.Trimesh) or not \
            full_satellite_trimesh_for_intersector.faces.shape[0] > 0:
                print("  Error: Failed to create concatenated mesh for intersector.");
                full_satellite_trimesh_for_intersector = None  # Keep error

        intersector = None
        if full_satellite_trimesh_for_intersector:
            print(
                f"Intersector Mesh: {len(full_satellite_trimesh_for_intersector.faces)} faces, {len(full_satellite_trimesh_for_intersector.vertices)} vertices.")  # Keep key info
            # print(f"    Is watertight: {full_satellite_trimesh_for_intersector.is_watertight}") # Informational, can remove
            # print(f"    Volume: {full_satellite_trimesh_for_intersector.volume if full_satellite_trimesh_for_intersector.is_watertight else 'N/A (not watertight)'}") # Can remove
            # print(f"    Scale: {full_satellite_trimesh_for_intersector.scale}") # Can remove
            # print(f"    Centroid: {full_satellite_trimesh_for_intersector.centroid}") # Can remove
            # print(f"    Bounds: {full_satellite_trimesh_for_intersector.bounds}") # Can remove

            try:
                intersector = RayMeshIntersector(full_satellite_trimesh_for_intersector)
                # print("  RayMeshIntersector created successfully.") # Can remove
            except Exception as e:
                print(f"  ERROR creating RayMeshIntersector: {e}")  # Keep error
                intersector = None
        else:
            print(
                "  Skipping RayMeshIntersector creation as full_satellite_trimesh_for_intersector is invalid.")  # Keep

        triangle_shadow_status = {}
        ray_direction_from_satellite_to_sun = np.array(sun_direction_in_body_frame, dtype=float)

        if intersector:
            print("Calculating shadow status for each component's triangles...")  # Keep
            for comp_name, mesh_for_shadow_calc in component_trimesh_objects.items():
                if mesh_for_shadow_calc is None: continue

                num_triangles_in_comp = len(mesh_for_shadow_calc.faces)
                comp_back_culled_count = 0
                comp_rays_cast_count = 0
                comp_no_hits_count = 0
                comp_too_close_hits_count = 0
                comp_occluded_hits_count = 0

                # print(f"    Processing component for shadow: {comp_name} ({num_triangles_in_comp} triangles)") # Can remove
                mesh_for_shadow_calc.fix_normals(multibody=True)

                for tri_idx in range(num_triangles_in_comp):
                    tri_centroid = mesh_for_shadow_calc.triangles_center[tri_idx]
                    tri_normal = mesh_for_shadow_calc.face_normals[tri_idx]
                    dot_product = np.dot(tri_normal, ray_direction_from_satellite_to_sun)

                    current_status = 0.0

                    if dot_product <= 1e-6:
                        comp_back_culled_count += 1
                        triangle_shadow_status[(comp_name, tri_idx)] = 1.0
                        continue

                    comp_rays_cast_count += 1

                    effective_scale = full_satellite_trimesh_for_intersector.scale if full_satellite_trimesh_for_intersector else 1.0
                    if effective_scale < 1e-9: effective_scale = 1.0

                    epsilon_offset_val = effective_scale * 1e-3
                    ray_origin = tri_centroid + tri_normal * epsilon_offset_val

                    min_occlusion_dist = epsilon_offset_val * 5.0

                    # Removed the per-ray DEBUG RAY CASTING prints here

                    locations, index_ray, index_tri_hit = intersector.intersects_location(
                        ray_origins=np.array([ray_origin]),
                        ray_directions=np.array([ray_direction_from_satellite_to_sun]),
                        multiple_hits=False
                    )

                    is_truly_occluded = False
                    hit_was_too_close = False

                    if len(locations) > 0:
                        first_hit_location = locations[0]
                        distance_to_hit = np.linalg.norm(first_hit_location - ray_origin)
                        if distance_to_hit > min_occlusion_dist:
                            is_truly_occluded = True
                            comp_occluded_hits_count += 1
                        else:
                            hit_was_too_close = True
                            comp_too_close_hits_count += 1
                    else:
                        comp_no_hits_count += 1

                    if is_truly_occluded:
                        current_status = 1.0
                    elif hit_was_too_close:
                        current_status = 2.0
                    else:
                        current_status = 0.0

                    triangle_shadow_status[(comp_name, tri_idx)] = current_status

                # Keep these per-component stats as they are very informative
                print(f"  Stats for '{comp_name}':")
                print(f"    - Total Triangles: {num_triangles_in_comp}")
                print(f"    - Back-face (Marked Shadowed): {comp_back_culled_count}")
                print(f"    - Rays Cast (Front-facing): {comp_rays_cast_count}")
                print(f"    - Cast Rays with NO Hits (Marked Lit): {comp_no_hits_count}")
                print(f"    - Cast Rays with TOO CLOSE Hits (Marked Yellow): {comp_too_close_hits_count}")
                print(f"    - Cast Rays with OCCLUDING Hits (Marked Red): {comp_occluded_hits_count}")

            # print("  Shadow status calculation complete.") # Can remove, implied by next step
        else:
            print("  Skipping shadow calculation as intersector is not available.")  # Keep

        # --- Task 3.3: Visualize Shadowed Satellite (REVISED) ---
        if triangle_shadow_status:
            # print("\n  Preparing scene for shadowed satellite visualization (REVISED)...") # Can remove
            shadow_visualization_scene = trimesh.Scene()

            illuminated_color_rgba = [0, 255, 0, 255]
            shadowed_color_rgba = [255, 0, 0, 255]
            ignored_hit_color_rgba = [255, 255, 0, 255]

            component_centers_for_sunray = {}

            for comp_name, mesh_that_was_shadow_calculated in component_trimesh_objects.items():
                if mesh_that_was_shadow_calculated is None: continue
                component_centers_for_sunray[comp_name] = mesh_that_was_shadow_calculated.bounds.mean(axis=0)

                mesh_to_display_in_scene = mesh_that_was_shadow_calculated.copy()
                num_faces_in_display_mesh = len(mesh_to_display_in_scene.faces)

                expected_faces_from_calc_mesh = len(mesh_that_was_shadow_calculated.faces)
                if num_faces_in_display_mesh != expected_faces_from_calc_mesh:  # Keep this important warning
                    print(
                        f"    !!!! VISUALIZATION WARNING for {comp_name}: Display mesh face count ({num_faces_in_display_mesh}) "
                        f"differs from calculation mesh face count ({expected_faces_from_calc_mesh}) !!!!")

                if not isinstance(mesh_to_display_in_scene.visual, trimesh.visual.ColorVisuals):
                    mesh_to_display_in_scene.visual = trimesh.visual.ColorVisuals(mesh_to_display_in_scene)
                current_mesh_face_colors = np.zeros((num_faces_in_display_mesh, 4), dtype=np.uint8)

                for face_idx in range(num_faces_in_display_mesh):
                    status = triangle_shadow_status.get((comp_name, face_idx), 0.0)
                    if status == 1.0:
                        current_mesh_face_colors[face_idx] = shadowed_color_rgba
                    elif status == 2.0:
                        current_mesh_face_colors[face_idx] = ignored_hit_color_rgba
                    else:
                        current_mesh_face_colors[face_idx] = illuminated_color_rgba

                mesh_to_display_in_scene.visual.face_colors = current_mesh_face_colors
                shadow_visualization_scene.add_geometry(mesh_to_display_in_scene, geom_name=comp_name + "_shadowed_viz")

                if mesh_to_display_in_scene.vertices.shape[0] > 0 and mesh_to_display_in_scene.edges.shape[0] > 0:
                    edges_path_s = trimesh.load_path(
                        mesh_to_display_in_scene.vertices[mesh_to_display_in_scene.edges_unique])
                    shadow_visualization_scene.add_geometry(edges_path_s, geom_name=comp_name + "_shadowed_edges")

            if not shadow_visualization_scene.is_empty:
                try:
                    axis_length_shadow = shadow_visualization_scene.scale * 0.75
                except:
                    axis_length_shadow = 10.0
                if axis_length_shadow < 1e-6: axis_length_shadow = 10.0
                x_ax_s = trimesh.path.Path3D(entities=[trimesh.path.entities.Line(points=[0, 1])],
                                             vertices=np.array([[0, 0, 0], [axis_length_shadow, 0, 0]]),
                                             colors=[[255, 0, 0, 255]]);
                shadow_visualization_scene.add_geometry(x_ax_s, "X_Axis_S")
                y_ax_s = trimesh.path.Path3D(entities=[trimesh.path.entities.Line(points=[0, 1])],
                                             vertices=np.array([[0, 0, 0], [0, axis_length_shadow, 0]]),
                                             colors=[[0, 255, 0, 255]]);
                shadow_visualization_scene.add_geometry(y_ax_s, "Y_Axis_S")
                z_ax_s = trimesh.path.Path3D(entities=[trimesh.path.entities.Line(points=[0, 1])],
                                             vertices=np.array([[0, 0, 0], [0, 0, axis_length_shadow]]),
                                             colors=[[0, 0, 255, 255]]);
                shadow_visualization_scene.add_geometry(z_ax_s, "Z_Axis_S")

                if sun_direction_in_body_frame is not None:
                    light_direction_vector = -np.array(sun_direction_in_body_frame, dtype=float)
                    ray_display_length = axis_length_shadow * 1.0
                    for comp_name, comp_center in component_centers_for_sunray.items():
                        ray_start_point = comp_center - light_direction_vector * ray_display_length
                        arrow_vector = comp_center - ray_start_point
                        try:
                            sun_ray_arrow = trimesh.creation.arrow(
                                origin_point=ray_start_point, vector=arrow_vector, sections=4,
                                radius_shaft=0.02 * axis_length_shadow, radius_head=0.05 * axis_length_shadow,
                                head_length=0.1 * axis_length_shadow)
                            sun_ray_arrow.visual.face_colors = [255, 255, 0, 200]
                            shadow_visualization_scene.add_geometry(sun_ray_arrow, geom_name=f"SunRay_to_{comp_name}")
                        except AttributeError:
                            ray_path = trimesh.path.Path3D(
                                entities=[trimesh.path.entities.Line(points=[0, 1])],
                                vertices=np.array([ray_start_point, comp_center]),
                                colors=[[255, 255, 0, 255]])
                            shadow_visualization_scene.add_geometry(ray_path, geom_name=f"SunRayLine_to_{comp_name}")
                    # print("  Added Sun Ray indicators pointing to component centers.") # Can remove

            if not shadow_visualization_scene.is_empty:
                print("Displaying final shadowed model (flat shading). Close window to exit.")  # Keep
                shadow_visualization_scene.show(smooth=False)
            else:
                print("Shadow visualization scene is empty.")  # Keep
        else:
            print("No shadow status data to visualize.")  # Keep

        if triangle_shadow_status:
            print("\nFinal Shadow Status Summary:")  # Keep
            total_shadowed_1 = sum(1 for stat in triangle_shadow_status.values() if stat == 1.0)
            total_ignored_2 = sum(1 for stat in triangle_shadow_status.values() if stat == 2.0)
            total_lit_0 = sum(1 for stat in triangle_shadow_status.values() if stat == 0.0)
            print(
                f"  Overall: Lit (Green): {total_lit_0}, Shadowed (Red): {total_shadowed_1}, Ignored Self-Hit (Yellow): {total_ignored_2}")  # Keep

            # Keep this selective print for detailed checking if needed, but can be commented out for normal runs
            # print("  Detailed status for selected triangles:")
            # for (cn, ti), stat in sorted(triangle_shadow_status.items()):
            #     s_str = "Lit (0.0)" if stat == 0.0 else ("Shadowed (1.0)" if stat == 1.0 else "IgnoredSelfHit (2.0)")
            #     if "Bus" in cn and ti < 15 :
            #          print(f"    {cn}, Triangle {ti}: {s_str}")
            #     elif "Bus" not in cn :
            #          print(f"    {cn}, Triangle {ti}: {s_str}")

        print("--- Phase 3 Complete ---")
    else:
        print("Cannot proceed to Phase 3 (Sun vector or component meshes missing).")  # Keep
    print("\n--- End of POC Script ---")

