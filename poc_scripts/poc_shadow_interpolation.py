#!/usr/bin/env python3
# poc_scripts/poc_shadow_interpolation.py

import os
import sys
import logging
import numpy as np
import quaternion
import trimesh
from typing import List, Dict, Tuple, Optional
import pickle
import imageio.v2 as imageio
import shutil
# Matplotlib imports
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Ensure the src directory is in the Python path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.append(PROJECT_ROOT)

from src.models.model_definitions import Satellite, Component, Facet, BRDFMaterialProperties
from src.spice.spice_handler import SpiceHandler
from src.core.common_types import Quaternion as CommonQuaternion, Vector3D

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants and Configuration based on User's POC Snippet ---
# Bus Dimensions
D_BUS_HALF = 5.6 / 2
W_BUS_HALF = 3.5 / 2
H_BUS_HALF = 2.8 / 2
BUS_DIMENSIONS = np.array([D_BUS_HALF * 2, W_BUS_HALF * 2, H_BUS_HALF * 2])

# Solar Panel Calculations
TOTAL_SPAN_ALONG_Z = 31.0
BUS_DIMENSION_ALONG_Z = H_BUS_HALF * 2
STRUT_LENGTH = 1.5
TOTAL_PANEL_ASSEMBLY_LENGTH = TOTAL_SPAN_ALONG_Z - BUS_DIMENSION_ALONG_Z
SINGLE_PANEL_ASSEMBLY_LENGTH = TOTAL_PANEL_ASSEMBLY_LENGTH / 2
PANEL_ACTUAL_LENGTH = SINGLE_PANEL_ASSEMBLY_LENGTH - STRUT_LENGTH
PANEL_ACTUAL_WIDTH = W_BUS_HALF * 2
SOLAR_PANEL_THICKNESS = 0.1

SOLAR_PANEL_BOX_EXTENTS = np.array([SOLAR_PANEL_THICKNESS, PANEL_ACTUAL_WIDTH, PANEL_ACTUAL_LENGTH])
L_PANEL_HALF = PANEL_ACTUAL_LENGTH / 2

BUS_REL_POS = np.array([0.0, 0.0, 0.0])
BUS_REL_ORIENT_NPQ = np.quaternion(1, 0, 0, 0)

PANEL_CENTER_OFFSET_Z = H_BUS_HALF + STRUT_LENGTH + L_PANEL_HALF
SP1_REL_POS = np.array([0.0, 0.0, PANEL_CENTER_OFFSET_Z])
SP1_REL_ORIENT_NPQ = np.quaternion(1, 0, 0, 0)
SP2_REL_POS = np.array([0.0, 0.0, -PANEL_CENTER_OFFSET_Z])
SP2_REL_ORIENT_NPQ = np.quaternion(1, 0, 0, 0)

# --- Database Configuration ---
SHADOW_DATABASE_FILENAME = "shadow_database.pkl"
NUM_AZIMUTH_SAMPLES = 8
NUM_ELEVATION_SAMPLES = 5

# --- Animation Configuration ---
ANIMATION_START_UTC = "2020-02-05T12:00:00"
ANIMATION_END_UTC = "2020-02-05T16:00:00"
ANIMATION_TIME_STEP_SEC = 600  # 10 minutes
ANIMATION_METAKERNEL_PATH = os.path.join(PROJECT_ROOT, "data", "spice_kernels", "missions", "dst-is901",
                                         "INTELSAT_901-metakernel.tm")
ANIMATION_OUTPUT_FILENAME = "shadow_animation_mpl_fixed_cam_axes.gif"
ANIMATION_FRAMES_DIR = os.path.join(SCRIPT_DIR, "animation_frames_mpl")
FRAMES_PER_SECOND_GIF = 5


# --- Helper Functions ---
def create_triangular_facets_from_conceptual_face(
        conceptual_face_vertices: List[np.ndarray],
        component_name: str,
        face_id_prefix: str,
        material_props: BRDFMaterialProperties
) -> List[Facet]:
    facets = []
    if len(conceptual_face_vertices) < 3:
        logging.warning(f"Conceptual face for {component_name} has < 3 vertices. Skipping.")
        return facets
    v0 = np.array(conceptual_face_vertices[0])
    for i in range(1, len(conceptual_face_vertices) - 1):
        v1 = np.array(conceptual_face_vertices[i])
        v2 = np.array(conceptual_face_vertices[i + 1])
        triangle_vertices_np = [v0, v1, v2]
        vec1 = v1 - v0
        vec2 = v2 - v0
        normal = np.cross(vec1, vec2)
        norm_mag = np.linalg.norm(normal)
        if norm_mag < 1e-9:
            logging.warning(f"Degenerate triangle in {component_name} {face_id_prefix}_{i - 1}. Normal is near zero.")
            normal = np.array([0.0, 0.0, 1.0])
        else:
            normal /= norm_mag
        area = 0.5 * norm_mag
        facet_id = f"{face_id_prefix}_{i - 1}"
        facets.append(Facet(
            id=facet_id,
            vertices=triangle_vertices_np,
            normal=normal,
            area=area,
            material_properties=material_props
        ))
    return facets


def create_box_trimesh(dimensions: np.ndarray, transform: Optional[np.ndarray] = None) -> trimesh.Trimesh:
    box = trimesh.creation.box(extents=dimensions, transform=transform)
    return box


def spherical_to_cartesian(azimuth_rad: float, elevation_rad: float, r: float = 1.0) -> np.ndarray:
    x = r * np.cos(elevation_rad) * np.cos(azimuth_rad)
    y = r * np.cos(elevation_rad) * np.sin(azimuth_rad)
    z = r * np.sin(elevation_rad)
    return np.array([x, y, z])


def define_satellite_model() -> Tuple[Satellite, Dict[str, trimesh.Trimesh]]:
    logging.info("Defining satellite model based on user's POC snippet...")
    components = []
    component_local_meshes: Dict[str, trimesh.Trimesh] = {}
    default_brdf = BRDFMaterialProperties(r_d=0.6, r_s=0.2, n_phong=10.0)
    bus_name = "IS901_Bus"
    bus_mesh_local = create_box_trimesh(BUS_DIMENSIONS)
    bus_mesh_local.metadata['name'] = bus_name
    component_local_meshes[bus_name] = bus_mesh_local
    bus_conceptual_faces_defs = {
        "X+": [[D_BUS_HALF, -W_BUS_HALF, -H_BUS_HALF], [D_BUS_HALF, W_BUS_HALF, -H_BUS_HALF],
               [D_BUS_HALF, W_BUS_HALF, H_BUS_HALF], [D_BUS_HALF, -W_BUS_HALF, H_BUS_HALF]],
        "X-": [[-D_BUS_HALF, -W_BUS_HALF, H_BUS_HALF], [-D_BUS_HALF, W_BUS_HALF, H_BUS_HALF],
               [-D_BUS_HALF, W_BUS_HALF, -H_BUS_HALF], [-D_BUS_HALF, -W_BUS_HALF, -H_BUS_HALF]],
        "Y+": [[D_BUS_HALF, W_BUS_HALF, -H_BUS_HALF], [-D_BUS_HALF, W_BUS_HALF, -H_BUS_HALF],
               [-D_BUS_HALF, W_BUS_HALF, H_BUS_HALF], [D_BUS_HALF, W_BUS_HALF, H_BUS_HALF]],
        "Y-": [[D_BUS_HALF, -W_BUS_HALF, H_BUS_HALF], [-D_BUS_HALF, -W_BUS_HALF, H_BUS_HALF],
               [-D_BUS_HALF, -W_BUS_HALF, -H_BUS_HALF], [D_BUS_HALF, -W_BUS_HALF, -H_BUS_HALF]],
        "Z+": [[D_BUS_HALF, -W_BUS_HALF, H_BUS_HALF], [D_BUS_HALF, W_BUS_HALF, H_BUS_HALF],
               [-D_BUS_HALF, W_BUS_HALF, H_BUS_HALF], [-D_BUS_HALF, -W_BUS_HALF, H_BUS_HALF]],
        "Z-": [[D_BUS_HALF, W_BUS_HALF, -H_BUS_HALF], [D_BUS_HALF, -W_BUS_HALF, -H_BUS_HALF],
               [-D_BUS_HALF, -W_BUS_HALF, -H_BUS_HALF], [-D_BUS_HALF, W_BUS_HALF, -H_BUS_HALF]],
    }
    bus_facets: List[Facet] = []
    for face_id, vertices_list_of_lists in bus_conceptual_faces_defs.items():
        vertices_list_of_nparrays = [np.array(v_list) for v_list in vertices_list_of_lists]
        bus_facets.extend(
            create_triangular_facets_from_conceptual_face(vertices_list_of_nparrays, bus_name, face_id, default_brdf))
    bus_component = Component(
        id=f"{bus_name}_comp_id", name=bus_name, facets=bus_facets, mesh_path=None,
        relative_position=BUS_REL_POS,
        relative_orientation=CommonQuaternion(BUS_REL_ORIENT_NPQ.w, BUS_REL_ORIENT_NPQ.x, BUS_REL_ORIENT_NPQ.y,
                                              BUS_REL_ORIENT_NPQ.z)
    )
    components.append(bus_component)
    solar_panel_names = ["IS901_SolarPanel_1", "IS901_SolarPanel_2"]
    solar_panel_rel_pos_map = {solar_panel_names[0]: SP1_REL_POS, solar_panel_names[1]: SP2_REL_POS}
    solar_panel_rel_orient_map_npq = {solar_panel_names[0]: SP1_REL_ORIENT_NPQ,
                                      solar_panel_names[1]: SP2_REL_ORIENT_NPQ}
    sp_x_half, sp_y_half, sp_z_half = SOLAR_PANEL_BOX_EXTENTS[0] / 2, SOLAR_PANEL_BOX_EXTENTS[1] / 2, \
                                      SOLAR_PANEL_BOX_EXTENTS[2] / 2
    sp_conceptual_faces_defs = {
        "X+": [[sp_x_half, -sp_y_half, -sp_z_half], [sp_x_half, sp_y_half, -sp_z_half],
               [sp_x_half, sp_y_half, sp_z_half], [sp_x_half, -sp_y_half, sp_z_half]],
        "X-": [[-sp_x_half, -sp_y_half, sp_z_half], [-sp_x_half, sp_y_half, sp_z_half],
               [-sp_x_half, sp_y_half, -sp_z_half], [-sp_x_half, -sp_y_half, -sp_z_half]],
        "Y+": [[sp_x_half, sp_y_half, -sp_z_half], [-sp_x_half, sp_y_half, -sp_z_half],
               [-sp_x_half, sp_y_half, sp_z_half], [sp_x_half, sp_y_half, sp_z_half]],
        "Y-": [[sp_x_half, -sp_y_half, sp_z_half], [-sp_x_half, -sp_y_half, sp_z_half],
               [-sp_x_half, -sp_y_half, -sp_z_half], [sp_x_half, -sp_y_half, -sp_z_half]],
        "Z+": [[sp_x_half, -sp_y_half, sp_z_half], [sp_x_half, sp_y_half, sp_z_half],
               [-sp_x_half, sp_y_half, sp_z_half], [-sp_x_half, -sp_y_half, sp_z_half]],
        "Z-": [[sp_x_half, sp_y_half, -sp_z_half], [sp_x_half, -sp_y_half, -sp_z_half],
               [-sp_x_half, -sp_y_half, -sp_z_half], [-sp_x_half, sp_y_half, -sp_z_half]],
    }
    for sp_name in solar_panel_names:
        sp_mesh_local = create_box_trimesh(SOLAR_PANEL_BOX_EXTENTS)
        sp_mesh_local.metadata['name'] = sp_name
        component_local_meshes[sp_name] = sp_mesh_local
        sp_facets: List[Facet] = []
        for face_id, vertices_list_of_lists in sp_conceptual_faces_defs.items():
            vertices_list_of_nparrays = [np.array(v_list) for v_list in vertices_list_of_lists]
            sp_facets.extend(create_triangular_facets_from_conceptual_face(vertices_list_of_nparrays, sp_name, face_id,
                                                                           default_brdf))
        current_rel_pos, current_rel_orient_npq = solar_panel_rel_pos_map[sp_name], solar_panel_rel_orient_map_npq[
            sp_name]
        sp_component = Component(
            id=f"{sp_name}_comp_id", name=sp_name, facets=sp_facets, mesh_path=None,
            relative_position=current_rel_pos,
            relative_orientation=CommonQuaternion(current_rel_orient_npq.w, current_rel_orient_npq.x,
                                                  current_rel_orient_npq.y, current_rel_orient_npq.z)
        )
        components.append(sp_component)
    satellite_obj = Satellite(
        id="SAT_IS901_POC_ID", name="Intelsat_901_POC_Corrected", components=components,
        body_frame_name="IS901_BUS_FRAME"
    )
    logging.info(f"Satellite model '{satellite_obj.name}' defined with {len(components)} components.")
    for comp in components:
        logging.info(f"  Component '{comp.name}' (ID: {comp.id}) with {len(comp.facets)} facets.")
        assert component_local_meshes[comp.name] is not None, f"Mesh not created for {comp.name}"
    return satellite_obj, component_local_meshes


def get_component_meshes_in_body_frame(
        satellite_obj: Satellite, component_local_meshes: Dict[str, trimesh.Trimesh],
        subdivide: bool = False
) -> Dict[str, trimesh.Trimesh]:
    component_body_meshes: Dict[str, trimesh.Trimesh] = {}
    for component_model in satellite_obj.components:
        local_mesh = component_local_meshes[component_model.name].copy()
        if subdivide:
            logging.debug(f"Subdividing mesh for component {component_model.name} for intersector.")
            local_mesh = local_mesh.subdivide()

        position = component_model.relative_position
        orientation_q_common = component_model.relative_orientation
        orientation_q_np = np.quaternion(orientation_q_common.w, orientation_q_common.x, orientation_q_common.y,
                                         orientation_q_common.z)
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = quaternion.as_rotation_matrix(orientation_q_np)
        transform_matrix[:3, 3] = position
        body_mesh = local_mesh.apply_transform(transform_matrix)
        component_body_meshes[component_model.name] = body_mesh
    return component_body_meshes


def calculate_shadow_status(
        satellite_obj: Satellite, intersector_mesh: trimesh.Trimesh, sun_direction_body_frame: np.ndarray
) -> Dict[str, List[float]]:
    shadow_statuses_by_component: Dict[str, List[float]] = {comp.name: [] for comp in satellite_obj.components}
    if not isinstance(intersector_mesh, trimesh.Trimesh) or not intersector_mesh.vertices.size > 0:
        logging.error("Intersector mesh is invalid or empty.")
        for comp_model in satellite_obj.components:
            shadow_statuses_by_component[comp_model.name] = [0.0] * len(comp_model.facets)
        return shadow_statuses_by_component
    logging.debug("Ensuring intersector mesh is processed for shadow calculation.")
    processed_intersector_mesh = intersector_mesh.copy().process()
    if not processed_intersector_mesh.is_watertight:
        logging.warning("Combined intersector mesh is not watertight after processing.")
    ray_intersector = trimesh.ray.ray_triangle.RayMeshIntersector(processed_intersector_mesh)
    effective_scale = np.linalg.norm(
        processed_intersector_mesh.bounding_box.extents) if not processed_intersector_mesh.is_empty else 1.0
    if effective_scale < 1e-6: effective_scale = 1.0
    epsilon_offset_val = effective_scale * 1e-4
    min_true_occlusion_dist = epsilon_offset_val * 2.0

    for comp_model in satellite_obj.components:
        comp_shadow_status: List[float] = []
        comp_pos_vec, comp_orient_q_common = comp_model.relative_position, comp_model.relative_orientation
        comp_orient_q_np = np.quaternion(comp_orient_q_common.w, comp_orient_q_common.x, comp_orient_q_common.y,
                                         comp_orient_q_common.z)
        comp_local_to_body_rot_matrix = quaternion.as_rotation_matrix(comp_orient_q_np)
        for facet_idx, facet in enumerate(comp_model.facets):
            facet_normal_local, facet_vertices_local_np = facet.normal, facet.vertices
            facet_normal_body = comp_local_to_body_rot_matrix @ facet_normal_local
            facet_centroid_local = np.mean(facet_vertices_local_np, axis=0)
            facet_centroid_body = comp_local_to_body_rot_matrix @ facet_centroid_local + comp_pos_vec
            dot_product = np.dot(facet_normal_body, sun_direction_body_frame)
            if dot_product <= 1e-6:
                comp_shadow_status.append(1.0)
                continue

            ray_origin = facet_centroid_body + facet_normal_body * epsilon_offset_val
            ray_direction = sun_direction_body_frame

            locations, index_ray, index_tri = ray_intersector.intersects_location(
                ray_origins=np.array([ray_origin]), ray_directions=np.array([ray_direction]), multiple_hits=False)

            is_occluded = False
            if len(locations) > 0:
                hit_distance = np.linalg.norm(locations[0] - ray_origin)
                if hit_distance > min_true_occlusion_dist:
                    is_occluded = True

            if is_occluded:
                comp_shadow_status.append(1.0)
            else:
                comp_shadow_status.append(0.0)
        shadow_statuses_by_component[comp_model.name] = comp_shadow_status
    return shadow_statuses_by_component


def generate_shadowing_database(
        satellite_obj: Satellite, intersector_mesh: trimesh.Trimesh,
        num_az_samples: int = NUM_AZIMUTH_SAMPLES, num_el_samples: int = NUM_ELEVATION_SAMPLES
) -> List[Tuple[np.ndarray, Dict[str, List[float]]]]:
    logging.info(
        f"Generating shadowing database with {num_az_samples} azimuth samples and {num_el_samples} elevation samples...")
    shadow_database: List[Tuple[np.ndarray, Dict[str, List[float]]]] = []
    azimuths_rad = np.linspace(0, 2 * np.pi, num_az_samples, endpoint=False)
    elevations_rad = np.linspace(-np.pi / 2, np.pi / 2, num_el_samples)
    total_samples, current_sample = len(azimuths_rad) * len(elevations_rad), 0
    for az_rad in azimuths_rad:
        for el_rad in elevations_rad:
            current_sample += 1
            sun_direction_body = spherical_to_cartesian(az_rad, el_rad)
            sun_direction_body /= (np.linalg.norm(sun_direction_body) + 1e-9)
            logging.debug(
                f"Processing sample {current_sample}/{total_samples}: Az={np.rad2deg(az_rad):.1f} deg, El={np.rad2deg(el_rad):.1f} deg, SunVec={sun_direction_body}")
            shadow_status = calculate_shadow_status(satellite_obj, intersector_mesh, sun_direction_body)
            shadow_database.append((sun_direction_body, shadow_status))
            if current_sample % 10 == 0 or current_sample == total_samples:
                logging.info(f"Database generation progress: {current_sample}/{total_samples} samples processed.")
    logging.info(f"Shadowing database generation complete. Total entries: {len(shadow_database)}")
    return shadow_database


# --- Phase 3: Animation - Interpolation Function ---
def get_interpolated_shadows(
        actual_sun_body_frame: np.ndarray,
        database: List[Tuple[np.ndarray, Dict[str, List[float]]]]
) -> Optional[Dict[str, List[float]]]:
    """
    Retrieves shadow status from the database using nearest-neighbor interpolation.
    """
    if not database:
        logging.warning("Shadow database is empty. Cannot interpolate.")
        return None

    actual_sun_body_frame_normalized = actual_sun_body_frame / (np.linalg.norm(actual_sun_body_frame) + 1e-9)

    best_match_shadow_status = None
    max_dot_product = -np.inf  # Initialize with a very small number

    for db_sun_vector, db_shadow_status in database:
        # db_sun_vector should already be normalized from database generation
        dot_product = np.dot(actual_sun_body_frame_normalized, db_sun_vector)
        if dot_product > max_dot_product:
            max_dot_product = dot_product
            best_match_shadow_status = db_shadow_status

    if best_match_shadow_status is None:
        logging.warning(
            f"No suitable match found in database for sun vector {actual_sun_body_frame_normalized}. Using first entry as fallback.")
        return database[0][1]  # Fallback to the first entry's shadow status

    return best_match_shadow_status


def visualize_scene_with_shadows(
        satellite_obj: Satellite, component_local_meshes: Dict[str, trimesh.Trimesh],
        shadow_statuses_by_component: Dict[str, List[float]],
        transform_satellite_body_to_world: Optional[np.ndarray] = None,
        sun_direction_world: Optional[np.ndarray] = None,
        camera_elev_azim: Tuple[float, float] = (30, -60),
        C_j2000_to_body_rotation: Optional[np.ndarray] = None,
        show_scene: bool = True, scene_title: str = "Satellite Shadow Visualization",
        save_path: Optional[str] = None
) -> None:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    if transform_satellite_body_to_world is None:
        transform_satellite_body_to_world = np.eye(4)

    all_verts_world = []

    for comp_model in satellite_obj.components:
        local_mesh_trimesh = component_local_meshes[comp_model.name]
        comp_pos = comp_model.relative_position
        comp_orient_q_common = comp_model.relative_orientation
        comp_orient_q_np = np.quaternion(comp_orient_q_common.w, comp_orient_q_common.x,
                                         comp_orient_q_common.y, comp_orient_q_common.z)
        comp_to_body_transform = np.eye(4)
        comp_to_body_transform[:3, :3] = quaternion.as_rotation_matrix(comp_orient_q_np)
        comp_to_body_transform[:3, 3] = comp_pos
        final_transform = transform_satellite_body_to_world @ comp_to_body_transform
        world_vertices = trimesh.transform_points(local_mesh_trimesh.vertices.copy(), final_transform)
        all_verts_world.extend(world_vertices)
        shadows = shadow_statuses_by_component.get(comp_model.name)
        num_defined_facets = len(comp_model.facets)
        num_mesh_faces = len(local_mesh_trimesh.faces)

        if shadows is None or len(shadows) != num_defined_facets:
            logging.warning(f"Shadow status list length mismatch for {comp_model.name}. Defaulting to grey.")
            face_colors_for_plot = ['gray'] * num_mesh_faces
        elif num_defined_facets != num_mesh_faces:
            logging.warning(
                f"Defined Facet count ({num_defined_facets}) != trimesh face count ({num_mesh_faces}) for {comp_model.name}. Coloring grey.")
            face_colors_for_plot = ['gray'] * num_mesh_faces
        else:
            face_colors_for_plot = []
            # Shadowed is red, illuminated is green
            for s_status in shadows:
                if s_status == 1.0:
                    face_colors_for_plot.append('red')
                else:
                    face_colors_for_plot.append('lime')

        poly_collection = Poly3DCollection(world_vertices[local_mesh_trimesh.faces],
                                           facecolors=face_colors_for_plot,
                                           edgecolor='k', linewidths=0.3, alpha=0.9)
        ax.add_collection3d(poly_collection)

    plot_max_range = 5.0
    if all_verts_world:
        all_verts_world_np = np.array(all_verts_world)
        min_coords = all_verts_world_np.min(axis=0)
        max_coords = all_verts_world_np.max(axis=0)

        mid_x = (max_coords[0] + min_coords[0]) / 2.0
        mid_y = (max_coords[1] + min_coords[1]) / 2.0
        mid_z = (max_coords[2] + min_coords[2]) / 2.0

        ranges = max_coords - min_coords
        plot_max_range = max(ranges.max() / 2.0, 5.0)

        ax.set_xlim(mid_x - plot_max_range, mid_x + plot_max_range)
        ax.set_ylim(mid_y - plot_max_range, mid_y + plot_max_range)
        ax.set_zlim(mid_z - plot_max_range, mid_z + plot_max_range)
    else:
        ax.set_xlim([-plot_max_range, plot_max_range]);
        ax.set_ylim([-plot_max_range, plot_max_range]);
        ax.set_zlim([-plot_max_range, plot_max_range])

    ax.set_xlabel("X (Body Frame)")
    ax.set_ylabel("Y (Body Frame)")
    ax.set_zlabel("Z (Body Frame)")
    ax.set_title(scene_title)
    ax.view_init(elev=camera_elev_azim[0], azim=camera_elev_azim[1])
    ax.grid(True)

    # Plot Body Frame Axes (RGB for XYZ) from origin
    body_axis_len = plot_max_range * 0.5
    origin_plot = transform_satellite_body_to_world[:3, 3]

    ax.quiver(origin_plot[0], origin_plot[1], origin_plot[2], 1, 0, 0, length=body_axis_len, color='red',
              label='X_Body', arrow_length_ratio=0.1)
    ax.quiver(origin_plot[0], origin_plot[1], origin_plot[2], 0, 1, 0, length=body_axis_len, color='green',
              label='Y_Body', arrow_length_ratio=0.1)
    ax.quiver(origin_plot[0], origin_plot[1], origin_plot[2], 0, 0, 1, length=body_axis_len, color='blue',
              label='Z_Body', arrow_length_ratio=0.1)

    if C_j2000_to_body_rotation is not None:
        j2000_axis_len = plot_max_range * 0.4
        x_j2000_basis, y_j2000_basis, z_j2000_basis = np.array([1., 0, 0]), np.array([0, 1., 0]), np.array([0, 0, 1.])
        x_j2000_in_body = C_j2000_to_body_rotation @ x_j2000_basis
        y_j2000_in_body = C_j2000_to_body_rotation @ y_j2000_basis
        z_j2000_in_body = C_j2000_to_body_rotation @ z_j2000_basis
        ax.quiver(origin_plot[0], origin_plot[1], origin_plot[2], x_j2000_in_body[0], x_j2000_in_body[1],
                  x_j2000_in_body[2], length=j2000_axis_len, color='cyan', label='X_J2000', arrow_length_ratio=0.1)
        ax.quiver(origin_plot[0], origin_plot[1], origin_plot[2], y_j2000_in_body[0], y_j2000_in_body[1],
                  y_j2000_in_body[2], length=j2000_axis_len, color='magenta', label='Y_J2000', arrow_length_ratio=0.1)
        ax.quiver(origin_plot[0], origin_plot[1], origin_plot[2], z_j2000_in_body[0], z_j2000_in_body[1],
                  z_j2000_in_body[2], length=j2000_axis_len, color='yellow', label='Z_J2000', arrow_length_ratio=0.1)

    legend_handles = [plt.Line2D([0], [0], c='r', lw=2, label='X_Body'),
                      plt.Line2D([0], [0], c='g', lw=2, label='Y_Body'),
                      plt.Line2D([0], [0], c='b', lw=2, label='Z_Body')]
    if C_j2000_to_body_rotation is not None:
        legend_handles.extend(
            [plt.Line2D([0], [0], c='c', lw=2, label='X_J2000'), plt.Line2D([0], [0], c='m', lw=2, label='Y_J2000'),
             plt.Line2D([0], [0], c='y', lw=2, label='Z_J2000')])

    if sun_direction_world is not None:
        # sun_direction_world is the normalized vector FROM satellite (origin_plot) TO sun
        # The light comes FROM sun TO satellite, so along -sun_direction_world
        light_direction_vector = -sun_direction_world

        # Position the tail of the arrow "at the sun" relative to the satellite
        arrow_tail_position = origin_plot + sun_direction_world * (plot_max_range * 0.8)

        ax.quiver(arrow_tail_position[0], arrow_tail_position[1], arrow_tail_position[2],
                  light_direction_vector[0], light_direction_vector[1], light_direction_vector[2],
                  length=plot_max_range * 0.8,  # Length to approximately reach the origin
                  color='orange', pivot='tail', arrow_length_ratio=0.2)
        legend_handles.append(plt.Line2D([0], [0], color='orange', lw=2, label='Sunlight Direction'))

    ax.legend(handles=legend_handles)

    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=100)
            logging.debug(f"Scene saved to {save_path} using matplotlib.pyplot.savefig")
        except Exception as e:
            logging.error(f"Error saving scene image to {save_path} using matplotlib: {e}", exc_info=True)
        finally:
            plt.close(fig)

    if show_scene and not save_path:
        logging.info(f"Displaying scene: {scene_title}")
        plt.show()
    elif show_scene and save_path:
        logging.info(f"Scene saved to {save_path}. To view, set show_scene=True and save_path=None.")
        if 'fig' in locals() and plt.fignum_exists(fig.number):
            plt.close(fig)


def effective_scale_for_scene(satellite_obj: Satellite, component_local_meshes: Dict[str, trimesh.Trimesh]) -> float:
    all_verts_body_frame = []
    for comp_model in satellite_obj.components:
        local_mesh = component_local_meshes[comp_model.name]
        comp_pos, comp_orient_q_common = comp_model.relative_position, comp_model.relative_orientation
        comp_orient_q_np = np.quaternion(comp_orient_q_common.w, comp_orient_q_common.x, comp_orient_q_common.y,
                                         comp_orient_q_common.z)
        comp_to_body_transform = np.eye(4)
        comp_to_body_transform[:3, :3] = quaternion.as_rotation_matrix(comp_orient_q_np)
        comp_to_body_transform[:3, 3] = comp_pos
        transformed_verts = trimesh.transform_points(local_mesh.vertices.copy(), comp_to_body_transform)
        all_verts_body_frame.extend(transformed_verts)
    if not all_verts_body_frame: return 1.0
    full_satellite_cloud_body = trimesh.PointCloud(np.array(all_verts_body_frame))
    if full_satellite_cloud_body.is_empty: return 1.0
    scale = np.linalg.norm(full_satellite_cloud_body.bounding_box.extents)
    return scale if scale > 1e-6 else 1.0


if __name__ == "__main__":
    logging.info("--- Running POC Shadow Interpolation Script ---")

    logging.info("--- Phase 1: Model Definition and Initial Setup ---")
    satellite_definition, component_local_meshes = define_satellite_model()
    logging.info(f"Satellite '{satellite_definition.name}' (ID: {satellite_definition.id}) defined.")

    component_body_meshes_map_coarse = get_component_meshes_in_body_frame(satellite_definition, component_local_meshes,
                                                                          subdivide=False)

    subdivided_body_meshes_for_intersector = []
    for name, mesh in component_body_meshes_map_coarse.items():
        if not mesh.is_empty:
            subdivided_mesh = mesh.copy().subdivide()
            subdivided_mesh.fix_normals(multibody=True)
            if not subdivided_mesh.is_watertight:
                logging.warning(f"Subdivided mesh for component {name} is not watertight.")
            subdivided_body_meshes_for_intersector.append(subdivided_mesh)
        else:
            logging.warning(f"Coarse mesh for component {name} was empty before subdivision.")

    if not subdivided_body_meshes_for_intersector:
        logging.error("No valid meshes to combine for intersector after subdivision. Exiting.")
        sys.exit(1)
    try:
        intersector_mesh_combined = trimesh.util.concatenate(subdivided_body_meshes_for_intersector)
        if intersector_mesh_combined.is_empty:
            logging.error("Combined intersector mesh is empty after subdivision. Exiting.")
            sys.exit(1)
        intersector_mesh_combined = intersector_mesh_combined.process()
        if not intersector_mesh_combined.is_watertight:
            logging.warning("Final subdivided intersector mesh not watertight after processing.")
        logging.info(
            f"Combined and subdivided intersector mesh created: {len(intersector_mesh_combined.vertices)}V, {len(intersector_mesh_combined.faces)}F.")
    except Exception as e:
        logging.error(f"Failed to create subdivided intersector mesh: {e}")
        sys.exit(1)

    db_filepath = os.path.join(SCRIPT_DIR, SHADOW_DATABASE_FILENAME)
    logging.info("Forcing database regeneration for testing with subdivision/camera changes.")
    shadow_db = None  # Force regeneration for this test

    if not shadow_db:
        logging.info("--- Phase 2: Generating Shadowing Database (with subdivided intersector) ---")
        shadow_db = generate_shadowing_database(satellite_definition, intersector_mesh_combined)
        try:
            with open(db_filepath, 'wb') as f:
                pickle.dump(shadow_db, f)
            logging.info(f"Shadowing database saved to: {db_filepath}")
        except Exception as e:
            logging.error(f"Error saving database: {e}")

    if not shadow_db:
        logging.error("Failed to generate or load shadow database. Exiting.")
        sys.exit(1)

    logging.info("--- Phase 3: Generating Animation ---")

    if os.path.exists(ANIMATION_FRAMES_DIR):
        shutil.rmtree(ANIMATION_FRAMES_DIR)
    os.makedirs(ANIMATION_FRAMES_DIR, exist_ok=True)

    frame_filenames = []

    try:
        with SpiceHandler() as spice:
            spice.load_metakernel(ANIMATION_METAKERNEL_PATH)
            start_et = spice.utc_to_et(ANIMATION_START_UTC)
            end_et = spice.utc_to_et(ANIMATION_END_UTC)
            num_steps = int((end_et - start_et) / ANIMATION_TIME_STEP_SEC) + 1
            epochs_et = np.linspace(start_et, end_et, num_steps)
            logging.info(
                f"Generating {len(epochs_et)} frames for animation from {ANIMATION_START_UTC} to {ANIMATION_END_UTC}.")

            if satellite_definition.name == "Intelsat_901_POC_Corrected":
                actual_satellite_spice_id_for_spice = "-126824"
            else:
                actual_satellite_spice_id_for_spice = satellite_definition.id
            logging.info(f"Using SPICE ID for satellite: {actual_satellite_spice_id_for_spice}")

            observer_spice_id_str = 'SUN'
            reference_frame_j2000 = 'J2000'

            fixed_camera_elev_azim = (10, 20)  # Head-on with slight elevation

            for i, epoch_et in enumerate(epochs_et):
                utc_time_str = spice.et_to_utc(epoch_et, "C", 3)
                logging.info(f"Processing frame {i + 1}/{len(epochs_et)} for ET {epoch_et:.2f} ({utc_time_str})")

                sun_pos_j2000_wrt_sat, _ = spice.get_body_position(
                    target='SUN', et=epoch_et, frame=reference_frame_j2000,
                    aberration_correction='NONE', observer=actual_satellite_spice_id_for_spice)
                vec_sat_to_sun_j2000 = np.array(sun_pos_j2000_wrt_sat)

                try:
                    C_j2000_to_body_3x3 = spice.get_target_orientation(
                        from_frame=reference_frame_j2000,
                        to_frame=satellite_definition.body_frame_name,
                        et=epoch_et)
                except Exception as e_spice_orient:
                    logging.warning(
                        f"Could not get C_j2000_to_body for {satellite_definition.body_frame_name}. Assuming identity. Error: {e_spice_orient}")
                    C_j2000_to_body_3x3 = np.eye(3)

                sun_vector_body_frame = C_j2000_to_body_3x3 @ vec_sat_to_sun_j2000
                sun_vector_body_frame_normalized = sun_vector_body_frame / (
                            np.linalg.norm(sun_vector_body_frame) + 1e-9)

                # Log for every frame
                logging.info(
                    f"Frame {i + 1} ({utc_time_str}): Sat->Sun (Body Frame, Normalized) = {sun_vector_body_frame_normalized}")

                current_shadow_status = get_interpolated_shadows(sun_vector_body_frame_normalized, shadow_db)
                if current_shadow_status is None:
                    logging.error(f"Could not get shadow status for ET {epoch_et}. Skipping frame.")
                    continue

                frame_filename = os.path.join(ANIMATION_FRAMES_DIR, f"frame_{i:04d}.png")

                visualize_scene_with_shadows(
                    satellite_obj=satellite_definition,
                    component_local_meshes=component_local_meshes,
                    shadow_statuses_by_component=current_shadow_status,
                    transform_satellite_body_to_world=np.eye(4),
                    sun_direction_world=sun_vector_body_frame_normalized,
                    camera_elev_azim=fixed_camera_elev_azim,
                    C_j2000_to_body_rotation=C_j2000_to_body_3x3,
                    show_scene=False,
                    scene_title=f"Frame {i + 1}: {utc_time_str}",
                    save_path=frame_filename
                )
                if os.path.exists(frame_filename):
                    frame_filenames.append(frame_filename)
                else:
                    logging.warning(f"Frame {frame_filename} was not saved.")

            if frame_filenames:
                logging.info(f"Compiling {len(frame_filenames)} frames into GIF: {ANIMATION_OUTPUT_FILENAME}")
                gif_path = os.path.join(SCRIPT_DIR, ANIMATION_OUTPUT_FILENAME)
                with imageio.get_writer(gif_path, mode='I', duration=1.0 / FRAMES_PER_SECOND_GIF, loop=0) as writer:
                    for filename in frame_filenames:
                        try:
                            image = imageio.imread(filename)
                            writer.append_data(image)
                        except Exception as e_img:
                            logging.error(f"Could not read frame {filename} for GIF: {e_img}")
                logging.info(f"Animation saved to {gif_path}")

                # Keep frames for inspection
                # try:
                #     shutil.rmtree(ANIMATION_FRAMES_DIR)
                #     logging.info(f"Cleaned up frames directory: {ANIMATION_FRAMES_DIR}")
                # except Exception as e_clean:
                #     logging.error(f"Error cleaning up frames directory: {e_clean}")
            else:
                logging.warning("No frames generated for animation. Check frame saving process.")
    except Exception as e:
        logging.error(f"An error occurred during animation generation: {e}", exc_info=True)

    logging.info("--- POC Script Complete ---")
