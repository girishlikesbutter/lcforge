# lcforge/src/models/model_definitions.py

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Any

# Import base types from common_types
# Vector3D is np.ndarray, Quaternion is np.quaternion as defined in common_types.py
from src.core.common_types import Vector3D, Quaternion


@dataclass
class BRDFMaterialProperties:
    """
    Represents the Bidirectional Reflectance Distribution Function (BRDF)
    material properties for a facet, based on a Phong model. (FR1.4)

    Attributes:
        r_d: Diffuse reflectivity coefficient (0 to 1). (FR1.4)
        r_s: Specular reflectivity coefficient (0 to 1). (FR1.4)
        n_phong: Phong exponent (shininess). (FR1.4)
    """
    r_d: float = 0.0
    r_s: float = 0.0
    n_phong: float = 1.0

@dataclass
class Facet:
    """
    Represents a single reflective facet of a satellite component.
    Used for light curve calculations. Shadowing effects from a detailed
    mesh model will be incorporated via pre-calculated factors. (FR1.1)

    Attributes:
        id: Unique identifier for the facet.
        vertices: List of 3D vectors defining the facet's corners/vertices. (FR1.1 related)
                  Order matters for normal calculation if not provided explicitly.
        normal: The outward-pointing normal vector of the facet. (FR1.1 related)
        area: Surface area of the facet. (FR1.1 related)
        material_properties: BRDF properties of the facet. (FR1.4)
    """
    id: str
    vertices: List[Vector3D] = field(default_factory=list)
    normal: Vector3D = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    area: float = 0.0
    material_properties: BRDFMaterialProperties = field(default_factory=BRDFMaterialProperties)

@dataclass
class Component:
    """
    Represents a distinct component of a satellite (e.g., bus, solar panel, antenna). (FR1.2)
    Components have geometry, position/orientation relative to the satellite body,
    material properties, and potentially articulation rules.

    Attributes:
        id: Unique identifier for the component.
        name: Descriptive name of the component.
        mesh_path: Optional path to a 3D mesh file (e.g., STL, OBJ)
                   representing this component for detailed self-shadowing analysis. (FR1.1)
        facets: List of simplified Facet objects used for light curve generation. (FR1.1)
                These facets might be derived from the mesh or defined separately.
        relative_position: 3D vector defining the component's origin offset from the
                           satellite's main body-fixed reference frame origin. (FR1.3)
        relative_orientation: Quaternion defining the component's orientation
                              relative to the satellite's main body-fixed reference frame. (FR1.3)
        articulation_rule: Placeholder for defining how this component might articulate
                           (e.g., point towards the Sun). This could be a string
                           referencing a SPICE frame or a more complex rule object.
                           (FR1.3, FR3.4)
    """
    id: str
    name: str
    mesh_path: Optional[str] = None
    facets: List[Facet] = field(default_factory=list)
    relative_position: Vector3D = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    relative_orientation: Quaternion = field(default_factory=lambda: np.quaternion(1, 0, 0, 0)) # Identity quaternion
    articulation_rule: Optional[Any] = None # To be defined more concretely

@dataclass
class Satellite:
    """
    Represents the overall satellite model, composed of multiple components. (FR1.2)
    Defines the satellite's primary body-fixed reference frame.

    Attributes:
        id: Unique identifier for the satellite.
        name: Descriptive name of the satellite.
        components: List of Component objects that make up the satellite. (FR1.2)
        body_frame_name: The name of the SPICE reference frame rigidly attached
                         to the satellite's main body. Component positions and
                         orientations are defined relative to this frame. (FR2.2 related)
    """
    id: str
    name: str
    components: List[Component] = field(default_factory=list)
    body_frame_name: str = "" # Should be a valid SPICE frame name


if __name__ == '__main__':
    # This section is for example usage and basic testing.
    # It ensures that the Quaternion type is understood as np.quaternion
    # For this example to run directly, ensure numpy and numpy-quaternion are importable.
    # The 'np.quaternion' type comes from the 'numpy-quaternion' library.

    # 1. Define Material Properties
    solar_panel_material = BRDFMaterialProperties(r_d=0.2, r_s=0.6, n_phong=10.0)
    bus_material = BRDFMaterialProperties(r_d=0.4, r_s=0.1, n_phong=5.0)

    # 2. Define Facets for a Solar Panel
    panel_facet1_vertices = [np.array([0,0,0]), np.array([1,0,0]), np.array([1,1,0]), np.array([0,1,0])]
    panel_facet1 = Facet(id="SP1_F1",
                         vertices=panel_facet1_vertices,
                         normal=np.array([0,0,1.0]), # Assuming outward normal
                         area=1.0,
                         material_properties=solar_panel_material)

    panel_facet2 = Facet(id="SP1_F2",
                         vertices=[np.array([0,0,0]), np.array([-1,0,0]), np.array([-1,-1,0]), np.array([0,-1,0])], # Different vertices
                         normal=np.array([0,0,-1.0]), # Back side, if modeled
                         area=1.0,
                         material_properties=solar_panel_material)

    # 3. Define a Component (Solar Panel)
    solar_panel_A = Component(id="SP_A",
                              name="SolarPanel_Port",
                              mesh_path="data/meshes/solar_panel_detailed.stl", # Example path
                              facets=[panel_facet1, panel_facet2],
                              relative_position=np.array([0.0, 1.5, 0.0]), # Positioned 1.5m along Y-axis of sat body
                              relative_orientation=np.quaternion(1,0,0,0), # No rotation relative to sat body initially
                              articulation_rule="TRACK_SUN_FRAME") # Example rule

    # 4. Define another Component (Satellite Bus)
    bus_facet_vertices = [np.array([-0.5,-0.5,-0.5]), np.array([0.5,-0.5,-0.5]), np.array([0.5,0.5,-0.5]), np.array([-0.5,0.5,-0.5])]
    bus_facet1 = Facet(id="BUS_F1_NEG_Z",
                       vertices=bus_facet_vertices,
                       normal=np.array([0,0,-1.0]),
                       area=1.0,
                       material_properties=bus_material)

    satellite_bus = Component(id="BUS",
                              name="SatelliteBus",
                              mesh_path="data/meshes/bus_detailed.obj",
                              facets=[bus_facet1], # Simplified with one facet
                              relative_position=np.array([0.0, 0.0, 0.0]), # At satellite origin
                              relative_orientation=np.quaternion(1,0,0,0)) # Identity

    # 5. Define the Satellite
    my_satellite = Satellite(id="SAT001",
                             name="MyResearchSat",
                             components=[satellite_bus, solar_panel_A],
                             body_frame_name="MYSAT_BODY_FIXED")

    print("--- Example Satellite Configuration ---")
    print(f"Satellite: {my_satellite.name} (ID: {my_satellite.id})")
    print(f"  Body Frame: {my_satellite.body_frame_name}")
    for comp in my_satellite.components:
        print(f"  Component: {comp.name} (ID: {comp.id})")
        print(f"    Mesh Path: {comp.mesh_path}")
        print(f"    Relative Position: {comp.relative_position}")
        print(f"    Relative Orientation: {comp.relative_orientation}")
        print(f"    Articulation: {comp.articulation_rule}")
        print(f"    Number of Facets: {len(comp.facets)}")
        for i, facet in enumerate(comp.facets):
            print(f"      Facet {i+1}: {facet.id}, Area: {facet.area:.2f}") # Added .2f for area
            print(f"        Normal: {facet.normal}")
            # print(f"        Vertices: {facet.vertices}") # Can be verbose
            print(f"        Material: Rd={facet.material_properties.r_d}, Rs={facet.material_properties.r_s}, n={facet.material_properties.n_phong}")

    print("\n--- End Example ---")