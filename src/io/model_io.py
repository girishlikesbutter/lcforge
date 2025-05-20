# lcforge/src/io/model_io.py

import yaml
import numpy as np
import quaternion as npq  # For np.quaternion
from dataclasses import is_dataclass, fields, asdict

from src.models.model_definitions import Satellite, Component, Facet, BRDFMaterialProperties
from src.core.common_types import Vector3D, Quaternion  # For type hinting and ensuring types are loaded


# --- Custom YAML Representers ---

def numpy_array_representer(dumper: yaml.Dumper, data: np.ndarray) -> yaml.Node:
    """Custom representer for numpy.ndarray."""
    return dumper.represent_list(data.tolist())


def numpy_quaternion_representer(dumper: yaml.Dumper, data: npq.quaternion) -> yaml.Node:
    """Custom representer for numpy.quaternion."""
    return dumper.represent_list([data.w, data.x, data.y, data.z])


# Add representers to the Dumper
yaml.add_representer(np.ndarray, numpy_array_representer)
yaml.add_representer(npq.quaternion, numpy_quaternion_representer)


# --- Custom YAML Constructors ---

def numpy_array_constructor(loader: yaml.Loader, node: yaml.SequenceNode) -> np.ndarray:
    """Custom constructor for numpy.ndarray."""
    return np.array(loader.construct_sequence(node, deep=True))


def numpy_quaternion_constructor(loader: yaml.Loader, node: yaml.SequenceNode) -> npq.quaternion:
    """Custom constructor for numpy.quaternion."""
    components = loader.construct_sequence(node, deep=True)
    return npq.quaternion(components[0], components[1], components[2], components[3])


# We need to use FullLoader or UnsafeLoader to allow constructing arbitrary Python objects.
# Let's define a loader that includes our custom constructors.
class ModelLoader(yaml.FullLoader):  # Or yaml.UnsafeLoader if FullLoader isn't available/sufficient
    pass


ModelLoader.add_constructor('!numpy.ndarray', numpy_array_constructor)
ModelLoader.add_constructor('!numpy.quaternion', numpy_quaternion_constructor)


# --- Helper to handle dataclasses specifically for saving ---
# PyYAML's default dataclass handling is okay, but explicit tagging can be more robust.
# However, for simplicity with standard dataclasses, we can often rely on asdict
# if all fields are serializable or have representers.
# For loading, we'll need to reconstruct dataclasses.

def satellite_model_representer(dumper: yaml.Dumper, data) -> yaml.Node:
    """Represents Satellite and its nested dataclasses with explicit tags."""
    if is_dataclass(data) and not isinstance(data, type):
        # Use a tag like !package.module.ClassName
        tag = f"!{data.__class__.__module__}.{data.__class__.__name__}"
        # Convert dataclass to dict, ensuring nested dataclasses are also processed
        # This requires that asdict is called recursively or that nested objects
        # are also handled by their own representers if needed.
        # PyYAML's represent_dict will handle fields.
        return dumper.represent_mapping(tag, asdict(data))
    return dumper.represent_undefined(data)


yaml.add_representer(Satellite, satellite_model_representer)
yaml.add_representer(Component, satellite_model_representer)
yaml.add_representer(Facet, satellite_model_representer)
yaml.add_representer(BRDFMaterialProperties, satellite_model_representer)

# --- Constructor for all our model dataclasses ---
# This generic constructor approach assumes that the YAML tags match the
# pattern !module.Class and that the class can be imported.

# Store a mapping of known model types for the constructor
_KNOWN_MODEL_CLASSES = {
    f"{cls.__module__}.{cls.__name__}": cls
    for cls in [Satellite, Component, Facet, BRDFMaterialProperties]
}


def model_dataclass_constructor(loader: yaml.Loader, tag_suffix: str, node: yaml.MappingNode) -> object:
    """
    Constructs an instance of our model dataclasses (Satellite, Component, etc.)
    The tag_suffix is expected to be 'module.ClassName'.
    """
    cls_name_full = tag_suffix.lstrip('!')  # Remove the initial '!' if present
    cls = _KNOWN_MODEL_CLASSES.get(cls_name_full)

    if cls is None:
        raise yaml.YAMLError(f"Unknown model class for tag: !{cls_name_full}")

    # Construct a dictionary from the YAML mapping
    data_dict = loader.construct_mapping(node, deep=True)

    # Ensure all fields are present with None if missing and not required by constructor
    # or handle appropriately based on dataclass defaults.
    # For dataclasses, PyYAML's construct_mapping usually does a good job if types are simple.
    # We might need to recursively ensure nested custom types are handled if not already by other constructors.

    # For dataclasses, we can often instantiate directly if the keys in data_dict
    # match the field names and their values are of the correct (or constructible) types.
    try:
        return cls(**data_dict)
    except TypeError as e:
        raise yaml.YAMLError(f"Failed to instantiate {cls.__name__} with data {data_dict}. Error: {e}")


# Register the constructor for each known model class using its full !module.Class tag
for full_cls_name, cls_obj in _KNOWN_MODEL_CLASSES.items():
    ModelLoader.add_constructor(f"!{full_cls_name}",
                                lambda l, n, c=cls_obj: model_dataclass_constructor(l, f"!{c.__module__}.{c.__name__}",
                                                                                    n))


# --- Main Save and Load Functions ---

def save_satellite_to_yaml(satellite: Satellite, file_path: str) -> None:
    """
    Saves a Satellite object to a YAML file.

    Args:
        satellite: The Satellite object to save.
        file_path: The path to the YAML file where the satellite will be saved.
    """
    try:
        with open(file_path, 'w') as f:
            # Use Dumper that knows about our custom representers
            # The explicit representers for dataclasses should now be used.
            yaml.dump(satellite, f, sort_keys=False, Dumper=yaml.Dumper)
        print(f"Satellite model saved to {file_path}")
    except Exception as e:
        print(f"Error saving satellite model to {file_path}: {e}")
        raise


def load_satellite_from_yaml(file_path: str) -> Optional[Satellite]:
    """
    Loads a Satellite object from a YAML file.

    Args:
        file_path: The path to the YAML file.

    Returns:
        The loaded Satellite object, or None if loading fails.
    """
    try:
        with open(file_path, 'r') as f:
            # Use our custom ModelLoader which knows about numpy and quaternion constructors
            # and the generic model dataclass constructor.
            data = yaml.load(f, Loader=ModelLoader)
        if isinstance(data, Satellite):
            print(f"Satellite model loaded successfully from {file_path}")
            return data
        else:
            print(f"Loaded data from {file_path} is not a Satellite instance.")
            return None
    except FileNotFoundError:
        print(f"Error: YAML file not found at {file_path}")
        return None
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file {file_path}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while loading {file_path}: {e}")
        return None


if __name__ == '__main__':
    # Example Usage (requires model_definitions.py to be in the Python path)

    # Create a dummy satellite model for testing
    # (Using the example from model_definitions.py for consistency)
    solar_panel_material_ex = BRDFMaterialProperties(r_d=0.2, r_s=0.6, n_phong=10.0)
    bus_material_ex = BRDFMaterialProperties(r_d=0.4, r_s=0.1, n_phong=5.0)

    panel_facet1_vertices_ex = [np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), np.array([1.0, 1.0, 0.0]),
                                np.array([0.0, 1.0, 0.0])]
    panel_facet1_ex = Facet(id="SP1_F1",
                            vertices=panel_facet1_vertices_ex,
                            normal=np.array([0.0, 0.0, 1.0]),
                            area=1.0,
                            material_properties=solar_panel_material_ex)

    solar_panel_A_ex = Component(id="SP_A",
                                 name="SolarPanel_Port",
                                 mesh_path="data/meshes/solar_panel_detailed.stl",
                                 facets=[panel_facet1_ex],
                                 relative_position=np.array([0.0, 1.5, 0.0]),
                                 relative_orientation=npq.quaternion(0.9659258, 0.0, 0.258819, 0.0),
                                 # Example: 30 deg rot around y
                                 articulation_rule="TRACK_SUN_FRAME")

    bus_facet_vertices_ex = [np.array([-0.5, -0.5, -0.5]), np.array([0.5, -0.5, -0.5]), np.array([0.5, 0.5, -0.5]),
                             np.array([-0.5, 0.5, -0.5])]
    bus_facet1_ex = Facet(id="BUS_F1_NEG_Z",
                          vertices=bus_facet_vertices_ex,
                          normal=np.array([0.0, 0.0, -1.0]),
                          area=1.0,
                          material_properties=bus_material_ex)

    satellite_bus_ex = Component(id="BUS",
                                 name="SatelliteBus",
                                 mesh_path="data/meshes/bus_detailed.obj",
                                 facets=[bus_facet1_ex],
                                 relative_position=np.array([0.0, 0.0, 0.0]),
                                 relative_orientation=npq.quaternion(1, 0, 0, 0))

    my_satellite_ex = Satellite(id="SAT001_IO_TEST",
                                name="MyResearchSatForIO",
                                components=[satellite_bus_ex, solar_panel_A_ex],
                                body_frame_name="MYSAT_IO_BODY_FIXED")

    # Define file path for testing
    test_yaml_file = "test_satellite_model.yaml"

    # Save the satellite
    print(f"\n--- Saving satellite to {test_yaml_file} ---")
    save_satellite_to_yaml(my_satellite_ex, test_yaml_file)

    # Load the satellite
    print(f"\n--- Loading satellite from {test_yaml_file} ---")
    loaded_satellite = load_satellite_from_yaml(test_yaml_file)

    if loaded_satellite:
        print("\n--- Verification of Loaded Satellite ---")
        print(f"Loaded Satellite ID: {loaded_satellite.id}, Name: {loaded_satellite.name}")
        print(f"Original Satellite ID: {my_satellite_ex.id}, Name: {my_satellite_ex.name}")
        assert loaded_satellite.id == my_satellite_ex.id
        assert loaded_satellite.name == my_satellite_ex.name
        assert loaded_satellite.body_frame_name == my_satellite_ex.body_frame_name
        assert len(loaded_satellite.components) == len(my_satellite_ex.components)

        # Basic check of a nested field
        original_pos = my_satellite_ex.components[1].relative_position
        loaded_pos = loaded_satellite.components[1].relative_position
        assert np.array_equal(loaded_pos, original_pos), f"Position mismatch: {loaded_pos} vs {original_pos}"
        print(f"Component '{loaded_satellite.components[1].name}' position verified: {loaded_pos}")

        original_orient = my_satellite_ex.components[1].relative_orientation
        loaded_orient = loaded_satellite.components[1].relative_orientation
        assert loaded_orient == original_orient, f"Orientation mismatch: {loaded_orient} vs {original_orient}"

        print(f"Component '{loaded_satellite.components[1].name}' orientation verified: {loaded_orient}")

        original_facet_normal = my_satellite_ex.components[0].facets[0].normal
        loaded_facet_normal = loaded_satellite.components[0].facets[0].normal
        assert np.array_equal(loaded_facet_normal, original_facet_normal)
        print(f"Facet '{loaded_satellite.components[0].facets[0].id}' normal verified: {loaded_facet_normal}")
        print("\nBasic verification passed.")
    else:
        print("Failed to load and verify satellite.")

    # Clean up test file
    import os

    if os.path.exists(test_yaml_file):
        os.remove(test_yaml_file)
        print(f"\nCleaned up {test_yaml_file}")