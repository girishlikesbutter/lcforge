# lcforge/src/utils/mesh_utils.py

import trimesh
import logging
from typing import Optional

# Configure a logger for this module
logger = logging.getLogger(__name__)

def load_mesh_from_file(file_path: str) -> Optional[trimesh.Trimesh]:
    """
    Loads a 3D mesh from the specified file path using the trimesh library.

    This function supports various mesh formats like STL, OBJ, etc.,
    as supported by trimesh.

    Args:
        file_path: The absolute or relative path to the mesh file.

    Returns:
        A trimesh.Trimesh object if the mesh is loaded successfully,
        otherwise None if an error occurs (e.g., file not found,
        unsupported format, or trimesh internal error).
    """
    if not file_path:
        logger.warning("File path is empty or None. Cannot load mesh.")
        return None

    try:
        # Attempt to load the mesh. Trimesh will attempt to infer the file type.
        # Forcing a specific type is also possible, e.g., file_type='stl'
        mesh = trimesh.load_mesh(file_path)
        if mesh.is_empty:
            logger.warning(f"Loaded mesh from '{file_path}' is empty.")
            # Depending on desired behavior, an empty mesh might be an error or not.
            # For now, let's treat it as a non-successful load for a distinct object.
            return None
        logger.info(f"Successfully loaded mesh from '{file_path}' "
                    f"with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces.")
        return mesh
    except FileNotFoundError:
        logger.error(f"Mesh file not found at path: {file_path}")
        return None
    except Exception as e:
        # Catching a broad exception here as trimesh can raise various errors
        # depending on the file format and its internal state.
        logger.error(f"Failed to load mesh from '{file_path}'. Error: {e}", exc_info=True)
        return None

if __name__ == '__main__':
    # Example Usage:
    # To run this example, you'll need a sample mesh file.
    # Create a dummy OBJ file for testing if you don't have one.
    # For example, create 'dummy_cube.obj' in the same directory with simple cube data.
    #
    # Example dummy_cube.obj content:
    # # Vertices
    # v 0.0 0.0 0.0
    # v 1.0 0.0 0.0
    # v 1.0 1.0 0.0
    # v 0.0 1.0 0.0
    # v 0.0 0.0 1.0
    # v 1.0 0.0 1.0
    # v 1.0 1.0 1.0
    # v 0.0 1.0 1.0
    # # Faces (simple quad, trimesh will triangulate)
    # f 1 2 3 4
    # f 5 6 7 8
    # f 1 2 6 5
    # f 2 3 7 6
    # f 3 4 8 7
    # f 4 1 5 8

    logging.basicConfig(level=logging.INFO) # Setup basic logging for the example

    # Create a dummy OBJ file for testing
    dummy_obj_content = """
# Vertices
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 1.0 1.0 0.0
v 0.0 1.0 0.0
v 0.0 0.0 1.0
v 1.0 0.0 1.0
v 1.0 1.0 1.0
v 0.0 1.0 1.0
# Faces
f 1 2 3 4
f 5 8 7 6
f 1 5 6 2
f 2 6 7 3
f 3 7 8 4
f 4 8 5 1
    """
    dummy_file_path = "dummy_cube.obj"
    with open(dummy_file_path, "w") as f:
        f.write(dummy_obj_content)

    logger.info(f"Attempting to load mesh from: {dummy_file_path}")
    loaded_mesh = load_mesh_from_file(dummy_file_path)

    if loaded_mesh:
        print(f"\nMesh loaded successfully:")
        print(f"  Number of vertices: {len(loaded_mesh.vertices)}")
        print(f"  Number of faces: {len(loaded_mesh.faces)}")
        print(f"  Is watertight: {loaded_mesh.is_watertight}")
        print(f"  Volume: {loaded_mesh.volume if loaded_mesh.is_watertight else 'N/A (not watertight)'}")
        # You can access mesh.vertices, mesh.faces, mesh.face_normals etc.
    else:
        print(f"\nFailed to load mesh from {dummy_file_path}")

    # Test with a non-existent file
    logger.info("\nAttempting to load mesh from a non-existent file:")
    non_existent_mesh = load_mesh_from_file("non_existent_file.stl")
    if not non_existent_mesh:
        print("Correctly handled non-existent file.")

    # Clean up dummy file
    import os
    os.remove(dummy_file_path)
    logger.info(f"Cleaned up {dummy_file_path}")