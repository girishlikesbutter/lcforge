# lcforge/tests/spice/test_spice_handler.py

import pytest
import numpy as np
import os
import spiceypy

# Adjust the import path based on your project structure (src layout)
from src.spice.spice_handler import SpiceHandler

# --- Test Data and Constants ---
# These should match the values confirmed to work in your __main__ tests
TEST_ET = 6.34169191e+08
EXPECTED_UTC_FROM_TEST_ET = "2020-02-05T10:05:21.815116"  # From your previous successful log

IS901_NAIF_ID_STR = "-126824"
OBSERVER_DST_NAIF_ID_STR = "399999"
EARTH_NAIF_ID_STR = "399"
INERTIAL_FRAME_NAME = "J2000"

USER_IS901_BUS_FRAME_NAME_STR = "IS901_BUS_FRAME"
USER_IS901_BUS_FRAME_ID_INT = -999824

# Expected position for IS901 from your log (use for np.allclose)
EXPECTED_IS901_POS = np.array([-30460.03777213, 29576.62061818, 900.06049665])
# Expected orientation matrix for IS901_BUS_FRAME from your log
EXPECTED_IS901_BUS_ORIENTATION = np.array([
    [0.717282328, -0.696462578, -0.0211172815],
    [0.0152161223, -0.0146430219, 0.999777001],
    [-0.696616488, -0.717443698, 0.0000942789690]
])


# --- Fixtures ---

@pytest.fixture(scope="session")
def project_root_dir():
    """Returns the absolute path to the project root."""
    # Assumes tests are in lcforge/tests/spice/
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


@pytest.fixture(scope="session")
def metakernel_path(project_root_dir):
    """Returns the absolute path to the metakernel."""
    return os.path.join(
        project_root_dir,
        "data",
        "spice_kernels",
        "missions",
        "dst-is901",
        "INTELSAT_901-metakernel.tm"
    )


@pytest.fixture
def spice_handler_loaded(metakernel_path):
    """
    Provides a SpiceHandler instance with the metakernel loaded.
    Ensures kernels are cleared after the test.
    """
    handler = SpiceHandler()
    try:
        handler.load_metakernel(metakernel_path)
        yield handler  # Provide the handler to the test
    finally:
        handler.unload_all_kernels()  # Ensure cleanup


# --- Test Functions ---

def test_metakernel_loading(spice_handler_loaded):
    """Tests if the metakernel and its constituent kernels are loaded."""
    assert spice_handler_loaded._loaded_kernels  # Check if the set is not empty
    # Check if the metakernel itself is in the loaded set
    # Note: metakernel_path fixture returns absolute path, SpiceHandler stores what was passed.
    # For this test, we're more interested that *something* got loaded.
    # A more robust check would be to query ktotal.
    assert spiceypy.ktotal('ALL') > 0  # Check if SPICE has kernels loaded


def test_et_to_utc_conversion(spice_handler_loaded):
    """Tests ET to UTC conversion."""
    utc_str = spice_handler_loaded.et_to_utc(TEST_ET, precision=6)
    assert utc_str == EXPECTED_UTC_FROM_TEST_ET


def test_utc_to_et_conversion(spice_handler_loaded):
    """Tests UTC to ET conversion."""
    # Use the UTC string we know corresponds to TEST_ET
    et = spice_handler_loaded.utc_to_et(EXPECTED_UTC_FROM_TEST_ET)
    np.testing.assert_allclose(et, TEST_ET, rtol=1e-7)  # Compare floats carefully


def test_get_is901_position(spice_handler_loaded):
    """Tests retrieving IS901 position."""
    pos_vec, _ = spice_handler_loaded.get_body_position(
        target=IS901_NAIF_ID_STR,
        et=TEST_ET,
        frame=INERTIAL_FRAME_NAME,
        observer=EARTH_NAIF_ID_STR,
        aberration_correction="LT+S"
    )
    np.testing.assert_allclose(pos_vec, EXPECTED_IS901_POS, rtol=1e-5)


def test_get_is901_bus_orientation(spice_handler_loaded):
    """Tests retrieving IS901_BUS_FRAME orientation."""
    # First, ensure the frame name is known
    try:
        frame_id = spice_handler_loaded.get_frame_id_from_name(USER_IS901_BUS_FRAME_NAME_STR)
        assert frame_id == USER_IS901_BUS_FRAME_ID_INT
    except ValueError:
        pytest.fail(f"Frame name '{USER_IS901_BUS_FRAME_NAME_STR}' not found by get_frame_id_from_name.")

    # Then, get the orientation
    rot_matrix = spice_handler_loaded.get_target_orientation(
        from_frame=INERTIAL_FRAME_NAME,
        to_frame=USER_IS901_BUS_FRAME_NAME_STR,
        et=TEST_ET
    )
    np.testing.assert_allclose(rot_matrix, EXPECTED_IS901_BUS_ORIENTATION, rtol=1e-7, atol=1e-7)


def test_get_frame_id_and_name_mapping(spice_handler_loaded):
    """Tests frame name to ID and ID to name mappings."""
    # Test name to ID
    resolved_id = spice_handler_loaded.get_frame_id_from_name(USER_IS901_BUS_FRAME_NAME_STR)
    assert resolved_id == USER_IS901_BUS_FRAME_ID_INT

    # Test ID to name
    resolved_name = spice_handler_loaded.get_frame_name_from_id(USER_IS901_BUS_FRAME_ID_INT)
    # Note: frmnam might return a slightly different canonical name than defined in FK,
    # but it should be consistent. For user-defined frames, it should match.
    # The previous log showed frinfo(-999824) returned '-126824' as name.
    # Let's check what frmnam returns for the ID.
    # If your FK defines IS901_BUS_FRAME for -999824, frmnam should return IS901_BUS_FRAME.
    assert resolved_name == USER_IS901_BUS_FRAME_NAME_STR


def test_get_frame_info_by_id(spice_handler_loaded):
    """Tests retrieving frame information by ID."""
    frname, center, frclass, frclss_id_list = spice_handler_loaded.get_frame_info_by_id(USER_IS901_BUS_FRAME_ID_INT)

    # Based on your log: frinfo(-999824) returned (-126824, 3, -999824)
    # This means:
    # frname = '-126824' (the S/C ID, which is unusual but what SPICE returns for this ID's "name" via frinfo)
    # center = 3 (Class of frame, e.g., CK-based)
    # frclass = -999824 (Class ID, often the frame ID itself for CK frames of type 3)

    assert frname == str(
        IS901_NAIF_ID_STR)  # Or check against the S/C ID if that's what frinfo consistently returns as name
    # More robustly, check if the ID resolved by namfrm(USER_IS901_BUS_FRAME_NAME_STR) is what we expect
    assert spiceypy.namfrm(USER_IS901_BUS_FRAME_NAME_STR) == USER_IS901_BUS_FRAME_ID_INT

    # We can also check the center and class if those are known and stable for this frame.
    # For example, if IS901_BUS_FRAME is a CK frame (type 3), then frclass should be 3.
    # The 'center' from frinfo is the NAIF ID of the object at the center of the frame.
    # The 'class ID' from frinfo is often the frame ID itself for CK frames.
    # Example checks (these might need adjustment based on your FK definition):
    # assert center == int(IS901_NAIF_ID_STR) # If the frame is centered on the spacecraft
    assert frclass == 3  # Common for CK frames
    assert frclss_id_list == []  # As frinfo returned 3 items, this should be empty

