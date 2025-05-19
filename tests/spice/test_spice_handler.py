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
OBSERVER_DST_NAIF_ID_STR = "399999"  # Example, ensure covered by kernels if tested
EARTH_NAIF_ID_STR = "399"
INERTIAL_FRAME_NAME = "J2000"

USER_IS901_BUS_FRAME_NAME_STR = "IS901_BUS_FRAME"
USER_IS901_BUS_FRAME_ID_INT = -999824  # This is the ID resolved by namfrm

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
    path = os.path.join(
        project_root_dir,
        "data",
        "spice_kernels",
        "missions",
        "dst-is901",
        "INTELSAT_901-metakernel.tm"
    )
    if not os.path.exists(path):
        pytest.fail(f"Metakernel not found at {path}. Ensure the path is correct relative to the test file.")
    return path


@pytest.fixture
def spice_handler_loaded(metakernel_path):
    """
    Provides a SpiceHandler instance with the metakernel loaded.
    Ensures kernels are cleared after the test using its __exit__ method.
    """
    # Using the context manager feature of SpiceHandler
    with SpiceHandler() as handler:
        handler.load_metakernel(metakernel_path)
        yield handler
    # Kernels are automatically unloaded by __exit__


# --- Test Functions ---

def test_metakernel_loading(spice_handler_loaded):
    """Tests if the metakernel and its constituent kernels are loaded."""
    assert spice_handler_loaded._loaded_kernels  # Check if the internal set is not empty
    # Check if SPICE toolkit reports kernels loaded
    # ktotal('ALL') counts all types of kernels.
    # For text kernels like FK, LSK, SCLK, ktotal('text') can be used.
    # For binary kernels SPK, CK, PCK, EK, DSK, ktotal('binary') can be used.
    assert spiceypy.ktotal('ALL') > 0


def test_context_manager(metakernel_path):
    """Tests the context manager ensures kernels are loaded and then cleared."""
    assert spiceypy.ktotal('ALL') == 0  # Should be 0 before context
    with SpiceHandler() as sh:
        sh.load_metakernel(metakernel_path)
        assert spiceypy.ktotal('ALL') > 0  # Should be > 0 inside context after loading
    # After exiting context, ktotal should be 0 due to unload_all_kernels in __exit__
    assert spiceypy.ktotal('ALL') == 0


def test_et_to_utc_conversion(spice_handler_loaded):
    """Tests ET to UTC conversion."""
    utc_str = spice_handler_loaded.et_to_utc(TEST_ET, precision=6)
    assert utc_str == EXPECTED_UTC_FROM_TEST_ET


def test_utc_to_et_conversion(spice_handler_loaded):
    """Tests UTC to ET conversion."""
    et = spice_handler_loaded.utc_to_et(EXPECTED_UTC_FROM_TEST_ET)
    np.testing.assert_allclose(et, TEST_ET, rtol=1e-7)


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
    try:
        frame_id = spice_handler_loaded.get_frame_id_from_name(USER_IS901_BUS_FRAME_NAME_STR)
        assert frame_id == USER_IS901_BUS_FRAME_ID_INT
    except ValueError:
        pytest.fail(f"Frame name '{USER_IS901_BUS_FRAME_NAME_STR}' not found by get_frame_id_from_name.")

    rot_matrix = spice_handler_loaded.get_target_orientation(
        from_frame=INERTIAL_FRAME_NAME,
        to_frame=USER_IS901_BUS_FRAME_NAME_STR,
        et=TEST_ET
    )
    np.testing.assert_allclose(rot_matrix, EXPECTED_IS901_BUS_ORIENTATION, rtol=1e-7, atol=1e-7)


def test_get_frame_id_and_name_mapping(spice_handler_loaded):
    """Tests frame name to ID and ID to name mappings."""
    resolved_id = spice_handler_loaded.get_frame_id_from_name(USER_IS901_BUS_FRAME_NAME_STR)
    assert resolved_id == USER_IS901_BUS_FRAME_ID_INT

    resolved_name = spice_handler_loaded.get_frame_name_from_id(USER_IS901_BUS_FRAME_ID_INT)
    assert resolved_name == USER_IS901_BUS_FRAME_NAME_STR


def test_get_frame_info_by_id(spice_handler_loaded):
    """
    Tests retrieving frame information by ID,
    reflecting the current SpiceHandler.get_frame_info_by_id behavior.
    """
    # USER_IS901_BUS_FRAME_ID_INT is -999824
    # spice_handler.get_frame_info_by_id will:
    # 1. Call spiceypy.frmnam(-999824) -> should return 'IS901_BUS_FRAME'
    # 2. Call spiceypy.frinfo(-999824) -> your log showed it returns (-126824, 3, -999824)
    #    This corresponds to (center, frclass, clssid_int)

    frname, center, frclass, frclss_id_list = spice_handler_loaded.get_frame_info_by_id(USER_IS901_BUS_FRAME_ID_INT)

    assert frname == USER_IS901_BUS_FRAME_NAME_STR  # From frmnam
    assert center == int(IS901_NAIF_ID_STR)  # -126824, from the first element of frinfo's 3-tuple
    assert frclass == 3  # From the second element of frinfo's 3-tuple

    # frclss_id_list is derived from clssid_int (-999824), the third element from frinfo's 3-tuple
    assert frclss_id_list == [USER_IS901_BUS_FRAME_ID_INT]

# Consider adding more tests for:
# - Unloading individual kernels (SpiceHandler.unload_kernel)
# - Error handling (e.g., requesting data for unloaded kernels or invalid IDs)
# - Loading a kernel that doesn't exist (should raise an error)