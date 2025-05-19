# lcforge/src/spice/spice_handler.py

import spiceypy
import numpy as np
from typing import Union, List, Tuple, Set
import logging
import os
import re  # For parsing
from spiceypy.utils.support_types import SPICEINT_CELL


class SpiceHandler:
    def __init__(self):
        self._loaded_kernels: Set[str] = set()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.info("SpiceHandler instance initialized.")

    def _parse_text_kernel_list(self, line_content: str) -> List[str]:
        """Helper to parse a SPICE list like ('item1', 'item2', ...)."""
        items = []
        # Remove parentheses and split by comma, then strip quotes and whitespace
        content_inside_parens = line_content.strip()[1:-1]  # Remove ( and )

        # Regex to find quoted strings, handling potential spaces around commas
        # This is more robust than simple split by comma if paths have spaces (though unlikely for SPICE)
        for item_match in re.finditer(r"'([^']*)'", content_inside_parens):
            items.append(item_match.group(1).strip())
        return items

    def load_metakernel_programmatically(self, metakernel_path: str):
        """
        Reads a metakernel, resolves paths relative to the metakernel's location,
        and furnishes each kernel individually.
        """
        self.logger.info(f"Programmatically loading metakernel: {metakernel_path}")
        metakernel_dir = os.path.dirname(os.path.abspath(metakernel_path))

        path_values_map = {}
        kernels_to_load_raw = []

        try:
            with open(metakernel_path, 'r') as mk_file:
                reading_paths = False
                reading_symbols = False
                reading_kernels = False

                for line in mk_file:
                    line_stripped = line.strip()
                    if not line_stripped or line_stripped.startswith('\\begintext') or line_stripped.startswith(
                            '\\begindata'):
                        continue

                    if 'PATH_VALUES' in line_stripped:
                        pv_line = line_stripped.split('=')[1].strip()
                        path_values_list = self._parse_text_kernel_list(pv_line)
                        reading_paths = True  # Assuming single line for simplicity now
                    elif 'PATH_SYMBOLS' in line_stripped and reading_paths:
                        ps_line = line_stripped.split('=')[1].strip()
                        path_symbols_list = self._parse_text_kernel_list(ps_line)
                        if len(path_values_list) == len(path_symbols_list):
                            for i, sym in enumerate(path_symbols_list):
                                # Resolve path in PATH_VALUES relative to metakernel_dir
                                resolved_pv_path = os.path.normpath(os.path.join(metakernel_dir, path_values_list[i]))
                                path_values_map[f"${sym}"] = resolved_pv_path  # Store with $ prefix
                        else:
                            self.logger.error("Mismatch between PATH_VALUES and PATH_SYMBOLS count.")
                        reading_paths = False
                        reading_symbols = True
                    elif 'KERNELS_TO_LOAD' in line_stripped:
                        ktl_line = line_stripped.split('=')[1].strip()
                        kernels_to_load_raw = self._parse_text_kernel_list(ktl_line)
                        reading_kernels = True  # Assuming single line for simplicity
                        break  # Assuming KERNELS_TO_LOAD is the last relevant part for this parser

            self.logger.debug(f"Parsed PATH_VALUES_MAP: {path_values_map}")
            self.logger.debug(f"Parsed KERNELS_TO_LOAD_RAW: {kernels_to_load_raw}")

            # Resolve and furnish kernels
            for kernel_entry in kernels_to_load_raw:
                resolved_kernel_path = kernel_entry
                # Substitute path symbols
                for symbol, path_prefix in path_values_map.items():
                    if kernel_entry.startswith(symbol):
                        resolved_kernel_path = kernel_entry.replace(symbol, path_prefix)
                        break  # Symbol found and replaced

                # If no symbol was found, and it's a relative path, assume it's relative to metakernel_dir
                if not os.path.isabs(resolved_kernel_path) and not any(
                        kernel_entry.startswith(s) for s in path_values_map.keys()):
                    resolved_kernel_path = os.path.normpath(os.path.join(metakernel_dir, kernel_entry))

                self.logger.info(f"Attempting to furnish individual kernel: {resolved_kernel_path}")
                self.load_kernel(resolved_kernel_path)  # Use existing load_kernel method

            # Add the metakernel itself to the list of "loaded" kernels for tracking,
            # even though its content was processed programmatically.
            if metakernel_path not in self._loaded_kernels:
                self._loaded_kernels.add(metakernel_path)
            self.logger.info(f"Finished programmatic processing of metakernel: {metakernel_path}")

        except Exception as e:
            self.logger.error(f"Error programmatically parsing or loading metakernel {metakernel_path}: {e}")
            raise

    # Keep original load_metakernel for now, or replace its call sites
    def load_metakernel(self, metakernel_path: str):
        # Option 1: Call the new programmatic loader
        self.load_metakernel_programmatically(metakernel_path)

        # Option 2: Keep the old direct furnsh for comparison if needed, but programmatic is preferred now
        # if metakernel_path not in self._loaded_kernels:
        #     try:
        #         spiceypy.furnsh(metakernel_path)
        #         self._loaded_kernels.add(metakernel_path)
        #         self.logger.info(f"Loaded SPICE metakernel (direct furnsh): {metakernel_path}")
        #     except spiceypy.utils.exceptions.SpiceyError as e:
        #         self.logger.error(f"SPICE error loading metakernel (direct furnsh) {metakernel_path}: {e}")
        #         raise
        # else:
        #     self.logger.debug(f"Metakernel already loaded, skipping: {metakernel_path}")

    # ... (all other methods: load_kernel, unload_kernel, unload_all_kernels, utc_to_et, et_to_utc, etc. remain THE SAME)
    def load_kernel(self, kernel_path: Union[
        str, List[str]]):  # Duplicated for completeness, ensure it's only once in actual file
        if isinstance(kernel_path, str):
            kernel_paths_to_load = [kernel_path]
        elif isinstance(kernel_path, list):
            kernel_paths_to_load = kernel_path
        else:
            msg = "kernel_path must be a string or a list of strings."
            self.logger.error(msg)
            raise TypeError(msg)
        for path in kernel_paths_to_load:
            if path not in self._loaded_kernels:
                try:
                    spiceypy.furnsh(path)  # Individual kernels are furnished directly
                    self._loaded_kernels.add(path)
                    self.logger.info(f"Loaded SPICE kernel: {path}")
                except spiceypy.utils.exceptions.SpiceyError as e:
                    self.logger.error(f"SPICE error loading kernel {path}: {e}")
                    raise
            else:
                self.logger.debug(f"Kernel already loaded, skipping: {path}")

    def unload_kernel(self, kernel_path: str):
        if kernel_path in self._loaded_kernels:
            try:
                spiceypy.unload(kernel_path)
                self._loaded_kernels.remove(kernel_path)
                self.logger.info(f"Unloaded SPICE kernel: {kernel_path}")
            except spiceypy.utils.exceptions.SpiceyError as e:
                self.logger.error(f"SPICE error unloading kernel {kernel_path}: {e}")
                raise
        else:
            self.logger.warning(f"Attempted to unload kernel not in loaded set: {kernel_path}")

    def unload_all_kernels(self):
        try:
            spiceypy.kclear()
            self._loaded_kernels.clear()
            self.logger.info("All SPICE kernels unloaded and pool cleared (kclear).")
        except spiceypy.utils.exceptions.SpiceyError as e:
            self.logger.error(f"SPICE error during kclear: {e}")
            raise

    def utc_to_et(self, utc_time_str: str) -> float:
        try:
            et = spiceypy.utc2et(utc_time_str)
            self.logger.debug(f"Converted UTC '{utc_time_str}' to ET {et}.")
            return et
        except spiceypy.utils.exceptions.SpiceyError as e:
            self.logger.error(f"SPICE error converting UTC '{utc_time_str}' to ET: {e}")
            raise

    def et_to_utc(self, et: float, time_format: str = "ISOC", precision: int = 3) -> str:
        try:
            utc_str = spiceypy.et2utc(et, time_format, precision)
            self.logger.debug(f"Converted ET {et} to UTC '{utc_str}' (Format: {time_format}, Precision: {precision}).")
            return utc_str
        except spiceypy.utils.exceptions.SpiceyError as e:
            self.logger.error(f"SPICE error converting ET {et} to UTC: {e}")
            raise

    def get_body_position(self, target: str, et: float, frame: str, observer: str,
                          aberration_correction: str = 'NONE') -> Tuple[np.ndarray, float]:
        try:
            target_str = str(target)
            observer_str = str(observer)
            position_vector, light_time = spiceypy.spkpos(targ=target_str, et=et, ref=frame,
                                                          abcorr=aberration_correction, obs=observer_str)
            self.logger.debug(
                f"Position of '{target_str}' relative to '{observer_str}' in '{frame}' at ET {et} (abcorr: '{aberration_correction}'): {position_vector}, LT: {light_time}s.")
            return np.array(position_vector), light_time
        except spiceypy.utils.exceptions.SpiceyError as e:
            self.logger.error(
                f"SPICE error getting position for target='{target}', observer='{observer}', frame='{frame}', et={et}: {e}")
            raise

    def get_target_orientation(self, from_frame: str, to_frame: str, et: float) -> np.ndarray:
        try:
            self.logger.debug(f"Attempting pxform from '{from_frame}' to '{to_frame}' at ET {et}")
            rotation_matrix = spiceypy.pxform(from_frame, to_frame, et)
            self.logger.info(f"Successfully obtained rotation matrix from '{from_frame}' to '{to_frame}'.")
            return np.array(rotation_matrix)
        except spiceypy.utils.exceptions.SpiceyError as e:
            self.logger.error(
                f"SPICE error getting orientation matrix for from_frame='{from_frame}', to_frame='{to_frame}', et={et}: {e}")
            raise

    def get_frame_name_from_id(self, frame_id: int) -> str:
        try:
            frame_name = spiceypy.frmnam(frame_id)
            if not frame_name:
                raise ValueError(f"No frame name found for ID {frame_id} by frmnam.")
            self.logger.debug(f"Frame ID {frame_id} resolved to name '{frame_name}' by frmnam.")
            return frame_name
        except spiceypy.utils.exceptions.SpiceyError as e:
            self.logger.error(f"SPICE error getting frame name for ID {frame_id}: {e}")
            raise ValueError(f"Could not get frame name for ID {frame_id} due to SPICE error.") from e

    def get_frame_id_from_name(self, frame_name: str) -> int:
        try:
            frame_id = spiceypy.namfrm(frame_name)
            if frame_id == 0:
                raise ValueError(f"No frame ID found for name '{frame_name}'.")
            self.logger.debug(f"Frame name '{frame_name}' resolved to ID {frame_id} by namfrm.")
            return frame_id
        except spiceypy.utils.exceptions.SpiceyError as e:
            self.logger.error(f"SPICE error getting frame ID for name '{frame_name}': {e}")
            raise ValueError(f"Could not get frame ID for name '{frame_name}' due to SPICE error.") from e

    def get_frame_info_by_id(self, frame_id: int) -> Tuple[str, int, int, List[int]]:
        try:
            self.logger.debug(f"Calling spiceypy.frinfo with ID: {frame_id}")
            raw_frinfo_result = spiceypy.frinfo(frame_id)
            self.logger.debug(f"Raw result from spiceypy.frinfo({frame_id}): {raw_frinfo_result}")
            frclss_id_list = []
            frname, center, frclass = "", 0, 0
            if isinstance(raw_frinfo_result, tuple):
                if len(raw_frinfo_result) == 4:
                    frname, center, frclass, frclss_id_cell = raw_frinfo_result
                    if isinstance(frclss_id_cell, spiceypy.utils.support_types.SpiceCell):
                        frclss_id_list = [frclss_id_cell[i] for i in range(spiceypy.card(frclss_id_cell))]
                elif len(raw_frinfo_result) == 3:
                    frname, center, frclass = raw_frinfo_result
                else:
                    raise ValueError(f"frinfo returned unexpected items: {len(raw_frinfo_result)}")
            else:
                raise ValueError(f"frinfo did not return tuple: {raw_frinfo_result}")
            self.logger.debug(
                f"Frame Info for ID {frame_id}: Name='{frname}', Center={center}, Class={frclass}, ClassIDList={frclss_id_list}")
            return frname, center, frclass, frclss_id_list
        except spiceypy.utils.exceptions.SpiceyError as e:
            self.logger.error(f"SPICE error getting frame info for ID {frame_id}: {e}")
            raise

    def __enter__(self):
        self.logger.debug("SpiceHandler context entered.")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unload_all_kernels()
        self.logger.debug("SpiceHandler context exited, all kernels unloaded.")


# --- Main execution block for testing (can be kept for direct script testing) ---
if __name__ == '__main__':
    # ... (The __main__ block from the previous version can remain largely the same,
    #      as it calls sh.load_metakernel(), which now uses the programmatic loader)
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    main_logger = logging.getLogger(f"{__name__}.__main__")
    main_logger.info("Running SpiceHandler example with programmatic metakernel loading.")

    current_script_path = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_script_path, '..', '..'))
    data_dir = os.path.join(project_root, "data", "spice_kernels")
    is901_metakernel_path = os.path.join(data_dir, "missions", "dst-is901", "INTELSAT_901-metakernel.tm")

    IS901_NAIF_ID_STR = "-126824"
    OBSERVER_DST_NAIF_ID_STR = "399999"
    USER_IS901_BUS_FRAME_NAME_STR = "IS901_BUS_FRAME"
    USER_IS901_BUS_FRAME_ID_INT = -999824
    TARGET_ET_FOR_TEST = 6.34169191e+08
    EARTH_NAIF_ID_STR = "399"
    INERTIAL_FRAME_NAME = "J2000"

    try:
        with SpiceHandler() as sh:
            # This now calls load_metakernel_programmatically
            sh.load_metakernel(is901_metakernel_path)
            main_logger.info(f"Programmatic processing of metakernel completed: {is901_metakernel_path}")

            # Rest of the tests from the previous __main__ block...
            test_utc_time_str_from_et = "UNKNOWN_UTC"
            try:
                test_utc_time_str_from_et = sh.et_to_utc(TARGET_ET_FOR_TEST, precision=6)
                main_logger.info(
                    f"--- Test Time (from user's ET {TARGET_ET_FOR_TEST}): {test_utc_time_str_from_et} ---")
            except Exception as e:
                main_logger.error(f"Error converting TARGET_ET_FOR_TEST ({TARGET_ET_FOR_TEST}) to UTC: {e}. Check LSK.")

            current_et_for_tests = TARGET_ET_FOR_TEST

            main_logger.info(f"--- Testing IS901 Position (ID: '{IS901_NAIF_ID_STR}') ---")
            try:
                pos_vec, light_time_sec = sh.get_body_position(
                    target=IS901_NAIF_ID_STR, et=current_et_for_tests, frame=INERTIAL_FRAME_NAME,
                    observer=EARTH_NAIF_ID_STR, aberration_correction="LT+S")
                main_logger.info(f"IS901 position rel to Earth: {pos_vec} km (LT: {light_time_sec:.6f}s)")
            except Exception as e:
                main_logger.error(f"SPICE error getting IS901 position: {e}")

            main_logger.info(f"--- Testing DST Observer Position (ID: '{OBSERVER_DST_NAIF_ID_STR}') ---")
            try:
                pos_vec_dst, lt_dst = sh.get_body_position(
                    target=OBSERVER_DST_NAIF_ID_STR, et=current_et_for_tests, frame=INERTIAL_FRAME_NAME,
                    observer=EARTH_NAIF_ID_STR)
                main_logger.info(f"DST Observer position rel to Earth: {pos_vec_dst} km (LT: {lt_dst:.6f}s)")
            except Exception as e:
                main_logger.error(f"SPICE error getting DST Observer position: {e}")

            main_logger.info(
                f"--- Testing IS901 Orientation (Frame Name: '{USER_IS901_BUS_FRAME_NAME_STR}', Expected ID: {USER_IS901_BUS_FRAME_ID_INT}) ---")
            try:
                resolved_id_from_name = sh.get_frame_id_from_name(USER_IS901_BUS_FRAME_NAME_STR)
                if resolved_id_from_name == USER_IS901_BUS_FRAME_ID_INT:
                    main_logger.info(
                        f"Confirmed: Name '{USER_IS901_BUS_FRAME_NAME_STR}' maps to ID {USER_IS901_BUS_FRAME_ID_INT} via namfrm.")
                else:
                    main_logger.error(
                        f"CRITICAL MISMATCH: Name '{USER_IS901_BUS_FRAME_NAME_STR}' maps to {resolved_id_from_name}, but expected ID was {USER_IS901_BUS_FRAME_ID_INT}.")

                frname_info, center_info, frclass_info, frclss_id_list_info = sh.get_frame_info_by_id(
                    USER_IS901_BUS_FRAME_ID_INT)
                main_logger.info(
                    f"Info for Frame ID {USER_IS901_BUS_FRAME_ID_INT}: Official Name (from frinfo)='{frname_info}', Center={center_info}, Class={frclass_info}, ClassID={frclss_id_list_info}")

                main_logger.info(
                    f"Attempting to get orientation for frame '{USER_IS901_BUS_FRAME_NAME_STR}' relative to '{INERTIAL_FRAME_NAME}'")
                rot_matrix = sh.get_target_orientation(
                    from_frame=INERTIAL_FRAME_NAME, to_frame=USER_IS901_BUS_FRAME_NAME_STR, et=current_et_for_tests)
                main_logger.info(f"IS901 Bus ('{USER_IS901_BUS_FRAME_NAME_STR}') orientation matrix:\n{rot_matrix}")

            except ValueError as e:
                main_logger.error(
                    f"Frame setup error for '{USER_IS901_BUS_FRAME_NAME_STR}' (ID: {USER_IS901_BUS_FRAME_ID_INT}): {e}.")
            except spiceypy.utils.exceptions.SpiceyError as e:
                main_logger.error(
                    f"SPICE error during frame processing or orientation for '{USER_IS901_BUS_FRAME_NAME_STR}' (ID: {USER_IS901_BUS_FRAME_ID_INT}): {e}")

        main_logger.info("Exited SpiceHandler context. Kernels unloaded.")
        main_logger.info("Kernel unload verification step (previously here) has been removed.")

    except FileNotFoundError as e:
        main_logger.error(f"CRITICAL FILE NOT FOUND during setup: {e}.")
    except Exception as e:
        main_logger.exception(f"An unexpected error occurred: {e}")
