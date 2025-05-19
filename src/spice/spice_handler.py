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

    def _parse_text_kernel_list(self, buffer_content: str) -> List[str]:
        """
        Parses a SPICE list like ('item1' 'item2'...) which might have been read
        across multiple lines and accumulated into buffer_content.
        """
        items = []
        # Clean the buffer: remove leading/trailing whitespace.
        content_to_parse = buffer_content.strip()

        # Expects format like "('itemA' 'itemB')" or "('itemA',\n 'itemB')"
        # Remove the outer parentheses if they are definitely the start and end
        if content_to_parse.startswith('(') and content_to_parse.endswith(')'):
            content_to_parse = content_to_parse[1:-1].strip()  # Get content within ()
        else:
            # This might occur if the metakernel format is unexpected or buffer isn't formed correctly.
            self.logger.warning(
                f"Parsing list content that does not strictly start with '(' and end with ')': '{buffer_content}'"
            )
            # Attempt to parse anyway if it seems to contain quoted items.

        # Regex to find all single-quoted strings
        # This pattern handles 'item1' 'item2' by finding each item.
        for item_match in re.finditer(r"'([^']*)'", content_to_parse):
            items.append(item_match.group(1).strip())

        if not items and content_to_parse:  # If regex found nothing but there was content
            self.logger.warning(
                f"Could not parse any items from content: '{content_to_parse}'. Original buffer: '{buffer_content}'"
            )
        return items

    def load_metakernel_programmatically(self, metakernel_path: str):
        """
        Reads a metakernel, resolves paths relative to the metakernel's location,
        and furnishes each kernel individually. Handles multi-line assignments.
        """
        self.logger.info(f"Programmatically loading metakernel: {metakernel_path}")
        metakernel_dir = os.path.dirname(os.path.abspath(metakernel_path))

        path_values_map = {}
        # These will store the fully parsed lists from the metakernel
        path_values_list_from_mk = []
        path_symbols_list_from_mk = []
        kernels_to_load_raw = []  # This is used by later logic to furnish kernels

        parsing_section = None  # None, "PATH_VALUES", "PATH_SYMBOLS", "KERNELS_TO_LOAD"
        section_buffer = ""

        try:
            with open(metakernel_path, 'r') as mk_file:
                for line_number, line in enumerate(mk_file, 1):
                    line_stripped = line.strip()

                    # Skip empty lines, SPICE \begin* directives, and comments (if any)
                    if not line_stripped or line_stripped.startswith('\\') or line_stripped.startswith('*'):
                        continue

                    # If we are not currently accumulating a section, look for a new one
                    if parsing_section is None:
                        if "PATH_VALUES" in line_stripped and '=' in line_stripped:
                            parsing_section = "PATH_VALUES"
                            section_buffer = line_stripped.split('=', 1)[1].strip()
                        elif "PATH_SYMBOLS" in line_stripped and '=' in line_stripped:
                            parsing_section = "PATH_SYMBOLS"
                            section_buffer = line_stripped.split('=', 1)[1].strip()
                        elif "KERNELS_TO_LOAD" in line_stripped and '=' in line_stripped:
                            parsing_section = "KERNELS_TO_LOAD"
                            section_buffer = line_stripped.split('=', 1)[1].strip()
                    else:  # We are accumulating for an active parsing_section
                        # Concatenate current line to the buffer. Add a space for separation.
                        # This handles items on new lines.
                        section_buffer += " " + line_stripped

                    # If a section is active and buffer seems complete (starts with '(' and ends with ')')
                    if parsing_section and \
                            section_buffer.strip().startswith("(") and \
                            section_buffer.strip().endswith(")"):

                        self.logger.debug(
                            f"Completed buffer for {parsing_section} at line {line_number}: {section_buffer}")
                        parsed_list = self._parse_text_kernel_list(section_buffer)

                        if parsing_section == "PATH_VALUES":
                            path_values_list_from_mk = parsed_list
                            self.logger.debug(f"Parsed PATH_VALUES: {path_values_list_from_mk}")
                        elif parsing_section == "PATH_SYMBOLS":
                            path_symbols_list_from_mk = parsed_list
                            self.logger.debug(f"Parsed PATH_SYMBOLS: {path_symbols_list_from_mk}")
                        elif parsing_section == "KERNELS_TO_LOAD":
                            kernels_to_load_raw = parsed_list  # This variable is used by subsequent logic
                            self.logger.debug(f"Parsed KERNELS_TO_LOAD_RAW: {kernels_to_load_raw}")
                            # Retain original logic of breaking after KERNELS_TO_LOAD for this specific fix.
                            # For a more general parser, this break might be removed.
                            self.logger.info("Finished parsing KERNELS_TO_LOAD section, stopping metakernel scan.")
                            parsing_section = None
                            section_buffer = ""
                            break

                        parsing_section = None  # Reset for next section
                        section_buffer = ""

            # In case the file ends before KERNELS_TO_LOAD or before a section is properly closed by ')'
            if parsing_section and section_buffer.strip():
                self.logger.warning(
                    f"Metakernel processing ended with active section '{parsing_section}' and buffer '{section_buffer}'. Attempting final parse.")
                if section_buffer.strip().startswith("(") and section_buffer.strip().endswith(")"):
                    parsed_list = self._parse_text_kernel_list(section_buffer)
                    if parsing_section == "PATH_VALUES":
                        path_values_list_from_mk = parsed_list
                    elif parsing_section == "PATH_SYMBOLS":
                        path_symbols_list_from_mk = parsed_list
                    elif parsing_section == "KERNELS_TO_LOAD":
                        kernels_to_load_raw = parsed_list

            # Process PATH_VALUES and PATH_SYMBOLS to create path_values_map
            if path_values_list_from_mk and path_symbols_list_from_mk:
                if len(path_values_list_from_mk) == len(path_symbols_list_from_mk):
                    for i, sym_raw in enumerate(path_symbols_list_from_mk):
                        symbol_clean = sym_raw.strip()
                        path_value_clean = path_values_list_from_mk[i].strip()

                        if not symbol_clean:
                            self.logger.warning(f"Empty symbol found at index {i} in PATH_SYMBOLS. Skipping.")
                            continue
                        if not path_value_clean:
                            self.logger.warning(
                                f"Empty path value found at index {i} for symbol '{symbol_clean}' in PATH_VALUES. Skipping.")
                            continue

                        resolved_pv_path = os.path.normpath(os.path.join(metakernel_dir, path_value_clean))
                        path_values_map[f"${symbol_clean}"] = resolved_pv_path
                else:
                    self.logger.error(
                        f"Mismatch between parsed PATH_VALUES ({len(path_values_list_from_mk)}) and "
                        f"PATH_SYMBOLS ({len(path_symbols_list_from_mk)}) count."
                    )
            elif path_values_list_from_mk or path_symbols_list_from_mk:  # If one list is empty and the other is not
                self.logger.warning(
                    "PATH_VALUES or PATH_SYMBOLS were parsed, but not both. Cannot create path symbol map. "
                    f"PATH_VALUES items: {len(path_values_list_from_mk)}, PATH_SYMBOLS items: {len(path_symbols_list_from_mk)}"
                )

            self.logger.debug(f"Constructed PATH_VALUES_MAP: {path_values_map}")

            # Resolve and furnish kernels from KERNELS_TO_LOAD_RAW
            for kernel_entry in kernels_to_load_raw:
                resolved_kernel_path = kernel_entry  # Default to the entry itself
                # Substitute path symbols
                for symbol, path_prefix in path_values_map.items():
                    if kernel_entry.startswith(symbol):
                        resolved_kernel_path = kernel_entry.replace(symbol, path_prefix)
                        break  # Symbol found and replaced

                # If no symbol was matched and path is not absolute, assume it's relative to metakernel_dir
                # Check if kernel_entry starts with any of the keys from path_values_map to confirm symbol substitution was attempted
                is_symbol_substituted = any(kernel_entry.startswith(s) for s in path_values_map.keys())
                if not os.path.isabs(resolved_kernel_path) and not is_symbol_substituted:
                    resolved_kernel_path = os.path.normpath(os.path.join(metakernel_dir, kernel_entry))

                # If after symbol replacement, path is still not absolute (e.g. symbol mapped to a relative path)
                # This case should ideally be handled by ensuring path_prefix in path_values_map are absolute.
                # The current logic for resolved_pv_path already makes them absolute.

                self.logger.info(f"Attempting to furnish individual kernel: {resolved_kernel_path}")
                self.load_kernel(resolved_kernel_path)

            # Add the metakernel itself to the list of "loaded" kernels for tracking,
            if metakernel_path not in self._loaded_kernels:
                self._loaded_kernels.add(metakernel_path)  # Mark metakernel as processed
            self.logger.info(f"Finished programmatic processing of metakernel: {metakernel_path}")

        except Exception as e:
            self.logger.error(f"Error programmatically parsing or loading metakernel {metakernel_path}: {e}",
                              exc_info=True)
            raise

    # ... (all other methods: load_metakernel, load_kernel, unload_kernel, etc. remain THE SAME) ...
    # Ensure the load_metakernel method correctly calls this programmatic one.
    def load_metakernel(self, metakernel_path: str):
        self.load_metakernel_programmatically(metakernel_path)

    def load_kernel(self, kernel_path: Union[
        str, List[str]]):
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
                    spiceypy.furnsh(path)
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
            if not frame_name:  # frmnam returns empty string if ID not found
                raise ValueError(f"No frame name found for ID {frame_id} by frmnam.")
            self.logger.debug(f"Frame ID {frame_id} resolved to name '{frame_name}' by frmnam.")
            return frame_name
        except spiceypy.utils.exceptions.SpiceyError as e:  # Catch potential SPICE errors
            self.logger.error(f"SPICE error getting frame name for ID {frame_id}: {e}")
            # Reraise as ValueError to be consistent with other ID/Name not found cases
            raise ValueError(f"Could not get frame name for ID {frame_id} due to SPICE error: {e}")

    def get_frame_id_from_name(self, frame_name: str) -> int:
        try:
            frame_id = spiceypy.namfrm(frame_name)
            if frame_id == 0:  # namfrm returns 0 if name not found
                raise ValueError(f"No frame ID found for name '{frame_name}'.")
            self.logger.debug(f"Frame name '{frame_name}' resolved to ID {frame_id} by namfrm.")
            return frame_id
        except spiceypy.utils.exceptions.SpiceyError as e:  # Catch potential SPICE errors
            self.logger.error(f"SPICE error getting frame ID for name '{frame_name}': {e}")
            # Reraise as ValueError
            raise ValueError(f"Could not get frame ID for name '{frame_name}' due to SPICE error: {e}")

    def get_frame_info_by_id(self, frame_id: int) -> Tuple[str, int, int, List[int]]:
        frname = ""
        center = 0
        frclass = 0
        clssid_int = 0  # The single integer class ID from frinfo's 3-tuple return
        frclss_id_list = []

        try:
            # 1. Get frame name using frmnam, as the 3-tuple frinfo output doesn't include it.
            try:
                frname = spiceypy.frmnam(frame_id)
                if not frname:
                    self.logger.warning(
                        f"spiceypy.frmnam({frame_id}) returned an empty name. This might indicate the frame ID is not defined or name is missing from FK.")
                    # We will proceed, as frinfo might still return data if the ID is known numerically.
            except spiceypy.utils.exceptions.SpiceyError as e_frmnam:
                self.logger.error(f"SPICE error from spiceypy.frmnam({frame_id}): {e_frmnam}")
                # If frmnam fails with a SPICE error, it's more serious.
                raise ValueError(
                    f"Could not get frame name for ID {frame_id} via frmnam due to SPICE error.") from e_frmnam

            # 2. Call frinfo and expect 3 items: (center, frclass, clssid_int)
            self.logger.debug(f"Calling spiceypy.frinfo with ID: {frame_id}")
            raw_frinfo_result = spiceypy.frinfo(frame_id)
            self.logger.info(
                f"Raw result from spiceypy.frinfo({frame_id}): {raw_frinfo_result} (type: {type(raw_frinfo_result)})")

            if isinstance(raw_frinfo_result, tuple) and len(raw_frinfo_result) == 3:
                center, frclass, clssid_int = raw_frinfo_result
                self.logger.debug(
                    f"Unpacked 3 items from frinfo: Center={center}, Class={frclass}, ClassID(int)={clssid_int}")

                # If frmnam returned an empty name but frinfo gave data, it implies a numeric ID exists
                # but might not have a name defined in the loaded text kernels.
                if not frname and center != 0:  # center != 0 is a heuristic that frinfo found *something*
                    self.logger.warning(
                        f"Frame ID {frame_id} yielded info from frinfo (Center: {center}) but no name from frmnam. Frame might be unnamed.")
                    # Proceeding without a name, or you could assign a placeholder like f"UNNAMED_FRAME_{frame_id}"
                    # For now, frname remains empty; the caller should handle this.
            else:
                self.logger.error(
                    f"spiceypy.frinfo({frame_id}) returned an unexpected result: {raw_frinfo_result}. Expected a 3-tuple.")
                raise ValueError(
                    f"spiceypy.frinfo for ID {frame_id} did not return the expected 3-tuple. Got: {raw_frinfo_result}")

            # Populate frclss_id_list based on the single clssid_int.
            # Your .tf file defines FRAME_-999824_CLASS_ID = -999824, so clssid_int should be -999824.
            # A value of 0 for clssid_int might mean "no specific class ID assigned" or "default".
            if clssid_int != 0:
                frclss_id_list = [clssid_int]

            self.logger.debug(
                f"Frame Info for ID {frame_id}: Name='{frname}', Center={center}, Class={frclass}, ClassID(int)={clssid_int} -> Derived ClassIDList={frclss_id_list}")

            # Ensure a name is present if the frame implies it should have one (e.g. center is not 0)
            if not frname and (center != 0 or frclass != 0 or clssid_int != 0):
                self.logger.warning(
                    f"Frame ID {frame_id} has attributes from frinfo but no name from frmnam. Proceeding with empty name.")

            return frname, center, frclass, frclss_id_list

        except ValueError as e:  # Handle ValueErrors raised within this function
            self.logger.error(f"ValueError in get_frame_info_by_id for ID {frame_id}: {e}", exc_info=True)
            raise
        except spiceypy.utils.exceptions.SpiceyError as e:  # Handle other SPICE errors
            self.logger.error(f"Generic SPICE error in get_frame_info_by_id for ID {frame_id}: {e}", exc_info=True)
            raise ValueError(f"Underlying SPICE error in get_frame_info_by_id for ID {frame_id}: {e}")
        except Exception as e:  # Catch any other unexpected errors
            self.logger.error(f"Unexpected generic error in get_frame_info_by_id for ID {frame_id}: {e}", exc_info=True)
            raise ValueError(f"Unexpected error in get_frame_info_by_id for ID {frame_id}: {e}")

    def __enter__(self):
        self.logger.debug("SpiceHandler context entered.")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unload_all_kernels()
        self.logger.debug("SpiceHandler context exited, all kernels unloaded.")


# --- Main execution block for testing (can be kept for direct script testing) ---
if __name__ == '__main__':
    if not logging.getLogger().hasHandlers():  # Ensure basicConfig is only called once
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    main_logger = logging.getLogger(f"{__name__}.__main__")  # Use __name__ for the script itself
    main_logger.info("Running SpiceHandler example with programmatic metakernel loading.")

    # Determine paths relative to the script's location
    try:
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        # Assuming project structure src/spice/spice_handler.py and data/spice_kernels/...
        project_root = os.path.abspath(os.path.join(current_script_dir, '..', '..'))
        data_dir = os.path.join(project_root, "data", "spice_kernels")
        is901_metakernel_path = os.path.join(data_dir, "missions", "dst-is901", "INTELSAT_901-metakernel.tm")

        if not os.path.exists(is901_metakernel_path):
            main_logger.error(f"CRITICAL: Metakernel not found at resolved path: {is901_metakernel_path}")
            # Try a relative path from CWD as a fallback for different execution contexts
            # This assumes CWD is project root
            fallback_path = os.path.join("data", "spice_kernels", "missions", "dst-is901", "INTELSAT_901-metakernel.tm")
            if os.path.exists(fallback_path):
                is901_metakernel_path = fallback_path
                main_logger.info(f"Using fallback metakernel path: {is901_metakernel_path}")
            else:
                raise FileNotFoundError(f"Metakernel not found at primary or fallback path: {is901_metakernel_path}")

        IS901_NAIF_ID_STR = "-126824"
        OBSERVER_DST_NAIF_ID_STR = "399999"  # Example ID, ensure it's correct for your kernels
        USER_IS901_BUS_FRAME_NAME_STR = "IS901_BUS_FRAME"  # From your .tf file
        USER_IS901_BUS_FRAME_ID_INT = -999824  # Example, ensure this matches your .tf definition
        TARGET_ET_FOR_TEST = 6.34169191e+08  # Example ET
        EARTH_NAIF_ID_STR = "399"
        INERTIAL_FRAME_NAME = "J2000"

        with SpiceHandler() as sh:
            sh.load_metakernel(is901_metakernel_path)  # This now calls the improved programmatic loader
            main_logger.info(f"Metakernel processing completed: {is901_metakernel_path}")

            test_utc_time_str_from_et = "UNKNOWN_UTC"
            try:
                test_utc_time_str_from_et = sh.et_to_utc(TARGET_ET_FOR_TEST, precision=6)
                main_logger.info(
                    f"--- Test Time (from ET {TARGET_ET_FOR_TEST}): {test_utc_time_str_from_et} ---")
            except Exception as e:
                main_logger.error(f"Error converting TARGET_ET_FOR_TEST ({TARGET_ET_FOR_TEST}) to UTC: {e}. Check LSK.")

            current_et_for_tests = TARGET_ET_FOR_TEST  # Use the same ET for subsequent tests for consistency

            main_logger.info(f"--- Testing IS901 Position (ID: '{IS901_NAIF_ID_STR}') ---")
            try:
                pos_vec, light_time_sec = sh.get_body_position(
                    target=IS901_NAIF_ID_STR, et=current_et_for_tests, frame=INERTIAL_FRAME_NAME,
                    observer=EARTH_NAIF_ID_STR, aberration_correction="LT+S")
                main_logger.info(f"IS901 position rel to Earth: {pos_vec} km (LT: {light_time_sec:.6f}s)")
            except Exception as e:
                main_logger.error(f"SPICE error getting IS901 position: {e}")

            # Test for DST Observer if defined and relevant
            # main_logger.info(f"--- Testing DST Observer Position (ID: '{OBSERVER_DST_NAIF_ID_STR}') ---")
            # try:
            #     pos_vec_dst, lt_dst = sh.get_body_position(
            #         target=OBSERVER_DST_NAIF_ID_STR, et=current_et_for_tests, frame=INERTIAL_FRAME_NAME,
            #         observer=EARTH_NAIF_ID_STR) # Assuming DST observer relative to Earth
            #     main_logger.info(f"DST Observer position rel to Earth: {pos_vec_dst} km (LT: {lt_dst:.6f}s)")
            # except Exception as e:
            #     main_logger.error(f"SPICE error getting DST Observer position: {e}")

            main_logger.info(
                f"--- Testing IS901 Orientation (Frame Name: '{USER_IS901_BUS_FRAME_NAME_STR}') ---")
            try:
                # Verify frame name to ID mapping
                resolved_id_from_name = sh.get_frame_id_from_name(USER_IS901_BUS_FRAME_NAME_STR)
                main_logger.info(
                    f"Name '{USER_IS901_BUS_FRAME_NAME_STR}' maps to ID {resolved_id_from_name} via namfrm.")

                # If you have an expected ID, you can assert/check it.
                # if resolved_id_from_name == USER_IS901_BUS_FRAME_ID_INT:
                #     main_logger.info(f"Confirmed: ID matches expected {USER_IS901_BUS_FRAME_ID_INT}.")
                # else:
                #     main_logger.warning(f"ID MISMATCH: Name '{USER_IS901_BUS_FRAME_NAME_STR}' resolved to {resolved_id_from_name}, expected {USER_IS901_BUS_FRAME_ID_INT if USER_IS901_BUS_FRAME_ID_INT else 'N/A'}.")

                # Get frame info using the resolved ID (or expected ID if confident)
                frname_info, center_info, frclass_info, frclss_id_list_info = sh.get_frame_info_by_id(
                    resolved_id_from_name)
                main_logger.info(
                    f"Info for Frame ID {resolved_id_from_name} (from frinfo): Official Name='{frname_info}', Center={center_info}, Class={frclass_info}, ClassIDList={frclss_id_list_info}")

                main_logger.info(
                    f"Attempting to get orientation for frame '{USER_IS901_BUS_FRAME_NAME_STR}' relative to '{INERTIAL_FRAME_NAME}'")
                rot_matrix = sh.get_target_orientation(
                    from_frame=INERTIAL_FRAME_NAME, to_frame=USER_IS901_BUS_FRAME_NAME_STR, et=current_et_for_tests)
                main_logger.info(
                    f"IS901 Bus ('{USER_IS901_BUS_FRAME_NAME_STR}') orientation matrix relative to {INERTIAL_FRAME_NAME}:\n{rot_matrix}")

            except ValueError as e:  # Catches explicit ValueError from get_frame_id/name/info
                main_logger.error(
                    f"Frame setup error for '{USER_IS901_BUS_FRAME_NAME_STR}': {e}.")
            except spiceypy.utils.exceptions.SpiceyError as e:  # Catches other SPICE errors
                main_logger.error(
                    f"SPICE error during frame processing or orientation for '{USER_IS901_BUS_FRAME_NAME_STR}': {e}")
            except Exception as e:  # Catch any other unexpected errors
                main_logger.error(f"Unexpected error during orientation test: {e}", exc_info=True)


    except FileNotFoundError as e:
        main_logger.error(f"CRITICAL FILE NOT FOUND during setup: {e}. Please check paths.")
        main_logger.error(
            f"Attempted metakernel path: {is901_metakernel_path if 'is901_metakernel_path' in locals() else 'Not resolved'}")
        main_logger.error(f"Current working directory: {os.getcwd()}")
        main_logger.error(f"Project root (estimated): {project_root if 'project_root' in locals() else 'Not resolved'}")

    except Exception as e:
        main_logger.exception(f"An unexpected error occurred in the main execution block: {e}")

    finally:
        main_logger.info("SpiceHandler example finished.")