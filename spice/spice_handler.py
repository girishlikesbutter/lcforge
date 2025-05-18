# lcforge/spice/spice_handler.py

import spiceypy
import numpy as np
from typing import Union, List, Tuple, Set
import logging

# Get a logger for this module
logger = logging.getLogger(__name__)


class SpiceHandler:
    """
    A class to handle SPICE operations, including kernel management,
    time conversions, and ephemeris data retrieval.
    """

    def __init__(self):
        """
        Initializes the SpiceHandler.
        Keeps track of loaded kernels to prevent redundant loading
        and to facilitate unloading.
        """
        self._loaded_kernels: Set[str] = set()
        logger.info("SpiceHandler initialized.")

    def load_kernel(self, kernel_path: Union[str, List[str]]):
        """
        Loads one or more SPICE kernels.

        Args:
            kernel_path: A string path to a single kernel file or a list of paths.

        Raises:
            spiceypy.utils.exceptions.SpiceyError: If SPICE encounters an error.
            FileNotFoundError: If a kernel file is not found.
        """
        if isinstance(kernel_path, str):
            kernel_paths_to_load = [kernel_path]
        elif isinstance(kernel_path, list):
            kernel_paths_to_load = kernel_path
        else:
            msg = "kernel_path must be a string or a list of strings."
            logger.error(msg)
            raise TypeError(msg)

        for path in kernel_paths_to_load:
            if path not in self._loaded_kernels:
                try:
                    spiceypy.furnsh(path)
                    self._loaded_kernels.add(path)
                    logger.info(f"Loaded SPICE kernel: {path}")
                except spiceypy.utils.exceptions.SpiceyError as e:
                    logger.error(f"SPICE error loading kernel {path}: {e}")
                    raise
                except FileNotFoundError as e:  # This might be caught by SPICE, but good to be explicit
                    logger.error(f"Kernel file not found: {path}: {e}")
                    raise
            else:
                logger.debug(f"Kernel already loaded, skipping: {path}")

    def load_metakernel(self, metakernel_path: str):
        """
        Loads a SPICE metakernel (which itself lists other kernels to load).
        The metakernel path itself is also tracked.

        Args:
            metakernel_path: Path to the metakernel file (.tm).

        Raises:
            spiceypy.utils.exceptions.SpiceyError: If SPICE encounters an error.
            FileNotFoundError: If the metakernel file is not found.
        """
        # Metakernels are loaded via furnsh, just like individual kernels.
        # We treat the metakernel itself as a kernel we've "loaded" for tracking.
        if metakernel_path not in self._loaded_kernels:
            try:
                spiceypy.furnsh(metakernel_path)
                self._loaded_kernels.add(metakernel_path)  # Track the metakernel itself
                logger.info(f"Loaded SPICE metakernel: {metakernel_path}")
            except spiceypy.utils.exceptions.SpiceyError as e:
                logger.error(f"SPICE error loading metakernel {metakernel_path}: {e}")
                raise
            except FileNotFoundError as e:
                logger.error(f"Metakernel file not found: {metakernel_path}: {e}")
                raise
        else:
            logger.debug(f"Metakernel already loaded, skipping: {metakernel_path}")

    def unload_kernel(self, kernel_path: str):
        """
        Unloads a specific SPICE kernel.

        Args:
            kernel_path: Path to the kernel file to unload.

        Raises:
            spiceypy.utils.exceptions.SpiceyError: If SPICE encounters an error.
        """
        if kernel_path in self._loaded_kernels:
            try:
                spiceypy.unload(kernel_path)
                self._loaded_kernels.remove(kernel_path)
                logger.info(f"Unloaded SPICE kernel: {kernel_path}")
            except spiceypy.utils.exceptions.SpiceyError as e:
                logger.error(f"SPICE error unloading kernel {kernel_path}: {e}")
                raise
        else:
            logger.warning(f"Attempted to unload kernel not in loaded set: {kernel_path}")

    def unload_all_kernels(self):
        """
        Unloads all SPICE kernels that were loaded through this handler
        and then clears the SPICE kernel pool using kclear.
        This is the most robust way to ensure a clean SPICE state.
        """
        # Unload individually tracked kernels first (optional, as kclear is comprehensive)
        # for k in list(self._loaded_kernels): # Iterate over a copy
        #     try:
        #         spiceypy.unload(k)
        #         logger.debug(f"Individually unloaded {k} before kclear.")
        #     except spiceypy.utils.exceptions.SpiceyError:
        #         logger.warning(f"Could not unload kernel {k} individually. kclear will handle it.")

        try:
            spiceypy.kclear()  # Clears all kernels from SPICE system
            self._loaded_kernels.clear()
            logger.info("All SPICE kernels unloaded and pool cleared (kclear).")
        except spiceypy.utils.exceptions.SpiceyError as e:
            logger.error(f"SPICE error during kclear: {e}")
            raise

    def utc_to_et(self, utc_time_str: str) -> float:
        """
        Converts a UTC time string to Ephemeris Time (ET).

        Args:
            utc_time_str: UTC time string in a format recognizable by SPICE
                          (e.g., "YYYY-MM-DDTHH:MM:SS.sssZ").
                          Requires a leap seconds kernel (LSK) to be loaded.

        Returns:
            The Ephemeris Time (TDB) as a float.

        Raises:
            spiceypy.utils.exceptions.SpiceyError: If SPICE encounters an error
                                                 (e.g., LSK not loaded, invalid time format).
        """
        try:
            et = spiceypy.utc2et(utc_time_str)
            logger.debug(f"Converted UTC '{utc_time_str}' to ET {et}.")
            return et
        except spiceypy.utils.exceptions.SpiceyError as e:
            logger.error(f"SPICE error converting UTC '{utc_time_str}' to ET: {e}")
            raise

    def et_to_utc(self, et: float, time_format: str = "ISOC", precision: int = 3) -> str:
        """
        Converts an Ephemeris Time (ET) to a UTC time string.

        Args:
            et: Ephemeris Time (TDB) as a float.
            time_format: The desired format of the output UTC string.
                         Common formats: "ISOC" (ISO_FORMAT), "CALFMT", "JULIAN".
                         Refer to SPICE documentation for et2utc_ for more formats.
            precision: The number of decimal places for the seconds component (if applicable).
                       Requires a leap seconds kernel (LSK) to be loaded.
        Returns:
            The UTC time string.

        Raises:
            spiceypy.utils.exceptions.SpiceyError: If SPICE encounters an error.
        """
        try:
            utc_str = spiceypy.et2utc(et, time_format, precision)
            logger.debug(f"Converted ET {et} to UTC '{utc_str}' (Format: {time_format}, Precision: {precision}).")
            return utc_str
        except spiceypy.utils.exceptions.SpiceyError as e:
            logger.error(f"SPICE error converting ET {et} to UTC: {e}")
            raise

    def get_body_position(self,
                          target: str,
                          et: float,
                          frame: str,
                          observer: str,
                          aberration_correction: str = 'NONE'
                          ) -> Tuple[np.ndarray, float]:
        """
        Retrieves the position of a target body relative to an observing body.

        Args:
            target: Name or ID of the target body (e.g., "SUN", "EARTH", "MOON", "-901" for a spacecraft).
            et: Ephemeris Time (TDB) at which to get the position.
            frame: The reference frame in which the position is to be expressed (e.g., "J2000", "IAU_EARTH").
            observer: Name or ID of the observing body (e.g., "EARTH", "399").
            aberration_correction: Aberration correction flag (e.g., "NONE", "LT", "LT+S").
                                   Requires appropriate SPK kernels to be loaded.
        Returns:
            A tuple containing:
                - position_vector (np.ndarray): 3D position vector (x, y, z) in km.
                - light_time (float): One-way light time in seconds.

        Raises:
            spiceypy.utils.exceptions.SpiceyError: If SPICE encounters an error
                                                 (e.g., kernels not loaded, unknown body/frame).
        """
        try:
            position_vector, light_time = spiceypy.spkpos(
                targ=target,
                et=et,
                ref=frame,
                abcorr=aberration_correction,
                obs=observer
            )
            logger.debug(
                f"Position of '{target}' relative to '{observer}' in '{frame}' at ET {et} "
                f"(abcorr: '{aberration_correction}'): {position_vector}, LT: {light_time}s."
            )
            return np.array(position_vector), light_time
        except spiceypy.utils.exceptions.SpiceyError as e:
            logger.error(
                f"SPICE error getting position for target='{target}', observer='{observer}', "
                f"frame='{frame}', et={et}: {e}"
            )
            raise

    def get_target_orientation(self,
                               from_frame: str,
                               to_frame: str,
                               et: float
                               ) -> np.ndarray:
        """
        Retrieves the 3x3 rotation matrix that transforms vectors from one
        reference frame to another at a specified ephemeris time.

        This is typically used to get the orientation of a body-fixed frame
        (e.g., a spacecraft's CK frame) relative to an inertial frame (e.g., J2000).

        Args:
            from_frame: The name of the frame to transform from (e.g., "J2000").
            to_frame: The name of the frame to transform to (e.g., "IAU_MARS", a spacecraft CK frame ID).
            et: Ephemeris Time (TDB) at which the transformation is desired.
                Requires appropriate FK, PCK, CK, SCLK kernels to be loaded.

        Returns:
            np.ndarray: A 3x3 rotation matrix. If `r_matrix` is this matrix, then a vector `v_from`
                        in `from_frame` can be transformed to `v_to` in `to_frame` by:
                        `v_to = r_matrix @ v_from` (if using NumPy for matrix multiplication).
                        Note: SPICE's pxform gives from_frame -> to_frame. If you need to_frame -> from_frame, use spiceypy.mxform.
                        The matrix returned by pxform directly transforms vectors *from* from_frame *to* to_frame.
                        To transform a vector *in* from_frame *to* a vector *in* to_frame, use spiceypy.pxform.
                        The matrix returned by spiceypy.pxform(from_frame, to_frame, et) is C_to<-from.
                        So, vec_to = C_to<-from * vec_from.

        Raises:
            spiceypy.utils.exceptions.SpiceyError: If SPICE encounters an error.
        """
        try:
            rotation_matrix = spiceypy.pxform(from_frame, to_frame, et)
            logger.debug(f"Rotation matrix from '{from_frame}' to '{to_frame}' at ET {et} obtained.")
            return np.array(rotation_matrix)
        except spiceypy.utils.exceptions.SpiceyError as e:
            logger.error(
                f"SPICE error getting orientation matrix for from_frame='{from_frame}', "
                f"to_frame='{to_frame}', et={et}: {e}"
            )
            raise

    def __enter__(self):
        """Support for context manager protocol."""
        logger.debug("SpiceHandler context entered.")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Support for context manager protocol.
        Ensures all kernels are unloaded when the context is exited.
        """
        self.unload_all_kernels()
        logger.debug("SpiceHandler context exited, all kernels unloaded.")


if __name__ == '__main__':
    # This is a basic example of how to use the SpiceHandler
    # You'll need to download a generic LSK kernel and place it in a known path.
    # E.g., from https://naif.jpl.nasa.gov/naif/data_generic.html -> LSK -> naifXXXX.tls
    # And a generic PCK, e.g., pckXXXXX.tpc

    # Setup basic logging for the example
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # --- Replace with the actual path to your LSK kernel ---
    lsk_kernel_path = "path_to_your_kernels/naif0012.tls"
    # --- Replace with the actual path to your PCK kernel (optional for this basic example but good practice) ---
    # pck_kernel_path = "path_to_your_kernels/pck00010.tpc"

    try:
        # Example using context manager
        with SpiceHandler() as sh:
            # Load a leap seconds kernel (LSK) - REQUIRED for time conversions
            try:
                sh.load_kernel(lsk_kernel_path)
                # sh.load_kernel(pck_kernel_path) # Optional for this specific example
            except FileNotFoundError:
                logger.error(f"CRITICAL: LSK kernel not found at '{lsk_kernel_path}'. "
                             "Time conversions will fail. Download it and update the path.")
                exit()
            except spiceypy.utils.exceptions.SpiceyError as e:
                logger.error(f"SPICE error loading initial kernels: {e}")
                exit()

            # Time conversion example
            utc_time = "2024-05-20T12:00:00Z"
            try:
                et_time = sh.utc_to_et(utc_time)
                logger.info(f"UTC: {utc_time} -> ET: {et_time}")

                retrieved_utc = sh.et_to_utc(et_time, "ISOC", 3)
                logger.info(f"ET: {et_time} -> UTC: {retrieved_utc}")

            except spiceypy.utils.exceptions.SpiceyError as e:
                logger.error(f"Error during time conversion example: {e}")
                # This usually means LSK is not loaded or SPICE doesn't know about it.

            # Ephemeris example (requires SPK kernels for Sun and Earth, e.g., de430.bsp or similar)
            # For this example to run, you'd need to load an SPK.
            # sh.load_kernel("path_to_your_spk/de430.bsp")
            # try:
            #     sun_pos_wrt_earth, lt = sh.get_body_position(
            #         target="SUN",
            #         et=et_time,
            #         frame="J2000",
            #         observer="EARTH",
            #         aberration_correction="LT+S"
            #     )
            #     logger.info(f"Position of SUN wrt EARTH at ET {et_time} in J2000: {sun_pos_wrt_earth} km (LT: {lt} s)")
            # except spiceypy.utils.exceptions.SpiceyError as e:
            #     logger.warning(f"Could not get Sun position. "
            #                    "This likely means an SPK kernel (e.g., de430.bsp) is not loaded. Details: {e}")

            logger.info(f"Currently loaded kernels: {sh._loaded_kernels}")

        logger.info("Exited SpiceHandler context. Kernels should be unloaded.")
        # To verify, try a SPICE command that requires a kernel (it should fail if LSK is unloaded)
        try:
            spiceypy.utc2et("2024-01-01T00:00:00Z")
        except spiceypy.utils.exceptions.SpiceyError as e:
            logger.info(f"Successfully verified kernel unload (expected error): {e}")

    except Exception as e:
        logger.exception(f"An unexpected error occurred in the example: {e}")

