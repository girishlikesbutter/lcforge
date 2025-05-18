# Development Plan: Satellite Light Curve Generation & Estimation (Rewrite)

**Based on PRD Version:** 1.1
**Date:** 2025-05-19

---

## Overview

This development plan outlines the phases and tasks required to build the new satellite light curve generation and estimation system (`lcforge`), as specified in the Product Requirements Document (PRD v1.1). The plan emphasizes a modular approach, iterative development, and continuous testing.

---

## Phase 0: Project Setup & Foundation

* **Goal:** Establish the development environment, version control, core SPICE integration, basic data structures, and the overall project skeleton.
* **Duration:** 1-2 Weeks
* **Key Tasks:**
    1.  **Environment & Version Control (NFR9, NFR8.6):**
        * Initialize a new Git repository.
        * Define and implement a branching strategy (e.g., Gitflow or a simpler feature branch workflow).
        * Set up Python virtual environment (e.g., using `pyenv` and `venv`) and dependency management (e.g., `requirements.txt`).
        * Install core dependencies: Python (3.10.8), NumPy, SpiceyPy, pytest.
    2.  **Project Structure (NFR1.1):**
        * Define the top-level directory structure for modules (e.g., `lcforge/`, `tests/`, `docs/`, `examples/`, `data/spice_kernels/`).
        * Create initial placeholder files/modules for key components identified in NFR1.1 (e.g., `spice_handler.py`, `model_definitions.py`).
    3.  **Core SPICE Handler (FR2.1):**
        * Implement initial `spice_handler` module:
            * Functionality to load/unload SPICE metakernels or individual kernels.
            * Basic time system conversions (e.g., UTC to ET).
            * Functions to retrieve positions of major celestial bodies (Sun, Earth) at given epochs.
    4.  **Basic Data Structures (FR1):**
        * Define preliminary Python classes or data structures for:
            * `Epoch` (time handling, wrapping SPICE ET).
            * `Vector3D`, `Quaternion`, `RotationMatrix` (potentially using NumPy or dedicated libraries).
            * Initial placeholders for `Satellite`, `Component`, `Facet`.
    5.  **Testing & CI Setup (NFR7):**
        * Set up a testing framework (e.g., `pytest`).
        * Write initial simple tests for the `spice_handler`.
        * (Optional) Configure a basic Continuous Integration (CI) pipeline (e.g., GitHub Actions).
    6.  **Logging Setup (FR7.4):**
        * Implement a basic logging configuration for the project.
* **Deliverables:**
    * Initialized and configured Git repository.
    * A basic, runnable project skeleton.
    * `spice_handler` module capable of loading SPICE kernels and retrieving basic ephemeris data.
    * Initial data class definitions.
    * Working development environment with testing configured.

---

## Phase 1: Satellite Modeling Core

* **Goal:** Implement the ability to define, load, and manage satellite models, including components, their base geometry (mesh for shadowing, facets for light curve), material properties, and basic articulation.
* **Duration:** 3-4 Weeks
* **Key Tasks:**
    1.  **Satellite & Component Definition (`models.model_definitions` module - FR1.1, FR1.2, FR1.4):**
        * Develop classes: `Satellite`, `Component`.
        * Define how a `Component` stores its geometry:
            * Reference to a mesh file (e.g., STL, OBJ path) for shadowing.
            * Definition of simplified facets (e.g., list of vertices, normal vector, area) for light curve calculation. This might involve a process to derive/approximate facets from the mesh or define them separately.
        * Implement assignment of BRDF material properties ($R_d, R_s, n$) to each `Component` or its facets.
    2.  **Mesh Handling (FR1.1, NFR8.3):**
        * Integrate a mesh handling library (e.g., Trimesh).
        * Implement functionality to load mesh files and extract basic properties (vertices, faces).
    3.  **Component Geometry & Hierarchy (FR1.3):**
        * Implement logic for defining a `Component`'s position (offset vector) and orientation (quaternion or rotation matrix) relative to the main `Satellite` body-fixed frame.
    4.  **Component Articulation (`articulation.articulator` module - FR1.3, FR3.4):**
        * Define how articulation rules are specified (e.g., a component's orientation is tied to a specific SPICE frame, like a solar panel tracking a "SUN_POINTING_FRAME").
        * Implement functions within `spice_handler` or a new `attitude_handler` to resolve these articulated component orientations at a given epoch using SPICE.
    5.  **Model Persistence (FR1.5):**
        * Design and implement a file format (e.g., JSON or YAML) for saving and loading complete satellite model definitions (including components, geometry references, material properties, articulation rules).
    6.  **Unit Testing (NFR7):**
        * Write unit tests for model creation, component addition, property assignment, articulation logic, and model persistence.
* **Deliverables:**
    * `models.model_definitions` module: Ability to programmatically define a multi-component satellite with distinct geometry for shadowing (mesh) and light curves (facets), and assign material properties.
    * `articulation.articulator` module: Basic functionality to articulate components based on SPICE frame definitions.
    * Ability to save and load these satellite model definitions from files.
    * Comprehensive unit tests for all new modeling functionalities.

---

## Phase 2: Kinematics, Attitude & Basic Illumination

* **Goal:** Fully integrate SPICE for all satellite and observer kinematics and attitude. Implement basic illumination calculations (direct solar, observer visibility, simple body shadowing).
* **Duration:** 2-3 Weeks
* **Key Tasks:**
    1.  **Advanced SPICE Integration (`spice.spice_handler` enhancements - FR2, FR3):**
        * Robust functions for retrieving satellite state (position, velocity) and attitude (orientation matrix/quaternion) from SPK and CK kernels at any epoch (FR2.2, FR3.2).
        * Handle observer definitions (SPICE ID or kernels) and retrieve observer state (FR2.3).
        * Ensure all necessary reference frame transformations are handled correctly via SPICE (e.g., inertial to body-fixed, body-fixed to component-fixed).
        * Utilize SPICE interpolation for all time-dependent data (FR3.3).
    2.  **Illumination Calculations (Initial `illumination_engine` or within `simulation.lightcurve_engine` - FR4.1, FR4.4):**
        * Calculate the Sun vector in the satellite body frame and relative to each facet.
        * Implement observer visibility checks for facets (back-face culling).
    3.  **Body Shadowing (Eclipse - FR4.3):**
        * Implement basic eclipse checking using SPICE geometry functions (e.g., `gfoclt`) to determine if the satellite is in Earth's (or other bodies') umbra/penumbra.
        * Represent this as a simple binary (shadowed/illuminated) or fractional factor for now.
    4.  **Unit Testing (NFR7):**
        * Test kinematic calculations against known scenarios or SPICE utility outputs.
        * Test illumination logic (Sun vector, visibility, basic eclipse).
* **Deliverables:**
    * A comprehensive `spice.spice_handler` capable of providing all necessary time, position, and attitude information for the satellite, its components, the Sun, and the observer.
    * Functions to determine direct solar illumination angles, facet visibility to observer, and basic eclipse state.
    * Unit tests verifying kinematic and basic illumination accuracy.

---

## Phase 3: Self-Shadowing Analysis

* **Goal:** Implement the mesh-based self-shadowing analysis to calculate shadowing factors for light curve facets.
* **Duration:** 3-5 Weeks (can be complex)
* **Key Tasks:**
    1.  **`illumination.shadowing_analyzer` Module (FR4.2):**
        * Design the interface for the shadowing module: input satellite mesh model, Sun direction(s); output shadowing factors per light curve facet.
        * Integrate chosen mesh library (e.g., Trimesh's `RayMeshIntersector`) for efficient ray-casting.
    2.  **Shadowing Algorithm Implementation (FR4.2):**
        * Implement a ray-tracing or rasterization-based algorithm to determine, for each light curve facet, the fraction of its area that is occluded by other parts of the satellite's mesh.
        * This may involve casting multiple rays from points on the facet towards the Sun or from the Sun towards the facet.
    3.  **Shadowing Factor Calculation & Mapping:**
        * Develop logic to map the mesh-based shadowing results to the (potentially simpler) light curve facets.
        * Decide on the granularity of shadowing analysis (e.g., compute for a set of representative Sun angles and interpolate, or compute on-the-fly if performance allows for specific epochs).
    4.  **Performance Considerations (NFR3.1, NFR3.3):**
        * Profile and optimize the shadowing calculations.
        * Design the algorithm with potential future parallelization in mind (e.g., independent ray casts).
    5.  **Testing & Validation (NFR4, NFR7):**
        * Test with simple geometric configurations (e.g., one plate shadowing another) where results can be manually verified.
        * Visualize shadowing results if possible.
* **Deliverables:**
    * An `illumination.shadowing_analyzer` module that can compute per-facet shadowing factors for a given satellite model and Sun vector.
    * Strategies for managing the computational cost of shadowing (e.g., pre-computation, interpolation).
    * Performance benchmarks for the shadowing analysis.
    * Unit and integration tests, including visual validation where appropriate.

---

## Phase 4: Light Curve Generation

* **Goal:** Develop the engine to generate satellite light curves, incorporating BRDF, all illumination factors (direct, self-shadowing, body-shadowing), and observer visibility.
* **Duration:** 2-3 Weeks
* **Key Tasks:**
    1.  **`simulation.lightcurve_engine` Module (FR5):**
        * Implement the Phong BRDF model (using $R_d, R_s, n$) to calculate reflected light from a facet given illumination and observer geometry (FR5.2).
    2.  **Integration of Illumination Factors (FR5.1, FR5.3):**
        * Combine direct solar illumination, observer visibility, body shadowing (eclipse) status, and self-shadowing factors to determine the effective illumination on each facet contributing to the light curve.
    3.  **Brightness Calculation & Aggregation (FR5.4):**
        * Calculate the brightness contribution from each relevant facet.
        * Sum these contributions to get the total apparent brightness of the satellite at each epoch.
        * Handle units (e.g., flux, magnitudes).
    4.  **Light Curve Generation Over Time (FR5.5):**
        * Implement logic to iterate over a series of epochs, calculate brightness at each, and compile the light curve.
    5.  **Output Formatting (FR5.5, FR7.3):**
        * Provide functionality to output generated light curves in standard formats (e.g., CSV, NumPy array).
    6.  **Testing & Validation (NFR4, NFR7):**
        * Test BRDF implementation.
        * Validate light curves against simple cases, and if possible, against results from your previous `lcg-lowfid` project for comparable scenarios (understanding the model differences).
* **Deliverables:**
    * A `simulation.lightcurve_engine` capable of generating synthetic light curves.
    * Integration of all illumination and shadowing effects into the brightness calculation.
    * Ability to output light curves.
    * Comprehensive tests for the light curve generation process.

---

## Phase 5: Parameter Estimation

* **Goal:** Implement the framework for estimating specified satellite parameters by fitting generated light curves to observational data.
* **Duration:** 3-4 Weeks
* **Key Tasks:**
    1.  **`estimation.estimator` Module (FR6):**
        * Design the structure for the estimation module.
    2.  **Observational Data Input (FR6.5):**
        * Implement functionality to load observed light curve data (time, brightness, uncertainty, observer info).
    3.  **Objective/Likelihood Function (FR6.4):**
        * Define and implement the objective function (e.g., chi-squared) to compare model-generated light curves with observations.
    4.  **Integration with Optimization Libraries (FR6.2, NFR8.5):**
        * Interface with chosen optimization libraries (e.g., `scipy.optimize.least_squares`, NLopt Python bindings).
        * Set up the estimation loop:
            * The optimizer proposes a set of parameters (BRDF: $R_d, R_s, n$; Attitude: offsets/adjustments - FR6.3).
            * The `lightcurve_engine` generates a light curve using these parameters.
            * The objective function evaluates the fit.
    5.  **Parameter Handling:**
        * Manage parameter bounds, fixed vs. free parameters.
        * Output estimated parameters, uncertainties (if provided by optimizer), and goodness-of-fit metrics.
    6.  **Testing (NFR7):**
        * Test with synthetic "observed" data (generated by the model itself with known parameters) to verify parameter recovery.
* **Deliverables:**
    * An `estimation.estimator` module capable of fitting model light curves to observations.
    * Support for estimating BRDF and basic attitude parameters.
    * Examples demonstrating parameter estimation runs.
    * Tests for the estimation pipeline.

---

## Phase 6: Utilities, Visualization, Documentation & Refinement

* **Goal:** Develop supporting utilities, visualization tools, comprehensive documentation, and refine the overall system. This phase often runs partly in parallel with earlier phases and continues post-core feature completion.
* **Duration:** Ongoing, with a focused effort of 2-4 Weeks after core features.
* **Key Tasks:**
    1.  **`io.config_loader` Module (FR7.1, FR7.2, FR7.3):**
        * Develop robust configuration file management (e.g., using YAML or TOML) for simulation scenarios, model paths, SPICE kernel lists, etc.
        * Finalize data input/output utilities for all data types.
    2.  **`visualization_tools` Module (FR8, NFR8.2):**
        * Implement 3D satellite visualization (e.g., using Matplotlib 3D, PyVista, or Open3D) showing the mesh, orientation, component articulation, and optionally illumination/shadowing.
        * Develop flexible light curve plotting functions (observed vs. model, residuals).
        * (Optional) Visualization for estimation results (e.g., corner plots if MCMC is added later, parameter convergence).
    3.  **Comprehensive Documentation (NFR6):**
        * Write detailed API documentation (docstrings, Sphinx).
        * Create user guides, tutorials, and illustrative examples.
        * Document the theoretical models and algorithms used.
    4.  **Testing Suite Finalization (NFR7):**
        * Ensure high test coverage (unit, integration, regression tests).
        * Add end-to-end tests for common use cases.
    5.  **Performance Optimization (NFR3.1):**
        * Profile critical code sections (shadowing, light curve inner loops, estimation).
        * Apply optimizations where necessary.
    6.  **Parallelization Exploration (NFR3.3):**
        * Based on profiling, identify and (if time permits) implement parallelization for key bottlenecks (e.g., using `multiprocessing` or `concurrent.futures` for batch epoch processing or independent ray casts).
    7.  **(New Task) Dockerization (NFR8.1 - related):**
        * Create `Dockerfile` for building the application environment.
        * Develop `docker-compose.yml` for easier local execution if applicable.
        * Add documentation for building and running with Docker.
* **Deliverables:**
    * User-friendly configuration system.
    * Helpful visualization tools.
    * Complete and clear project documentation.
    * A comprehensive and robust test suite.
    * An optimized and well-performing application.
    * Docker setup for portable execution.

---

## Phase 7: Final Testing, Validation & Release Preparation

* **Goal:** Conduct final end-to-end testing, validate against known cases or real data if possible, and prepare the software for an initial release or deployment.
* **Duration:** 1-2 Weeks
* **Key Tasks:**
    1.  **End-to-End Testing (NFR4.2):**
        * Test complete workflows: from model definition and SPICE setup to light curve generation and parameter estimation.
    2.  **Validation:**
        * Validate results against simplified analytical cases.
        * Compare against outputs from other trusted software or (if available and appropriate) real observational data for known objects.
    3.  **Bug Fixing:**
        * Address any remaining bugs or issues identified during testing.
    4.  **Packaging & Distribution:**
        * Prepare the software for distribution (e.g., as a Python package on PyPI).
        * Finalize README and installation instructions.
* **Deliverables:**
    * A thoroughly tested and validated version of the software.
    * A packaged version ready for distribution/use.
    * Finalized user and developer documentation.

---

## Cross-Cutting Concerns (Addressed Throughout All Phases)

* **Modularity (NFR1):** Maintain strict adherence to modular design with clear APIs.
* **Clean Code & Readability (NFR2.2):** Follow PEP 8, write clear comments, and aim for maintainable code. Consider code reviews.
* **Iterative Refinement:** Revisit and refine designs and implementations as the project progresses and understanding deepens.

This development plan provides a roadmap. Flexibility will be needed, and priorities might shift, but this structure should guide the project towards achieving the goals set out in the PRD.

