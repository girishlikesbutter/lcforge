# Product Requirements Document (PRD): Satellite Light Curve Generation & Estimation (Rewrite)

**Version:** 1.1
**Date:** 2025-05-19

---

## 1. Introduction & Goals

**1.1. Project Overview:**
This document outlines the requirements for a new Python-based software system designed for the simulation of satellite light curves and the subsequent estimation of satellite physical and attitude parameters. This project is a complete rewrite of the existing `lcg-lowfid` system, aiming for a more modular, extensible, accurate, and maintainable codebase. The project will be named `lcforge`.

**1.2. Project Goals:**
* **Develop a Core Simulation Engine:** Create a robust engine capable of generating realistic satellite light curves based on detailed physical models.
* **Implement Advanced Modeling:** Incorporate mesh-based geometry for accurate self-shadowing calculations, precise component-level modeling with articulation capabilities, and standard BRDF models.
* **Leverage SPICE Exclusively:** Utilize the SPICE toolkit (via SpiceyPy) as the sole source for all ephemeris, trajectory, attitude, and reference frame data, eliminating the need for other propagators like SGP4.
* **Build a Modular Architecture:** Design the system with distinct, loosely coupled modules to enhance maintainability, testability, and future expandability.
* **Enable Parameter Estimation:** Integrate a flexible framework for estimating specified satellite parameters (BRDF coefficients, attitude) from observed light curve data.
* **Prioritize Performance and Scalability:** Design the system with future parallelization in mind to handle complex models and long simulation runs efficiently.
* **Ensure Accuracy and Validation:** Strive for high physical accuracy and provide mechanisms for validating simulation outputs.

**1.3. Target Audience:**
* Researchers in astrophysics, space situational awareness (SSA), and satellite characterization.
* Satellite operators and engineers requiring tools for signature analysis.
* Students and academics learning about satellite photometry and modeling.

---

## 2. User Stories (Illustrative)

* **As a researcher, I want to** define a complex satellite model using an imported STL mesh and specify its component parts (bus, solar panels, antennas) with their individual positions and orientations, **so that I can** accurately represent the satellite's geometry.
* **As an analyst, I want to** articulate the solar panels of my satellite model to track the Sun according to SPICE data, **so that I can** simulate realistic power generation and thermal effects on the light curve.
* **As a researcher, I want to** load SPICE kernels for satellite trajectory, attitude, observer location, and solar system bodies, **so that I can** have a consistent and high-fidelity kinematic model for my simulation.
* **As an SSA specialist, I want to** compute the self-shadowing on a satellite using its detailed mesh model for various illumination conditions, **so that I can** understand which parts are illuminated at any given time.
* **As a developer, I want to** use the pre-computed shadowing factors for each satellite facet when generating a light curve, **so that I can** efficiently calculate brightness without re-running full mesh ray-tracing for every epoch.
* **As a researcher, I want to** generate a synthetic light curve for a defined satellite model, orbit, attitude, and observer over a specified time window, **so that I can** compare it with actual observations.
* **As an analyst, I want to** estimate the diffuse reflectivity ($R_d$), specular reflectivity ($R_s$), Phong exponent ($n$), and attitude offsets of a satellite by fitting a model to an observed light curve, **so that I can** characterize the satellite.
* **As a developer, I want to** easily extend the system with new BRDF models or estimation algorithms, **so that I can** adapt the software to new research needs.

---

## 3. Functional Requirements (FR)

These define *what* the system should do.

**FR1: Satellite Modeling**
* **FR1.1 Mesh-Based Geometry for Shadowing:** The system must support defining satellite geometry using 3D meshes (e.g., import from common formats like OBJ, STL). These meshes are primarily used for detailed self-shadowing analysis. For light curve generation, a simplified facet representation (derived from or related to the mesh) will be used, with shadowing effects incorporated via pre-calculated factors.
* **FR1.2 Component-Based Architecture:** Satellites shall be modeled as a collection of distinct components (e.g., bus, solar arrays, antennas, sensors), each potentially represented by one or more meshes or facet groups.
* **FR1.3 Precise and Articulable Component Geometry:**
    * Each component (and its constituent mesh/facets) must have a defined 3D position and orientation (transformation matrix or vector and quaternion) relative to a common satellite body-fixed reference frame. Component centers are not assumed to be at the satellite's COM.
    * The system must allow for the articulation of components *after* initial model definition/import. This means the orientation of specific components (e.g., solar panels, antennas) relative to the main satellite body can be dynamically modified based on time or control laws (e.g., solar panels tracking the Sun as defined by SPICE frames).
* **FR1.4 Material Properties:** The system must allow assigning optical material properties to each component or facet group. This includes parameters for a Phong-based BRDF model:
    * $R_d$: Diffuse reflectivity coefficient.
    * $R_s$: Specular reflectivity coefficient.
    * $n$: Phong exponent (specular shininess).
* **FR1.5 Model Persistence:** Ability to define, save, and load satellite models (including geometry references, components, relative positions/orientations, articulation rules, and material properties) from/to a structured file format (e.g., JSON, YAML).
* **FR1.6 Multiple Satellite Instances:** The system should be capable of handling and simulating multiple distinct satellite models.

**FR2: Kinematics & Ephemeris (SPICE-Driven)**
* **FR2.1 Exclusive SPICE Integration:** All positional and orientational data for celestial bodies (Sun, Earth, Moon, etc.), spacecraft, and observers shall be derived exclusively from SPICE kernels (SPK, CK, PCK, LSK, FK) via the SpiceyPy library. No other propagators (e.g., SGP4) will be implemented for trajectory generation.
* **FR2.2 State Vectors & Reference Frames:** Calculate and provide time-tagged positions, velocities, and orientation matrices/quaternions of the satellite, Sun, and observer(s) in consistent celestial and body-fixed reference frames, all managed through SPICE. This includes handling all necessary reference frame transformations.
* **FR2.3 Observer Definition:** Allow defining one or more observers by their SPICE body name/ID or by providing SPICE kernels that define their trajectory (e.g., an SPK for a ground station).

**FR3: Attitude Modeling & Dynamics (SPICE-Driven)**
* **FR3.1 Attitude Representation:** Support various attitude representations (quaternions, Euler angles, rotation matrices) and conversions between them, primarily using SPICE functionalities.
* **FR3.2 Attitude Profiles from SPICE:**
    * Satellite attitude as a function of time must be defined and ingested from SPICE CK kernels.
    * Support for attitude derived from SPICE frames, including nominal pointing (e.g., nadir, Sun-pointing) and specific maneuver sequences defined in CK kernels.
* **FR3.3 Attitude Interpolation (SPICE):** Rely on SPICE's interpolation capabilities (e.g., for CK kernels) to determine satellite and component attitude at arbitrary times.
* **FR3.4 Component Articulation Kinematics:** The orientation of articulable components (see FR1.3) relative to the main satellite body will also be driven by SPICE data (e.g., custom frames defined in FKs and orientations provided in CKs) or by rules that can be translated into SPICE frame definitions.

**FR4: Illumination, Shadowing, and Visibility**
* **FR4.1 Direct Illumination:** Calculate direct solar illumination on each facet, considering its normal and the Sun vector (derived from SPICE).
* **FR4.2 Self-Shadowing Analysis (Mesh-Based):**
    * Implement accurate self-shadowing algorithms where parts of the satellite (represented by detailed meshes) can cast shadows on other parts.
    * This analysis will be performed as a distinct step to determine, for each facet used in light curve calculations, a "shadowing factor" (e.g., fraction of area illuminated) at various epochs or under representative geometric conditions.
    * Employ efficient techniques (e.g., ray tracing or a rasterization-like approach on the meshes) for this step.
* **FR4.3 Body Shadowing (Eclipse):** Determine if the satellite is shadowed by celestial bodies (e.g., Earth, Moon) using SPICE geometry finders, including umbra and penumbra transitions. This will also contribute to the overall illumination status of facets.
* **FR4.4 Observer Visibility:** Determine which facets are visible to a given observer, considering facet normals and the observer vector (back-face culling).
* **FR4.5 (Optional/Advanced) Earthshine & Albedo:** Capability to model illumination from Earthshine and reflected light from Earth (albedo).

**FR5: Light Curve Generation**
* **FR5.1 Brightness Calculation (Facet-Based with Shadow Factor):** Calculate the apparent brightness (e.g., in astronomical magnitudes or flux units) of the satellite as seen by a defined observer at specified epochs. This calculation will be based on simplified facets.
* **FR5.2 BRDF Application:** Apply the defined BRDF model (using $R_d, R_s, n$) to each visible facet to calculate the light reflected towards the observer.
* **FR5.3 Shadow Impact:** The flux contribution from each facet will be scaled by its pre-calculated shadowing factor (from FR4.2) and its eclipse status (FR4.3).
* **FR5.4 Integrated Brightness:** Sum the adjusted brightness contributions from all visible and illuminated facets of the satellite to get the total apparent brightness.
* **FR5.5 Light Curve Output:** Generate time series data of apparent brightness (light curves) and output them in a common, usable format (e.g., CSV, FITS, NumPy arrays).

**FR6: Parameter Estimation**
* **FR6.1 Estimation Framework:** Implement or interface with a flexible parameter estimation framework.
* **FR6.2 Algorithms:** Support various optimization algorithms (e.g., Levenberg-Marquardt, Nelder-Mead, NLopt bindings; potentially Bayesian methods like MCMC in the future).
* **FR6.3 Estimable Parameters:** The system shall allow estimation of the following parameters:
    * **BRDF Coefficients:** $R_d$ (diffuse reflectivity), $R_s$ (specular reflectivity), and $n$ (Phong exponent).
    * **Attitude Parameters:** e.g., initial attitude offsets, systematic pointing biases, or parameters defining simple attitude adjustments, if not fully constrained by SPICE CKs.
* **FR6.4 Objective/Likelihood Function:** Define and compute an objective function (e.g., chi-squared) or likelihood function for comparing generated light curves with observed data.
* **FR6.5 Observational Data Input:** Ingest observational light curve data (time, brightness, uncertainties, observer location metadata for SPICE setup) for use in the estimation process.

**FR7: Data Handling & I/O**
* **FR7.1 Configuration Management:** Use configuration files (e.g., YAML, JSON, TOML) to manage simulation scenarios, model parameters, estimation settings, and file paths (including SPICE kernel lists/metakernels).
* **FR7.2 Data Input:** Robustly load various input data types: SPICE kernels (metakernels, SPK, CK, LSK, FK, PCK), observation files, satellite model files.
* **FR7.3 Data Output:** Save all relevant outputs: generated light curves, estimated parameters, covariance matrices, intermediate calculation results (e.g., shadowing factors), logs.
* **FR7.4 Logging:** Implement comprehensive logging for diagnostics, debugging, and tracking simulation progress.

**FR8: Visualization (Recommended)**
* **FR8.1 3D Satellite Visualization:** Provide a utility for 3D visualization of the satellite model (meshes), its components, orientation, and illumination/shadowing conditions at specific epochs.
* **FR8.2 Light Curve Plotting:** Tools for plotting generated and observed light curves, residuals, and comparison plots.
* **FR8.3 Estimation Visualization:** Tools for visualizing parameter estimation results (e.g., parameter traces, corner plots if MCMC is used, convergence plots).

---

## 4. Non-Functional Requirements (NFR)

These define *how well* the system should perform its functions.

* **NFR1: Modularity & Reusability**
    * **NFR1.1 Logical Modules:** The codebase must be organized into distinct, logical modules with clear, well-defined APIs and minimal coupling (e.g., `satellite_model_definition`, `component_articulation`, `spice_handler`, `shadowing_analyzer` (mesh-based), `lightcurve_engine` (facet-based), `parameter_estimator`, `io_utils`, `visualization_tools`).
    * **NFR1.2 Reusable Components:** Design components to be reusable.
* **NFR2: Expandability & Maintainability**
    * **NFR2.1 Extensible Design:** The architecture should facilitate adding new functionalities (e.g., new BRDF models, different shadowing algorithms, new estimation techniques) with minimal refactoring.
    * **NFR2.2 Clean Code:** Adhere to good coding practices (e.g., PEP 8 for Python), with clear, concise, and well-commented code.
* **NFR3: Performance**
    * **NFR3.1 Efficiency:** Computationally intensive parts (especially mesh-based shadowing analysis) must be optimized. The two-stage approach (mesh for shadowing, facets for light curve) is intended to manage this.
    * **NFR3.2 Scalability:** The system should handle complex satellite models and long-duration simulations.
    * **NFR3.3 Designed for Parallelization:** The architecture, particularly for shadowing analysis and potentially for light curve generation over many epochs or parameter estimation runs, should be designed from the outset to allow for future parallel processing (multithreading, multiprocessing, or GPU acceleration) even if not implemented initially.
* **NFR4: Accuracy & Validation**
    * **NFR4.1 Physical Accuracy:** Strive for high physical accuracy in all models.
    * **NFR4.2 Validation:** Implement a strategy for validating simulation results.
* **NFR5: Usability**
    * **NFR5.1 API Design:** Provide a clear, intuitive, and well-documented Python API.
    * **NFR5.2 (Optional) CLI:** A command-line interface for common tasks.
* **NFR6: Documentation**
    * **NFR6.1 Code Documentation:** Comprehensive docstrings.
    * **NFR6.2 User Documentation:** README, tutorials, examples.
    * **NFR6.3 Theoretical Basis:** Documentation of models and algorithms.
* **NFR7: Testing**
    * **NFR7.1 Unit Tests:** Extensive unit tests.
    * **NFR7.2 Integration Tests:** Verify module interactions.
    * **NFR7.3 Regression Tests:** Prevent re-introduction of bugs.
* **NFR8: Dependencies & Environment**
    * **NFR8.1 Python Version:** Target Python 3.9+ (specifically 3.10.8 for this project).
    * **NFR8.2 Core Libraries:** NumPy, SciPy, Matplotlib.
    * **NFR8.3 Mesh Handling:** Trimesh, Open3D, or PyVista.
    * **NFR8.4 SPICE:** SpiceyPy.
    * **NFR8.5 Estimation Libraries:** `scipy.optimize`, NLopt bindings.
    * **NFR8.6 Dependency Management:** `requirements.txt` or `pyproject.toml`.
* **NFR9: Version Control**
    * **NFR9.1 Git Repository:** New Git repository.
    * **NFR9.2 Branching Strategy:** Suitable Git branching strategy.

---

