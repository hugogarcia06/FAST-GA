"""Estimation of cl/cm/oswald aero coefficients using OPENVSP."""
#  This file is part of FAST : A framework for rapid Overall Aircraft Design
#  Copyright (C) 2020  ONERA & ISAE-SUPAERO
#  FAST is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os
import math
import os.path as pth
import pandas as pd
import numpy as np

from importlib.resources import path

from openmdao.components.external_code_comp import ExternalCodeComp
from openmdao.utils.file_wrap import InputFileGenerator

from fastoad.model_base import FlightPoint

# noinspection PyProtectedMember
from fastoad._utils.resource_management.copy import copy_resource, copy_resource_folder

# noinspection PyProtectedMember
from fastoad.module_management._bundle_loader import BundleLoader
from fastoad.constants import EngineSetting

from stdatm import Atmosphere

from fastga.models.propulsion.fuel_propulsion.base import FuelEngineSet
from fastga.models.propulsion.fuel_propulsion.basicIC_engine.basicIC_engine import (
    PROPELLER_EFFICIENCY,
)

# noinspection PyProtectedMember
from fastga.command.api import _create_tmp_directory

from . import resources as local_resources
from . import openvsp3201
from fastga.models.handling_qualities import resources

DEFAULT_WING_AIRFOIL = "naca23012.af"
DEFAULT_HTP_AIRFOIL = "naca0012.af"
DEFAULT_VTP_AIRFOIL = "naca0012.af"
INPUT_WING_SCRIPT = "wing_openvsp.vspscript"
INPUT_WING_ROTOR_SCRIPT = "wing_rotor_openvsp.vspscript"
INPUT_HTP_SCRIPT = "ht_openvsp.vspscript"
INPUT_AIRCRAFT_SCRIPT = "wing_ht_vt_openvsp.vspscript"
INPUT_AIRCRAFT_FUSELAGE_SCRIPT = "fuselage_wing_ht_vt_openvsp.vspscript"
STDERR_FILE_NAME = "vspaero_calc.err"
VSPSCRIPT_EXE_NAME = "vspscript.exe"
VSPAERO_EXE_NAME = "vspaero.exe"


class OPENVSPSimpleGeometry(ExternalCodeComp):
    """Execution of OpenVSP for clean surfaces."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.stderr = None

    def initialize(self):
        self.options.declare("result_folder_path", default="", types=str)
        self.options.declare("openvsp_exe_path", default="", types=str, allow_none=True)
        self.options.declare(
            "wing_airfoil_file", default=DEFAULT_WING_AIRFOIL, types=str, allow_none=True
        )
        self.options.declare(
            "htp_airfoil_file", default=DEFAULT_HTP_AIRFOIL, types=str, allow_none=True
        )
        self.options.declare(
            "vtp_airfoil_file", default=DEFAULT_VTP_AIRFOIL, types=str, allow_none=True
        )
        self.options.declare(
            "add_fuselage", default=False, types=bool, allow_none=True
        )

    def setup(self):
        self.add_input("data:reference_flight_condition:CG:x", val=np.nan, units="m")
        self.add_input("data:reference_flight_condition:CG:z", val=np.nan, units="m")

        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:maximum_height", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:length", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:rear_length", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:front_length", val=np.nan, units="m")

        self.add_input("data:geometry:wing:sweep_25", val=np.nan, units="deg")
        self.add_input("data:geometry:wing:taper_ratio", val=np.nan)
        self.add_input("data:geometry:wing:aspect_ratio", val=np.nan)
        self.add_input("data:geometry:wing:dihedral", val=np.nan, units="deg")
        self.add_input("data:geometry:wing:twist", val=np.nan, units="deg")
        self.add_input("data:geometry:wing:MAC:leading_edge:x:local", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:y", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:y", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:sweep_0", val=np.nan, units="deg")
        self.add_input("data:geometry:wing:MAC:at25percent:x", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:y", val=np.nan, units="m")
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:wing:span", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:sweep_25", val=np.nan, units="deg")
        self.add_input("data:geometry:horizontal_tail:taper_ratio", val=np.nan)
        self.add_input("data:geometry:horizontal_tail:aspect_ratio", val=np.nan)
        self.add_input("data:geometry:horizontal_tail:dihedral", val=np.nan, units="deg")
        self.add_input("data:geometry:horizontal_tail:twist", val=np.nan, units="deg")
        self.add_input("data:geometry:horizontal_tail:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:horizontal_tail:span", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:tip:chord", val=np.nan, units="m")
        self.add_input(
            "data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25", val=np.nan, units="m"
        )
        self.add_input("data:geometry:horizontal_tail:MAC:length", val=np.nan, units="m")
        self.add_input(
            "data:geometry:horizontal_tail:MAC:at25percent:x:local", val=np.nan, units="m"
        )
        self.add_input("data:geometry:horizontal_tail:z:from_wingMAC25", val=np.nan, units="m")
        self.add_input("data:geometry:vertical_tail:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:vertical_tail:sweep_25", val=np.nan, units="deg")
        self.add_input("data:geometry:vertical_tail:aspect_ratio", val=np.nan)
        self.add_input("data:geometry:vertical_tail:taper_ratio", val=np.nan)
        self.add_input("data:geometry:vertical_tail:span", val=np.nan, units="m")
        self.add_input("data:geometry:vertical_tail:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:vertical_tail:tip:chord", val=np.nan, units="m")
        self.add_input("data:geometry:vertical_tail:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25", val=np.nan, units="m")
        self.add_input("data:geometry:vertical_tail:MAC:at25percent:x:local", val=np.nan, units="m")
        self.add_input("data:geometry:vertical_tail:MAC:z", val=np.nan, units="m")

    def check_config(self, logger):
        # let void to avoid logger error on "The command cannot be empty"
        pass

    def compute_stab_coef(self, inputs, outputs, altitude, mach, aoa_angle):
        """
        Function that computes in OpenVSP environment all the aerodynamic parameters @0° and
        aoa_angle and calculate the associated derivatives.

        @param inputs: inputs parameters defined within FAST-OAD-GA
        @param outputs: outputs parameters defined within FAST-OAD-GA
        @param altitude: altitude for aerodynamic calculation in meters
        @param mach: air speed expressed in mach
        @param aoa_angle: air speed angle of attack with respect to aircraft

        """

        # Fix mach number of digits to consider similar results
        mach = round(float(mach) * 1e3) / 1e3

        # Get inputs necessary to define global geometry
        sref_wing = float(inputs["data:geometry:wing:area"])
        sref_htp = float(inputs["data:geometry:horizontal_tail:area"])
        sref_vtp = float(inputs["data:geometry:vertical_tail:area"])
        area_ratio_htp = sref_htp / sref_wing
        area_ratio_vtp = sref_vtp / sref_wing
        sweep25_wing = float(inputs["data:geometry:wing:sweep_25"])
        taper_ratio_wing = float(inputs["data:geometry:wing:taper_ratio"])
        aspect_ratio_wing = float(inputs["data:geometry:wing:aspect_ratio"])
        dihedral_wing = float(inputs["data:geometry:wing:dihedral"])
        twist_wing = float(inputs["data:geometry:wing:twist"])
        sweep25_htp = float(inputs["data:geometry:horizontal_tail:sweep_25"])
        aspect_ratio_htp = float(inputs["data:geometry:horizontal_tail:aspect_ratio"])
        taper_ratio_htp = float(inputs["data:geometry:horizontal_tail:taper_ratio"])
        sweep25_vtp = float(inputs["data:geometry:vertical_tail:sweep_25"])
        aspect_ratio_vtp = float(inputs["data:geometry:vertical_tail:aspect_ratio"])
        taper_ratio_vtp = float(inputs["data:geometry:vertical_tail:taper_ratio"])
        fus_length = float(inputs["data:geometry:fuselage:length"])
        fus_front_length = float(inputs["data:geometry:fuselage:front_length"])
        fus_rear_length = float(inputs["data:geometry:fuselage:rear_length"])
        fus_max_width = float(inputs["data:geometry:fuselage:maximum_width"])
        fus_max_height = float(inputs["data:geometry:fuselage:maximum_height"])

        if self.options["add_fuselage"]:
            geometry_set = np.around(
                np.array(
                    [
                        sweep25_wing,
                        taper_ratio_wing,
                        aspect_ratio_wing,
                        dihedral_wing,
                        twist_wing,
                        sweep25_htp,
                        taper_ratio_htp,
                        aspect_ratio_htp,
                        sweep25_vtp,
                        taper_ratio_vtp,
                        aspect_ratio_vtp,
                        mach,
                        area_ratio_htp,
                        area_ratio_vtp,
                        fus_length,
                        fus_rear_length,
                        fus_front_length,
                        fus_max_width,
                        fus_max_height
                    ]
                ),
                decimals=6,
            )
        else:
            geometry_set = np.around(
                np.array(
                    [
                        sweep25_wing,
                        taper_ratio_wing,
                        aspect_ratio_wing,
                        dihedral_wing,
                        twist_wing,
                        sweep25_htp,
                        taper_ratio_htp,
                        aspect_ratio_htp,
                        sweep25_vtp,
                        taper_ratio_vtp,
                        aspect_ratio_vtp,
                        mach,
                        area_ratio_htp,
                        area_ratio_vtp
                    ]
                ),
                decimals=6,
            )

        # Search if results already exist:
        result_folder_path = self.options["result_folder_path"]
        result_file_path = None
        saved_area_ratio_htp = 1.0
        saved_area_ratio_vtp = 1.0
        if result_folder_path != "":
            if not self.options["add_fuselage"]:
                result_file_path, saved_area_ratio_htp, saved_area_ratio_vtp = self.search_results(
                    result_folder_path, geometry_set
                )
            else:
                result_file_path, saved_area_ratio_htp, saved_area_ratio_vtp = self.search_results_fuselage(
                    result_folder_path, geometry_set
                )
        # If no result saved for that geometry under this mach condition, computation is done
        if result_file_path is None:

            # Create result folder first (if it must fail, let it fail as soon as possible)
            if result_folder_path != "":
                if not os.path.exists(result_folder_path):
                    os.makedirs(pth.join(result_folder_path), exist_ok=True)

            # Save the geometry (result_file_path is None entering the function)
            if self.options["result_folder_path"] != "":
                if not self.options["add_fuselage"]:
                    result_file_path = self.save_geometry(result_folder_path, geometry_set)
                else:
                    result_file_path = self.save_geometry_fuselage(result_folder_path, geometry_set)

            # Compute complete aircraft @ 0°/X° angle of attack
            # NOTE: this is where the computation by OpenVSP is performed.
            aircraft_stab_coef = self.compute_aircraft(inputs, outputs, altitude, mach, aoa_angle)

            # Post-process aircraft stability data (dictionary) -------------------------------------------------------
            cL_u = aircraft_stab_coef["cL_u"]
            cD_u = aircraft_stab_coef["cD_u"]
            cm_u = aircraft_stab_coef["cm_u"]
            cL_alpha = aircraft_stab_coef["cL_alpha"]
            cD_alpha = aircraft_stab_coef["cD_alpha"]
            cm_alpha = aircraft_stab_coef["cm_alpha"]
            cL_q = aircraft_stab_coef["cL_q"]
            cD_q = aircraft_stab_coef["cD_q"]
            cm_q = aircraft_stab_coef["cm_q"]
            cY_beta = aircraft_stab_coef["cY_beta"]
            cl_beta = aircraft_stab_coef["cl_beta"]
            cn_beta = aircraft_stab_coef["cn_beta"]
            cY_p = aircraft_stab_coef["cY_p"]
            cl_p = aircraft_stab_coef["cl_p"]
            cn_p = aircraft_stab_coef["cn_p"]
            cY_r = aircraft_stab_coef["cY_r"]
            cl_r = aircraft_stab_coef["cl_r"]
            cn_r = aircraft_stab_coef["cn_r"]

            # Resize vectors -----------------------------------------------------------------------

            # Save results to defined path ---------------------------------------------------------
            if self.options["result_folder_path"] != "":
                results = [
                    cL_u,
                    cD_u,
                    cm_u,
                    cL_alpha,
                    cD_alpha,
                    cm_alpha,
                    cL_q,
                    cD_q,
                    cm_q,
                    cY_beta,
                    cl_beta,
                    cn_beta,
                    cY_p,
                    cl_p,
                    cn_p,
                    cY_r,
                    cl_r,
                    cn_r
                ]
                self.save_results(result_file_path, results)

        # Else retrieved results are used, eventually adapted with new area ratio
        else:
            # Read values from result file ---------------------------------------------------------
            data = self.read_results(result_file_path)
            cL_u = float(data.loc["cL_u", 0])
            cD_u = float(data.loc["cD_u", 0])
            cm_u = float(data.loc["cm_u", 0])
            cL_alpha = float(data.loc["cL_alpha", 0])
            cD_alpha = float(data.loc["cD_alpha", 0])
            cm_alpha = float(data.loc["cm_alpha", 0])
            cL_q = float(data.loc["cL_q", 0])
            cD_q = float(data.loc["cD_q", 0])
            cm_q = float(data.loc["cm_q", 0])
            cY_beta = float(data.loc["cY_beta", 0])
            cl_beta = float(data.loc["cl_beta", 0])
            cn_beta = float(data.loc["cn_beta", 0])
            cY_p = float(data.loc["cY_p", 0])
            cl_p = float(data.loc["cl_p", 0])
            cn_p = float(data.loc["cn_p", 0])
            cY_r = float(data.loc["cY_r", 0])
            cl_r = float(data.loc["cl_r", 0])
            cn_r = float(data.loc["cn_r", 0])

        return (
            cL_u,
            cD_u,
            cm_u,
            cL_alpha,
            cD_alpha,
            cm_alpha,
            cL_q,
            cD_q,
            cm_q,
            cY_beta,
            cl_beta,
            cn_beta,
            cY_p,
            cl_p,
            cn_p,
            cY_r,
            cl_r,
            cn_r
        )

    def compute_aircraft(self, inputs, outputs, altitude, mach, aoa_angle):
        """
        Function that computes in OpenVSP environment the complete aircraft (considering wing and
        horizontal tail plan) and returns the different aerodynamic stability coefficients. The downwash is
        done by OpenVSP considering far field.

        @param inputs: inputs parameters defined within FAST-OAD-GA
        @param outputs: outputs parameters defined within FAST-OAD-GA
        @param altitude: altitude for aerodynamic calculation in meters
        @param mach: air speed expressed in mach
        @param aoa_angle: air speed angle of attack with respect to aircraft
        @return: aircraft dictionary including its stability derivative coefficients:
        {   cL_u,
            cD_u,
            cm_u,
            cL_alpha,
            cD_alpha,
            cm_alpha,
            cL_q,
            cD_q,
            cm_q,
            cY_beta,
            cl_beta,
            cn_beta,
            cY_p,
            cl_p,
            cn_p,
            cY_r,
            cl_r,
            cn_r
        }
        """

        # STEP 1/XX - DEFINE OR CALCULATE INPUT DATA FOR AERODYNAMIC EVALUATION ####################
        ############################################################################################

        # Get inputs (and calculate missing ones)
        x_cg = inputs["data:reference_flight_condition:CG:x"]
        z_cg = inputs["data:reference_flight_condition:CG:z"]
        # Fuselage Geometry
        fus_length = float(inputs["data:geometry:fuselage:length"])
        fus_front_length = float(inputs["data:geometry:fuselage:front_length"])
        fus_rear_length = float(inputs["data:geometry:fuselage:rear_length"])
        width_max = inputs["data:geometry:fuselage:maximum_width"]
        height_max = inputs["data:geometry:fuselage:maximum_height"]
        # Wing Geometry
        sref_wing = float(inputs["data:geometry:wing:area"])
        x0_wing = inputs["data:geometry:wing:MAC:leading_edge:x:local"]
        l0_wing = inputs["data:geometry:wing:MAC:length"]
        y1_wing = width_max / 2.0
        y2_wing = inputs["data:geometry:wing:root:y"]
        l2_wing = inputs["data:geometry:wing:root:chord"]
        y4_wing = inputs["data:geometry:wing:tip:y"]
        l4_wing = inputs["data:geometry:wing:tip:chord"]
        y_MAC = inputs["data:geometry:wing:MAC:y"]
        sweep_0_wing = inputs["data:geometry:wing:sweep_0"]
        fa_length = inputs["data:geometry:wing:MAC:at25percent:x"]
        span_wing = inputs["data:geometry:wing:span"]
        dihedral_wing = inputs["data:geometry:wing:dihedral"]
        twist_wing = inputs["data:geometry:wing:twist"]
        # Horizontal Tail Geometry
        area_htp = inputs["data:geometry:horizontal_tail:area"]
        sweep_25_htp = inputs["data:geometry:horizontal_tail:sweep_25"]
        span_htp = inputs["data:geometry:horizontal_tail:span"]
        semispan_htp = span_htp / 2.0
        root_chord_htp = inputs["data:geometry:horizontal_tail:root:chord"]
        tip_chord_htp = inputs["data:geometry:horizontal_tail:tip:chord"]
        lp_htp = inputs["data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25"]
        l0_htp = inputs["data:geometry:horizontal_tail:MAC:length"]
        x0_htp = inputs["data:geometry:horizontal_tail:MAC:at25percent:x:local"]
        height_htp = inputs["data:geometry:horizontal_tail:z:from_wingMAC25"]
        height_MAC = (y_MAC - l2_wing) * math.sin(dihedral_wing * math.pi / 180.0)
        dihedral_htp = inputs["data:geometry:horizontal_tail:dihedral"]
        twist_htp = inputs["data:geometry:horizontal_tail:twist"]
        # Vertical Tail Geometry
        area_vtp = inputs["data:geometry:vertical_tail:area"]
        vtp_taper_ratio = inputs["data:geometry:vertical_tail:taper_ratio"]
        sweep_25_vtp = inputs["data:geometry:vertical_tail:sweep_25"]
        span_vtp = inputs["data:geometry:vertical_tail:span"]
        z_mac_local_vtp = span_vtp / 3.0 * ((1 + 2*vtp_taper_ratio)/(1 + vtp_taper_ratio))
        root_chord_vtp = inputs["data:geometry:vertical_tail:root:chord"]
        tip_chord_vtp = inputs["data:geometry:vertical_tail:tip:chord"]
        lp_vtp = inputs["data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25"]
        l0_vtp = inputs["data:geometry:vertical_tail:MAC:length"]
        x0_vtp = inputs["data:geometry:vertical_tail:MAC:at25percent:x:local"]
        height_mac_vtp = inputs["data:geometry:vertical_tail:MAC:z"]
        # TODO: needs corrections
        height_vtp = inputs["data:geometry:fuselage:maximum_height"] / 4.0
        # Compute remaining inputs
        atm = Atmosphere(altitude, altitude_in_feet=False)
        x_wing = fa_length - x0_wing - 0.25 * l0_wing
        # NOTE: where does this expression come from?
        z_wing = -(height_max - 0.12 * l2_wing) * 0.5
        span2_wing = y4_wing - y2_wing
        z_htp = z_wing + height_MAC + height_htp
        x_htp = fa_length + lp_htp - x0_htp  # distance from airplane nose to htp root leading edge
        z_vtp = 0.0
        x_vtp = fa_length + lp_vtp - x0_vtp
        rho = atm.density
        v_inf = max(atm.speed_of_sound * mach, 0.01)  # avoid V=0 m/s crashes
        reynolds = v_inf * l0_wing / atm.kinematic_viscosity

        # STEP 2/XX - DEFINE WORK DIRECTORY, COPY RESOURCES AND CREATE COMMAND BATCH ###############
        ############################################################################################

        # If a folder path is specified for openvsp .exe, it becomes working directory (target),
        # if not temporary folder is created
        if self.options["openvsp_exe_path"]:
            target_directory = pth.abspath(self.options["openvsp_exe_path"])
        else:
            tmp_directory = _create_tmp_directory()
            target_directory = tmp_directory.name
        # Define the list of necessary input files: geometry script and foil file for both wing/HTP
        input_file_list = [
            pth.join(target_directory, INPUT_AIRCRAFT_SCRIPT),
            pth.join(target_directory, self.options["wing_airfoil_file"]),
            pth.join(target_directory, self.options["htp_airfoil_file"]),
            pth.join(target_directory, self.options["vtp_airfoil_file"]),
        ]
        if self.options["add_fuselage"]:
            input_file_list = [
                pth.join(target_directory, INPUT_AIRCRAFT_FUSELAGE_SCRIPT),
                pth.join(target_directory, self.options["wing_airfoil_file"]),
                pth.join(target_directory, self.options["htp_airfoil_file"]),
                pth.join(target_directory, self.options["vtp_airfoil_file"]),
            ]
        self.options["external_input_files"] = input_file_list
        # Define standard error file by default to avoid error code return
        self.stderr = pth.join(target_directory, STDERR_FILE_NAME)
        # Copy resource in working (target) directory
        # noinspection PyTypeChecker
        copy_resource_folder(openvsp3201, target_directory)
        # noinspection PyTypeChecker
        copy_resource(resources, self.options["wing_airfoil_file"], target_directory)
        # noinspection PyTypeChecker
        copy_resource(resources, self.options["htp_airfoil_file"], target_directory)
        # noinspection PyTypeChecker
        copy_resource(resources, self.options["vtp_airfoil_file"], target_directory)
        # Create corresponding .bat files (one for each geometry configuration)
        self.options["command"] = [pth.join(target_directory, "vspscript.bat")]
        batch_file = open(self.options["command"][0], "w+")
        batch_file.write("@echo off\n")
        command = (
                pth.join(target_directory, VSPSCRIPT_EXE_NAME)
                + " -script "
                + pth.join(target_directory, INPUT_AIRCRAFT_SCRIPT)
                + " >nul 2>nul\n"
        )
        if self.options["add_fuselage"]:
            command = (
                    pth.join(target_directory, VSPSCRIPT_EXE_NAME)
                    + " -script "
                    + pth.join(target_directory, INPUT_AIRCRAFT_FUSELAGE_SCRIPT)
                    + " >nul 2>nul\n"
            )
        batch_file.write(command)
        batch_file.close()

        # STEP 3/XX - OPEN THE TEMPLATE SCRIPT FOR GEOMETRY GENERATION, MODIFY VALUES AND SAVE TO
        # WORKDIR ##################################################################################
        output_file_list = [
            pth.join(
                target_directory, INPUT_AIRCRAFT_SCRIPT.replace(".vspscript", "_DegenGeom.csv")
            )
        ]
        if self.options["add_fuselage"]:
            output_file_list = [
                pth.join(
                    target_directory, INPUT_AIRCRAFT_FUSELAGE_SCRIPT.replace(".vspscript", "_DegenGeom.csv")
                )
            ]
        parser = InputFileGenerator()
        if not self.options["add_fuselage"]:
            with path(local_resources, INPUT_AIRCRAFT_SCRIPT) as input_template_path:
                parser.set_template_file(str(input_template_path))
                parser.set_generated_file(input_file_list[0])
                # Modify WING parameters
                parser.mark_anchor("x_wing")
                parser.transfer_var(float(x_wing), 0, 5)
                parser.mark_anchor("z_wing")
                parser.transfer_var(float(z_wing), 0, 5)
                parser.mark_anchor("y1_wing")
                parser.transfer_var(float(y1_wing), 0, 5)
                for i in range(3):
                    parser.mark_anchor("l2_wing")
                    parser.transfer_var(float(l2_wing), 0, 5)
                parser.reset_anchor()
                parser.mark_anchor("span2_wing")
                parser.transfer_var(float(span2_wing), 0, 5)
                parser.mark_anchor("l4_wing")
                parser.transfer_var(float(l4_wing), 0, 5)
                parser.mark_anchor("sweep_0_wing")
                parser.transfer_var(float(sweep_0_wing), 0, 5)
                parser.mark_anchor("dihedral_wing")
                parser.transfer_var(float(dihedral_wing), 0, 5)
                parser.mark_anchor("twist_wing")
                parser.transfer_var(float(twist_wing), 0, 5)
                parser.mark_anchor("airfoil_0_file")
                parser.transfer_var('"' + input_file_list[-3].replace("\\", "/") + '"', 0, 3)
                parser.mark_anchor("airfoil_1_file")
                parser.transfer_var('"' + input_file_list[-3].replace("\\", "/") + '"', 0, 3)
                parser.mark_anchor("airfoil_2_file")
                parser.transfer_var('"' + input_file_list[-3].replace("\\", "/") + '"', 0, 3)
                # Modify HTP parameters
                parser.mark_anchor("x_htp")
                parser.transfer_var(float(x_htp), 0, 5)
                parser.mark_anchor("z_htp")
                parser.transfer_var(float(z_htp), 0, 5)
                parser.mark_anchor("semispan_htp")
                parser.transfer_var(float(semispan_htp), 0, 5)
                parser.mark_anchor("root_chord_htp")
                parser.transfer_var(float(root_chord_htp), 0, 5)
                parser.mark_anchor("tip_chord_htp")
                parser.transfer_var(float(tip_chord_htp), 0, 5)
                parser.mark_anchor("sweep_25_htp")
                parser.transfer_var(float(sweep_25_htp), 0, 5)
                parser.mark_anchor("dihedral_htp")
                parser.transfer_var(float(dihedral_htp), 0, 5)
                parser.mark_anchor("twist_htp")
                parser.transfer_var(float(twist_htp), 0, 5)
                parser.mark_anchor("htp_area")
                parser.transfer_var(float(area_htp), 0, 5)
                parser.mark_anchor("airfoil_3_file")
                parser.transfer_var('"' + input_file_list[-2].replace("\\", "/") + '"', 0, 3)
                parser.mark_anchor("airfoil_4_file")
                parser.transfer_var('"' + input_file_list[-2].replace("\\", "/") + '"', 0, 3)
                # Modify VTP parameters
                parser.mark_anchor("x_vtp")
                parser.transfer_var(float(x_vtp), 0, 5)
                parser.mark_anchor("z_vtp")
                parser.transfer_var(float(z_vtp), 0, 5)
                parser.mark_anchor("span_vtp")
                parser.transfer_var(float(span_vtp), 0, 5)
                parser.mark_anchor("root_chord_vtp")
                parser.transfer_var(float(root_chord_vtp), 0, 5)
                parser.mark_anchor("tip_chord_vtp")
                parser.transfer_var(float(tip_chord_vtp), 0, 5)
                parser.mark_anchor("sweep_25_vtp")
                parser.transfer_var(float(sweep_25_vtp), 0, 5)
                parser.mark_anchor("vtp_area")
                parser.transfer_var(float(area_vtp), 0, 5)
                parser.mark_anchor("airfoil_5_file")
                parser.transfer_var('"' + input_file_list[-1].replace("\\", "/") + '"', 0, 3)
                parser.mark_anchor("airfoil_6_file")
                parser.transfer_var('"' + input_file_list[-1].replace("\\", "/") + '"', 0, 3)
                parser.mark_anchor("csv_file")
                csv_name = output_file_list[0]
                parser.transfer_var('"' + csv_name.replace("\\", "/") + '"', 0, 3)
                parser.generate()

        else:
            with path(local_resources, INPUT_AIRCRAFT_FUSELAGE_SCRIPT) as input_template_path:
                parser.set_template_file(str(input_template_path))
                parser.set_generated_file(input_file_list[0])
                # Modify FUSELAGE parameters
                parser.mark_anchor("fus_length")
                parser.transfer_var(float(fus_length), 0, 5)
                fus_diameter = math.sqrt(width_max * height_max)
                parser.mark_anchor("fus_diameter")
                parser.transfer_var(float(fus_diameter), 0, 5)
                parser.mark_anchor("fus_front_length")
                parser.transfer_var(float(fus_front_length), 0, 5)
                parser.mark_anchor("fus_rear_length")
                parser.transfer_var(float(fus_rear_length), 0, 5)
                # Modify WING parameters
                parser.mark_anchor("x_wing")
                parser.transfer_var(float(x_wing), 0, 5)
                parser.mark_anchor("z_wing")
                parser.transfer_var(float(z_wing), 0, 5)
                parser.mark_anchor("y1_wing")
                parser.transfer_var(float(y1_wing), 0, 5)
                for i in range(3):
                    parser.mark_anchor("l2_wing")
                    parser.transfer_var(float(l2_wing), 0, 5)
                parser.reset_anchor()
                parser.mark_anchor("span2_wing")
                parser.transfer_var(float(span2_wing), 0, 5)
                parser.mark_anchor("l4_wing")
                parser.transfer_var(float(l4_wing), 0, 5)
                parser.mark_anchor("sweep_0_wing")
                parser.transfer_var(float(sweep_0_wing), 0, 5)
                parser.mark_anchor("dihedral_wing")
                parser.transfer_var(float(dihedral_wing), 0, 5)
                parser.mark_anchor("twist_wing")
                parser.transfer_var(float(twist_wing), 0, 5)
                parser.mark_anchor("airfoil_0_file")
                parser.transfer_var('"' + input_file_list[-3].replace("\\", "/") + '"', 0, 3)
                parser.mark_anchor("airfoil_1_file")
                parser.transfer_var('"' + input_file_list[-3].replace("\\", "/") + '"', 0, 3)
                parser.mark_anchor("airfoil_2_file")
                parser.transfer_var('"' + input_file_list[-3].replace("\\", "/") + '"', 0, 3)
                # Modify HTP parameters
                parser.mark_anchor("x_htp")
                parser.transfer_var(float(x_htp), 0, 5)
                parser.mark_anchor("z_htp")
                parser.transfer_var(float(z_htp), 0, 5)
                parser.mark_anchor("semispan_htp")
                parser.transfer_var(float(semispan_htp), 0, 5)
                parser.mark_anchor("root_chord_htp")
                parser.transfer_var(float(root_chord_htp), 0, 5)
                parser.mark_anchor("tip_chord_htp")
                parser.transfer_var(float(tip_chord_htp), 0, 5)
                parser.mark_anchor("sweep_25_htp")
                parser.transfer_var(float(sweep_25_htp), 0, 5)
                parser.mark_anchor("dihedral_htp")
                parser.transfer_var(float(dihedral_htp), 0, 5)
                parser.mark_anchor("twist_htp")
                parser.transfer_var(float(twist_htp), 0, 5)
                parser.mark_anchor("htp_area")
                parser.transfer_var(float(area_htp), 0, 5)
                parser.mark_anchor("airfoil_3_file")
                parser.transfer_var('"' + input_file_list[-2].replace("\\", "/") + '"', 0, 3)
                parser.mark_anchor("airfoil_4_file")
                parser.transfer_var('"' + input_file_list[-2].replace("\\", "/") + '"', 0, 3)
                # Modify VTP parameters
                parser.mark_anchor("x_vtp")
                parser.transfer_var(float(x_vtp), 0, 5)
                parser.mark_anchor("z_vtp")
                parser.transfer_var(float(z_vtp), 0, 5)
                parser.mark_anchor("span_vtp")
                parser.transfer_var(float(span_vtp), 0, 5)
                parser.mark_anchor("root_chord_vtp")
                parser.transfer_var(float(root_chord_vtp), 0, 5)
                parser.mark_anchor("tip_chord_vtp")
                parser.transfer_var(float(tip_chord_vtp), 0, 5)
                parser.mark_anchor("sweep_25_vtp")
                parser.transfer_var(float(sweep_25_vtp), 0, 5)
                parser.mark_anchor("vtp_area")
                parser.transfer_var(float(area_vtp), 0, 5)
                parser.mark_anchor("airfoil_5_file")
                parser.transfer_var('"' + input_file_list[-1].replace("\\", "/") + '"', 0, 3)
                parser.mark_anchor("airfoil_6_file")
                parser.transfer_var('"' + input_file_list[-1].replace("\\", "/") + '"', 0, 3)
                parser.mark_anchor("csv_file")
                csv_name = output_file_list[0]
                parser.transfer_var('"' + csv_name.replace("\\", "/") + '"', 0, 3)
                parser.generate()

        # STEP 4/XX - RUN BATCH TO GENERATE GEOMETRY .CSV FILE #####################################
        ############################################################################################

        self.options["external_output_files"] = output_file_list
        super().compute(inputs, outputs)

        # STEP 5/XX - DEFINE NEW INPUT/OUTPUT FILES LIST AND CREATE BATCH FOR VLM COMPUTATION ######
        ############################################################################################

        input_file_list = output_file_list
        input_file_list.append(input_file_list[0].replace("csv", "vspaero"))
        output_file_list = [
            input_file_list[0].replace("csv", "stab")
        ]
        self.options["external_input_files"] = input_file_list
        self.options["external_output_files"] = output_file_list
        self.options["command"] = [pth.join(target_directory, "vspaero.bat")]
        batch_file = open(self.options["command"][0], "w+")
        batch_file.write("@echo off\n")
        command = (
                pth.join(target_directory, VSPAERO_EXE_NAME)
                + " -stab "  # TO PERFORM STABILITY ANALYSIS
                + input_file_list[1].replace(".vspaero", "")
                + " >nul 2>nul\n"
        )
        batch_file.write(command)
        batch_file.close()

        # STEP 6/XX - OPEN THE TEMPLATE VSPAERO FOR COMPUTATION, MODIFY VALUES AND SAVE TO WORKDIR #
        ############################################################################################

        parser = InputFileGenerator()
        template_file = pth.split(input_file_list[1])[1]
        with path(local_resources, template_file) as input_template_path:
            parser.set_template_file(str(input_template_path))
            parser.set_generated_file(input_file_list[1])
            parser.reset_anchor()
            parser.mark_anchor("Sref")
            parser.transfer_var(float(sref_wing), 0, 3)
            parser.mark_anchor("Cref")
            parser.transfer_var(float(l0_wing), 0, 3)
            parser.mark_anchor("Bref")
            parser.transfer_var(float(span_wing), 0, 3)
            parser.mark_anchor("X_cg")
            parser.transfer_var(float(x_cg), 0, 3)
            # parser.transfer_var(float(fa_length), 0, 3)
            parser.mark_anchor("Z_cg")
            parser.transfer_var(float(z_cg), 0, 3)
            parser.mark_anchor("Mach")
            parser.transfer_var(float(mach), 0, 3)
            parser.mark_anchor("AOA")
            parser.transfer_var(float(aoa_angle), 0, 3)
            parser.mark_anchor("Vinf")
            parser.transfer_var(float(v_inf), 0, 3)
            parser.mark_anchor("Rho")
            parser.transfer_var(float(rho), 0, 3)
            parser.mark_anchor("ReCref")
            parser.transfer_var(float(reynolds), 0, 3)
            parser.generate()

        # STEP 7/XX - RUN BATCH TO GENERATE AERO OUTPUT FILES (.stab) ####################
        ############################################################################################

        super().compute(inputs, outputs)

        # STEP 8/XX - READ FILES, RETURN RESULTS (AND CLEAR TEMPORARY WORKDIR) #####################
        ############################################################################################

        # Open .stab file and extract data
        filename = output_file_list[0]
        (
            cL_u,
            cD_u,
            cm_u,
            cL_alpha,
            cD_alpha,
            cm_alpha,
            cL_q,
            cD_q,
            cm_q,
            cY_beta,
            cl_beta,
            cn_beta,
            cY_p,
            cl_p,
            cn_p,
            cY_r,
            cl_r,
            cn_r,
        ) = self.read_stab_file(filename)

        # Delete temporary directory
        if not (self.options["openvsp_exe_path"]):
            # noinspection PyUnboundLocalVariable
            tmp_directory.cleanup()

        # Return values
        aircraft = {
            "cL_u": cL_u,
            "cD_u": cD_u,
            "cm_u": cm_u,
            "cL_alpha": cL_alpha,
            "cD_alpha": cD_alpha,
            "cm_alpha": cm_alpha,
            "cL_q": cL_q,
            "cD_q": cD_q,
            "cm_q": cm_q,
            "cY_beta": cY_beta,
            "cl_beta": cl_beta,
            "cn_beta": cn_beta,
            "cY_p": cY_p,
            "cl_p": cl_p,
            "cn_p": cn_p,
            "cY_r": cY_r,
            "cl_r": cl_r,
            "cn_r": cn_r,
        }
        return aircraft

    @staticmethod
    def search_results(result_folder_path, geometry_set):
        """Search the results folder to see if the geometry has already been calculated."""
        if os.path.exists(result_folder_path):
            geometry_set_labels = [
                "sweep25_wing",
                "taper_ratio_wing",
                "aspect_ratio_wing",
                "dihedral_wing",
                "twist_wing",
                "sweep25_htp",
                "taper_ratio_htp",
                "aspect_ratio_htp",
                "sweep25_vtp",
                "taper_ratio_vtp",
                "aspect_ratio_vtp",
                "mach",
                "area_ratio_htp",
                "area_ratio_vtp",
            ]
            # If some results already stored search for corresponding geometry
            if pth.exists(pth.join(result_folder_path, "geometry_0.csv")):
                idx = 0
                while pth.exists(pth.join(result_folder_path, "geometry_" + str(idx) + ".csv")):
                    if pth.exists(pth.join(result_folder_path, "openvsp_" + str(idx) + ".csv")):
                        data = pd.read_csv(
                            pth.join(result_folder_path, "geometry_" + str(idx) + ".csv")
                        )
                        values = data.to_numpy()[:, 1].tolist()
                        labels = data.to_numpy()[:, 0].tolist()
                        data = pd.DataFrame(values, index=labels)
                        # noinspection PyBroadException
                        try:
                            if np.size(data.loc[geometry_set_labels[0:-1], 0].to_numpy()) == 7:
                                saved_set = np.around(
                                    data.loc[geometry_set_labels[0:-1], 0].to_numpy(), decimals=6
                                )
                                if np.sum(saved_set == geometry_set[0:-1]) == 7:
                                    result_file_path = pth.join(
                                        result_folder_path, "openvsp_" + str(idx) + ".csv"
                                    )
                                    saved_area_ratio_htp = data.loc["area_ratio_htp", 0]
                                    saved_area_ratio_vtp = data.loc["area_ratio_vtp", 0]
                                    return result_file_path, saved_area_ratio_htp, saved_area_ratio_vtp
                        except Exception:
                            break
                    idx += 1

        return None, 1.0, 1.0

    @staticmethod
    def search_results_fuselage(result_folder_path, geometry_set):
        """Search the results folder to see if the geometry (with fuselage) has already been calculated."""
        if os.path.exists(result_folder_path):
            geometry_set_labels = [
                "sweep25_wing",
                "taper_ratio_wing",
                "aspect_ratio_wing",
                "dihedral_wing",
                "twist_wing",
                "sweep25_htp",
                "taper_ratio_htp",
                "aspect_ratio_htp",
                "sweep25_vtp",
                "taper_ratio_vtp",
                "aspect_ratio_vtp",
                "mach",
                "area_ratio_htp",
                "area_ratio_vtp",
                "fus_length",
                "fus_front_length",
                "fus_rear_length",
                "fus_max_width",
                "fus_max_height",
            ]
            # If some results already stored search for corresponding geometry
            if pth.exists(pth.join(result_folder_path, "geometry_0.csv")):
                idx = 0
                while pth.exists(pth.join(result_folder_path, "geometry_" + str(idx) + ".csv")):
                    if pth.exists(pth.join(result_folder_path, "openvsp_" + str(idx) + ".csv")):
                        data = pd.read_csv(
                            pth.join(result_folder_path, "geometry_" + str(idx) + ".csv")
                        )
                        values = data.to_numpy()[:, 1].tolist()
                        labels = data.to_numpy()[:, 0].tolist()
                        data = pd.DataFrame(values, index=labels)
                        # noinspection PyBroadException
                        try:
                            if np.size(data.loc[geometry_set_labels[0:-1], 0].to_numpy()) == 7:
                                saved_set = np.around(
                                    data.loc[geometry_set_labels[0:-1], 0].to_numpy(), decimals=6
                                )
                                if np.sum(saved_set == geometry_set[0:-1]) == 7:
                                    result_file_path = pth.join(
                                        result_folder_path, "openvsp_" + str(idx) + ".csv"
                                    )
                                    saved_area_ratio_htp = data.loc["area_ratio_htp", 0]
                                    saved_area_ratio_vtp = data.loc["area_ratio_vtp", 0]
                                    return result_file_path, saved_area_ratio_htp, saved_area_ratio_vtp
                        except Exception:
                            break
                    idx += 1

        return None, 1.0, 1.0

    @staticmethod
    def save_geometry(result_folder_path, geometry_set):
        """Save geometry if not already computed by finding first available index."""
        geometry_set_labels = [
            "sweep25_wing",
            "taper_ratio_wing",
            "aspect_ratio_wing",
            "dihedral_wing",
            "twist_wing",
            "sweep25_htp",
            "taper_ratio_htp",
            "aspect_ratio_htp",
            "sweep25_vtp",
            "taper_ratio_vtp",
            "aspect_ratio_vtp",
            "mach",
            "area_ratio_htp",
            "area_ratio_vtp",
        ]
        data = pd.DataFrame(geometry_set, index=geometry_set_labels)
        idx = 0
        while pth.exists(pth.join(result_folder_path, "geometry_" + str(idx) + ".csv")):
            idx += 1
        data.to_csv(pth.join(result_folder_path, "geometry_" + str(idx) + ".csv"))
        result_file_path = pth.join(result_folder_path, "openvsp_" + str(idx) + ".csv")

        return result_file_path

    @staticmethod
    def save_geometry_fuselage(result_folder_path, geometry_set):
        """Save geometry if not already computed by finding first available index."""
        geometry_set_labels = [
            "sweep25_wing",
            "taper_ratio_wing",
            "aspect_ratio_wing",
            "dihedral_wing",
            "twist_wing",
            "sweep25_htp",
            "taper_ratio_htp",
            "aspect_ratio_htp",
            "sweep25_vtp",
            "taper_ratio_vtp",
            "aspect_ratio_vtp",
            "mach",
            "area_ratio_htp",
            "area_ratio_vtp",
            "fus_length",
            "fus_front_length",
            "fus_rear_length",
            "fus_max_width",
            "fus_max_height",
        ]
        data = pd.DataFrame(geometry_set, index=geometry_set_labels)
        idx = 0
        while pth.exists(pth.join(result_folder_path, "geometry_" + str(idx) + ".csv")):
            idx += 1
        data.to_csv(pth.join(result_folder_path, "geometry_" + str(idx) + ".csv"))
        result_file_path = pth.join(result_folder_path, "openvsp_" + str(idx) + ".csv")

        return result_file_path

    @staticmethod
    def save_results(result_file_path, results):
        """Reads saved results."""

        labels = [
            "cL_u",
            "cD_u",
            "cm_u",
            "cL_alpha",
            "cD_alpha",
            "cm_alpha",
            "cL_q",
            "cD_q",
            "cm_q",
            "cY_beta",
            "cl_beta",
            "cn_beta",
            "cY_p",
            "cl_p",
            "cn_p",
            "cY_r",
            "cl_r",
            "cn_r",
        ]
        data = pd.DataFrame(results, index=labels)
        data.to_csv(result_file_path)

    @staticmethod
    def read_results(result_file_path):

        data = pd.read_csv(result_file_path)
        values = data.to_numpy()[:, 1].tolist()
        labels = data.to_numpy()[:, 0].tolist()

        return pd.DataFrame(values, index=labels)

    @staticmethod
    def read_stab_file(filename):
        file = open(filename, 'r')

        # col are formated such that CL=with respect to [alpha, beta, p,q,r, Mach, U, dfl, da, dr]
        # We get rid of Mach. dfl, da and dr are not calculated (flaps, ailerons and rudder)
        #    col=(11,24,37,50,63,76,128,141) # includes base aero, doesn't take flap effects
        #    lign=(49,50,51,52,53,54)
        #    byte=8

        CL = []
        CD = []
        CY = []
        Cl = []
        Cm = []
        Cn = []

        CoefOrder = []

        for lines in file:
            words = lines.split()

            if len(words) >= 2:

                # coefficient order
                if words[0] == 'Coef':
                    for i in range(len(words) - 1):
                        CoefOrder.append(words[i + 1])

                # ----- Lift coefficients -------
                if words[0] == 'CL':
                    for i in range(len(words) - 1):
                        if CoefOrder[i] != 'Mach':
                            # skip Mach and U derivative, take all inputs
                            CL.append(float(words[i + 1]))

                # ----- Side coefficients -------
                if words[0] == 'CS':
                    for i in range(len(words) - 1):
                        if CoefOrder[i] != 'Mach':
                            # skip Mach and U derivative, take all inputs
                            CY.append(float(words[i + 1]))

                # ----- Drag coefficients -------
                if words[0] == 'CD':
                    for i in range(len(words) - 1):
                        if CoefOrder[i] != 'Mach':
                            # skip Mach and U derivative, take all inputs
                            CD.append(float(words[i + 1]))

                # ----- pitch coefficients -------
                if words[0] == 'CMm':
                    for i in range(len(words) - 1):
                        if CoefOrder[i] != 'Mach':
                            # skip Mach and U derivative, take all inputs
                            Cm.append(float(words[i + 1]))

                # ----- roll coefficients -------
                if words[0] == 'CMl':
                    for i in range(len(words) - 1):
                        if CoefOrder[i] != 'Mach':
                            # skip Mach and U derivative, take all inputs
                            Cl.append(float(words[i + 1]))

                # ----- yaw coefficients -------
                if words[0] == 'CMn':
                    for i in range(len(words) - 1):
                        if CoefOrder[i] != 'Mach':
                            # skip Mach and U derivative, take all inputs
                            Cn.append(float(words[i + 1]))

        file.close()

        cL_u = CL[6]
        cD_u = CD[6]
        cm_u = Cm[6]
        cL_alpha = CL[1]
        cD_alpha = CD[1]
        cm_alpha = Cm[1]
        cL_q = CL[4]
        cD_q = CD[4]
        cm_q = Cm[4]
        cY_beta = CY[2]
        cl_beta = Cl[2]
        cn_beta = Cn[2]
        cY_p = CY[3]
        cl_p = Cl[3]
        cn_p = Cn[3]
        cY_r = CY[5]
        cl_r = Cl[5]
        cn_r = Cn[5]

        # derivative=('alpha','beta','p','q','r','a','n') #names of derivatives

        return (
            cL_u,
            cD_u,
            cm_u,
            cL_alpha,
            cD_alpha,
            cm_alpha,
            cL_q,
            cD_q,
            cm_q,
            cY_beta,
            cl_beta,
            cn_beta,
            cY_p,
            cl_p,
            cn_p,
            cY_r,
            cl_r,
            cn_r,
        )


class OPENVSPSimpleGeometryDP(OPENVSPSimpleGeometry):
    """Execution of OpenVSP for surfaces with slipstream effects."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._engine_wrapper = None

    def initialize(self):
        super().initialize()
        self.options.declare("propulsion_id", default=None, types=str, allow_none=True)

    def setup(self):
        super().setup()
        self._engine_wrapper = BundleLoader().instantiate_component(self.options["propulsion_id"])
        self._engine_wrapper.setup(self)
        # nan_array = np.full(ENGINE_COUNT, np.nan)
        self.add_input("data:geometry:wing:tip:leading_edge:x:local", val=np.nan, units="m")
        self.add_input("data:propulsion:IC_engine:max_rpm", val=np.nan, units="1/min")
        self.add_input("data:geometry:propeller:diameter", val=np.nan, units="m")
        self.add_input("data:geometry:propulsion:nacelle:length", val=np.nan, units="m")
        self.add_input("data:geometry:propulsion:engine:count", val=np.nan)
        # self.add_input("data:geometry:propulsion:engine:y_ratio", shape=ENGINE_COUNT, val=nan_array)

    def compute_wing_rotor(self, inputs, outputs, altitude, mach, aoa_angle, thrust_rate):
        """
        Function that computes in OpenVSP environment the wing with a rotor and returns the
        different aerodynamic parameters.

        @param inputs: inputs parameters defined within FAST-OAD-GA
        @param outputs: outputs parameters defined within FAST-OAD-GA
        @param altitude: altitude for aerodynamic calculation in meters
        @param mach: air speed expressed in mach
        @param aoa_angle: air speed angle of attack with respect to wing (degree)
        @param thrust_rate: thrust rate for computation of the power and thrust coefficient
        @return: wing dictionary including aero parameters as keys: y_vector, cl_vector, cd_vector,
        cm_vector, cl, cdi, cm, coef_e
        """

        # STEP 1/XX - DEFINE OR CALCULATE INPUT DATA FOR AERODYNAMIC EVALUATION ####################
        ############################################################################################

        # Get inputs (and calculate missing ones)
        sref_wing = float(inputs["data:geometry:wing:area"])
        x0_wing = inputs["data:geometry:wing:MAC:leading_edge:x:local"]
        l0_wing = inputs["data:geometry:wing:MAC:length"]
        width_max = inputs["data:geometry:fuselage:maximum_width"]
        y1_wing = width_max / 2.0
        y2_wing = inputs["data:geometry:wing:root:y"]
        l2_wing = inputs["data:geometry:wing:root:chord"]
        y4_wing = inputs["data:geometry:wing:tip:y"]
        l4_wing = inputs["data:geometry:wing:tip:chord"]
        x4_wing = inputs["data:geometry:wing:tip:leading_edge:x:local"]
        sweep_0_wing = inputs["data:geometry:wing:sweep_0"]
        fa_length = inputs["data:geometry:wing:MAC:at25percent:x"]
        span_wing = inputs["data:geometry:wing:span"]
        height_max = inputs["data:geometry:fuselage:maximum_height"]
        engine_rpm = inputs["data:propulsion:IC_engine:max_rpm"]
        propeller_diameter = float(inputs["data:geometry:propeller:diameter"])
        nac_length = inputs["data:geometry:propulsion:nacelle:length"]
        engine_config = inputs["data:geometry:propulsion:engine:layout"]
        engine_count = int(float(inputs["data:geometry:propulsion:engine:count"]))
        semi_span = span_wing / 2.0

        if engine_config != 1.0:
            y_ratio_array = 0.0
        else:
            used_index = np.where(
                np.array(inputs["data:geometry:propulsion:engine:y_ratio"]) >= 0.0
            )[0]
            y_ratio_array = np.array(inputs["data:geometry:propulsion:engine:y_ratio"])[used_index]

        # Compute remaining inputs
        atm = Atmosphere(altitude, altitude_in_feet=False)
        x_wing = fa_length - x0_wing - 0.25 * l0_wing
        z_wing = -(height_max - 0.12 * l2_wing) * 0.5
        span2_wing = y4_wing - y2_wing
        rho = atm.density
        v_inf = max(atm.speed_of_sound * mach, 0.01)  # avoid V=0 m/s crashes
        reynolds = v_inf * l0_wing / atm.kinematic_viscosity

        # STEP 1.5/XX - COMPUTE THE PARAMETERS RELATED TO THE COMPUTATION OF THE SLIPSTREAM ########
        # EFFECTS ON THE WING ######################################################################

        propulsion_model = FuelEngineSet(
            self._engine_wrapper.get_model(inputs), inputs["data:geometry:propulsion:engine:count"]
        )
        flight_point = FlightPoint(
            mach=mach,
            altitude=altitude,
            engine_setting=EngineSetting.CLIMB,
            thrust_rate=thrust_rate,
        )
        propulsion_model.compute_flight_points(flight_point)
        thrust = float(flight_point.thrust)
        power = thrust * v_inf / PROPELLER_EFFICIENCY
        engine_rps = engine_rpm / 60.0
        # For now thrust is distributed equally on each engine
        thrust_coefficient = round(
            float(thrust / engine_count / (rho * engine_rps ** 2.0 * propeller_diameter ** 4.0)), 5
        )
        power_coefficient = round(
            float(power / engine_count / (rho * engine_rps ** 3.0 * propeller_diameter ** 5.0)), 5
        )

        prop_radius = round(propeller_diameter / 2.0, 3)
        prop_hub_radius = round(0.2 * prop_radius, 3)

        # Writing propeller properties
        motor_pos_x = np.zeros(engine_count)
        motor_pos_y = np.zeros(engine_count)
        motor_pos_z = np.zeros(engine_count)
        motor_rpm_signed = np.zeros(engine_count)
        eng_start = 0
        if engine_config != 1.0:  # For now, we will just put a motor on the nose of the aircraft
            motor_pos_x[0] = 0.0
            motor_pos_y[0] = 0.0
            motor_pos_z[0] = 0.0
            motor_rpm_signed[0] = engine_rpm
            eng_per_wing = 1
            # Even if there is no engine of the wing, we put one so that we pick the correct
            # template, the engine will be placed on the nose
        else:
            if (
                    engine_count % 2 == 1.0
            ):  # Put one motor on the nose if there is an odd number of engine
                motor_pos_x[0] = 0.0
                motor_pos_y[0] = 0.0
                motor_pos_z[0] = z_wing
                motor_rpm_signed[0] = engine_rpm
                eng_start += 1
                eng_per_wing = int((engine_count - 1) / 2)
            else:
                eng_per_wing = int(engine_count / 2)

            i = 0
            # We put engine on the wings now, later, their position will be described by an array
            # in the xml
            for y_ratio in y_ratio_array:

                y_engine = y_ratio * semi_span

                if y_engine > y2_wing:  # engine in the tapered part of the wing
                    l_wing_eng = l4_wing + (l2_wing - l4_wing) * (y4_wing - y_engine) / (
                            y4_wing - y2_wing
                    )
                    delta_x_eng = 0.05 * l_wing_eng
                    x_eng_rel = (
                            x4_wing * (y_engine - y2_wing) / (y4_wing - y2_wing)
                            - delta_x_eng
                            - nac_length
                    )
                    x_eng = fa_length - 0.25 * l0_wing - (x0_wing - x_eng_rel)

                else:  # engine in the straight part of the wing
                    l_wing_eng = l2_wing
                    delta_x_eng = 0.05 * l_wing_eng
                    x_eng_rel = -delta_x_eng - nac_length
                    x_eng = fa_length - 0.25 * l0_wing - (x0_wing - x_eng_rel)

                if i % 2 == 0:
                    prop_rpm_loop = -engine_rpm
                else:
                    prop_rpm_loop = engine_rpm

                motor_pos_x[eng_start + i] = round(float(x_eng), 2)
                motor_pos_y[eng_start + i] = round(float(y_engine), 2)
                motor_pos_z[eng_start + i] = round(float(z_wing), 2)
                motor_rpm_signed[eng_start + i] = float(prop_rpm_loop)
                motor_pos_x[eng_start + eng_per_wing + i] = round(float(x_eng), 2)
                motor_pos_y[eng_start + eng_per_wing + i] = round(-float(y_engine), 2)
                motor_pos_z[eng_start + eng_per_wing + i] = round(float(z_wing), 2)
                motor_rpm_signed[eng_start + eng_per_wing + i] = -float(prop_rpm_loop)
                i += 1

        # STEP 2/XX - DEFINE WORK DIRECTORY, COPY RESOURCES AND CREATE COMMAND BATCH ###############
        ############################################################################################

        # If a folder path is specified for openvsp .exe, it becomes working directory (target),
        # if not temporary folder is created
        if self.options["openvsp_exe_path"]:
            target_directory = pth.abspath(self.options["openvsp_exe_path"])
        else:
            tmp_directory = _create_tmp_directory()
            target_directory = tmp_directory.name
        # Define the list of necessary input files: geometry script and foil file for both wing/HTP
        input_file_list = [
            pth.join(target_directory, INPUT_WING_ROTOR_SCRIPT),
            pth.join(target_directory, self.options["wing_airfoil_file"]),
        ]
        self.options["external_input_files"] = input_file_list
        # Define standard error file by default to avoid error code return
        self.stderr = pth.join(target_directory, STDERR_FILE_NAME)
        # Copy resource in working (target) directory
        # noinspection PyTypeChecker
        copy_resource_folder(openvsp3201, target_directory)
        # noinspection PyTypeChecker
        copy_resource(resources, self.options["wing_airfoil_file"], target_directory)
        # Create corresponding .bat files (one for each geometry configuration)
        self.options["command"] = [pth.join(target_directory, "vspscript.bat")]
        batch_file = open(self.options["command"][0], "w+")
        batch_file.write("@echo off\n")
        command = (
                pth.join(target_directory, VSPSCRIPT_EXE_NAME)
                + " -script "
                + pth.join(target_directory, INPUT_WING_ROTOR_SCRIPT)
                + " >nul 2>nul\n"
        )
        batch_file.write(command)
        batch_file.close()

        # STEP 3/XX - OPEN THE TEMPLATE SCRIPT FOR GEOMETRY GENERATION, MODIFY VALUES AND SAVE TO
        # WORKDIR ##################################################################################

        output_file_list = [
            pth.join(
                target_directory, INPUT_WING_ROTOR_SCRIPT.replace(".vspscript", "_DegenGeom.csv")
            )
        ]
        parser = InputFileGenerator()
        with path(local_resources, INPUT_WING_ROTOR_SCRIPT) as input_template_path:
            parser.set_template_file(str(input_template_path))
            parser.set_generated_file(input_file_list[0])
            # Modify wing parameters
            parser.mark_anchor("x_wing")
            parser.transfer_var(float(x_wing), 0, 5)
            parser.mark_anchor("z_wing")
            parser.transfer_var(float(z_wing), 0, 5)
            parser.mark_anchor("y1_wing")
            parser.transfer_var(float(y1_wing), 0, 5)
            for i in range(3):
                parser.mark_anchor("l2_wing")
                parser.transfer_var(float(l2_wing), 0, 5)
            parser.reset_anchor()
            parser.mark_anchor("span2_wing")
            parser.transfer_var(float(span2_wing), 0, 5)
            parser.mark_anchor("l4_wing")
            parser.transfer_var(float(l4_wing), 0, 5)
            parser.mark_anchor("sweep_0_wing")
            parser.transfer_var(float(sweep_0_wing), 0, 5)
            parser.mark_anchor("airfoil_0_file")
            parser.transfer_var('"' + input_file_list[1].replace("\\", "/") + '"', 0, 3)
            parser.mark_anchor("airfoil_1_file")
            parser.transfer_var('"' + input_file_list[1].replace("\\", "/") + '"', 0, 3)
            parser.mark_anchor("airfoil_2_file")
            parser.transfer_var('"' + input_file_list[1].replace("\\", "/") + '"', 0, 3)
            parser.mark_anchor("csv_file")
            csv_name = output_file_list[0]
            parser.transfer_var('"' + csv_name.replace("\\", "/") + '"', 0, 3)
            parser.generate()

        # STEP 4/XX - RUN BATCH TO GENERATE GEOMETRY .CSV FILE #####################################
        ############################################################################################

        self.options["external_output_files"] = output_file_list
        super().compute(inputs, outputs)

        # STEP 5/XX - DEFINE NEW INPUT/OUTPUT FILES LIST AND CREATE BATCH FOR VLM COMPUTATION ######
        ############################################################################################

        input_file_list = output_file_list
        input_file_list.append(input_file_list[0].replace(".csv", ".vspaero"))
        output_file_list = [
            input_file_list[0].replace(".csv", ".lod"),
            input_file_list[0].replace(".csv", ".polar"),
        ]
        self.options["external_input_files"] = input_file_list
        self.options["external_output_files"] = output_file_list
        self.options["command"] = [pth.join(target_directory, "vspaero.bat")]
        batch_file = open(self.options["command"][0], "w+")
        batch_file.write("@echo off\n")
        command = (
                pth.join(target_directory, VSPAERO_EXE_NAME)
                + " "
                + input_file_list[1].replace(".vspaero", "")
                + " >nul 2>nul\n"
        )
        batch_file.write(command)
        batch_file.close()

        # STEP 6/XX - OPEN THE TEMPLATE VSPAERO FOR COMPUTATION, MODIFY VALUES AND SAVE TO WORKDIR #
        ############################################################################################

        parser = InputFileGenerator()

        if engine_config == 1.0:
            rotor_template_file_name = generate_wing_rotor_file(int(engine_count / 2.0))
        else:
            rotor_template_file_name = generate_wing_rotor_file(int(1))

        with path(local_resources, rotor_template_file_name) as input_template_path:
            parser.set_template_file(str(input_template_path))
            parser.set_generated_file(input_file_list[1])
            parser.reset_anchor()
            parser.mark_anchor("Sref")
            parser.transfer_var(float(sref_wing), 0, 3)
            parser.mark_anchor("Cref")
            parser.transfer_var(float(l0_wing), 0, 3)
            parser.mark_anchor("Bref")
            parser.transfer_var(float(span_wing), 0, 3)
            parser.mark_anchor("X_cg")
            parser.transfer_var(float(fa_length), 0, 3)
            parser.mark_anchor("Mach")
            parser.transfer_var(float(mach), 0, 3)
            parser.mark_anchor("AOA")
            parser.transfer_var(float(aoa_angle), 0, 3)
            parser.mark_anchor("Vinf")
            parser.transfer_var(float(v_inf), 0, 3)
            parser.mark_anchor("Rho")
            parser.transfer_var(float(rho), 0, 3)
            parser.mark_anchor("ReCref")
            parser.transfer_var(float(reynolds), 0, 3)
            for i in range(1, eng_per_wing + 1):
                parser.mark_anchor("Prop_" + str(i) + "_name")
                parser.transfer_var("Prop_element_" + str(i), 0, 1)
                parser.mark_anchor("Disc_" + str(i) + "_ID")
                parser.transfer_var(i, 0, 1)
                parser.mark_anchor("Disc_" + str(i) + "_x")
                parser.transfer_var(motor_pos_x[i - 1], 0, 1)
                parser.transfer_var(motor_pos_y[i - 1], 0, 2)
                parser.transfer_var(motor_pos_z[i - 1], 0, 3)
                parser.mark_anchor("Disc_" + str(i) + "_nx")
                parser.transfer_var(1.0, 0, 1)
                parser.transfer_var(0.0, 0, 2)
                parser.transfer_var(0.0, 0, 3)
                parser.mark_anchor("Disc_" + str(i) + "_radius")
                parser.transfer_var(prop_radius, 0, 1)
                parser.mark_anchor("Disc_" + str(i) + "_hub_radius")
                parser.transfer_var(prop_hub_radius, 0, 1)
                parser.mark_anchor("Disc_" + str(i) + "_rpm")
                parser.transfer_var(motor_rpm_signed[i - 1], 0, 1)
                parser.mark_anchor("Disc_" + str(i) + "_CT")
                parser.transfer_var(thrust_coefficient, 0, 1)
                parser.mark_anchor("Disc_" + str(i) + "_CP")
                parser.transfer_var(power_coefficient, 0, 1)
            parser.generate()

        os.remove(pth.join(local_resources.__path__[0], rotor_template_file_name))

        # STEP 7/XX - RUN BATCH TO GENERATE AERO OUTPUT FILES (.lod, .polar...) ####################
        ############################################################################################

        super().compute(inputs, outputs)

        # STEP 8/XX - READ FILES, RETURN RESULTS (AND CLEAR TEMPORARY WORKDIR) #####################
        ############################################################################################

        # Open .lod file and extract data
        wing_y_vect = []
        wing_chord_vect = []
        wing_cl_vect = []
        wing_cd_vect = []
        wing_cm_vect = []
        with open(output_file_list[0], "r") as lf:
            data = lf.readlines()
            for i in range(len(data)):
                line = data[i].split()
                line.append("**")
                if line[0] == "1":
                    wing_y_vect.append(float(line[2]))
                    wing_chord_vect.append(float(line[3]))
                    wing_cl_vect.append(float(line[5]))
                    wing_cd_vect.append(float(line[6]))
                    wing_cm_vect.append(float(line[12]))
                if line[0] == "Comp":
                    cl_wing = float(data[i + 1].split()[5]) + float(
                        data[i + 2].split()[5]
                    )  # sum CL left/right
                    cdi_wing = float(data[i + 1].split()[6]) + float(
                        data[i + 2].split()[6]
                    )  # sum CDi left/right
                    cm_wing = float(data[i + 1].split()[12]) + float(
                        data[i + 2].split()[12]
                    )  # sum CM left/right
                    break
        # Open .polar file and extract data
        with open(output_file_list[1], "r") as lf:
            data = lf.readlines()
            wing_e = float(data[1].split()[10])
        # Delete temporary directory
        if not (self.options["openvsp_exe_path"]):
            # noinspection PyUnboundLocalVariable
            tmp_directory.cleanup()
        # Return values
        wing_rotor = {
            "y_vector": wing_y_vect,
            "cl_vector": wing_cl_vect,
            "chord_vector": wing_chord_vect,
            "cd_vector": wing_cd_vect,
            "cm_vector": wing_cm_vect,
            "cl": cl_wing,
            "cdi": cdi_wing,
            "cm": cm_wing,
            "coef_e": wing_e,
            "ct": thrust_coefficient,
        }
        return wing_rotor


def generate_wing_rotor_file(engine_count: int):
    """
    Uses the base VSPAERO template file to generate a file with all the line required to launch
    OpenVSP with n rotors in the run

    :param engine_count: the number of engine in the run

    return the path to the new template file for the n rotor run.
    """

    rotor_template_file_name = "wing_" + str(engine_count) + "_rotor_openvsp_DegenGeom.vspaero"
    original_template = pth.join(local_resources.__path__[0], "wing_openvsp_DegenGeom.vspaero")
    new_template = pth.join(local_resources.__path__[0], rotor_template_file_name)

    file_to_copy = open(original_template, "r").readlines()
    file = open(new_template, "w")

    for i, _ in enumerate(file_to_copy):
        if "NumberOfRotors" in file_to_copy[i]:
            new_line = list(file_to_copy[i][:])
            new_line[-2] = str(engine_count)
            file.write("".join(new_line))
            for j in range(engine_count):
                engine_number = str(int(j + 1))
                file.write("Prop_" + engine_number + "_name\n")
                file.write("Disc_" + engine_number + "_ID\n")
                file.write(
                    "Disc_"
                    + engine_number
                    + "_x Disc_"
                    + engine_number
                    + "_y Disc_"
                    + engine_number
                    + "_z\n"
                )
                file.write(
                    "Disc_"
                    + engine_number
                    + "_nx Disc_"
                    + engine_number
                    + "_ny Disc_"
                    + engine_number
                    + "_nz\n"
                )
                file.write("Disc_" + engine_number + "_radius\n")
                file.write("Disc_" + engine_number + "_hub_radius\n")
                file.write("Disc_" + engine_number + "_rpm\n")
                file.write("Disc_" + engine_number + "_CT\n")
                file.write("Disc_" + engine_number + "_CP\n")
        else:
            file.write(file_to_copy[i])

    file.close()

    return rotor_template_file_name
