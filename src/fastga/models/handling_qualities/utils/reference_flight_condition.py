#  This file is part of FAST-OAD : A framework for rapid Overall Aircraft Design
#  Copyright (C) 2022 ONERA & ISAE-SUPAERO
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
import math
import numpy as np
import os

import openmdao.api as om
from fastoad.model_base import Atmosphere
from .. import resources as local_resources

from openmdao.utils.file_wrap import InputFileGenerator
from importlib.resources import path



class ReferenceFlightCondition(om.ExplicitComponent):
    """ Establishes the reference flight condition for the stability analysis  """

    def initialize(self):
        """Definition of the options of the group"""
        self.options.declare("airplane_file", default="", types=str)
        self.options.declare("reference_flight_condition", default={}, types=dict)
        self.options.declare("result_folder_path", default="", types=str)
        self.options.declare("add_fuselage", default=False, types=bool, allow_none=True)
        self.options.declare("use_openvsp", default=True, types=bool)


    def setup(self):

        self.add_input("data:weight:aircraft:MTOW", val=np.nan, units="kg")
        self.add_input("data:weight:aircraft:inertia:Iox", val=np.nan, units="kg*m**2")
        self.add_input("data:weight:aircraft:inertia:Ioy", val=np.nan, units="kg*m**2")
        self.add_input("data:weight:aircraft:inertia:Ioz", val=np.nan, units="kg*m**2")
        self.add_input("data:weight:aircraft:inertia:Ioxz", val=np.nan, units="kg*m**2")
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input("data:aerodynamics:aircraft:cruise:CD0", val=np.nan)
        self.add_input("data:aerodynamics:wing:cruise:induced_drag_coefficient", val=np.nan)

        self.add_input("data:weight:aircraft:CG:aft:x", val=np.nan, units="m")
        self.add_input("data:weight:aircraft:CG:fwd:x", val=np.nan, units="m")

        self.add_output("data:reference_flight_condition:mach")
        self.add_output("data:reference_flight_condition:alpha", units="deg")
        self.add_output("data:reference_flight_condition:theta", units="deg")
        self.add_output("data:reference_flight_condition:speed", units="m/s")
        self.add_output("data:reference_flight_condition:altitude", units="ft")
        self.add_output("data:reference_flight_condition:air_density", units="kg/m**3")
        self.add_output("data:reference_flight_condition:weight", units="kg")
        self.add_output("data:reference_flight_condition:Ixx", units="kg*m**2")
        self.add_output("data:reference_flight_condition:Iyy", units="kg*m**2")
        self.add_output("data:reference_flight_condition:Izz", units="kg*m**2")
        self.add_output("data:reference_flight_condition:Ixz", units="kg*m**2")
        self.add_output("data:reference_flight_condition:CL")
        self.add_output("data:reference_flight_condition:CD")
        self.add_output("data:reference_flight_condition:CD0")
        self.add_output("data:reference_flight_condition:CT")
        self.add_output("data:reference_flight_condition:dynamic_pressure", units="Pa")
        self.add_output("data:reference_flight_condition:CG:x", units="m")
        self.add_output("data:reference_flight_condition:CG:z", units="m")
        self.add_output("data:reference_flight_condition:flaps_deflection", units="deg")
        self.add_output("data:reference_flight_condition:flight_phase_category")



    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):


        g = 9.81
        S = inputs["data:geometry:wing:area"]
        flight_phase_category = 2.0
        MTOW = inputs["data:weight:aircraft:MTOW"]

        if self.options["reference_flight_condition"] == {}:

            mach = 0.201
            # NOTE: aoa in reference condition is always zero due to having chosen the stability axes (body-axis that
            # NOTE: confuse with wind axis in trim condition)
            alpha = 0.0   # deg
            theta = 0.0   # deg
            theta_rad = theta * math.pi / 180.0
            altitude = 5000.0    # altitude in feet
            sound_speed = Atmosphere(altitude, True).speed_of_sound   # m/s
            speed = mach * sound_speed   # m/s
            rho = Atmosphere(altitude, True).density   # kg/m**3
            q = 0.5 * rho * speed**2   # Pa

            weight = 0.9*MTOW   # kg
            L = math.cos(theta * math.pi / 180.0) * weight * g
            CL = L / (q * S)

            CD0 = inputs["data:aerodynamics:aircraft:cruise:CD0"]
            k = inputs["data:aerodynamics:wing:cruise:induced_drag_coefficient"]
            CD = CD0 + k * CL ** 2
            CT = CD + math.sin(theta * math.pi / 180.0) * weight / (q * S)

            flaps_deflection = 0.0

            # NOTE: for purposes of preliminary design, the most critical airplane configurations occur at the most forward
            # NOTE:  and at the most aft center of gravity locations.
            cg_aft = inputs["data:weight:aircraft:CG:aft:x"]
            cg_fwd = inputs["data:weight:aircraft:CG:fwd:x"]
            cg_x = (cg_fwd + cg_aft) / 2.0
            cg_z = 0.0

            Ixx_B = float(inputs["data:weight:aircraft:inertia:Iox"])
            Iyy_B = float(inputs["data:weight:aircraft:inertia:Ioy"])
            Izz_B = float(inputs["data:weight:aircraft:inertia:Ioz"])
            Ixz_B = float(inputs["data:weight:aircraft:inertia:Ioxz"])

            transformation_matrix = np.array([
                [(math.cos(theta_rad))**2, (math.sin(theta_rad))**2, -math.sin(2*theta_rad)],
                [(math.sin(theta_rad))**2, (math.cos(theta_rad))**2, math.sin(2*theta_rad)],
                [0.5*math.sin(2*theta_rad), -0.5*math.sin(2*theta_rad), math.cos(2*theta_rad)]
            ])

            body_inertias = np.array([Ixx_B, Izz_B, Ixz_B])
            stability_inertias = transformation_matrix.dot(body_inertias)
            [Ixx_S, Izz_S, Ixz_S] = stability_inertias
            Iyy_S = Iyy_B


        else:
            ref_flight_cond_dict = self.options["reference_flight_condition"]

            mach = ref_flight_cond_dict["mach"]
            alpha = 0.0
            altitude = ref_flight_cond_dict["altitude"]
            theta = ref_flight_cond_dict["theta"]
            theta_rad = theta * math.pi / 180.0
            sound_speed = Atmosphere(altitude, True).speed_of_sound   # m/s
            speed = mach * sound_speed   # m/s
            rho = Atmosphere(altitude, True).density   # kg/m**3
            q = 0.5 * rho * speed**2   # Pa

            weight = ref_flight_cond_dict["weight"]
            L = math.cos(theta * math.pi / 180.0) * weight * g
            CL = L / (q * S)

            CD0 = inputs["data:aerodynamics:aircraft:cruise:CD0"]
            k = inputs["data:aerodynamics:wing:cruise:induced_drag_coefficient"]
            CD = CD0 + k * CL ** 2
            CT = CD + math.sin(theta * math.pi / 180.0) * weight / (q * S)

            flaps_deflection = 0.0

            cg_x = ref_flight_cond_dict["cg_x"]
            cg_z = 0.0

            Ixx_B = float(inputs["data:weight:aircraft:inertia:Iox"])
            Iyy_B = float(inputs["data:weight:aircraft:inertia:Ioy"])
            Izz_B = float(inputs["data:weight:aircraft:inertia:Ioz"])
            Ixz_B = float(inputs["data:weight:aircraft:inertia:Ioxz"])

            transformation_matrix = np.array([
                [(math.cos(theta_rad)) ** 2, (math.sin(theta_rad)) ** 2, -math.sin(2 * theta_rad)],
                [(math.sin(theta_rad)) ** 2, (math.cos(theta_rad)) ** 2, math.sin(2 * theta_rad)],
                [0.5 * math.sin(2 * theta_rad), -0.5 * math.sin(2 * theta_rad), math.cos(2 * theta_rad)]
            ])

            body_inertias = np.array([Ixx_B, Izz_B, Ixz_B])
            stability_inertias = transformation_matrix.dot(body_inertias)
            [Ixx_S, Izz_S, Ixz_S] = stability_inertias
            Iyy_S = Iyy_B


        airplane_file = self.options["airplane_file"]

        file_name = "readme_test_data.txt"
        dir_name = self.options["result_folder_path"]
        RESULTS_FILE = os.path.join(dir_name, file_name)

        parser = InputFileGenerator()
        with path(local_resources, "readme_test_data.txt") as input_template_path:
            parser.set_template_file(str(input_template_path))
            parser.set_generated_file(RESULTS_FILE)
            parser.mark_anchor("airplane_xml_file")
            if airplane_file == "":
                parser.transfer_var("Not Specified", 0, 5)
            else:
                parser.transfer_var(airplane_file, 0, 5)
            parser.reset_anchor()
            parser.mark_anchor("add_fuselage_bool")
            if self.options["add_fuselage"] is True:
                parser.transfer_var("True", 0, 2)
            else:
                parser.transfer_var("False", 0, 2)
            parser.reset_anchor()
            parser.mark_anchor("use_openvsp_bool")
            if self.options["use_openvsp"] is True:
                parser.transfer_var("True", 0, 2)

            else:
                parser.transfer_var("False", 0, 2)
            parser.reset_anchor()
            parser.mark_anchor("mach_data")
            parser.transfer_var(round(float(mach), 5), 0, 2)
            parser.mark_anchor("alpha_data")
            parser.transfer_var(round(float(alpha), 5), 0, 2)
            parser.mark_anchor("theta_data")
            parser.transfer_var(round(float(theta), 5), 0, 2)
            parser.reset_anchor()
            parser.mark_anchor("altitude_data")
            parser.transfer_var(round(float(altitude), 5), 0, 2)
            parser.mark_anchor("speed_data")
            parser.transfer_var(round(float(speed), 5), 0, 2)
            parser.mark_anchor("rho_data")
            parser.transfer_var(round(float(rho), 5), 0, 2)
            parser.mark_anchor("flaps_deflection_data")
            parser.transfer_var(round(float(flaps_deflection), 5), 0, 2)
            parser.reset_anchor()
            parser.mark_anchor("CL_data")
            parser.transfer_var(round(float(CL), 5), 0, 2)
            parser.mark_anchor("CD_data")
            parser.transfer_var(round(float(CD), 5), 0, 2)
            parser.mark_anchor("CT_data")
            parser.transfer_var(round(float(CT), 5), 0, 2)
            parser.reset_anchor()
            parser.mark_anchor("weight_data")
            parser.transfer_var(round(float(weight), 5), 0, 2)
            parser.mark_anchor("cg_x_position_data")
            parser.transfer_var(round(float(cg_x), 5), 0, 2)
            parser.mark_anchor("cg_z_position_data")
            parser.transfer_var(round(float(cg_z), 5), 0, 2)
            parser.mark_anchor("Ixx_B_data")
            parser.transfer_var(round(float(Ixx_B), 5), 0, 3)
            parser.mark_anchor("Iyy_B_data")
            parser.transfer_var(round(float(Iyy_B), 5), 0, 3)
            parser.mark_anchor("Izz_B_data")
            parser.transfer_var(round(float(Izz_B), 5), 0, 3)
            parser.mark_anchor("Ixz_B_data")
            parser.transfer_var(round(float(Ixz_B), 5), 0, 3)
            parser.mark_anchor("Ixx_S_data")
            parser.transfer_var(round(float(Ixx_S), 5), 0, 3)
            parser.mark_anchor("Iyy_S_data")
            parser.transfer_var(round(float(Iyy_S), 5), 0, 3)
            parser.mark_anchor("Izz_S_data")
            parser.transfer_var(round(float(Izz_S), 5), 0, 3)
            parser.mark_anchor("Ixz_S_data")
            parser.transfer_var(round(float(Ixz_S), 5), 0, 3)
            parser.reset_anchor()

            parser.generate()

        outputs["data:reference_flight_condition:mach"] = mach
        outputs["data:reference_flight_condition:alpha"] = alpha
        outputs["data:reference_flight_condition:theta"] = theta
        outputs["data:reference_flight_condition:speed"] = speed
        outputs["data:reference_flight_condition:weight"] = weight
        outputs["data:reference_flight_condition:Ixx"] = Ixx_S
        outputs["data:reference_flight_condition:Iyy"] = Iyy_S
        outputs["data:reference_flight_condition:Izz"] = Izz_S
        outputs["data:reference_flight_condition:Ixz"] = Ixz_S
        outputs["data:reference_flight_condition:altitude"] = altitude
        outputs["data:reference_flight_condition:air_density"] = rho
        outputs["data:reference_flight_condition:CL"] = CL
        outputs["data:reference_flight_condition:CD"] = CD
        outputs["data:reference_flight_condition:CT"] = CT
        outputs["data:reference_flight_condition:dynamic_pressure"] = q
        outputs["data:reference_flight_condition:CG:x"] = cg_x
        outputs["data:reference_flight_condition:CG:z"] = cg_z
        outputs["data:reference_flight_condition:flaps_deflection"] = flaps_deflection
        outputs["data:reference_flight_condition:CD0"] = CD0
        outputs["data:reference_flight_condition:flight_phase_category"] = flight_phase_category
