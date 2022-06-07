"""
    Estimation of stability derivatives coefficients using OpenVSP.
"""
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


import logging

import numpy as np
from openmdao.core.group import Group

from .openvsp import OPENVSPSimpleGeometry, DEFAULT_WING_AIRFOIL, DEFAULT_HTP_AIRFOIL, DEFAULT_VTP_AIRFOIL


_LOGGER = logging.getLogger(__name__)


class ComputeSTABopenvsp(Group):
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

    def setup(self):

        self.add_subsystem(
            "stab_openvsp",
            _ComputeSTABopenvsp(
                result_folder_path=self.options["result_folder_path"],
                openvsp_exe_path=self.options["openvsp_exe_path"],
                wing_airfoil_file=self.options["wing_airfoil_file"],
                htp_airfoil_file=self.options["htp_airfoil_file"],
                vtp_airfoil_file=self.options["vtp_airfoil_file"]
            ),
            promotes=["*"],
        )


class _ComputeSTABopenvsp(OPENVSPSimpleGeometry):
    def initialize(self):
        super().initialize()

    def setup(self):
        super().setup()

        self.add_input("data:reference_flight_condition:mach")
        self.add_input("data:reference_flight_condition:altitude", units="m")
        self.add_input("data:reference_flight_condition:alpha", units="deg")

        self.add_output("data:handling_qualities:longitudinal:derivatives:CL:speed", units="rad**-1")
        self.add_output("data:handling_qualities:longitudinal:derivatives:CD:speed", units="rad**-1")
        self.add_output("data:handling_qualities:longitudinal:derivatives:Cm:speed", units="rad**-1")
        self.add_output("data:handling_qualities:longitudinal:derivatives:CL:alpha", units="rad**-1")
        self.add_output("data:handling_qualities:longitudinal:derivatives:CD:alpha", units="rad**-1")
        self.add_output("data:handling_qualities:longitudinal:derivatives:Cm:alpha", units="rad**-1")
        self.add_output("data:handling_qualities:longitudinal:derivatives:CL:pitchrate", units="rad**-1")
        self.add_output("data:handling_qualities:longitudinal:derivatives:CD:pitchrate", units="rad**-1")
        self.add_output("data:handling_qualities:longitudinal:derivatives:Cm:pitchrate", units="rad**-1")
        self.add_output("data:handling_qualities:lateral:derivatives:CY:beta", units="rad**-1")
        self.add_output("data:handling_qualities:lateral:derivatives:Cl:beta", units="rad**-1")
        self.add_output("data:handling_qualities:lateral:derivatives:Cn:beta", units="rad**-1")
        self.add_output("data:handling_qualities:lateral:derivatives:CY:rollrate", units="rad**-1")
        self.add_output("data:handling_qualities:lateral:derivatives:Cl:rollrate", units="rad**-1")
        self.add_output("data:handling_qualities:lateral:derivatives:Cn:rollrate", units="rad**-1")
        self.add_output("data:handling_qualities:lateral:derivatives:CY:yawrate", units="rad**-1")
        self.add_output("data:handling_qualities:lateral:derivatives:Cl:yawrate", units="rad**-1")
        self.add_output("data:handling_qualities:lateral:derivatives:Cn:yawrate", units="rad**-1")



    def check_config(self, logger):
        # let void to avoid logger error on "The command cannot be empty"
        pass

    def compute(self, inputs, outputs):

        INPUT_AOA = float(inputs["data:reference_flight_condition:alpha"])  # only one value given since calculation is done by default around 0.0!

        _LOGGER.debug("Entering aerodynamic computation")

        # Check AOA input is float
        if not isinstance(INPUT_AOA, float):
            raise TypeError("INPUT_AOA should be a float!")

        altitude = float(inputs["data:reference_flight_condition:altitude"])
        mach = float(inputs["data:reference_flight_condition:mach"])

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
            cn_r
        ) = self.compute_stab_coef(inputs, outputs, altitude, mach, INPUT_AOA)


        # Defining outputs
        outputs["data:handling_qualities:longitudinal:derivatives:CL:speed"] = cL_u
        outputs["data:handling_qualities:longitudinal:derivatives:CD:speed"] = cD_u
        outputs["data:handling_qualities:longitudinal:derivatives:Cm:speed"] = cm_u
        outputs["data:handling_qualities:longitudinal:derivatives:CL:alpha"] = cL_alpha
        outputs["data:handling_qualities:longitudinal:derivatives:CD:alpha"] = cD_alpha
        outputs["data:handling_qualities:longitudinal:derivatives:Cm:alpha"] = cm_alpha
        outputs["data:handling_qualities:longitudinal:derivatives:CL:pitchrate"] = cL_q
        outputs["data:handling_qualities:longitudinal:derivatives:CD:pitchrate"] = cD_q
        outputs["data:handling_qualities:longitudinal:derivatives:Cm:pitchrate"] = cm_q
        outputs["data:handling_qualities:lateral:derivatives:CY:beta"] = cY_beta
        outputs["data:handling_qualities:lateral:derivatives:Cl:beta"] = cl_beta
        outputs["data:handling_qualities:lateral:derivatives:Cn:beta"] = cn_beta
        outputs["data:handling_qualities:lateral:derivatives:CY:rollrate"] = cY_p
        outputs["data:handling_qualities:lateral:derivatives:Cl:rollrate"] = cl_p
        outputs["data:handling_qualities:lateral:derivatives:Cn:rollrate"] = cn_p
        outputs["data:handling_qualities:lateral:derivatives:CY:yawrate"] = cY_r
        outputs["data:handling_qualities:lateral:derivatives:Cl:yawrate"] = cl_r
        outputs["data:handling_qualities:lateral:derivatives:Cn:yawrate"] = cn_r

