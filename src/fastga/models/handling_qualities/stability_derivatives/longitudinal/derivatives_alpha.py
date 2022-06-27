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

import openmdao.api as om
import numpy as np
import math


# TODO: register the class
class AoADerivatives(om.ExplicitComponent):
    """
    Computes de stability angle of attack derivatives of the CD, CL and Cm coefficients. The expressions used for the
    computation of the different coefficients were obtained from Roskam - Estimating Stability and Control Derivatives.
    Same expressions can be found in Roskam - Aircraft Design Part VI: Preliminary Calculation of Aerodynamic Thrust and
    Power Characteristics.
    """

    def setup(self):
        # Reference Flight Condition
        self.add_input("data:reference_flight_condition:weight", val=np.nan, units="kg")
        self.add_input("data:reference_flight_condition:dynamic_pressure", val=np.nan, units="Pa")
        self.add_input("data:reference_flight_condition:CG:x", val=np.nan, units="m")

        self.add_input("data:aerodynamics:cruise:neutral_point:stick_fixed:x", val=np.nan)

        # Wing Geometry
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:wing:aspect_ratio", val=np.nan)
        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:at25percent:x", val=np.nan, units="m")

        # Wing Aerodynamics
        self.add_input("data:aerodynamics:wing:cruise:induced_drag_coefficient", val=np.nan)

        # Horizontal Tail Aerodynamics
        self.add_input("data:aerodynamics:horizontal_tail:efficiency", val=np.nan,)

        self.add_input("data:handling_qualities:stick_fixed_static_margin", val=np.nan)

        self.add_input(
            "data:handling_qualities:longitudinal:derivatives:wing:CL:alpha", val=np.nan, units="rad**-1")
        self.add_input(
            "data:handling_qualities:longitudinal:derivatives:horizontal_tail:CL:alpha", val=np.nan, units="rad**-1"
                       )

        self.add_output("data:handling_qualities:longitudinal:derivatives:CL:alpha", units="rad**-1")
        self.add_output("data:handling_qualities:longitudinal:derivatives:CD:alpha", units="rad**-1")
        self.add_output("data:handling_qualities:longitudinal:derivatives:Cm:alpha", units="rad**-1")


    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        # Reference flight conditions
        weight = inputs["data:reference_flight_condition:weight"]
        q = inputs["data:reference_flight_condition:dynamic_pressure"]

        # Wing Geometry
        S_W = inputs["data:geometry:wing:area"]
        A_W = inputs["data:geometry:wing:aspect_ratio"]

        mac_length = inputs["data:geometry:wing:MAC:length"]
        aero_center = inputs["data:aerodynamics:cruise:neutral_point:stick_fixed:x"]
        cg_x = inputs["data:reference_flight_condition:CG:x"]
        mac_leading_edge_x = inputs["data:geometry:wing:MAC:at25percent:x"] - 0.25 * mac_length
        cg_x_local = cg_x - mac_leading_edge_x
        static_margin = aero_center - cg_x_local / mac_length

        # static_margin = inputs["data:handling_qualities:stick_fixed_static_margin"]

        # Aerodynamics
        k = inputs["data:aerodynamics:wing:cruise:induced_drag_coefficient"]
        e = 1 / (math.pi * A_W * k)

        # ---- CL alpha derivative ----
        CL_alpha_WB = inputs["data:handling_qualities:longitudinal:derivatives:wing:CL:alpha"]
        CL_alpha_H = inputs["data:handling_qualities:longitudinal:derivatives:horizontal_tail:CL:alpha"]
        CL_alpha = CL_alpha_WB + CL_alpha_H

        # ---- CD alpha derivative ----
        CL = weight/(q * S_W)
        CD_alpha = 2 * CL * CL_alpha * (1/(math.pi * A_W * e))

        # ---- Cm alpha derivative ----
        Cm_alpha = - static_margin * CL_alpha

        outputs["data:handling_qualities:longitudinal:derivatives:CD:alpha"] = CD_alpha
        outputs["data:handling_qualities:longitudinal:derivatives:CL:alpha"] = CL_alpha
        outputs["data:handling_qualities:longitudinal:derivatives:Cm:alpha"] = Cm_alpha