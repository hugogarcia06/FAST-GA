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

import numpy as np
import math

from fastga.models.handling_qualities.utils.figure_digitization import FigureDigitization2


# TODO:register class
class ClRollRateHT(FigureDigitization2):

    def setup(self):
        # Flight reference condition
        self.add_input("data:reference_flight_condition:mach", val=np.nan)

        # wing Geometry
        self.add_input("data:geometry:wing:taper_ratio", val=np.nan)
        self.add_input("data:geometry:wing:span", val=np.nan, units="m")
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")

        # Horizontal Tail Geometry
        self.add_input("data:geometry:horizontal_tail:aspect_ratio", val=np.nan)
        self.add_input("data:geometry:horizontal_tail:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:horizontal_tail:span", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:sweep_25", val=np.nan, units="rad")

        # Horizontal Tail Aerodynamics
        self.add_input("data:aerodynamics:horizontal_tail:cruise:induced_drag_coefficient", val=np.nan, units="rad**-1")

        self.add_output("data:handling_qualities:lateral:derivatives:horizontal_tail:Cl:rollrate", units="rad**-1")


    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        # Flight reference condition
        mach = inputs["data:reference_flight_condition:mach"]
        beta = math.sqrt(1 - mach ** 2)

        # Wing geometry
        b = inputs["data:geometry:wing:span"]
        S_W = inputs["data:geometry:wing:area"]
        taper_ratio_W = inputs["data:geometry:wing:taper_ratio"]

        # Horizontal tail geometry
        A_H = inputs["data:geometry:horizontal_tail:aspect_ratio"]
        S_H = inputs["data:geometry:horizontal_tail:area"]
        b_H = inputs["data:geometry:horizontal_tail:span"]
        sweep_25_H = inputs["data:geometry:horizontal_tail:sweep_25"]

        # Horizontal tail aerodynamics
        k_H = inputs["data:aerodynamics:horizontal_tail:cruise:induced_drag_coefficient"]

        # Horizontal tail contribution to Cl_p. It should be calculated in the same way as wing contribution, but it
        # will be simplified to just one term (Expression 8.3 from Roskam - Methods for Estimating Stability and Control
        # Derivatives of Conventional Subsonic Airplanes)
        beta_Clp_k_H = self.get_beta_Clp_k(mach, k_H, sweep_25_H, A_H, taper_ratio_W)
        Cl_p_H = 0.5 * beta_Clp_k_H * k_H / beta * (S_H / S_W) * (b_H / b) ** 2

        outputs["data:handling_qualities:lateral:derivatives:horizontal_tail:Cl:rollrate"] = Cl_p_H