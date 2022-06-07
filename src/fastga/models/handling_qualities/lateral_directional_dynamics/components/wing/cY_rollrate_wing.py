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
class CYRollRateWing(FigureDigitization2):

    def setup(self):
        # Flight reference condition
        self.add_input("data:reference_flight_condition:mach", val=np.nan)
        self.add_input("data:reference_flight_condition:alpha", val=np.nan, units="rad")

        # Wing Geometry
        self.add_input("data:geometry:wing:span", val=np.nan, units="m")
        self.add_input("data:geometry:wing:aspect_ratio", val=np.nan)
        self.add_input("data:geometry:wing:taper_ratio", val=np.nan)
        self.add_input("data:geometry:wing:dihedral", val=np.nan, units="rad")
        self.add_input("data:geometry:wing:sweep_25", val=np.nan, units="rad")

        # Wing Aerodynamics
        self.add_input("data:aerodynamics:wing:cruise:induced_drag_coefficient", val=np.nan)

        # Vertical tail Geometry
        self.add_input("data:geometry:vertical_tail:MAC:z", val=np.nan, units="m")
        self.add_input("data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25", val=np.nan, units="m")

        self.add_output("data:handling_qualities:lateral:derivatives:wing:CY:rollrate", units="rad**-1")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        # Flight reference condition
        mach = inputs["data:reference_flight_condition:mach"]
        alpha = inputs["data:reference_flight_condition:alpha"]

        # Wing geometry
        b = inputs["data:geometry:wing:span"]
        A_W = inputs["data:geometry:wing:aspect_ratio"]
        taper_ratio_W = inputs["data:geometry:wing:taper_ratio"]
        dihedral_W = inputs["data:geometry:wing:dihedral"]
        sweep_25_W = inputs["data:geometry:wing:sweep_25"]

        # Wing aerodynamics
        k = inputs["data:aerodynamics:wing:cruise:induced_drag_coefficient"]

        # Vertical tail geometry
        # TODO: compute l_v as the distance between the vertical tail aerodynamic center and the CG of the aircraft
        l_v = inputs["data:geometry:vertical_tail:MAC:at25percent:x:from_wingMAC25"]
        z_v = inputs["data:geometry:vertical_tail:MAC:z"]
        beta = math.sqrt(1 - mach ** 2 * (math.cos(sweep_25_W)) ** 2)

        z = z_v * math.cos(alpha) - l_v * math.sin(alpha)
        # Figure 10.35
        beta_Clp_k = self.get_beta_Clp_k(mach, k, sweep_25_W, A_W, taper_ratio_W)
        Cl_p_0 = k / beta * beta_Clp_k
        CY_p_W = 3 * math.sin(dihedral_W) * (1 - math.sin(dihedral_W) * (4 * z / b)) * Cl_p_0

        outputs["data:handling_qualities:lateral:derivatives:wing:CY:rollrate"] = CY_p_W
