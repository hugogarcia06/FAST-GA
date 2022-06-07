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

from fastga.models.handling_qualities.utils.lift_curve_slope import get_lift_curve_slope

from fastga.models.handling_qualities.utils.figure_digitization import FigureDigitization2


# TODO:register class
class ClRollRateWing(FigureDigitization2):

    def setup(self):
        # Flight reference condition
        self.add_input("data:reference_flight_condition:mach", val=np.nan)
        self.add_input("data:reference_flight_condition:CL", val=np.nan)
        self.add_input("data:reference_flight_condition:CD0", val=np.nan)

        # Wing Geometry
        self.add_input("data:geometry:wing:span", val=np.nan, units="m")
        self.add_input("data:geometry:wing:aspect_ratio", val=np.nan)
        self.add_input("data:geometry:wing:taper_ratio", val=np.nan)
        self.add_input("data:geometry:wing:dihedral", val=np.nan, units="rad")
        self.add_input("data:geometry:wing:sweep_25", val=np.nan, units="rad")
        self.add_input("data:geometry:wing:sweep_0", val=np.nan, units="rad")
        self.add_input("data:geometry:wing:vertical_position", val=np.nan, units="m")

        # Wing Aerodynamics
        self.add_input("data:aerodynamics:wing:cruise:induced_drag_coefficient", val=np.nan)
        self.add_input("data:aerodynamics:wing:airfoil:CL_alpha", val=np.nan, units="rad**-1")

        self.add_output("data:handling_qualities:lateral:derivatives:wing:Cl:rollrate", units="rad**-1")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        # Flight reference condition
        mach = inputs["data:reference_flight_condition:mach"]
        CL_s = inputs["data:reference_flight_condition:CL"]
        CL_W = CL_s
        CD0_W = inputs["data:reference_flight_condition:CD0"]

        # Wing geometry
        b = inputs["data:geometry:wing:span"]
        A_W = inputs["data:geometry:wing:aspect_ratio"]
        taper_ratio_W = inputs["data:geometry:wing:taper_ratio"]
        dihedral_W = inputs["data:geometry:wing:dihedral"]
        sweep_25_W = inputs["data:geometry:wing:sweep_25"]
        # sweep_0_W = inputs["data:geometry:wing:sweep_0"]
        # sweep_50_W = math.atan(math.tan(sweep_0_W) - 2 / A_W * ((1 - taper_ratio_W) / (1 + taper_ratio_W)))
        z_w = inputs["data:geometry:wing:vertical_position"]

        # Wing Aerodynamics
        k = inputs["data:aerodynamics:wing:cruise:induced_drag_coefficient"]
        # cl_alpha_w = inputs["data:aerodynamics:wing:airfoil:CL_alpha"]

        beta = math.sqrt(1 - mach ** 2)
        # Wing contribution to Cl_p. Equation 10.52 from Roskam - Airplane Design Part VI
        # Roll damping parameter obtained from Roskam. Figure 10.35
        beta_Clp_k = self.get_beta_Clp_k(mach, k, sweep_25_W, A_W, taper_ratio_W)
        # Wing lift-curve slope at any lift coefficient. Section 8.1.3.5 or 8.1.4.4.
        # CL_alpha_W = get_lift_curve_slope(cl_alpha_w, A_W, mach, sweep_50_W)
        # Wing lift-curve slope at zero lift obtained from Equation 8.22
        # CL_alpha_W_0 = get_lift_curve_slope(cl_alpha_w, A_W, mach, sweep_50_W)
        dihedral_W_rad = dihedral_W * math.pi / 180.0
        Clp_dihedral = 1 - (4 * z_w / b) * math.sin(dihedral_W_rad) + 12 * (z_w / b) ** 2 * (
            math.sin(dihedral_W_rad)) ** 2
        # Figure 10.36
        Clp_CDL_CLW = self.get_Clp_CDL_CLW(A_W, sweep_25_W)
        delta_Clp_drag = Clp_CDL_CLW * CL_W ** 2 - 0.125 * CD0_W
        # CL_alpha_W__CL_alpha_W_0 = CL_alpha_W / CL_alpha_W_0
        # We assumed that this quotient is equal to 1.0:
        CL_alpha_W__CL_alpha_W_0 = 1.0
        Cl_p_W = beta_Clp_k * (k / beta) * CL_alpha_W__CL_alpha_W_0 * Clp_dihedral + delta_Clp_drag

        outputs["data:handling_qualities:lateral:derivatives:wing:Cl:rollrate"] = Cl_p_W
