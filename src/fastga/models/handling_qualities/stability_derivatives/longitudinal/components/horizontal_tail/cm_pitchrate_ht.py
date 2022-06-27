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

import openmdao.api as om
from scipy.interpolate import interp1d

from fastga.models.handling_qualities.utils.lift_curve_slope import get_lift_curve_slope


class CmPitchRateHT(om.ExplicitComponent):

    def setup(self):
        # Reference Flight Condition
        # self.add_input(weight)
        # self.add_input(dynamic_pressure)
        self.add_input("data:reference_flight_condition:mach", val=np.nan)

        # Wing Geometry
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")

        # Horizontal Tail Geometry
        self.add_input("data:geometry:horizontal_tail:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:horizontal_tail:aspect_ratio", val=np.nan)
        self.add_input("data:geometry:horizontal_tail:taper_ratio", val=np.nan)
        self.add_input("data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:sweep_0", val=np.nan, units="rad")
        self.add_input("data:geometry:horizontal_tail:sweep_50", val=np.nan, units="rad")

        # Horizontal Tail Aerodynamics
        self.add_input("data:aerodynamics:horizontal_tail:airfoil:CL_alpha", val=np.nan, units="rad**-1")
        self.add_input("data:aerodynamics:horizontal_tail:efficiency", val=np.nan)

        self.add_output("data:handling_qualities:longitudinal:derivatives:horizontal_tail:Cm:pitchrate",
                        units="rad**-1")


    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        # Reference Flight Conditions
        mach = inputs["data:reference_flight_condition:mach"]
        
        # Wing Geometry
        S_W = inputs["data:geometry:wing:area"]
        mac_W = inputs["data:geometry:wing:MAC:length"]
        
        # Horizontal Tail Geometry
        l_H = inputs["data:geometry:horizontal_tail:MAC:at25percent:x:from_wingMAC25"]
        S_H = inputs["data:geometry:horizontal_tail:area"]
        A_H = inputs["data:geometry:horizontal_tail:aspect_ratio"]
        taper_ratio_H = inputs["data:geometry:horizontal_tail:taper_ratio"]
        sweep_0_H = inputs["data:geometry:horizontal_tail:sweep_0"]
        sweep_50_H = inputs["data:geometry:horizontal_tail:sweep_50"]
        if sweep_50_H == np.nan and sweep_0_H != np.nan:
            sweep_50_H = sweep_0_H - 2 / A_H * ((1 - taper_ratio_H) / (1 + taper_ratio_H))

        # Horizontal tail volume coefficient: V_H = (xac_h - xg) * S_H/(S * mac)
        # For (xac_h - xg) is very often acceptable tu use the distance between the quarter MAC of the wing and the
        # quarter MAc of the horizontal tail
        V_H = l_H * S_H / (S_W * mac_W)
        X_H = l_H
        
        # Horizontal tail Aerodynamics
        cl_alpha_h = inputs["data:aerodynamics:horizontal_tail:airfoil:CL_alpha"]
        # Roskam presents a method of computing this ratio in his Airplane Design PART IV: Preliminary Calculation...
        # This method comes from DATCOM 1978.
        # The dynamic pressure ratio at the horizontal tail can usually be assumed in the range 0.9 - 1.0.
        # A value of .9 being on the conservative side in power-off flight.
        eta_H = inputs["data:aerodynamics:horizontal_tail:efficiency"]

        CL_alpha_H = get_lift_curve_slope(cl_alpha_h, A_H, mach, sweep_50_H)
        Cm_q_H = -2 * CL_alpha_H * eta_H * V_H * X_H / mac_W

        # NOTE: DATCOM's uses exposed surfaces for this computation. Shall we?

        outputs["data:handling_qualities:longitudinal:derivatives:horizontal_tail:Cm:pitchrate"] = Cm_q_H