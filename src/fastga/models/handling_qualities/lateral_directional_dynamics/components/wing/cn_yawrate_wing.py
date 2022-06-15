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

from fastga.models.handling_qualities.utils.figure_digitization import FigureDigitization2


# TODO: register class
class CnYawRateWing(FigureDigitization2):

    def setup(self):
        # Flight reference condition
        self.add_input("data:reference_flight_condition:CL", val=np.nan)
        self.add_input("data:reference_flight_condition:CD0", val=np.nan)

        # Wing Geometry
        self.add_input("data:geometry:wing:aspect_ratio", val=np.nan)
        self.add_input("data:geometry:wing:taper_ratio", val=np.nan)
        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:wing:sweep_25", val=np.nan, units="rad")

        self.add_input("data:handling_qualities:stick_fixed_static_margin", val=np.nan)

        self.add_output("data:handling_qualities:lateral:derivatives:wing:Cn:yawrate", units="rad**-1")


    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        # Flight reference condition
        CL_s = inputs["data:reference_flight_condition:CL"]
        CL_W = CL_s
        CD0_W = inputs["data:reference_flight_condition:CD0"]

        # Wing geometry
        A_W = inputs["data:geometry:wing:aspect_ratio"]
        taper_ratio_W = inputs["data:geometry:wing:taper_ratio"]
        mac_W = inputs["data:geometry:wing:MAC:length"]
        sweep_25_W = inputs["data:geometry:wing:sweep_25"]

        static_margin = inputs["data:handling_qualities:stick_fixed_static_margin"]

        x = static_margin * mac_W

        # Wing contribution: Equation 10.87 Roskam Airplane Design Part VI
        # Wing yaw damping derivative: lift effect. Figure digitization of Figure 10.44
        Cnr_CL2 = self.get_Cnr_CL2(A_W, sweep_25_W, taper_ratio_W, x, mac_W)
        # Drag effect. Figure digitization of Figure 10.45
        Cnr_CD0 = self.get_Cnr_CD0(A_W, sweep_25_W, x, mac_W)
        Cn_r_W = Cnr_CL2 * CL_W**2 + Cnr_CD0 * CD0_W

        outputs["data:handling_qualities:lateral:derivatives:wing:Cn:yawrate"] = Cn_r_W
