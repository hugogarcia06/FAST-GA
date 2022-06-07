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

from openmdao.core.explicitcomponent import ExplicitComponent
from fastga.models.weight.utils.figure_digitization import FigureDigitization


class ComputeHorizontalTailInertia(ExplicitComponent):
    # TODOC
    """

    """

    def setup(self):
        self.add_input("data:weight:airframe:horizontal_tail:mass", val=np.nan, units="lb")
        self.add_input("data:geometry:horizontal_tail:root:chord", val=np.nan, units="inch")
        self.add_input("data:geometry:horizontal_tail:tip:chord", val=np.nan, units="inch")
        self.add_input("data:geometry:horizontal_tail:span", val=np.nan, units="inch")
        self.add_input("data:geometry:horizontal_tail:sweep_0", val=np.nan, units="rad")
        self.add_input("data:geometry:horizontal_tail:z:from_wingMAC25", val=np.nan, units="inch")

        self.add_input("data:weight:airframe:horizontal_tail:CG:x", val=np.nan, units="inch")
        self.add_input("data:weight:airframe:half_horizontal_tail:CG:y", val=np.nan, units="inch")

        self.add_output("data:weight:airframe:horizontal_tail:inertia:Iox", units="lb*inch**2")
        self.add_output("data:weight:airframe:horizontal_tail:inertia:Ioy", units="lb*inch**2")
        self.add_output("data:weight:airframe:horizontal_tail:inertia:Ioz", units="lb*inch**2")

        self.add_output("data:weight:airframe:horizontal_tail:inertia:Wx", units="lb*inch")
        self.add_output("data:weight:airframe:horizontal_tail:inertia:Wy", units="lb*inch")
        self.add_output("data:weight:airframe:horizontal_tail:inertia:Wz", units="lb*inch")

        self.add_output("data:weight:airframe:horizontal_tail:inertia:Ix", units="lb*inch**2")
        self.add_output("data:weight:airframe:horizontal_tail:inertia:Iy", units="lb*inch**2")
        self.add_output("data:weight:airframe:horizontal_tail:inertia:Iz", units="lb*inch**2")
        self.add_output("data:weight:airframe:horizontal_tail:inertia:Ixz", units="lb*inch**2")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        ht_weight = inputs["data:weight:airframe:horizontal_tail:mass"]
        cr = inputs["data:geometry:horizontal_tail:root:chord"]
        ct = inputs["data:geometry:horizontal_tail:tip:chord"]
        span = inputs["data:geometry:horizontal_tail:span"]
        sweep_0 = inputs["data:geometry:horizontal_tail:sweep_0"]

        x_cg = inputs["data:weight:airframe:horizontal_tail:CG:x"]
        y_cg = inputs["data:weight:airframe:half_horizontal_tail:CG:y"]

        values = [cr, span * math.tan(sweep_0) / 2, ct + span * math.tan(sweep_0) / 2]
        values.sort()
        ca = values[0]
        cb = values[1]
        cc = values[2]

        # Ratio of weight to chord for wing shapes
        ro = 2 * ht_weight / (- ca + cb + cc)

        ###### Horizontal Tail Pitching Ioy ######
        k_o = 0.771
        weight_x = ro / 6 * (-ca ** 2 + cb ** 2 + cc * cb + cc ** 2)
        I = ro / 12 * (-ca ** 3 + cb ** 3 + cc ** 2 * cb + cc * cb ** 2 + cc ** 3)
        Io_y = k_o * (I - weight_x ** 2 / ht_weight)

        ###### Horizontal Tail Rolling Iox ######
        # Figure digitization 8.1-26 (DATCOM)
        k_4 = FigureDigitization.get_k4(y_cg, span, cr, ct)
        Io_x = ht_weight * span ** 2 * k_4 / 24.0 * ((cr + 3 * ct) / (cr + ct))

        ###### Horizontal Tail Yawing Ioz ######
        Io_z = Io_x + Io_y

        outputs["data:weight:airframe:horizontal_tail:inertia:Iox"] = Io_x
        outputs["data:weight:airframe:horizontal_tail:inertia:Ioy"] = Io_y
        outputs["data:weight:airframe:horizontal_tail:inertia:Ioz"] = Io_z

        # x, y and z are the position of the ht cg with respect to the axis where everything is calculated.
        x = x_cg
        y = 0.0
        z = inputs["data:geometry:horizontal_tail:z:from_wingMAC25"]
        I_x = Io_x + ht_weight * (y ** 2 + z ** 2)
        I_y = Io_y + ht_weight * (x ** 2 + z ** 2)
        I_z = Io_z + ht_weight * (x ** 2 + y ** 2)
        I_xz = ht_weight * x * z

        outputs["data:weight:airframe:horizontal_tail:inertia:Wx"] = ht_weight * x
        outputs["data:weight:airframe:horizontal_tail:inertia:Wy"] = ht_weight * y
        outputs["data:weight:airframe:horizontal_tail:inertia:Wz"] = ht_weight * z

        outputs["data:weight:airframe:horizontal_tail:inertia:Ix"] = I_x
        outputs["data:weight:airframe:horizontal_tail:inertia:Iy"] = I_y
        outputs["data:weight:airframe:horizontal_tail:inertia:Iz"] = I_z
        outputs["data:weight:airframe:horizontal_tail:inertia:Ixz"] = I_xz
