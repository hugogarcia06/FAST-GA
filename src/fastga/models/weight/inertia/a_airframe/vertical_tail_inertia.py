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


class ComputeVerticalTailInertia(ExplicitComponent):
    # TODOC
    """

    """

    def setup(self):
        self.add_input("data:weight:airframe:vertical_tail:mass", val=np.nan, units="lb")
        self.add_input("data:geometry:vertical_tail:root:chord", val=np.nan, units="inch")
        self.add_input("data:geometry:vertical_tail:tip:chord", val=np.nan, units="inch")
        self.add_input("data:geometry:vertical_tail:span", val=np.nan, units="inch")
        self.add_input("data:geometry:vertical_tail:sweep_0", val=np.nan, units="rad")

        self.add_input("data:geometry:fuselage:maximum_height", val=np.nan, units="inch")

        self.add_input("data:weight:airframe:vertical_tail:CG:x", val=np.nan, units="inch")
        self.add_input("data:weight:airframe:vertical_tail:CG:z", val=np.nan, units="inch")

        self.add_output("data:weight:airframe:vertical_tail:inertia:Iox", units="lb*inch**2")
        self.add_output("data:weight:airframe:vertical_tail:inertia:Ioy", units="lb*inch**2")
        self.add_output("data:weight:airframe:vertical_tail:inertia:Ioz", units="lb*inch**2")

        self.add_output("data:weight:airframe:vertical_tail:inertia:Wx", units="lb*inch")
        self.add_output("data:weight:airframe:vertical_tail:inertia:Wy", units="lb*inch")
        self.add_output("data:weight:airframe:vertical_tail:inertia:Wz", units="lb*inch")

        self.add_output("data:weight:airframe:vertical_tail:inertia:Ix", units="lb*inch**2")
        self.add_output("data:weight:airframe:vertical_tail:inertia:Iy", units="lb*inch**2")
        self.add_output("data:weight:airframe:vertical_tail:inertia:Iz", units="lb*inch**2")
        self.add_output("data:weight:airframe:vertical_tail:inertia:Ixz", units="lb*inch**2")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        vt_weight = inputs["data:weight:airframe:vertical_tail:mass"]
        cr = inputs["data:geometry:vertical_tail:root:chord"]
        ct = inputs["data:geometry:vertical_tail:tip:chord"]
        span = inputs["data:geometry:vertical_tail:span"]
        sweep_0 = inputs["data:geometry:vertical_tail:sweep_0"]
        fus_height = inputs["data:geometry:fuselage:maximum_height"]

        x_cg = inputs["data:weight:airframe:vertical_tail:CG:x"]
        z_cg = inputs["data:weight:airframe:vertical_tail:CG:z"]

        ###### Vertical Tail Rolling Iox ######
        # Figure digitization 8.1-27 (DATCOM)
        k_5 = FigureDigitization.get_k5(z_cg, span, cr, ct)
        Io_x = vt_weight * span ** 2 * k_5 / 18.0 * (1 + (2 * cr * ct) / (cr + ct) ** 2)

        ###### Vertical Tail Yawing Ioz ######
        values = [cr, 2 * span * math.tan(sweep_0) / 2, ct + 2 * span * math.tan(sweep_0) / 2]
        values.sort()
        ca = values[0]
        cb = values[1]
        cc = values[2]

        # Ratio of weight to chord for wing shapes
        ro = 2 * vt_weight / (- ca + cb + cc)

        k_o = 0.771
        weight_x = ro / 6 * (-ca ** 2 + cb ** 2 + cc * cb + cc ** 2)
        I = ro / 12 * (-ca ** 3 + cb ** 3 + cc ** 2 * cb + cc * cb ** 2 + cc ** 3)
        Io_z = k_o * (I - weight_x ** 2 / vt_weight)

        ###### Vertical Tail Pitching Ioy ######
        Io_y = Io_x + Io_z

        outputs["data:weight:airframe:vertical_tail:inertia:Iox"] = Io_x
        outputs["data:weight:airframe:vertical_tail:inertia:Ioy"] = Io_y
        outputs["data:weight:airframe:vertical_tail:inertia:Ioz"] = Io_z

        # x, y and z are the position of the vt cg with respect to the axis where everything is calculated.
        x = x_cg
        y = 0.0
        z = z_cg + fus_height / 2.0
        I_x = Io_x + vt_weight * (y ** 2 + z ** 2)
        I_y = Io_y + vt_weight * (x ** 2 + z ** 2)
        I_z = Io_z + vt_weight * (x ** 2 + y ** 2)
        I_xz = vt_weight * x * z
        
        outputs["data:weight:airframe:vertical_tail:inertia:Wx"] = vt_weight * x
        outputs["data:weight:airframe:vertical_tail:inertia:Wy"] = vt_weight * y
        outputs["data:weight:airframe:vertical_tail:inertia:Wz"] = vt_weight * z

        outputs["data:weight:airframe:vertical_tail:inertia:Ix"] = I_x
        outputs["data:weight:airframe:vertical_tail:inertia:Iy"] = I_y
        outputs["data:weight:airframe:vertical_tail:inertia:Iz"] = I_z
        outputs["data:weight:airframe:vertical_tail:inertia:Ixz"] = I_xz
