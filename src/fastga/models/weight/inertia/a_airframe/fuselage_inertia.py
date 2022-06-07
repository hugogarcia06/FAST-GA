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

class ComputeFuselageInertia(ExplicitComponent):
    # TODOC
    """

    """

    def setup(self):
        self.add_input("data:weight:airframe:fuselage:mass", val=np.nan, units="lb")
        self.add_input("data:geometry:fuselage:wet_area", val=np.nan, units="inch**2")
        self.add_input("data:geometry:fuselage:maximum_height", val=np.nan, units="inch")
        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="inch")
        self.add_input("data:geometry:fuselage:length", val=np.nan, units="inch")

        self.add_input("data:weight:airframe:fuselage:CG:x", val=np.nan, units="inch")

        self.add_input("data:weight:airframe:fuselage:shell:mass", val=np.nan, units="lb")
        self.add_input("data:weight:airframe:fuselage:bulkhead:mass", val=np.nan, units="lb")

        self.add_output("data:weight:airframe:fuselage:inertia:Iox", units="lb*inch**2")
        self.add_output("data:weight:airframe:fuselage:inertia:Ioy", units="lb*inch**2")
        self.add_output("data:weight:airframe:fuselage:inertia:Ioz", units="lb*inch**2")
        
        self.add_output("data:weight:airframe:fuselage:inertia:Wx", units="lb*inch")
        self.add_output("data:weight:airframe:fuselage:inertia:Wy", units="lb*inch")
        self.add_output("data:weight:airframe:fuselage:inertia:Wz", units="lb*inch")

        self.add_output("data:weight:airframe:fuselage:inertia:Ix", units="lb*inch**2")
        self.add_output("data:weight:airframe:fuselage:inertia:Iy", units="lb*inch**2")
        self.add_output("data:weight:airframe:fuselage:inertia:Iz", units="lb*inch**2")
        self.add_output("data:weight:airframe:fuselage:inertia:Ixz", units="lb*inch**2")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        fus_weight = inputs["data:weight:airframe:fuselage:mass"]
        S_wet = inputs["data:geometry:fuselage:wet_area"]
        max_height = inputs["data:geometry:fuselage:maximum_height"]
        max_width = inputs["data:geometry:fuselage:maximum_width"]
        d = (max_height + max_width) / 2.0
        l_b = inputs["data:geometry:fuselage:length"]

        x_cg = inputs["data:weight:airframe:fuselage:CG:x"]

        # TODO: compute fuselage structure weight. Floor?
        # Length fuselage structure (Gudmonsson): forward bulkhead to aft frame
        # Fuselage structure weight includes: bulkhead, frames, stringers, skin
        # Shell weight includes skin, stringer and frames mas
        shell_weight = inputs["data:weight:airframe:fuselage:shell:mass"]
        bulkhead_weight = inputs["data:weight:airframe:fuselage:bulkhead:mass"]
        # floor_weight = inputs[]
        fus_struct_weight = shell_weight + bulkhead_weight


        ###### Fuselage Pitching Ioy ######
        # Figure digitization 8.1-24
        k_2 = FigureDigitization.get_k2(x_cg, l_b)
        Io_y = fus_weight * S_wet * k_2 / 37.68 * (3*d/(2*l_b) + l_b/d)

        ###### Fuselage Rolling Iox ######
        # Figure digitization 8.1-25
        k_3 = FigureDigitization.get_k3(d, fus_weight, fus_struct_weight)
        Io_x = fus_weight * k_3 / 4.0 * (S_wet / (math.pi * l_b)) ** 2


        ###### Fuselage Yawing Ioz ######
        Io_z = Io_y

        outputs["data:weight:airframe:fuselage:inertia:Iox"] = Io_x
        outputs["data:weight:airframe:fuselage:inertia:Ioy"] = Io_y
        outputs["data:weight:airframe:fuselage:inertia:Ioz"] = Io_z

        x = x_cg
        y = 0.0
        z = 0.0
        I_x = Io_x + fus_weight * (y ** 2 + z ** 2)
        I_y = Io_y + fus_weight * (x ** 2 + z ** 2)
        I_z = Io_z + fus_weight * (x ** 2 + y ** 2)
        I_xz = fus_weight * x * z
        
        outputs["data:weight:airframe:fuselage:inertia:Wx"] = fus_weight * x
        outputs["data:weight:airframe:fuselage:inertia:Wy"] = fus_weight * y
        outputs["data:weight:airframe:fuselage:inertia:Wz"] = fus_weight * z

        outputs["data:weight:airframe:fuselage:inertia:Ix"] = I_x
        outputs["data:weight:airframe:fuselage:inertia:Iy"] = I_y
        outputs["data:weight:airframe:fuselage:inertia:Iz"] = I_z
        outputs["data:weight:airframe:fuselage:inertia:Ixz"] = I_xz







