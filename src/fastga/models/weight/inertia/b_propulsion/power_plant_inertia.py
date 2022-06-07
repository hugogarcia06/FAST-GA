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

from openmdao.core.explicitcomponent import ExplicitComponent


class ComputePowerPlantInertia(ExplicitComponent):
    # TODOC
    """

    """

    def setup(self):
        self.add_input("data:weight:propulsion:mass", val=np.nan, units="lb")
        self.add_input("data:geometry:propulsion:nacelle:length", val=np.nan, units="inch")
        self.add_input("data:geometry:propulsion:nacelle:height", val=np.nan, units="inch")
        self.add_input("data:geometry:propulsion:nacelle:width", val=np.nan, units="inch")
        self.add_input("data:weight:propulsion:engine:mass", val=np.nan, units="lb")
        self.add_input("data:geometry:propulsion:engine:layout", val=np.nan)

        self.add_input("data:weight:propulsion:CG:x", val=np.nan, units="inch")

        self.add_output("data:weight:propulsion:powerplant:inertia:Iox", units="lb*inch**2")
        self.add_output("data:weight:propulsion:powerplant:inertia:Ioy", units="lb*inch**2")
        self.add_output("data:weight:propulsion:powerplant:inertia:Ioz", units="lb*inch**2")

        self.add_output("data:weight:propulsion:powerplant:inertia:Wx", units="lb*inch")
        self.add_output("data:weight:propulsion:powerplant:inertia:Wy", units="lb*inch")
        self.add_output("data:weight:propulsion:powerplant:inertia:Wz", units="lb*inch")

        self.add_output("data:weight:propulsion:powerplant:inertia:Ix", units="lb*inch**2")
        self.add_output("data:weight:propulsion:powerplant:inertia:Iy", units="lb*inch**2")
        self.add_output("data:weight:propulsion:powerplant:inertia:Iz", units="lb*inch**2")
        self.add_output("data:weight:propulsion:powerplant:inertia:Ixz", units="lb*inch**2")


    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        w_p = inputs["data:weight:propulsion:mass"]
        l_p = inputs["data:geometry:propulsion:nacelle:length"]
        nacelle_height = inputs["data:geometry:propulsion:nacelle:height"]
        nacelle_width = inputs["data:geometry:propulsion:nacelle:width"]
        nacelle_diameter = (nacelle_height + nacelle_width) / 2.0
        eng_layout = inputs["data:geometry:propulsion:engine:layout"]

        x_cg = inputs["data:weight:propulsion:CG:x"]

        w_e = inputs["data:weight:propulsion:engine:mass"]
        d_e = 0.8 * nacelle_diameter
        # TODO: obtain engine length (different from nacelle length)
        # Estimation of engine length with respect to nacelle length.
        l_e = 0.6 * l_p
        # l_e = inputs[]

        ###### Fuselage Pitching Ioy ######
        Io_y = 0.061 * (0.75 * w_p * d_e ** 2 + w_e * l_e ** 2 + (w_p - w_e) * l_p ** 2)

        ###### Fuselage Rolling Iox ######
        Io_x = 0.083 * w_p * d_e ** 2

        ###### Fuselage Yawing Ioz ######
        Io_z = Io_y

        outputs["data:weight:propulsion:powerplant:inertia:Iox"] = Io_x
        outputs["data:weight:propulsion:powerplant:inertia:Ioy"] = Io_y
        outputs["data:weight:propulsion:powerplant:inertia:Ioz"] = Io_z
        
        x = x_cg
        y = 0.0
        if eng_layout == 2 or eng_layout == 3:   # engine at the rear fuselage or at the nose
            z = 0.0
        else:   # engine under the wings
            z = nacelle_height / 2.0
        I_x = Io_x + w_p * (y ** 2 + z ** 2)
        I_y = Io_y + w_p * (x ** 2 + z ** 2)
        I_z = Io_z + w_p * (x ** 2 + y ** 2)
        I_xz = w_p * x * z

        outputs["data:weight:propulsion:powerplant:inertia:Wx"] = w_p * x
        outputs["data:weight:propulsion:powerplant:inertia:Wy"] = w_p * y
        outputs["data:weight:propulsion:powerplant:inertia:Wz"] = w_p * z

        outputs["data:weight:airframe:powerplant:inertia:Ix"] = I_x
        outputs["data:weight:airframe:powerplant:inertia:Iy"] = I_y
        outputs["data:weight:airframe:powerplant:inertia:Iz"] = I_z
        outputs["data:weight:airframe:powerplant:inertia:Ixz"] = I_xz
