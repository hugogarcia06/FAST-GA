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


class ComputeFuelInertia(ExplicitComponent):
    # TODOC
    """
    Estimation of the fuel inertia as a rectangular flat plate as done in USAF DATCOM section 8.1 sample problem.
    """

    def setup(self):
        self.add_input("data:reference_flight_condition:fuel_weight", val=np.nan, units="lb")
        self.add_input("data:weight:airframe:wing:CG:x", val=np.nan, units="inch")
        self.add_input("data:weight:propulsion:tank:CG:x", val=np.nan, units="inch")

        self.add_input("data:geometry:wing:root:chord", val=np.nan, units="inch")
        self.add_input("data:geometry:wing:root:thickness_ratio", val=np.nan)
        self.add_input("data:geometry:wing:tip:chord", val=np.nan, units="inch")
        self.add_input("data:geometry:wing:tip:thickness_ratio", val=np.nan)
        self.add_input("data:geometry:wing:span", val=np.nan, units="inch")

        self.add_output("data:weight:propulsion:fuel:inertia:Iox", units="lb*inch**2")
        self.add_output("data:weight:propulsion:fuel:inertia:Ioy", units="lb*inch**2")
        self.add_output("data:weight:propulsion:fuel:inertia:Ioz", units="lb*inch**2")

        self.add_output("data:weight:propulsion:fuel:inertia:Wx", units="lb*inch")
        self.add_output("data:weight:propulsion:fuel:inertia:Wy", units="lb*inch")
        self.add_output("data:weight:propulsion:fuel:inertia:Wz", units="lb*inch")

        self.add_output("data:weight:propulsion:fuel:inertia:Ix", units="lb*inch**2")
        self.add_output("data:weight:propulsion:fuel:inertia:Iy", units="lb*inch**2")
        self.add_output("data:weight:propulsion:fuel:inertia:Iz", units="lb*inch**2")
        self.add_output("data:weight:propulsion:fuel:inertia:Ixz", units="lb*inch**2")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        fuel_weight = inputs["data:reference_flight_condition:fuel_weight"]
        # x_cg = inputs["data:weight:airframe:wing:CG:x"]
        x_cg = inputs["data:weight:propulsion:tank:CG:x"]

        cr = inputs["data:geometry:wing:root:chord"]
        t_c_root = inputs["data:geometry:wing:root:thickness_ratio"]
        t_root = t_c_root * cr   # thickness at the root
        ct = inputs["data:geometry:wing:tip:chord"]
        t_c_tip = inputs["data:geometry:wing:tip:thickness_ratio"]
        t_tip = t_c_tip * ct   # thickness at the tip
        t = (t_root + t_tip) / 2.0   # mean thickness of the wing
        c = (cr + ct)/2.0
        b = inputs["data:geometry:wing:span"]

        ###### Fuel Pitching Ioy ######
        Io_y = fuel_weight / 12.0 * (c**2 + t**2)

        ###### Fuel Rolling Iox ######
        Io_x = fuel_weight / 12.0 * (b**2 + t**2)

        ###### Fuel Yawing Ioz ######
        Io_z = fuel_weight / 12.0 * (c**2 + b**2)

        outputs["data:weight:propulsion:fuel:inertia:Iox"] = Io_x
        outputs["data:weight:propulsion:fuel:inertia:Ioy"] = Io_y
        outputs["data:weight:propulsion:fuel:inertia:Ioz"] = Io_z

        x = x_cg
        y = 0.0
        z = 0.0
        I_x = Io_x + fuel_weight * (y ** 2 + z ** 2)
        I_y = Io_y + fuel_weight * (x ** 2 + z ** 2)
        I_z = Io_z + fuel_weight * (x ** 2 + y ** 2)
        I_xz = fuel_weight * x * z

        outputs["data:weight:propulsion:fuel:inertia:Wx"] = fuel_weight * x
        outputs["data:weight:propulsion:fuel:inertia:Wy"] = fuel_weight * y
        outputs["data:weight:propulsion:fuel:inertia:Wz"] = fuel_weight * z

        outputs["data:weight:propulsion:fuel:inertia:Ix"] = I_x
        outputs["data:weight:propulsion:fuel:inertia:Iy"] = I_y
        outputs["data:weight:propulsion:fuel:inertia:Iz"] = I_z
        outputs["data:weight:propulsion:fuel:inertia:Ixz"] = I_xz
