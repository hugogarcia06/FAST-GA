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


class ComputeLandingGearInertia(ExplicitComponent):
    # TODOC
    """
    Estimation of the landing gear inertia considering it as a point mass.
    """

    def setup(self):
        self.add_input("data:weight:airframe:landing_gear:main:mass", val=np.nan, units="lb")
        self.add_input("data:weight:airframe:landing_gear:front:mass", val=np.nan, units="lb")
        self.add_input("data:weight:airframe:landing_gear:main:CG:x", val=np.nan, units="inch")
        self.add_input("data:weight:airframe:landing_gear:main:CG:x", val=np.nan, units="inch")
        self.add_input("data:geometry:landing_gear:height", val=np.nan, units="inch")

        self.add_output("data:weight:airframe:landing_gear:front:inertia:Wx", units="lb*inch")
        self.add_output("data:weight:airframe:landing_gear:front:inertia:Wy", units="lb*inch")
        self.add_output("data:weight:airframe:landing_gear:front:inertia:Wz", units="lb*inch")

        self.add_output("data:weight:airframe:landing_gear:front:inertia:Ix", units="lb*inch**2")
        self.add_output("data:weight:airframe:landing_gear:front:inertia:Iy", units="lb*inch**2")
        self.add_output("data:weight:airframe:landing_gear:front:inertia:Iz", units="lb*inch**2")
        self.add_output("data:weight:airframe:landing_gear:front:inertia:Ixz", units="lb*inch**2")

        self.add_output("data:weight:airframe:landing_gear:main:inertia:Wx", units="lb*inch")
        self.add_output("data:weight:airframe:landing_gear:main:inertia:Wy", units="lb*inch")
        self.add_output("data:weight:airframe:landing_gear:main:inertia:Wz", units="lb*inch")

        self.add_output("data:weight:airframe:landing_gear:main:inertia:Ix", units="lb*inch**2")
        self.add_output("data:weight:airframe:landing_gear:main:inertia:Iy", units="lb*inch**2")
        self.add_output("data:weight:airframe:landing_gear:main:inertia:Iz", units="lb*inch**2")
        self.add_output("data:weight:airframe:landing_gear:main:inertia:Ixz", units="lb*inch**2")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        ### Main Landing Gear ###
        main_lg_weight = inputs["data:weight:airframe:landing_gear:main:mass"]
        x_cg_main = inputs["data:weight:airframe:landing_gear:main:CG:x"]
        x = x_cg_main
        y = 0.0
        z = inputs["data:geometry:landing_gear:height"] / 2.0
        I_x_main = main_lg_weight * (y ** 2 + z ** 2)
        I_y_main = main_lg_weight * (x ** 2 + z ** 2)
        I_z_main = main_lg_weight * (x ** 2 + y ** 2)
        I_xz_main = main_lg_weight * x * z
        
        outputs["data:weight:airframe:landing_gear:main:inertia:Wx"] = main_lg_weight * x
        outputs["data:weight:airframe:landing_gear:main:inertia:Wy"] = main_lg_weight * y
        outputs["data:weight:airframe:landing_gear:main:inertia:Wz"] = main_lg_weight * z

        outputs["data:weight:airframe:landing_gear:inertia:main:Ix"] = I_x_main
        outputs["data:weight:airframe:landing_gear:inertia:main:Iy"] = I_y_main
        outputs["data:weight:airframe:landing_gear:inertia:main:Iz"] = I_z_main
        outputs["data:weight:airframe:landing_gear:inertia:main:Ixz"] = I_xz_main

        ### Front Landing Gear ###
        front_lg_weight = inputs["data:weight:airframe:landing_gear:front:mass"]
        x_cg_front = inputs["data:weight:airframe:landing_gear:front:CG:x"]
        x = x_cg_front
        y = 0.0
        z = inputs["data:geometry:landing_gear:height"] / 2.0
        I_x_front = front_lg_weight * (y ** 2 + z ** 2)
        I_y_front = front_lg_weight * (x ** 2 + z ** 2)
        I_z_front = front_lg_weight * (x ** 2 + y ** 2)
        I_xz_front = front_lg_weight * x * z
        
        outputs["data:weight:airframe:landing_gear:front:inertia:Wx"] = front_lg_weight * x
        outputs["data:weight:airframe:landing_gear:front:inertia:Wy"] = front_lg_weight * y
        outputs["data:weight:airframe:landing_gear:front:inertia:Wz"] = front_lg_weight * z

        outputs["data:weight:airframe:landing_gear:inertia:front:Ix"] = I_x_front
        outputs["data:weight:airframe:landing_gear:inertia:front:Iy"] = I_y_front
        outputs["data:weight:airframe:landing_gear:inertia:front:Iz"] = I_z_front
        outputs["data:weight:airframe:landing_gear:inertia:front:Ixz"] = I_xz_front
