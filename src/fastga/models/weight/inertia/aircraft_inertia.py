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


class ComputeAircraftInertia(ExplicitComponent):
    # TODOC
    """

    """

    def setup(self):
        # Wing
        self.add_input("data:weight:airframe:wing:inertia:Wx", val=np.nan, units="lb*inch")
        self.add_input("data:weight:airframe:wing:inertia:Wy", val=np.nan, units="lb*inch")
        self.add_input("data:weight:airframe:wing:inertia:Wz", val=np.nan, units="lb*inch")

        self.add_input("data:weight:airframe:wing:inertia:Ix", val=np.nan, units="lb*inch**2")
        self.add_input("data:weight:airframe:wing:inertia:Iy", val=np.nan, units="lb*inch**2")
        self.add_input("data:weight:airframe:wing:inertia:Iz", val=np.nan, units="lb*inch**2")
        self.add_input("data:weight:airframe:wing:inertia:Ixz", val=np.nan, units="lb*inch**2")

        # Horizontal tail
        self.add_input("data:weight:airframe:horizontal_tail:inertia:Wx", val=np.nan, units="lb*inch")
        self.add_input("data:weight:airframe:horizontal_tail:inertia:Wy", val=np.nan, units="lb*inch")
        self.add_input("data:weight:airframe:horizontal_tail:inertia:Wz", val=np.nan, units="lb*inch")

        self.add_input("data:weight:airframe:horizontal_tail:inertia:Ix", val=np.nan, units="lb*inch**2")
        self.add_input("data:weight:airframe:horizontal_tail:inertia:Iy", val=np.nan, units="lb*inch**2")
        self.add_input("data:weight:airframe:horizontal_tail:inertia:Iz", val=np.nan, units="lb*inch**2")
        self.add_input("data:weight:airframe:horizontal_tail:inertia:Ixz", val=np.nan, units="lb*inch**2")

        # Vertical tail
        self.add_input("data:weight:airframe:vertical_tail:inertia:Wx", val=np.nan, units="lb*inch")
        self.add_input("data:weight:airframe:vertical_tail:inertia:Wy", val=np.nan, units="lb*inch")
        self.add_input("data:weight:airframe:vertical_tail:inertia:Wz", val=np.nan, units="lb*inch")

        self.add_input("data:weight:airframe:vertical_tail:inertia:Ix", val=np.nan, units="lb*inch**2")
        self.add_input("data:weight:airframe:vertical_tail:inertia:Iy", val=np.nan, units="lb*inch**2")
        self.add_input("data:weight:airframe:vertical_tail:inertia:Iz", val=np.nan, units="lb*inch**2")
        self.add_input("data:weight:airframe:vertical_tail:inertia:Ixz", val=np.nan, units="lb*inch**2")

        # Fuselage
        self.add_input("data:weight:airframe:fuselage:inertia:Wx", val=np.nan, units="lb*inch")
        self.add_input("data:weight:airframe:fuselage:inertia:Wy", val=np.nan, units="lb*inch")
        self.add_input("data:weight:airframe:fuselage:inertia:Wz", val=np.nan, units="lb*inch")

        self.add_input("data:weight:airframe:fuselage:inertia:Ix", val=np.nan, units="lb*inch**2")
        self.add_input("data:weight:airframe:fuselage:inertia:Iy", val=np.nan, units="lb*inch**2")
        self.add_input("data:weight:airframe:fuselage:inertia:Iz", val=np.nan, units="lb*inch**2")
        self.add_input("data:weight:airframe:fuselage:inertia:Ixz", val=np.nan, units="lb*inch**2")

        # Landing Gear
        self.add_input("data:weight:airframe:landing_gear:front:inertia:Wx", val=np.nan, units="lb*inch")
        self.add_input("data:weight:airframe:landing_gear:front:inertia:Wy", val=np.nan, units="lb*inch")
        self.add_input("data:weight:airframe:landing_gear:front:inertia:Wz", val=np.nan, units="lb*inch")

        self.add_input("data:weight:airframe:landing_gear:front:inertia:Ix", val=np.nan, units="lb*inch**2")
        self.add_input("data:weight:airframe:landing_gear:front:inertia:Iy", val=np.nan, units="lb*inch**2")
        self.add_input("data:weight:airframe:landing_gear:front:inertia:Iz", val=np.nan, units="lb*inch**2")
        self.add_input("data:weight:airframe:landing_gear:front:inertia:Ixz", val=np.nan, units="lb*inch**2")

        self.add_input("data:weight:airframe:landing_gear:main:inertia:Wx", val=np.nan, units="lb*inch")
        self.add_input("data:weight:airframe:landing_gear:main:inertia:Wy", val=np.nan, units="lb*inch")
        self.add_input("data:weight:airframe:landing_gear:main:inertia:Wz", val=np.nan, units="lb*inch")

        self.add_input("data:weight:airframe:landing_gear:main:inertia:Ix", val=np.nan, units="lb*inch**2")
        self.add_input("data:weight:airframe:landing_gear:main:inertia:Iy", val=np.nan, units="lb*inch**2")
        self.add_input("data:weight:airframe:landing_gear:main:inertia:Iz", val=np.nan, units="lb*inch**2")
        self.add_input("data:weight:airframe:landing_gear:main:inertia:Ixz", val=np.nan, units="lb*inch**2")

        # Power plant
        self.add_input("data:weight:propulsion:powerplant:inertia:Wx", val=np.nan, units="lb*inch")
        self.add_input("data:weight:propulsion:powerplant:inertia:Wy", val=np.nan, units="lb*inch")
        self.add_input("data:weight:propulsion:powerplant:inertia:Wz", val=np.nan, units="lb*inch")

        self.add_input("data:weight:propulsion:powerplant:inertia:Ix", val=np.nan, units="lb*inch**2")
        self.add_input("data:weight:propulsion:powerplant:inertia:Iy", val=np.nan, units="lb*inch**2")
        self.add_input("data:weight:propulsion:powerplant:inertia:Iz", val=np.nan, units="lb*inch**2")
        self.add_input("data:weight:propulsion:powerplant:inertia:Ixz", val=np.nan, units="lb*inch**2")

        # Fuel
        self.add_input("data:weight:propulsion:fuel:inertia:Wx", val=np.nan, units="lb*inch")
        self.add_input("data:weight:propulsion:fuel:inertia:Wy", val=np.nan, units="lb*inch")
        self.add_input("data:weight:propulsion:fuel:inertia:Wz", val=np.nan, units="lb*inch")

        self.add_input("data:weight:propulsion:fuel:inertia:Ix", val=np.nan, units="lb*inch**2")
        self.add_input("data:weight:propulsion:fuel:inertia:Iy", val=np.nan, units="lb*inch**2")
        self.add_input("data:weight:propulsion:fuel:inertia:Iz", val=np.nan, units="lb*inch**2")
        self.add_input("data:weight:propulsion:fuel:inertia:Ixz", val=np.nan, units="lb*inch**2")

        # OUTPUTS
        self.add_output("data:weight:aircraft:inertia:Ix", units="lb*inch**2")
        self.add_output("data:weight:aircraft:inertia:Iy", units="lb*inch**2")
        self.add_output("data:weight:aircraft:inertia:Iz", units="lb*inch**2")
        self.add_output("data:weight:aircraft:inertia:Ixz", units="lb*inch**2")

        self.add_output("data:weight:aircraft:inertia:Iox", units="lb*inch**2")
        self.add_output("data:weight:aircraft:inertia:Ioy", units="lb*inch**2")
        self.add_output("data:weight:aircraft:inertia:Ioz", units="lb*inch**2")
        self.add_output("data:weight:aircraft:inertia:Ioxz", units="lb*inch**2")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        ### STEP 1: COMPUTE TOTAL AIRPLANE INERTIAS Ix, Iy, Iz and Ixz ###
        # Wing
        Ix_wing = inputs["data:weight:airframe:wing:inertia:Ix"]
        Iy_wing = inputs["data:weight:airframe:wing:inertia:Iy"]
        Iz_wing = inputs["data:weight:airframe:wing:inertia:Iz"]
        Ixz_wing = inputs["data:weight:airframe:wing:inertia:Ixz"]

        # Horizontal Tail
        Ix_ht = inputs["data:weight:airframe:horizontal_tail:inertia:Ix"]
        Iy_ht = inputs["data:weight:airframe:horizontal_tail:inertia:Iy"]
        Iz_ht = inputs["data:weight:airframe:horizontal_tail:inertia:Iz"]
        Ixz_ht = inputs["data:weight:airframe:horizontal_tail:inertia:Ixz"]

        # Vertical Tail
        Ix_vt = inputs["data:weight:airframe:vertical_tail:inertia:Ix"]
        Iy_vt = inputs["data:weight:airframe:vertical_tail:inertia:Iy"]
        Iz_vt = inputs["data:weight:airframe:vertical_tail:inertia:Iz"]
        Ixz_vt = inputs["data:weight:airframe:vertical_tail:inertia:Ixz"]

        # Fuselage
        Ix_fus = inputs["data:weight:airframe:fuselage:inertia:Ix"]
        Iy_fus = inputs["data:weight:airframe:fuselage:inertia:Iy"]
        Iz_fus = inputs["data:weight:airframe:fuselage:inertia:Iz"]
        Ixz_fus = inputs["data:weight:airframe:fuselage:inertia:Ixz"]

        # Landing Gear
        Ix_lg_main = inputs["data:weight:airframe:landing_gear:main:inertia:Ix"]
        Iy_lg_main = inputs["data:weight:airframe:landing_gear:main:inertia:Iy"]
        Iz_lg_main = inputs["data:weight:airframe:landing_gear:main:inertia:Iz"]
        Ixz_lg_main = inputs["data:weight:airframe:landing_gear:main:inertia:Ixz"]

        Ix_lg_front = inputs["data:weight:airframe:landing_gear:front:inertia:Ix"]
        Iy_lg_front = inputs["data:weight:airframe:landing_gear:front:inertia:Iy"]
        Iz_lg_front = inputs["data:weight:airframe:landing_gear:front:inertia:Iz"]
        Ixz_lg_front = inputs["data:weight:airframe:landing_gear:front:inertia:Ixz"]

        # Power Plant
        Ix_pp = inputs["data:weight:propulsion:powerplant:inertia:Ix"]
        Iy_pp = inputs["data:weight:propulsion:powerplant:inertia:Iy"]
        Iz_pp = inputs["data:weight:propulsion:powerplant:inertia:Iz"]
        Ixz_pp = inputs["data:weight:propulsion:powerplant:inertia:Ixz"]

        # Fuel
        Ix_fuel = inputs["data:weight:variable_loads:fuel:inertia:Ix"]
        Iy_fuel = inputs["data:weight:variable_loads:fuel:inertia:Iy"]
        Iz_fuel = inputs["data:weight:variable_loads:fuel:inertia:Iz"]
        Ixz_fuel = inputs["data:weight:variable_loads:fuel:inertia:Ixz"]

        # Cargo
        # TODO: add cargo's inertias

        # Airplane Total Inertias
        Ix = Ix_wing + Ix_ht + Ix_vt + Ix_fus + Ix_lg_main + Ix_lg_front + Ix_pp + Ix_fuel
        Iy = Iy_wing + Iy_ht + Iy_vt + Iy_fus + Iy_lg_main + Iy_lg_front + Iy_pp + Iy_fuel
        Iz = Iz_wing + Iz_ht + Iz_vt + Iz_fus + Iz_lg_main + Iz_lg_front + Iz_pp + Iz_fuel
        Ixz = Ixz_wing + Ixz_ht + Ixz_vt + Ixz_fus + Ixz_lg_main + Ixz_lg_front + Ixz_pp + Ixz_fuel

        ### STEP 2: COMPUTE INERTIAS ABOUT THE AIRPLANE CENTROID
        # Wing
        wing_weight = inputs["data:weight:airframe:wing:mass"]
        W_x_wing = inputs["data:weight:airframe:wing:inertia:Wx"]
        W_y_wing = inputs["data:weight:airframe:wing:inertia:Wy"]
        W_z_wing = inputs["data:weight:airframe:wing:inertia:Wz"]

        # Horizontal Tail
        ht_weight = inputs["data:weight:airframe:horizontal_tail:mass"]
        W_x_ht = inputs["data:weight:airframe:horizontal_tail:inertia:Wx"]
        W_y_ht = inputs["data:weight:airframe:horizontal_tail:inertia:Wy"]
        W_z_ht = inputs["data:weight:airframe:horizontal_tail:inertia:Wz"]

        # Vertical Tail
        vt_weight = inputs["data:weight:airframe:vertical_tail:mass"]
        W_x_vt = inputs["data:weight:airframe:vertical_tail:inertia:Wx"]
        W_y_vt = inputs["data:weight:airframe:vertical_tail:inertia:Wy"]
        W_z_vt = inputs["data:weight:airframe:vertical_tail:inertia:Wz"]

        # Fuselage
        fus_weight = inputs["data:weight:airframe:fuselage:mass"]
        W_x_fus = inputs["data:weight:airframe:fuselage:inertia:Wx"]
        W_y_fus = inputs["data:weight:airframe:fuselage:inertia:Wy"]
        W_z_fus = inputs["data:weight:airframe:fuselage:inertia:Wz"]

        # Landing Gear
        lg_main_weight = inputs["data:weight:airframe:landing_gear:main:mass"]
        W_x_lg_main = inputs["data:weight:airframe:landing_gear:main:inertia:Wx"]
        W_y_lg_main = inputs["data:weight:airframe:landing_gear:main:inertia:Wx"]
        W_z_lg_main = inputs["data:weight:airframe:landing_gear:main:inertia:Wx"]

        lg_front_weight = inputs["data:weight:airframe:landing_gear:front:mass"]
        W_x_lg_front = inputs["data:weight:airframe:landing_gear:front:inertia:Wx"]
        W_y_lg_front = inputs["data:weight:airframe:landing_gear:front:inertia:Wx"]
        W_z_lg_front = inputs["data:weight:airframe:landing_gear:front:inertia:Wx"]

        # Power Plant
        pp_weight = inputs["data:weight:propulsion:mass"]
        W_x_pp = inputs["data:weight:propulsion:powerplant:inertia:Wx"]
        W_y_pp = inputs["data:weight:propulsion:powerplant:inertia:Wx"]
        W_z_pp = inputs["data:weight:propulsion:powerplant:inertia:Wx"]

        # Fuel
        fuel_weight = inputs["data:reference_flight_condition:fuel_weight"]
        W_x_fuel = inputs["data:weight:propulsion:fuel:inertia:Wx"]
        W_y_fuel = inputs["data:weight:propulsion:fuel:inertia:Wx"]
        W_z_fuel = inputs["data:weight:propulsion:fuel:inertia:Wx"]

        W_x_2 = (W_x_wing ** 2 + W_x_ht ** 2 + W_x_vt ** 2 +
                 W_x_fus ** 2 + W_x_lg_front ** 2 + W_x_lg_main ** 2 +
                 W_x_pp ** 2 + W_x_fuel ** 2
                 )
        W_y_2 = (W_y_wing ** 2 + W_y_ht ** 2 + W_y_vt ** 2 +
                 W_y_fus ** 2 + W_y_lg_front ** 2 + W_y_lg_main ** 2 +
                 W_y_pp ** 2 + W_y_fuel ** 2
                 )
        W_z_2 = (W_z_wing ** 2 + W_z_ht ** 2 + W_z_vt ** 2 +
                 W_z_fus ** 2 + W_z_lg_front ** 2 + W_z_lg_main ** 2 +
                 W_z_pp ** 2 + W_z_fuel ** 2
                 )
        W = (wing_weight + ht_weight + vt_weight +
             fus_weight + lg_front_weight + lg_main_weight +
             pp_weight + fuel_weight
             )

        Iox = Ix - (W_y_2 + W_z_2) / W
        Ioy = Iy - (W_x_2 + W_z_2) / W
        Ioz = Iz - (W_x_2 + W_y_2) / W
        # TODO: moment of inertia Ixz?

        outputs["data:weight:aircraft:inertia:Ix"] = Ix
        outputs["data:weight:aircraft:inertia:Iy"] = Iy
        outputs["data:weight:aircraft:inertia:Iz"] = Iz
        outputs["data:weight:aircraft:inertia:Ixz"] = Ixz

        outputs["data:weight:aircraft:inertia:Iox"] = Iox
        outputs["data:weight:aircraft:inertia:Ioy"] = Ioy
        outputs["data:weight:aircraft:inertia:Ioz"] = Ioz
        # outputs["data:weight:aircraft:inertia:Ioxz"] = Ioxz

