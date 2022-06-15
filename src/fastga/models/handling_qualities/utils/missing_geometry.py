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
import openmdao.api as om
import numpy as np


class MissingGeometry(om.ExplicitComponent):

    def setup(self):
        self.add_input("data:geometry:wing:sweep_0", val=np.nan, units="rad")
        self.add_input("data:geometry:wing:taper_ratio", val=np.nan)
        self.add_input("data:geometry:wing:aspect_ratio", val=np.nan)
        self.add_input("data:geometry:wing:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:leading_edge:x:local", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:at25percent:x", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:leading_edge:x:local", val=np.nan, units="m")

        self.add_input("data:geometry:horizontal_tail:sweep_0", val=np.nan, units="rad")
        self.add_input("data:geometry:horizontal_tail:taper_ratio", val=np.nan)
        self.add_input("data:geometry:horizontal_tail:aspect_ratio", val=np.nan)

        self.add_input("data:geometry:vertical_tail:sweep_0", val=np.nan, units="rad")
        self.add_input("data:geometry:vertical_tail:taper_ratio", val=np.nan)
        self.add_input("data:geometry:vertical_tail:aspect_ratio", val=np.nan)
        self.add_input("data:geometry:vertical_tail:root:chord", val=np.nan, units="m")

        self.add_input("data:geometry:fuselage:maximum_width", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:maximum_height", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:rear_length", val=np.nan, units="m")

        self.add_output("data:geometry:horizontal_tail:sweep_50", units="rad")
        self.add_output("data:geometry:vertical_tail:sweep_50", units="rad")
        self.add_output("data:geometry:wing:sweep_50", units="rad")

        self.add_output("data:geometry:wing:root:leading_edge:x:absolut", units="m")
        self.add_output("data:geometry:wing:tip:half_chord_point:x:absolut", units="m")

        self.add_output("data:geometry:wing:vertical_position", units="m")

        self.add_output("data:geometry:fuselage:depth_quarter_vt", units="m")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        sweep_0_W = inputs["data:geometry:wing:sweep_0"]
        sweep_0_H = inputs["data:geometry:horizontal_tail:sweep_0"]
        sweep_0_V = inputs["data:geometry:vertical_tail:sweep_0"]

        taper_ratio_W = inputs["data:geometry:wing:taper_ratio"]
        taper_ratio_H = inputs["data:geometry:horizontal_tail:taper_ratio"]
        taper_ratio_V = inputs["data:geometry:vertical_tail:taper_ratio"]

        A_W = inputs["data:geometry:wing:aspect_ratio"]
        A_H = inputs["data:geometry:horizontal_tail:aspect_ratio"]
        A_V = inputs["data:geometry:vertical_tail:aspect_ratio"]

        sweep_50_W = math.atan(math.tan(sweep_0_W) - 2 / A_W * ((1 - taper_ratio_W) / (1 + taper_ratio_W)))
        sweep_50_H = math.atan(math.tan(sweep_0_H) - 2 / A_H * ((1 - taper_ratio_H) / (1 + taper_ratio_H)))
        sweep_50_V = math.atan(math.tan(sweep_0_V) - 2 / A_V * ((1 - taper_ratio_V) / (1 + taper_ratio_H)))

        height_max = inputs["data:geometry:fuselage:maximum_height"]
        l2_wing = inputs["data:geometry:wing:root:chord"]
        wing_vertical_position = (height_max - 0.12 * l2_wing) * 0.5

        fa_length = inputs["data:geometry:wing:MAC:at25percent:x"]
        x0_wing = inputs["data:geometry:wing:MAC:leading_edge:x:local"]
        l0_wing = inputs["data:geometry:wing:MAC:length"]
        x_wing_root_absolut = fa_length - x0_wing - 0.25 * l0_wing
        x_wing_tip_local = inputs["data:geometry:wing:tip:leading_edge:x:local"]
        x_tip_wing_absolut = x_wing_root_absolut + x_wing_tip_local
        tip_chord = inputs["data:geometry:wing:tip:chord"]
        x_half_tip_wing_absolut = x_tip_wing_absolut + tip_chord / 2.0

        l_ar = inputs["data:geometry:fuselage:rear_length"]
        width_max = inputs["data:geometry:fuselage:maximum_width"]
        root_chord_vt = inputs["data:geometry:vertical_tail:root:chord"]
        avg_fus_depth = np.sqrt(width_max * height_max) * root_chord_vt / (2.0 * l_ar)

        outputs["data:geometry:wing:sweep_50"] = sweep_50_W
        outputs["data:geometry:horizontal_tail:sweep_50"] = sweep_50_H
        outputs["data:geometry:vertical_tail:sweep_50"] = sweep_50_V
        outputs["data:geometry:wing:vertical_position"] = wing_vertical_position
        outputs["data:geometry:wing:root:leading_edge:x:absolut"] = x_wing_root_absolut
        outputs["data:geometry:wing:tip:half_chord_point:x:absolut"] = x_half_tip_wing_absolut
        outputs["data:geometry:fuselage:depth_quarter_vt"] = avg_fus_depth
