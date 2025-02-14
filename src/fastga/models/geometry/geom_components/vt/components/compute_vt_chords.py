"""
    Estimation of vertical tail chords and span
"""
#  This file is part of FAST : A framework for rapid Overall Aircraft Design
#  Copyright (C) 2020  ONERA & ISAE-SUPAERO
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
from fastoad.module_management.service_registry import RegisterSubmodel

from ..constants import SUBMODEL_VT_CHORD


@RegisterSubmodel(SUBMODEL_VT_CHORD, "fastga.submodel.geometry.vertical_tail.chord.legacy")
class ComputeVTChords(ExplicitComponent):
    # TODO: Document equations. Cite sources
    """Vertical tail chords and span estimation"""

    def setup(self):
        self.add_input("data:geometry:vertical_tail:aspect_ratio", val=np.nan)
        self.add_input("data:geometry:vertical_tail:area", val=np.nan, units="m**2")
        self.add_input("data:geometry:vertical_tail:taper_ratio", val=np.nan)

        self.add_output("data:geometry:vertical_tail:span", units="m")
        self.add_output("data:geometry:vertical_tail:root:chord", units="m")
        self.add_output("data:geometry:vertical_tail:tip:chord", units="m")

        self.declare_partials(
            "data:geometry:vertical_tail:span",
            ["data:geometry:vertical_tail:aspect_ratio", "data:geometry:vertical_tail:area"],
            method="fd",
        )
        self.declare_partials("data:geometry:vertical_tail:root:chord", "*", method="fd")
        self.declare_partials("data:geometry:vertical_tail:tip:chord", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        lambda_vt = float(inputs["data:geometry:vertical_tail:aspect_ratio"])
        s_v = float(inputs["data:geometry:vertical_tail:area"])
        taper_v = inputs["data:geometry:vertical_tail:taper_ratio"]

        b_v = np.sqrt(max(lambda_vt * s_v, 0.1))  # !!!: to avoid 0 division if s_h initialised to 0
        root_chord = s_v * 2 / (1 + taper_v) / b_v
        tip_chord = root_chord * taper_v

        outputs["data:geometry:vertical_tail:span"] = b_v
        outputs["data:geometry:vertical_tail:root:chord"] = root_chord
        outputs["data:geometry:vertical_tail:tip:chord"] = tip_chord
