"""
    Estimation of horizontal tail sweeps and aspect ratio.
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

import math
import numpy as np

from openmdao.core.explicitcomponent import ExplicitComponent
from fastoad.module_management.service_registry import RegisterSubmodel

from ..constants import SUBMODEL_HT_SWEEP


@RegisterSubmodel(SUBMODEL_HT_SWEEP, "fastga.submodel.geometry.horizontal_tail.sweep.legacy")
class ComputeHTSweep(ExplicitComponent):
    # TODO: Document equations. Cite sources
    """Horizontal tail sweeps and aspect ratio estimation"""

    def setup(self):
        self.add_input("data:geometry:horizontal_tail:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:tip:chord", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:span", val=np.nan, units="m")
        self.add_input("data:geometry:horizontal_tail:sweep_25", val=np.nan, units="deg")

        self.add_output("data:geometry:horizontal_tail:sweep_0", units="deg")
        self.add_output("data:geometry:horizontal_tail:sweep_100", units="deg")

        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        b_h = inputs["data:geometry:horizontal_tail:span"]
        root_chord = inputs["data:geometry:horizontal_tail:root:chord"]
        tip_chord = inputs["data:geometry:horizontal_tail:tip:chord"]
        sweep_25 = inputs["data:geometry:horizontal_tail:sweep_25"]

        half_span = b_h / 2.0
        # TODO: The unit conversion can be handled by OpenMDAO
        sweep_0 = (
            (
                math.pi / 2
                - math.atan(
                    half_span
                    / (
                        0.25 * root_chord
                        - 0.25 * tip_chord
                        + half_span * math.tan(sweep_25 / 180.0 * math.pi)
                    )
                )
            )
            / math.pi
            * 180.0
        )
        sweep_100 = (
            (
                math.pi / 2
                - math.atan(
                    half_span
                    / (
                        half_span * math.tan(sweep_25 / 180.0 * math.pi)
                        - 0.75 * root_chord
                        + 0.75 * tip_chord
                    )
                )
            )
            / math.pi
            * 180.0
        )

        outputs["data:geometry:horizontal_tail:sweep_0"] = sweep_0
        outputs["data:geometry:horizontal_tail:sweep_100"] = sweep_100
