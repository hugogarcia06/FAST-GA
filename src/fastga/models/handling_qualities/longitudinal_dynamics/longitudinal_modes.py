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

import openmdao.api as om
import numpy as np


class LongitudinalModes(om.ExplicitComponent):

    def setup(self):
        self.add_input("data:handling_qualities:longitudinal:spacestate:matrixA", val=np.nan, units="rad**-1")

        self.add_output("data:handling_qualities:longitudinal:spacestate:eigenvalues")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        A = inputs["data:handling_qualities:longitudinal:spacestate:matrixA"]

        w, v = np.linalg.eig(A)

        outputs["data:handling_qualities:longitudinal:spacestate:eigenvalues"] = w
