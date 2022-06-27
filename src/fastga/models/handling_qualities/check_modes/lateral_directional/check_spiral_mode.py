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
import matplotlib.pyplot as plt


class CheckSpiralMode(om.ExplicitComponent):
    """
    # TODOC:
    """

    def setup(self):
        self.add_input("data:reference_flight_condition:flight_phase_category", val=np.nan)

        self.add_input("data:handling_qualities:lateral:modes:spiral:real_part", units="s**-1")
        self.add_input("data:handling_qualities:lateral:modes:spiral:imag_part", units="s**-1")

        self.add_output(
            "data:handling_qualities:lateral:modes:spiral:check:time_double:satisfaction_level")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        flight_phase_category = inputs["data:reference_flight_condition:flight_phase_category"]

        real_spiral = inputs["data:handling_qualities:lateral:modes:spiral:real_part"]
        imag_spiral = inputs["data:handling_qualities:lateral:modes:spiral:imag_part"]

        # GET REQUIREMENTS
        t_2_req = self.get_roll_mode_requirements(flight_phase_category)

        # CHECK
        check_spiral = self.check_spiral_requirements(real_spiral, t_2_req)

        # PLOT
        self.plot_s_plane(real_spiral, imag_spiral, t_2_req)

        outputs["data:handling_qualities:lateral:modes:spiral:check:time_double:satisfaction_level"] = check_spiral



    @staticmethod
    def get_roll_mode_requirements(flight_phase_category):
        """
        there are no specific requirements for spiral stability in any airplane. However, the military requirements
        place limits on the allowable divergence of the spiral mode. These limits are presented in the military
        regulation.

        """

        t_2_req_1 = 0.0
        t_2_req_2 = 0.0
        t_2_req_3 = 0.0
        if flight_phase_category == 1.0 or flight_phase_category == 3.0:
            t_2_req_1 = 12.0
            t_2_req_2 = 8.0
            t_2_req_3 = 4.0
        elif flight_phase_category == 2.0:
            t_2_req_1 = 20.0
            t_2_req_2 = 8.0
            t_2_req_3 = 4.0

        t_2_req = [t_2_req_1, t_2_req_2, t_2_req_3]

        return t_2_req


    @staticmethod
    def check_spiral_requirements(real_spiral, t_2_req):

        t_2_req_1 = t_2_req[0]
        t_2_req_2 = t_2_req[1]
        t_2_req_3 = t_2_req[2]

        check_spiral = 0.0
        if real_spiral <= 0.0:
            check_spiral = 1.0
        elif real_spiral > 0.0:
            # Compute time to double
            t_2 = math.log(2) / real_spiral
            if t_2 > t_2_req_1:
                check_spiral = 1.0
            elif t_2_req_1 > t_2 > t_2_req_1:
                check_spiral = 2.0
            elif t_2_req_2 > t_2 > t_2_req_3:
                check_spiral = 3.0

        return check_spiral


    @staticmethod
    def plot_s_plane(real_spiral, imag_spiral, t_2_req):

        def get_vertical_limits(t_2_req):

            real_part = 1.0 / t_2_req
            x = real_part * np.ones(1000)
            y = np.linspace(-100, 100, 1000)

            return x, y

        t_2_req_1 = t_2_req[0]
        t_2_req_2 = t_2_req[1]
        t_2_req_3 = t_2_req[2]

        fig, ax = plt.subplots()
        ax.set_title("Check of Spiral Mode Characteristics Versus Flying Quality Requirements")
        ax.set_xlabel(r"$n$")
        ax.set_ylabel(r"$jw$")
        ax.grid(True, which="both")
        # x-axis
        ax.plot(np.linspace(-1000, 1000, 1000), 0.0 * np.linspace(-1000, 1000, 1000), color="black")
        # y-axis
        ax.plot(0.0 * np.linspace(-1000, 1000, 1000), np.linspace(-1000, 1000, 1000), color="black")
        ax.scatter(real_spiral, imag_spiral, label=r"$\lambda_{spiral}$")

        # Damping-frequency product limit
        x_1, y_1 = get_vertical_limits(t_2_req_1)
        ax.plot(x_1, y_1, linestyle="--", color="red", label="Level 1")
        x_2, y_2 = get_vertical_limits(t_2_req_2)
        ax.plot(x_2, y_2, linestyle="--", color="orange", label="Level 2")
        x_3, y_3 = get_vertical_limits(t_2_req_3)
        ax.plot(x_3, y_3, linestyle="--", color="yellow", label="Level 3")

        ax.legend(loc="upper right")
        ax.set_xbound(real_spiral * 3, 0.1)
        ax.set_ybound(-real_spiral * 3, real_spiral * 3)
        plt.show()




