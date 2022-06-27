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


class CheckDutchRoll(om.ExplicitComponent):
    """
    # TODOC:
    """

    def setup(self):
        self.add_input("data:geometry:aircraft:class", val=np.nan)
        self.add_input("data:reference_flight_condition:flight_phase_category", val=np.nan)
        self.add_input("data:reference_flight_condition:dynamic_pressure", val=np.nan, units="Pa")
        self.add_input("data:reference_flight_condition:weight", val=np.nan, units="kg")

        self.add_input("data:handling_qualities:lateral:modes:dutch_roll:real_part", val=np.nan, units="s**-1")
        self.add_input("data:handling_qualities:lateral:modes:dutch_roll:imag_part", val=np.nan,
                       units="s**-1")

        self.add_output(
            "data:handling_qualities:lateral:modes:dutch_roll:check:damping_ratio:satisfaction_level")
        self.add_output(
            "data:handling_qualities:lateral:modes:dutch_roll:check:undamped_frequency:satisfaction_level")
        self.add_output(
            "data:handling_qualities:lateral:modes:dutch_roll:check:damping_ratio_frequency_product:satisfaction_level")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        real_dr = inputs["data:handling_qualities:lateral:modes:dutch_roll:real_part"]
        imag_dr = inputs["data:handling_qualities:lateral:modes:dutch_roll:imag_part"]

        wn_dr = math.sqrt(real_dr ** 2 + imag_dr ** 2)
        z_dr = - real_dr / math.sqrt(real_dr ** 2 + imag_dr ** 2)
        z_wn_dr = z_dr * wn_dr

        aircraft_class = inputs["data:geometry:aircraft:class"]
        flight_phase_category = inputs["data:reference_flight_condition:flight_phase_category"]

        real_dr = - z_dr * wn_dr
        imag_dr = wn_dr * math.sqrt(1 - z_dr ** 2)

        level_1_dr_reqs, level_2_dr_reqs, level_3_dr_reqs = self.get_dutch_roll_requirements(
            aircraft_class, flight_phase_category)


        ### CHECK ###
        check_damping_ratio, check_undamped_frequency, check_product = self.check_dutch_roll_requirements(
            z_dr, wn_dr, z_wn_dr, level_1_dr_reqs, level_2_dr_reqs, level_3_dr_reqs)


        ### PLOT ###
        self.plot_s_plane(real_dr, imag_dr, level_1_dr_reqs, level_2_dr_reqs, level_3_dr_reqs)


        outputs["data:handling_qualities:lateral:modes:dutch_roll:check:damping_ratio:satisfaction_level"] = check_damping_ratio
        outputs["data:handling_qualities:lateral:modes:dutch_roll:check:undamped_frequency:satisfaction_level"] = check_undamped_frequency
        outputs["data:handling_qualities:lateral:modes:dutch_roll:check:damping_ratio_frequency_product:satisfaction_level"] = check_product


    @staticmethod
    def get_dutch_roll_requirements(aircraft_class, flight_phase_category):
        """
        # TODOC
        """

        # Level 1
        min_z_dr_1 = 0.0
        min_z_wn_dr_1 = 0.0
        min_wn_dr_1 = 0.0
        if flight_phase_category == 1.0:
            if aircraft_class == 1.0 or aircraft_class == 4.0:
                min_z_dr_1 = 0.19
                min_z_wn_dr_1 = 0.35
                min_wn_dr_1 = 1.0
            elif aircraft_class == 2.0 or aircraft_class == 3.0:
                min_z_dr_1 = 0.19
                min_z_wn_dr_1 = 0.35
                min_wn_dr_1 = 0.4
        elif flight_phase_category == 2.0:
            min_z_dr_1 = 0.08
            min_z_wn_dr_1 = 0.15
            min_wn_dr_1 = 0.4
        elif flight_phase_category == 3.0:
            if aircraft_class == 1.0 or aircraft_class == 4.0:
                min_z_dr_1 = 0.08
                min_z_wn_dr_1 = 0.15
                min_wn_dr_1 = 1.0
            elif aircraft_class == 2.0 or aircraft_class == 3.0:
                min_z_dr_1 = 0.08
                min_z_wn_dr_1 = 0.10
                min_wn_dr_1 = 0.4

        level_1_dr_reqs = [min_z_dr_1, min_z_wn_dr_1, min_wn_dr_1]

        # Level 2
        min_z_dr_2 = 0.02
        min_z_wn_dr_2 = 0.05
        min_wn_dr_2 = 0.4
        level_2_dr_reqs = [min_z_dr_2, min_z_wn_dr_2, min_wn_dr_2]

        # Level 3
        min_z_dr_3 = 0.0
        min_z_wn_dr_3 = 0.0
        min_wn_dr_3 = 0.4
        level_3_dr_reqs = [min_z_dr_3, min_z_wn_dr_3, min_wn_dr_3]


        return level_1_dr_reqs, level_2_dr_reqs, level_3_dr_reqs

    @staticmethod
    def check_dutch_roll_requirements(z_dr, wn_dr, z_wn_dr, level_1_dr_reqs, level_2_dr_reqs, level_3_dr_reqs):

        min_z_dr_1 = level_1_dr_reqs[0]
        min_z_wn_dr_1 = level_1_dr_reqs[1]
        min_wn_dr_1 = level_1_dr_reqs[2]

        min_z_dr_2 = level_2_dr_reqs[0]
        min_z_wn_dr_2 = level_2_dr_reqs[1]
        min_wn_dr_2 = level_2_dr_reqs[2]

        min_z_dr_3 = level_3_dr_reqs[0]
        min_z_wn_dr_3 = level_3_dr_reqs[1]
        min_wn_dr_3 = level_3_dr_reqs[2]

        ### CHECK ###
        # Damping ratio
        check_damping_ratio = 0.0
        if z_dr > min_z_dr_1:
            check_damping_ratio = 1.0
        elif min_z_dr_2 < z_dr <= min_z_dr_1:
            check_damping_ratio = 2.0
        elif min_z_dr_3 <= z_dr <= min_z_dr_2:
            check_damping_ratio = 3.0

        # Undamped frequency
        check_undamped_frequency = 0.0
        if wn_dr > min_wn_dr_1:
            check_undamped_frequency = 1.0
        elif min_wn_dr_2 < wn_dr <= min_wn_dr_1:
            check_undamped_frequency = 2.0
        elif min_wn_dr_3 <= wn_dr <= min_wn_dr_2:
            check_undamped_frequency = 3.0

        # Damping ratio and frequency product
        check_product = 0.0
        if z_wn_dr > min_z_wn_dr_1:
            check_product = 1.0
        elif min_z_wn_dr_2 < z_wn_dr <= min_z_wn_dr_1:
            check_product = 2.0
        elif min_z_wn_dr_3 <= z_wn_dr <= min_z_wn_dr_2:
            check_product = 3.0

        return check_damping_ratio, check_undamped_frequency, check_product

    @staticmethod
    def plot_s_plane(real_dr, imag_dr, level_1_dr_reqs, level_2_dr_reqs, level_3_dr_reqs):

        def get_damping_limits(damping_ratio):
            phi = math.asin(damping_ratio)
            phi = math.pi / 2 - phi
            x = np.linspace(-100, 100, 100)
            y = math.tan(-phi) * x

            return x, y

        def get_vertical_limits(z_wn_product):
            x = -z_wn_product * np.ones(1000)
            y = np.linspace(-100, 100, 1000)

            return x, y

        def get_frequency_limits(wn):
            angle = np.linspace(math.pi, math.pi / 2.0, 100)

            x = wn * np.cos(angle)
            y = wn * np.sin(angle)

            return x, y

        min_z_dr_1 = level_1_dr_reqs[0]
        min_z_wn_dr_1 = level_1_dr_reqs[1]
        min_wn_dr_1 = level_1_dr_reqs[2]

        min_z_dr_2 = level_2_dr_reqs[0]
        min_z_wn_dr_2 = level_2_dr_reqs[1]
        min_wn_dr_2 = level_2_dr_reqs[2]

        min_z_dr_3 = level_3_dr_reqs[0]
        min_z_wn_dr_3 = level_3_dr_reqs[1]
        min_wn_dr_3 = level_3_dr_reqs[2]

        ### PLOT ###
        ### s-plane ###
        fig, ax = plt.subplots()
        ax.set_title("Check of Dutch Roll Characteristics Versus Flying Quality Requirements")
        ax.set_xlabel(r"$n$")
        ax.set_ylabel(r"$jw$")
        ax.grid(True, which="both")
        # x-axis
        ax.plot(np.linspace(-1000, 1000, 1000), 0.0 * np.linspace(-1000, 1000, 1000), color="black")
        # y-axis
        ax.plot(0.0 * np.linspace(-1000, 1000, 1000), np.linspace(-1000, 1000, 1000), color="black")
        ax.scatter(real_dr, imag_dr, label=r"$\lambda_{dr}$")
        # Limits
        # Damping ratio limits
        x_1, y_1 = get_damping_limits(min_z_dr_1)
        ax.plot(x_1, y_1, linestyle="--", color="red", label="Level 1")
        x_2, y_2 = get_damping_limits(min_z_dr_2)
        ax.plot(x_2, y_2, linestyle="--", color="orange", label="Level 2")
        x_3, y_3 = get_damping_limits(min_z_dr_3)
        ax.plot(x_3, y_3, linestyle="--", color="yellow", label="Level 3")

        # Frequency limits
        x_1, y_1 = get_frequency_limits(min_wn_dr_1)
        ax.plot(x_1, y_1, linestyle="--", color="red", label="")
        x_2, y_2 = get_frequency_limits(min_wn_dr_2)
        ax.plot(x_2, y_2, linestyle="--", color="orange", label="")
        x_3, y_3 = get_frequency_limits(min_wn_dr_3)
        ax.plot(x_3, y_3, linestyle="--", color="yellow", label="")

        # Damping-frequency product limit
        x_1, y_1 = get_vertical_limits(min_z_wn_dr_1)
        ax.plot(x_1, y_1, linestyle="--", color="red")
        x_2, y_2 = get_vertical_limits(min_z_wn_dr_2)
        ax.plot(x_2, y_2, linestyle="--", color="orange")
        x_3, y_3 = get_vertical_limits(min_z_wn_dr_3)
        ax.plot(x_3, y_3, linestyle="--", color="yellow")

        ax.legend(loc="upper right")
        ax.set_xbound(real_dr * 3, 0.1)
        ax.set_ybound(-imag_dr, imag_dr * 4)
        plt.show()

        return


    @staticmethod
    def get_frequency_limits(wn):

        angle = np.linspace(math.pi, math.pi / 2.0, 100)

        x = wn * np.cos(angle)
        y = wn * np.sin(angle)

        return x, y

    @staticmethod
    def get_damping_limits(damping_ratio):

        phi = math.asin(damping_ratio)
        phi = math.pi / 2 - phi
        x = np.linspace(-100, 100, 100)
        y = math.tan(-phi) * x

        return x,y

    @staticmethod
    def get_vertical_limits(z_wn_product):

        x = -z_wn_product * np.ones(1000)
        y = np.linspace(-100, 100, 1000)

        return x,y






