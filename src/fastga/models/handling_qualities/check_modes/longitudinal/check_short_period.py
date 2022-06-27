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
import logging
import os.path as pth
from pandas import read_csv
from scipy import interpolate

import openmdao.api as om
import numpy as np
import matplotlib.pyplot as plt

from fastga.models.handling_qualities.resources import digit_figures

_LOGGER = logging.getLogger(__name__)


class CheckShortPeriod(om.ExplicitComponent):
    """
    # TODOC:
    """

    def setup(self):
        self.add_input("data:reference_flight_condition:dynamic_pressure", val=np.nan, units="Pa")
        self.add_input("data:handling_qualities:longitudinal:derivatives:CL:alpha", val=np.nan, units="rad**-1")
        self.add_input("data:geometry:wing:area", val=np.nan, units="m**2")
        self.add_input("data:reference_flight_condition:weight", val=np.nan, units="kg")

        self.add_input("data:handling_qualities:longitudinal:modes:short_period:real_part", val=np.nan, units="s**-1")
        self.add_input("data:handling_qualities:longitudinal:modes:short_period:imag_part", val=np.nan,
                       units="s**-1")
        self.add_input("data:reference_flight_condition:flight_phase_category", val=np.nan)

        self.add_output(
            "data:handling_qualities:longitudinal:modes:short_period:check:damping_ratio:satisfaction_level")
        self.add_output(
            "data:handling_qualities:longitudinal:modes:short_period:check:undamped_frequency:satisfaction_level")

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):

        real_sp = inputs["data:handling_qualities:longitudinal:modes:short_period:real_part"]
        imag_sp = inputs["data:handling_qualities:longitudinal:modes:short_period:imag_part"]

        wn_sp = math.sqrt(real_sp ** 2 + imag_sp ** 2)
        z_sp = - real_sp / math.sqrt(real_sp ** 2 + imag_sp ** 2)

        flight_phase_category = inputs["data:reference_flight_condition:flight_phase_category"]
        q = inputs["data:reference_flight_condition:dynamic_pressure"]
        CL_alpha = inputs["data:handling_qualities:longitudinal:derivatives:CL:alpha"]
        S = inputs["data:geometry:wing:area"]
        W = inputs["data:reference_flight_condition:weight"]
        n_alpha = q * CL_alpha * S / W

        ### SHORT PERIOD REQUIREMENTS ###
        z_sp_reqs, wn_sp_reqs = self.get_short_period_requirements(flight_phase_category, n_alpha)

        ### CHECK ###
        check_shortperiod_damping, check_shortperiod_frequency = self.check_short_period_requirements(
            z_sp, wn_sp, z_sp_reqs, wn_sp_reqs)

        ### PLOT ###
        ### Frequency vs. n/alpha plot ###
        # TODO: add rest of flight phase categories.
        file = pth.join(digit_figures.__path__[0], "sp_frequency_req_B.csv")
        db = read_csv(file)

        level1_up_x = db["level1_up_X"]
        level1_up_y = db["level1_up_Y"]
        errors = np.logical_or(np.isnan(level1_up_x), np.isnan(level1_up_y))
        level1_up_x = level1_up_x[np.logical_not(errors)].tolist()
        level1_up_y = level1_up_y[np.logical_not(errors)].tolist()

        level1_down_x = db["level1_down_X"]
        level1_down_y = db["level1_down_Y"]
        errors = np.logical_or(np.isnan(level1_down_x), np.isnan(level1_down_y))
        level1_down_x = level1_down_x[np.logical_not(errors)].tolist()
        level1_down_y = level1_down_y[np.logical_not(errors)].tolist()

        level2_up_x = db["level2_up_X"]
        level2_up_y = db["level2_up_Y"]
        errors = np.logical_or(np.isnan(level2_up_x), np.isnan(level2_up_y))
        level2_up_x = level2_up_x[np.logical_not(errors)].tolist()
        level2_up_y = level2_up_y[np.logical_not(errors)].tolist()

        level2_down_x = db["level2_down_X"]
        level2_down_y = db["level2_down_Y"]
        errors = np.logical_or(np.isnan(level2_down_x), np.isnan(level2_down_y))
        level2_down_x = level2_down_x[np.logical_not(errors)].tolist()
        level2_down_y = level2_down_y[np.logical_not(errors)].tolist()

        fig1, ax1 = plt.subplots()
        ax1.set_yscale('log')
        ax1.set_xscale('log')
        ax1.set_title("Check of Short Period Frequency Versus Flying Quality Requirements")
        ax1.set_xlabel(r"$n / \alpha$")
        ax1.set_ylabel("$w_n (rad/s)$")
        ax1.grid(True, which="both")

        ax1.scatter(n_alpha, wn_sp, label=r"$w_{n_{sp}}$")
        ax1.plot(level1_up_x, level1_up_y, linestyle="--", color="red", label="Level 1")
        ax1.plot(level1_down_x, level1_down_y, linestyle="--", color="red", label="")
        ax1.plot(level2_up_x, level2_up_y, linestyle="--", color="orange", label="Level 2")
        ax1.plot(level2_down_x, level2_down_y, linestyle="--", color="yellow", label="Level 2 & 3")
        ax1.legend(loc="upper right")
        ax1.set_xbound(1.0, 100)
        ax1.set_ybound(0.1, 100.0)
        # plt.show()

        ### s-plane plot ###
        self.plot_s_plane(real_sp, imag_sp, z_sp_reqs, wn_sp_reqs)


        outputs[
            "data:handling_qualities:longitudinal:modes:short_period:check:damping_ratio:satisfaction_level"
        ] = check_shortperiod_damping
        outputs[
            "data:handling_qualities:longitudinal:modes:short_period:check:undamped_frequency:satisfaction_level"
        ] = check_shortperiod_frequency


    @staticmethod
    def get_short_period_requirements(flight_phase_category, n_alpha):

        # Damping ratio requirements
        z_sp_min_req_1 = 0.0
        z_sp_max_req_1 = 0.0
        z_sp_min_req_2 = 0.0
        z_sp_max_req_2 = 0.0
        z_sp_min_req_3 = 0.0
        if flight_phase_category == 1.0 or flight_phase_category == 3.0:

            # Level 1
            z_sp_min_req_1 = 0.35
            z_sp_max_req_1 = 1.3

            # level 2
            z_sp_min_req_2 = 0.25
            z_sp_max_req_2 = 2.0

            # Level 3
            z_sp_min_req_3 = 0.15

        elif flight_phase_category == 2.0:
            # Level 1
            z_sp_min_req_1 = 0.3
            z_sp_max_req_1 = 2.0

            # level 2
            z_sp_min_req_2 = 0.2
            z_sp_max_req_2 = 2.0

            # Level 3
            z_sp_min_req_3 = 0.15

        z_reqs = [z_sp_min_req_1, z_sp_max_req_1, z_sp_min_req_2, z_sp_max_req_2, z_sp_min_req_3]

        # Frequency requirements
        # TODO: add rest of flight phase categories.
        file = pth.join(digit_figures.__path__[0], "sp_frequency_req_B.csv")
        db = read_csv(file)

        level1_up_x = db["level1_up_X"]
        level1_up_y = db["level1_up_Y"]
        errors = np.logical_or(np.isnan(level1_up_x), np.isnan(level1_up_y))
        level1_up_x = level1_up_x[np.logical_not(errors)].tolist()
        level1_up_y = level1_up_y[np.logical_not(errors)].tolist()

        level1_down_x = db["level1_down_X"]
        level1_down_y = db["level1_down_Y"]
        errors = np.logical_or(np.isnan(level1_down_x), np.isnan(level1_down_y))
        level1_down_x = level1_down_x[np.logical_not(errors)].tolist()
        level1_down_y = level1_down_y[np.logical_not(errors)].tolist()

        level2_up_x = db["level2_up_X"]
        level2_up_y = db["level2_up_Y"]
        errors = np.logical_or(np.isnan(level2_up_x), np.isnan(level2_up_y))
        level2_up_x = level2_up_x[np.logical_not(errors)].tolist()
        level2_up_y = level2_up_y[np.logical_not(errors)].tolist()

        level2_down_x = db["level2_down_X"]
        level2_down_y = db["level2_down_Y"]
        errors = np.logical_or(np.isnan(level2_down_x), np.isnan(level2_down_y))
        level2_down_x = level2_down_x[np.logical_not(errors)].tolist()
        level2_down_y = level2_down_y[np.logical_not(errors)].tolist()

        if float(n_alpha) != np.clip(float(n_alpha), min(level1_up_x), max(level1_up_x)):
            _LOGGER.warning("n/alpha parameter outside range, value clipped")
        if float(n_alpha) != np.clip(float(n_alpha), min(level1_down_x), max(level1_down_x)):
            _LOGGER.warning("n/alpha parameter outside range, value clipped")
        if float(n_alpha) != np.clip(float(n_alpha), min(level2_up_x), max(level2_up_x)):
            _LOGGER.warning("n/alpha parameter outside range, value clipped")
        if float(n_alpha) != np.clip(float(n_alpha), min(level2_down_x), max(level2_down_x)):
            _LOGGER.warning("n/alpha parameter outside range, value clipped")

        wn_level1_up = float(
            interpolate.interp1d(level1_up_x, level1_up_y)(np.clip(float(n_alpha), min(level1_up_x), max(level1_up_x))))
        wn_level1_down = float(
            interpolate.interp1d(level1_down_x, level1_down_y)(
                np.clip(float(n_alpha), min(level1_down_x), max(level1_down_x))))
        wn_level2_up = float(
            interpolate.interp1d(level2_up_x, level2_up_y)(np.clip(float(n_alpha), min(level2_up_x), max(level2_up_x))))
        wn_level2_down = float(
            interpolate.interp1d(level2_down_x, level2_down_y)(
                np.clip(float(n_alpha), min(level2_down_x), max(level2_down_x))))
        wn_level3 = wn_level2_down

        wn_reqs = [wn_level1_down, wn_level1_up, wn_level2_down, wn_level2_up, wn_level3]

        return z_reqs, wn_reqs


    @staticmethod
    def check_short_period_requirements(z_sp, wn_sp, z_sp_reqs, wn_sp_reqs):

        z_sp_min_req_1 = z_sp_reqs[0]
        z_sp_max_req_1 = z_sp_reqs[1]
        z_sp_min_req_2 = z_sp_reqs[2]
        z_sp_max_req_2 = z_sp_reqs[3]
        z_sp_min_req_3 = z_sp_reqs[4]

        wn_sp_min_req_1 = wn_sp_reqs[0]
        wn_sp_max_req_1 = wn_sp_reqs[1]
        wn_sp_min_req_2 = wn_sp_reqs[2]
        wn_sp_max_req_2 = wn_sp_reqs[3]
        wn_sp_min_req_3 = wn_sp_reqs[4]

        check_shortperiod_damping = 0.0
        if z_sp_min_req_1 <= z_sp <= z_sp_max_req_1:
            check_shortperiod_damping = 1.0
        elif z_sp_min_req_2 <= z_sp <= z_sp_max_req_2:
            check_shortperiod_damping = 2.0
        elif z_sp_min_req_3 <= z_sp:
            check_shortperiod_damping = 3.0

        check_shortperiod_frequency = 0.0
        if wn_sp_min_req_1 <= wn_sp <= wn_sp_max_req_1:
            check_shortperiod_damping = 1.0
        elif wn_sp_min_req_2 <= wn_sp <= wn_sp_max_req_2:
            check_shortperiod_damping = 2.0
        elif wn_sp_min_req_3 <= wn_sp:
            check_shortperiod_damping = 3.0

        return check_shortperiod_damping, check_shortperiod_frequency


    @staticmethod
    def plot_s_plane(real_sp, imag_sp, z_sp_reqs, wn_sp_reqs):

        def get_damping_limits(damping_ratio):
            if damping_ratio == 0.0:
                x = np.linspace(-100, 0.0, 100) * 0.0
                y = np.linspace(-100, 0.0, 100)
            elif damping_ratio == 1.0 or damping_ratio > 1.0:
                x = np.linspace(-100, 0.0, 100)
                y = np.linspace(-100, 0.0, 100) * 0.0
            else:
                phi = math.asin(damping_ratio)
                phi = math.pi / 2 - phi
                x = np.linspace(-100, 0.0, 100)
                y = math.tan(-phi) * x

            return x, y

        def get_frequency_limits(wn):
            angle = np.linspace(math.pi, math.pi / 2.0, 100)

            x = wn * np.cos(angle)
            y = wn * np.sin(angle)

            return x, y

        z_sp_min_req_1 = z_sp_reqs[0]
        z_sp_max_req_1 = z_sp_reqs[1]
        z_sp_min_req_2 = z_sp_reqs[2]
        z_sp_max_req_2 = z_sp_reqs[3]
        z_sp_min_req_3 = z_sp_reqs[4]

        wn_sp_min_req_1 = wn_sp_reqs[0]
        wn_sp_max_req_1 = wn_sp_reqs[1]
        wn_sp_min_req_2 = wn_sp_reqs[2]
        wn_sp_max_req_2 = wn_sp_reqs[3]
        wn_sp_min_req_3 = wn_sp_reqs[4]

        fig2, ax2 = plt.subplots()
        ax2.set_title("Check of Short Period Characteristics Versus Flying Quality Requirements")
        ax2.set_xlabel(r"$n$")
        ax2.set_ylabel(r"$jw$")
        ax2.grid(True, which="both")
        # x-axis
        ax2.plot(np.linspace(-1000, 1000, 1000), 0.0 * np.linspace(-1000, 1000, 1000), color="black")
        # y-axis
        ax2.plot(0.0 * np.linspace(-1000, 1000, 1000), np.linspace(-1000, 1000, 1000), color="black")

        ax2.scatter(real_sp, imag_sp, label=r"$\lambda_{sp}$")

        # Damping ratio limits
        # Level 1
        x_damp_min_1, y_damp_min_1 = get_damping_limits(z_sp_min_req_1)
        ax2.plot(x_damp_min_1, y_damp_min_1, linestyle="--", color="red", label="Level 1")
        x_damp_max_1, y_damp_max_1 = get_damping_limits(z_sp_max_req_1)
        ax2.plot(x_damp_max_1, y_damp_max_1, linestyle="--", color="red", label="")
        # Level 2
        x_damp_min_2, y_damp_min_2 = get_damping_limits(z_sp_min_req_2)
        ax2.plot(x_damp_min_2, y_damp_min_2, linestyle="--", color="orange", label="Level 2")
        # x_damp_max_2, y_damp_max_2 = get_damping_limits(z_sp_max_req_2)
        # ax2.plot(x_damp_max_2, y_damp_max_2, linestyle="--", color="orange", label="")
        # Level 3
        x_damp_min_3, y_damp_min_3 = get_damping_limits(z_sp_min_req_3)
        ax2.plot(x_damp_min_3, y_damp_min_3, linestyle="--", color="yellow", label="Level 3")

        # Frequency limits
        # Level 1
        x_freq_min_1, y_freq_min_1 = get_frequency_limits(wn_sp_min_req_1)
        ax2.plot(x_freq_min_1, y_freq_min_1, linestyle="--", color="red", label="")
        x_freq_max_1, y_freq_max_1 = get_frequency_limits(wn_sp_max_req_1)
        ax2.plot(x_freq_max_1, y_freq_max_1, linestyle="--", color="red", label="")

        # Level 2
        x_freq_min_2, y_freq_min_2 = get_frequency_limits(wn_sp_min_req_2)
        ax2.plot(x_freq_min_2, y_freq_min_2, linestyle="--", color="yellow", label="Level 2")
        x_freq_max_2, y_freq_max_2 = get_frequency_limits(wn_sp_max_req_2)
        ax2.plot(x_freq_max_2, y_freq_max_2, linestyle="--", color="orange", label="")

        ax2.legend(loc="upper right")
        ax2.set_xbound(real_sp * 3, 0.1)
        ax2.set_ybound(-imag_sp, imag_sp * 4)
        plt.show()


    @staticmethod
    def get_frequency_limits(wn):

        angle = np.linspace(math.pi, math.pi / 2.0, 100)

        x = wn * np.cos(angle)
        y = wn * np.sin(angle)

        return x, y

