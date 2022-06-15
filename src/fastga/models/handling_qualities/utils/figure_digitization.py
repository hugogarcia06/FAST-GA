"""
Generic class containing all the digitization needed to compute the stability
coefficient of the aircraft.
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

import logging
import math
import os.path as pth

import numpy as np
import openmdao.api as om
from pandas import read_csv

from scipy import interpolate

from fastga.models.handling_qualities import resources, digit_figures

_LOGGER = logging.getLogger(__name__)


class FigureDigitization2(om.ExplicitComponent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.phase = None

    @staticmethod
    def get_k2_k1(fus_length, fus_diameter) -> float:
        """
        Data from USAF DATCOM to estimate the apparent mass factor presented in Figure 4.2.1.1-2a as a function of the
        fuselage fineness ratio.

        :param fus_length:
        :param fus_diameter:
        """

        fineness_ratio = fus_length / fus_diameter

        file = pth.join(digit_figures.__path__[0], "4_2_1_1_20a.csv")
        db = read_csv(file)

        x = db["X"]
        y = db["Y"]
        errors = np.logical_or(np.isnan(x), np.isnan(y))
        x = x[np.logical_not(errors)].tolist()
        y = y[np.logical_not(errors)].tolist()

        if float(fineness_ratio) != np.clip(float(fineness_ratio), min(x), max(x)):
            _LOGGER.warning("Fuselage fineness ratio value outside of the range in Roskam's book, value clipped")

        k2_k1 = float(interpolate.interp1d(x, y)(np.clip(float(fineness_ratio), min(x), max(x))))

        return k2_k1


    @staticmethod
    def get_k_wb(body_diameter, wing_span) -> float:
        """
        Data from USAF DATCOM to estimate the apparent mass factor presented in Figure 4.2.1.1-2a as a function of the
        fuselage fineness ratio.

        :param body_diameter:
        :param wing_span:
        """

        body_span_ratio = body_diameter / wing_span

        file = pth.join(digit_figures.__path__[0], "4_3_1_2_14.csv")
        db = read_csv(file)

        x = db["K_WB_X"]
        y = db["K_WB_Y"]
        errors = np.logical_or(np.isnan(x), np.isnan(y))
        x = x[np.logical_not(errors)].tolist()
        y = y[np.logical_not(errors)].tolist()

        if float(body_span_ratio) != np.clip(float(body_span_ratio), min(x), max(x)):
            _LOGGER.warning("Body diameter and wing span ratio outside of the range in Roskam's book, value clipped")

        k_wb = float(interpolate.interp1d(x, y)(np.clip(float(body_span_ratio), min(x), max(x))))

        return k_wb


    @staticmethod
    def get_k_bw(body_diameter, wing_span) -> float:
        """
        Data from USAF DATCOM to estimate the apparent mass factor presented in Figure 4.2.1.1-2a as a function of the
        fuselage fineness ratio.

        :param body_diameter:
        :param wing_span:
        """

        body_span_ratio = body_diameter / wing_span

        file = pth.join(digit_figures.__path__[0], "4_3_1_2_14.csv")
        db = read_csv(file)

        x = db["K_BW_X"]
        y = db["K_BW_Y"]
        errors = np.logical_or(np.isnan(x), np.isnan(y))
        x = x[np.logical_not(errors)].tolist()
        y = y[np.logical_not(errors)].tolist()

        if float(body_span_ratio) != np.clip(float(body_span_ratio), min(x), max(x)):
            _LOGGER.warning("Fuselage fineness ratio value outside of the range in Roskam's book, value clipped")

        k_bw = float(interpolate.interp1d(x, y)(np.clip(float(body_span_ratio), min(x), max(x))))

        return k_bw


    @staticmethod
    def get_Clbeta_CL_sweep50(sweep_50, aspect_ratio, taper_ratio):
        """
        Data from Roskam - Part VI to compute the wing sweep contribution to rolling moment due to sideslip found in 
        Figure 10.20.

        :param sweep_50: surface mid-chord point line sweep angle in radians.
        :param aspect_ratio: surface aspect ratio.
        :param taper_ratio: surface taper_ratio
        """

        sweep_50 = sweep_50 * 180.0 / math.pi  # radians to degrees

        # ----- GRAPH FOR TAPER RATIO = 0.0  -----
        file = pth.join(digit_figures.__path__[0], "10_20_0.csv")
        db = read_csv(file)

        x_aspectratio1_graph_1 = db["ASPECT_RATIO_1_X"]
        y_aspectratio1_graph_1 = db["ASPECT_RATIO_1_Y"]
        errors = np.logical_or(np.isnan(x_aspectratio1_graph_1), np.isnan(y_aspectratio1_graph_1))
        x_aspectratio1_graph_1 = x_aspectratio1_graph_1[np.logical_not(errors)].tolist()
        y_aspectratio1_graph_1 = y_aspectratio1_graph_1[np.logical_not(errors)].tolist()

        x_aspectratio1_5_graph_1 = db["ASPECT_RATIO_1.5_X"]
        y_aspectratio1_5_graph_1 = db["ASPECT_RATIO_1.5_Y"]
        errors = np.logical_or(np.isnan(x_aspectratio1_5_graph_1), np.isnan(y_aspectratio1_5_graph_1))
        x_aspectratio1_5_graph_1 = x_aspectratio1_5_graph_1[np.logical_not(errors)].tolist()
        y_aspectratio1_5_graph_1 = y_aspectratio1_5_graph_1[np.logical_not(errors)].tolist()

        x_aspectratio2_graph_1 = db["ASPECT_RATIO_2_X"]
        y_aspectratio2_graph_1 = db["ASPECT_RATIO_2_Y"]
        errors = np.logical_or(np.isnan(x_aspectratio2_graph_1), np.isnan(y_aspectratio2_graph_1))
        x_aspectratio2_graph_1 = x_aspectratio2_graph_1[np.logical_not(errors)].tolist()
        y_aspectratio2_graph_1 = y_aspectratio2_graph_1[np.logical_not(errors)].tolist()

        x_aspectratio3_graph_1 = db["ASPECT_RATIO_3_X"]
        y_aspectratio3_graph_1 = db["ASPECT_RATIO_3_Y"]
        errors = np.logical_or(np.isnan(x_aspectratio3_graph_1), np.isnan(y_aspectratio3_graph_1))
        x_aspectratio3_graph_1 = x_aspectratio3_graph_1[np.logical_not(errors)].tolist()
        y_aspectratio3_graph_1 = y_aspectratio3_graph_1[np.logical_not(errors)].tolist()

        x_aspectratio6_graph_1 = db["ASPECT_RATIO_6_X"]
        y_aspectratio6_graph_1 = db["ASPECT_RATIO_6_Y"]
        errors = np.logical_or(np.isnan(x_aspectratio6_graph_1), np.isnan(y_aspectratio6_graph_1))
        x_aspectratio6_graph_1 = x_aspectratio6_graph_1[np.logical_not(errors)].tolist()
        y_aspectratio6_graph_1 = y_aspectratio6_graph_1[np.logical_not(errors)].tolist()

        k_aspectratio1_graph_1 = interpolate.interp1d(x_aspectratio1_graph_1, y_aspectratio1_graph_1)
        k_aspectratio1_5_graph_1 = interpolate.interp1d(x_aspectratio1_5_graph_1, y_aspectratio1_5_graph_1)
        k_aspectratio2_graph_1 = interpolate.interp1d(x_aspectratio2_graph_1, y_aspectratio2_graph_1)
        k_aspectratio3_graph_1 = interpolate.interp1d(x_aspectratio3_graph_1, y_aspectratio3_graph_1)
        k_aspectratio6_graph_1 = interpolate.interp1d(x_aspectratio6_graph_1, y_aspectratio6_graph_1)

        if (
                (sweep_50 != np.clip(sweep_50, min(x_aspectratio1_graph_1), max(x_aspectratio1_graph_1)))
                or (sweep_50 != np.clip(sweep_50, min(x_aspectratio1_5_graph_1), max(x_aspectratio1_5_graph_1)))
                or (sweep_50 != np.clip(sweep_50, min(x_aspectratio2_graph_1), max(x_aspectratio2_graph_1)))
                or (sweep_50 != np.clip(sweep_50, min(x_aspectratio3_graph_1), max(x_aspectratio3_graph_1)))
                or (sweep_50 != np.clip(sweep_50, min(x_aspectratio6_graph_1), max(x_aspectratio6_graph_1)))
        ):
            _LOGGER.warning("Sweep angle value outside of the range in Roskam's book, value clipped")

        k_aspectratio_graph_1 = [
            float(k_aspectratio1_graph_1(np.clip(sweep_50, min(x_aspectratio1_graph_1), max(x_aspectratio1_graph_1)))),
            float(k_aspectratio1_5_graph_1(
                np.clip(sweep_50, min(x_aspectratio1_5_graph_1), max(x_aspectratio1_5_graph_1)))),
            float(k_aspectratio2_graph_1(np.clip(sweep_50, min(x_aspectratio2_graph_1), max(x_aspectratio2_graph_1)))),
            float(k_aspectratio3_graph_1(np.clip(sweep_50, min(x_aspectratio3_graph_1), max(x_aspectratio3_graph_1)))),
            float(k_aspectratio6_graph_1(np.clip(sweep_50, min(x_aspectratio6_graph_1), max(x_aspectratio6_graph_1)))),

        ]

        if aspect_ratio != np.clip(aspect_ratio, 1.0, 6.0):
            _LOGGER.warning(
                "Aspect ratio value outside of the range in Roskam's book, value clipped"
            )

        Clbeta_graph_1 = float(
            interpolate.interp1d([1.0, 1.5, 2.0, 3.0, 6.0], k_aspectratio_graph_1)(
                np.clip(aspect_ratio, 1.0, 6.0)
            )
        )

        # ----- GRAPH FOR TAPER RATIO = 0.5 -----
        file = pth.join(digit_figures.__path__[0], "10_20_0_5.csv")
        db = read_csv(file)

        x_aspectratio1_graph_2 = db["ASPECT_RATIO_1_X"]
        y_aspectratio1_graph_2 = db["ASPECT_RATIO_1_Y"]
        errors = np.logical_or(np.isnan(x_aspectratio1_graph_2), np.isnan(y_aspectratio1_graph_2))
        x_aspectratio1_graph_2 = x_aspectratio1_graph_2[np.logical_not(errors)].tolist()
        y_aspectratio1_graph_2 = y_aspectratio1_graph_2[np.logical_not(errors)].tolist()

        x_aspectratio2_graph_2 = db["ASPECT_RATIO_2_X"]
        y_aspectratio2_graph_2 = db["ASPECT_RATIO_2_Y"]
        errors = np.logical_or(np.isnan(x_aspectratio2_graph_2), np.isnan(y_aspectratio2_graph_2))
        x_aspectratio2_graph_2 = x_aspectratio2_graph_2[np.logical_not(errors)].tolist()
        y_aspectratio2_graph_2 = y_aspectratio2_graph_2[np.logical_not(errors)].tolist()

        x_aspectratio4_graph_2 = db["ASPECT_RATIO_4_X"]
        y_aspectratio4_graph_2 = db["ASPECT_RATIO_4_Y"]
        errors = np.logical_or(np.isnan(x_aspectratio4_graph_2), np.isnan(y_aspectratio4_graph_2))
        x_aspectratio4_graph_2 = x_aspectratio4_graph_2[np.logical_not(errors)].tolist()
        y_aspectratio4_graph_2 = y_aspectratio4_graph_2[np.logical_not(errors)].tolist()

        x_aspectratio6_graph_2 = db["ASPECT_RATIO_6_X"]
        y_aspectratio6_graph_2 = db["ASPECT_RATIO_6_Y"]
        errors = np.logical_or(np.isnan(x_aspectratio6_graph_2), np.isnan(y_aspectratio6_graph_2))
        x_aspectratio6_graph_2 = x_aspectratio6_graph_2[np.logical_not(errors)].tolist()
        y_aspectratio6_graph_2 = y_aspectratio6_graph_2[np.logical_not(errors)].tolist()

        x_aspectratio8_graph_2 = db["ASPECT_RATIO_8_X"]
        y_aspectratio8_graph_2 = db["ASPECT_RATIO_8_Y"]
        errors = np.logical_or(np.isnan(x_aspectratio8_graph_2), np.isnan(y_aspectratio8_graph_2))
        x_aspectratio8_graph_2 = x_aspectratio8_graph_2[np.logical_not(errors)].tolist()
        y_aspectratio8_graph_2 = y_aspectratio8_graph_2[np.logical_not(errors)].tolist()

        k_aspectratio1_graph_2 = interpolate.interp1d(x_aspectratio1_graph_2, y_aspectratio1_graph_2)
        k_aspectratio2_graph_2 = interpolate.interp1d(x_aspectratio2_graph_2, y_aspectratio2_graph_2)
        k_aspectratio4_graph_2 = interpolate.interp1d(x_aspectratio4_graph_2, y_aspectratio4_graph_2)
        k_aspectratio6_graph_2 = interpolate.interp1d(x_aspectratio6_graph_2, y_aspectratio6_graph_2)
        k_aspectratio8_graph_2 = interpolate.interp1d(x_aspectratio8_graph_2, y_aspectratio8_graph_2)

        if (
                (sweep_50 != np.clip(sweep_50, min(x_aspectratio1_graph_2), max(x_aspectratio1_graph_2)))
                or (sweep_50 != np.clip(sweep_50, min(x_aspectratio2_graph_2), max(x_aspectratio2_graph_2)))
                or (sweep_50 != np.clip(sweep_50, min(x_aspectratio4_graph_2), max(x_aspectratio4_graph_2)))
                or (sweep_50 != np.clip(sweep_50, min(x_aspectratio6_graph_2), max(x_aspectratio6_graph_2)))
                or (sweep_50 != np.clip(sweep_50, min(x_aspectratio8_graph_2), max(x_aspectratio8_graph_2)))
        ):
            _LOGGER.warning("Sweep angle value outside of the range in Roskam's book, value clipped")

        k_aspectratio_graph_2 = [
            float(k_aspectratio1_graph_2(np.clip(sweep_50, min(x_aspectratio1_graph_2), max(x_aspectratio1_graph_2)))),
            float(k_aspectratio2_graph_2(np.clip(sweep_50, min(x_aspectratio2_graph_2), max(x_aspectratio2_graph_2)))),
            float(k_aspectratio4_graph_2(np.clip(sweep_50, min(x_aspectratio4_graph_2), max(x_aspectratio4_graph_2)))),
            float(k_aspectratio6_graph_2(np.clip(sweep_50, min(x_aspectratio6_graph_2), max(x_aspectratio6_graph_2)))),
            float(k_aspectratio8_graph_2(np.clip(sweep_50, min(x_aspectratio8_graph_2), max(x_aspectratio8_graph_2)))),

        ]

        if aspect_ratio != np.clip(aspect_ratio, 1.0, 8.0):
            _LOGGER.warning(
                "Aspect ratio value outside of the range in Roskam's book, value clipped"
            )

        Clbeta_graph_2 = float(
            interpolate.interp1d([1.0, 2.0, 4.0, 6.0, 8.0], k_aspectratio_graph_2)(
                np.clip(aspect_ratio, 1.0, 8.0)
            )
        )

        # ----- GRAPH FOR TAPER RATIO = 1.0 -----
        file = pth.join(digit_figures.__path__[0], "10_20_1.csv")
        db = read_csv(file)

        x_aspectratio1_graph_3 = db["ASPECT_RATIO_1_X"]
        y_aspectratio1_graph_3 = db["ASPECT_RATIO_1_Y"]
        errors = np.logical_or(np.isnan(x_aspectratio1_graph_3), np.isnan(y_aspectratio1_graph_3))
        x_aspectratio1_graph_3 = x_aspectratio1_graph_3[np.logical_not(errors)].tolist()
        y_aspectratio1_graph_3 = y_aspectratio1_graph_3[np.logical_not(errors)].tolist()

        x_aspectratio2_graph_3 = db["ASPECT_RATIO_2_X"]
        y_aspectratio2_graph_3 = db["ASPECT_RATIO_2_Y"]
        errors = np.logical_or(np.isnan(x_aspectratio2_graph_3), np.isnan(y_aspectratio2_graph_3))
        x_aspectratio2_graph_3 = x_aspectratio2_graph_3[np.logical_not(errors)].tolist()
        y_aspectratio2_graph_3 = y_aspectratio2_graph_3[np.logical_not(errors)].tolist()

        x_aspectratio4_graph_3 = db["ASPECT_RATIO_4_X"]
        y_aspectratio4_graph_3 = db["ASPECT_RATIO_4_Y"]
        errors = np.logical_or(np.isnan(x_aspectratio4_graph_3), np.isnan(y_aspectratio4_graph_3))
        x_aspectratio4_graph_3 = x_aspectratio4_graph_3[np.logical_not(errors)].tolist()
        y_aspectratio4_graph_3 = y_aspectratio4_graph_3[np.logical_not(errors)].tolist()

        x_aspectratio6_graph_3 = db["ASPECT_RATIO_6_X"]
        y_aspectratio6_graph_3 = db["ASPECT_RATIO_6_Y"]
        errors = np.logical_or(np.isnan(x_aspectratio6_graph_3), np.isnan(y_aspectratio6_graph_3))
        x_aspectratio6_graph_3 = x_aspectratio6_graph_3[np.logical_not(errors)].tolist()
        y_aspectratio6_graph_3 = y_aspectratio6_graph_3[np.logical_not(errors)].tolist()

        x_aspectratio8_graph_3 = db["ASPECT_RATIO_8_X"]
        y_aspectratio8_graph_3 = db["ASPECT_RATIO_8_Y"]
        errors = np.logical_or(np.isnan(x_aspectratio8_graph_3), np.isnan(y_aspectratio8_graph_3))
        x_aspectratio8_graph_3 = x_aspectratio8_graph_3[np.logical_not(errors)].tolist()
        y_aspectratio8_graph_3 = y_aspectratio8_graph_3[np.logical_not(errors)].tolist()

        k_aspectratio1_graph_3 = interpolate.interp1d(x_aspectratio1_graph_3, y_aspectratio1_graph_3)
        k_aspectratio2_graph_3 = interpolate.interp1d(x_aspectratio2_graph_3, y_aspectratio2_graph_3)
        k_aspectratio4_graph_3 = interpolate.interp1d(x_aspectratio4_graph_3, y_aspectratio4_graph_3)
        k_aspectratio6_graph_3 = interpolate.interp1d(x_aspectratio6_graph_3, y_aspectratio6_graph_3)
        k_aspectratio8_graph_3 = interpolate.interp1d(x_aspectratio8_graph_3, y_aspectratio8_graph_3)

        if (
                (sweep_50 != np.clip(sweep_50, min(x_aspectratio1_graph_3), max(x_aspectratio1_graph_3)))
                or (sweep_50 != np.clip(sweep_50, min(x_aspectratio2_graph_3), max(x_aspectratio2_graph_3)))
                or (sweep_50 != np.clip(sweep_50, min(x_aspectratio4_graph_3), max(x_aspectratio4_graph_3)))
                or (sweep_50 != np.clip(sweep_50, min(x_aspectratio6_graph_3), max(x_aspectratio6_graph_3)))
                or (sweep_50 != np.clip(sweep_50, min(x_aspectratio8_graph_3), max(x_aspectratio8_graph_3)))
        ):
            _LOGGER.warning("Sweep angle value outside of the range in Roskam's book, value clipped")

        k_aspectratio_graph_3 = [
            float(k_aspectratio1_graph_3(np.clip(sweep_50, min(x_aspectratio1_graph_3), max(x_aspectratio1_graph_3)))),
            float(k_aspectratio2_graph_3(np.clip(sweep_50, min(x_aspectratio2_graph_3), max(x_aspectratio2_graph_3)))),
            float(k_aspectratio4_graph_3(np.clip(sweep_50, min(x_aspectratio4_graph_3), max(x_aspectratio4_graph_3)))),
            float(k_aspectratio6_graph_3(np.clip(sweep_50, min(x_aspectratio6_graph_3), max(x_aspectratio6_graph_3)))),
            float(k_aspectratio8_graph_3(np.clip(sweep_50, min(x_aspectratio8_graph_3), max(x_aspectratio8_graph_3)))),

        ]

        if aspect_ratio != np.clip(aspect_ratio, 1.0, 8.0):
            _LOGGER.warning(
                "Aspect ratio value outside of the range in Roskam's book, value clipped"
            )

        Clbeta_graph_3 = float(
            interpolate.interp1d([1.0, 2.0, 4.0, 6.0, 8.0], k_aspectratio_graph_3)(
                np.clip(aspect_ratio, 1.0, 8.0)
            )
        )

        # INTERPOLATION BETWEEN THE 3 GRAPHS
        Clbeta_graph = [
            float(Clbeta_graph_1),
            float(Clbeta_graph_2),
            float(Clbeta_graph_3),
        ]
        Clbeta_CL_sweep = float(
            interpolate.interp1d([0.0, 0.5, 1.0], Clbeta_graph)(
                np.clip(taper_ratio, 0.0, 1.0)
            )
        )

        return Clbeta_CL_sweep

    @staticmethod
    def get_k_mach_sweep(mach, aspect_ratio, sweep_50):
        """
        Data from Roskam - Airplane Design Part VI to estimate the compressibility correction to wing sweep found in 
        Figure 10.21.

        :param mach: flight reference condition mach number
        :param aspect_ratio: surface aspect ratio
        :param sweep_50: surface mid-chord point line sweep angle in rdians.
        """

        ratio_aspect_ratio_sweep50 = aspect_ratio / math.cos(sweep_50)
        mach_sweep_50 = mach * math.cos(sweep_50)

        file = pth.join(digit_figures.__path__[0], "10_21.csv ")
        db = read_csv(file)

        x_ratio_2 = db["RATIO_AR_SWEEP_2_X"]
        y_ratio_2 = db["RATIO_AR_SWEEP_2_Y"]
        errors = np.logical_or(np.isnan(x_ratio_2), np.isnan(y_ratio_2))
        x_ratio_2 = x_ratio_2[np.logical_not(errors)].tolist()
        y_ratio_2 = y_ratio_2[np.logical_not(errors)].tolist()

        x_ratio_3 = db["RATIO_AR_SWEEP_3_X"]
        y_ratio_3 = db["RATIO_AR_SWEEP_3_Y"]
        errors = np.logical_or(np.isnan(x_ratio_3), np.isnan(y_ratio_3))
        x_ratio_3 = x_ratio_3[np.logical_not(errors)].tolist()
        y_ratio_3 = y_ratio_3[np.logical_not(errors)].tolist()

        x_ratio_4 = db["RATIO_AR_SWEEP_4_X"]
        y_ratio_4 = db["RATIO_AR_SWEEP_4_Y"]
        errors = np.logical_or(np.isnan(x_ratio_4), np.isnan(y_ratio_4))
        x_ratio_4 = x_ratio_4[np.logical_not(errors)].tolist()
        y_ratio_4 = y_ratio_4[np.logical_not(errors)].tolist()

        x_ratio_5 = db["RATIO_AR_SWEEP_5_X"]
        y_ratio_5 = db["RATIO_AR_SWEEP_5_Y"]
        errors = np.logical_or(np.isnan(x_ratio_5), np.isnan(y_ratio_5))
        x_ratio_5 = x_ratio_5[np.logical_not(errors)].tolist()
        y_ratio_5 = y_ratio_5[np.logical_not(errors)].tolist()

        x_ratio_6 = db["RATIO_AR_SWEEP_6_X"]
        y_ratio_6 = db["RATIO_AR_SWEEP_6_Y"]
        errors = np.logical_or(np.isnan(x_ratio_6), np.isnan(y_ratio_6))
        x_ratio_6 = x_ratio_6[np.logical_not(errors)].tolist()
        y_ratio_6 = y_ratio_6[np.logical_not(errors)].tolist()

        x_ratio_8 = db["RATIO_AR_SWEEP_8_X"]
        y_ratio_8 = db["RATIO_AR_SWEEP_8_Y"]
        errors = np.logical_or(np.isnan(x_ratio_8), np.isnan(y_ratio_8))
        x_ratio_8 = x_ratio_8[np.logical_not(errors)].tolist()
        y_ratio_8 = y_ratio_8[np.logical_not(errors)].tolist()

        x_ratio_10 = db["RATIO_AR_SWEEP_10_X"]
        y_ratio_10 = db["RATIO_AR_SWEEP_10_Y"]
        errors = np.logical_or(np.isnan(x_ratio_10), np.isnan(y_ratio_10))
        x_ratio_10 = x_ratio_10[np.logical_not(errors)].tolist()
        y_ratio_10 = y_ratio_10[np.logical_not(errors)].tolist()

        k_ratio_2 = interpolate.interp1d(x_ratio_2, y_ratio_2)
        k_ratio_3 = interpolate.interp1d(x_ratio_3, y_ratio_3)
        k_ratio_4 = interpolate.interp1d(x_ratio_4, y_ratio_4)
        k_ratio_5 = interpolate.interp1d(x_ratio_5, y_ratio_5)
        k_ratio_6 = interpolate.interp1d(x_ratio_6, y_ratio_6)
        k_ratio_8 = interpolate.interp1d(x_ratio_8, y_ratio_8)
        k_ratio_10 = interpolate.interp1d(x_ratio_10, y_ratio_10)

        if (
                (mach_sweep_50 != np.clip(mach_sweep_50, min(x_ratio_2), max(x_ratio_2)))
                or (mach_sweep_50 != np.clip(mach_sweep_50, min(x_ratio_3), max(x_ratio_3)))
                or (mach_sweep_50 != np.clip(mach_sweep_50, min(x_ratio_4), max(x_ratio_4)))
                or (mach_sweep_50 != np.clip(mach_sweep_50, min(x_ratio_5), max(x_ratio_5)))
                or (mach_sweep_50 != np.clip(mach_sweep_50, min(x_ratio_6), max(x_ratio_6)))
                or (mach_sweep_50 != np.clip(mach_sweep_50, min(x_ratio_8), max(x_ratio_8)))
                or (mach_sweep_50 != np.clip(mach_sweep_50, min(x_ratio_10), max(x_ratio_10)))

        ):
            _LOGGER.warning("Ratio between mach and sweep angle 50% value outside of the range in Roskam's book, "
                            "value clipped")

        k_ratio = [
            float(k_ratio_2(np.clip(mach_sweep_50, min(x_ratio_2), max(x_ratio_2)))),
            float(k_ratio_3(np.clip(mach_sweep_50, min(x_ratio_3), max(x_ratio_3)))),
            float(k_ratio_4(np.clip(mach_sweep_50, min(x_ratio_4), max(x_ratio_4)))),
            float(k_ratio_5(np.clip(mach_sweep_50, min(x_ratio_5), max(x_ratio_5)))),
            float(k_ratio_6(np.clip(mach_sweep_50, min(x_ratio_6), max(x_ratio_6)))),
            float(k_ratio_8(np.clip(mach_sweep_50, min(x_ratio_8), max(x_ratio_8)))),
            float(k_ratio_10(np.clip(mach_sweep_50, min(x_ratio_10), max(x_ratio_10)))),

        ]

        if ratio_aspect_ratio_sweep50 != np.clip(ratio_aspect_ratio_sweep50, 4.0, 8.0):
            _LOGGER.warning(
                "Ratio between aspect ratio and sweep angle outside of the range in Roskam's book, value clipped"
            )

        k_mach_sweep = float(
            interpolate.interp1d([2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0], k_ratio)(
                np.clip(ratio_aspect_ratio_sweep50, 2.0, 10.0)
            )
        )

        return k_mach_sweep

    @staticmethod
    def get_k_fuselage(l_f, b, aspect_ratio, sweep_50):
        """
        Data from Roskam - Airplane design Part VI to calculate de fuselage correction factor found in Figure 10.22.

        :param l_f: length of the fuselage from the cochpit tip to the end of the wing projection on the fuselage.
        :param b: wing or other surface span.
        :param aspect_ratio: wing or other surface aspect ratio.
        :param sweep_50: sweep angle of the mid-chord point line in radians.
        """

        ratio_aspect_ratio_sweep50 = aspect_ratio / math.cos(sweep_50)
        ratio_fuselage_span = l_f / b

        file = pth.join(digit_figures.__path__[0], "10_22.csv")
        db = read_csv(file)

        x_ratio_4 = db["RATIO_FUSELAGE_SPAN_4_X"]
        y_ratio_4 = db["RATIO_FUSELAGE_SPAN_4_Y"]
        errors = np.logical_or(np.isnan(x_ratio_4), np.isnan(y_ratio_4))
        x_ratio_4 = x_ratio_4[np.logical_not(errors)].tolist()
        y_ratio_4 = y_ratio_4[np.logical_not(errors)].tolist()

        x_ratio_4_5 = db["RATIO_FUSELAGE_SPAN_4.5_X"]
        y_ratio_4_5 = db["RATIO_FUSELAGE_SPAN_4.5_Y"]
        errors = np.logical_or(np.isnan(x_ratio_4_5), np.isnan(y_ratio_4_5))
        x_ratio_4_5 = x_ratio_4_5[np.logical_not(errors)].tolist()
        y_ratio_4_5 = y_ratio_4_5[np.logical_not(errors)].tolist()

        x_ratio_5_5 = db["RATIO_FUSELAGE_SPAN_5.5_X"]
        y_ratio_5_5 = db["RATIO_FUSELAGE_SPAN_5.5_Y"]
        errors = np.logical_or(np.isnan(x_ratio_5_5), np.isnan(y_ratio_5_5))
        x_ratio_5_5 = x_ratio_5_5[np.logical_not(errors)].tolist()
        y_ratio_5_5 = y_ratio_5_5[np.logical_not(errors)].tolist()

        x_ratio_6 = db["RATIO_FUSELAGE_SPAN_6_X"]
        y_ratio_6 = db["RATIO_FUSELAGE_SPAN_6_Y"]
        errors = np.logical_or(np.isnan(x_ratio_6), np.isnan(y_ratio_6))
        x_ratio_6 = x_ratio_6[np.logical_not(errors)].tolist()
        y_ratio_6 = y_ratio_6[np.logical_not(errors)].tolist()

        x_ratio_7 = db["RATIO_FUSELAGE_SPAN_7_X"]
        y_ratio_7 = db["RATIO_FUSELAGE_SPAN_7_Y"]
        errors = np.logical_or(np.isnan(x_ratio_7), np.isnan(y_ratio_7))
        x_ratio_7 = x_ratio_7[np.logical_not(errors)].tolist()
        y_ratio_7 = y_ratio_7[np.logical_not(errors)].tolist()

        x_ratio_8 = db["RATIO_FUSELAGE_SPAN_8_X"]
        y_ratio_8 = db["RATIO_FUSELAGE_SPAN_8_Y"]
        errors = np.logical_or(np.isnan(x_ratio_8), np.isnan(y_ratio_8))
        x_ratio_8 = x_ratio_8[np.logical_not(errors)].tolist()
        y_ratio_8 = y_ratio_8[np.logical_not(errors)].tolist()

        k_ratio_4 = interpolate.interp1d(x_ratio_4, y_ratio_4)
        k_ratio_4_5 = interpolate.interp1d(x_ratio_4_5, y_ratio_4_5)
        k_ratio_5_5 = interpolate.interp1d(x_ratio_5_5, y_ratio_5_5)
        k_ratio_6 = interpolate.interp1d(x_ratio_6, y_ratio_6)
        k_ratio_7 = interpolate.interp1d(x_ratio_7, y_ratio_7)
        k_ratio_8 = interpolate.interp1d(x_ratio_8, y_ratio_8)

        if (
                (ratio_fuselage_span != np.clip(ratio_fuselage_span, min(x_ratio_4), max(x_ratio_4)))
                or (ratio_fuselage_span != np.clip(ratio_fuselage_span, min(x_ratio_4_5), max(x_ratio_4_5)))
                or (ratio_fuselage_span != np.clip(ratio_fuselage_span, min(x_ratio_5_5), max(x_ratio_5_5)))
                or (ratio_fuselage_span != np.clip(ratio_fuselage_span, min(x_ratio_6), max(x_ratio_6)))
                or (ratio_fuselage_span != np.clip(ratio_fuselage_span, min(x_ratio_7), max(x_ratio_7)))
                or (ratio_fuselage_span != np.clip(ratio_fuselage_span, min(x_ratio_8), max(x_ratio_8)))
        ):
            _LOGGER.warning("Ratio between fuselage length and span outside of the range in Roskam's book, "
                            "value clipped")

        k_ratio = [
            float(k_ratio_4(np.clip(ratio_fuselage_span, min(x_ratio_4), max(x_ratio_4)))),
            float(k_ratio_4_5(np.clip(ratio_fuselage_span, min(x_ratio_4_5), max(x_ratio_4_5)))),
            float(k_ratio_5_5(np.clip(ratio_fuselage_span, min(x_ratio_5_5), max(x_ratio_5_5)))),
            float(k_ratio_6(np.clip(ratio_fuselage_span, min(x_ratio_6), max(x_ratio_6)))),
            float(k_ratio_7(np.clip(ratio_fuselage_span, min(x_ratio_7), max(x_ratio_7)))),
            float(k_ratio_8(np.clip(ratio_fuselage_span, min(x_ratio_8), max(x_ratio_8)))),

        ]

        if ratio_aspect_ratio_sweep50 != np.clip(ratio_aspect_ratio_sweep50, 4.0, 8.0):
            _LOGGER.warning(
                "Ratio of aspect ratio and sweep angle outside of the range in Roskam's book, value clipped"
            )

        k_fuselage = float(
            interpolate.interp1d([4.0, 4.5, 5.5, 6.0, 7.0, 8.0], k_ratio)(
                np.clip(ratio_aspect_ratio_sweep50, 4.0, 8.0)
            )
        )

        return k_fuselage

    @staticmethod
    def get_Clbeta_CL_aspect_ratio(aspect_ratio, taper_ratio):
        """
        Data frokm Roskam - Airplane Design Part VI to compute the aspect ratio contribution to rolling moment 
        side-slip derivative found in Figure 10.23.

        :param aspect_ratio: surface aspect ratio.
        :param taper_ratio: surface aspect ratio.
        """

        file = pth.join(digit_figures.__path__[0], "10_23.csv")
        db = read_csv(file)

        x_taperratio_0 = db["TAPER_RATIO_0_X"]
        y_taperratio_0 = db["TAPER_RATIO_0_Y"]
        errors = np.logical_or(np.isnan(x_taperratio_0), np.isnan(y_taperratio_0))
        x_taperratio_0 = x_taperratio_0[np.logical_not(errors)].tolist()
        y_taperratio_0 = y_taperratio_0[np.logical_not(errors)].tolist()

        x_taperratio_0_5 = db["TAPER_RATIO_0.5_X"]
        y_taperratio_0_5 = db["TAPER_RATIO_0.5_Y"]
        errors = np.logical_or(np.isnan(x_taperratio_0_5), np.isnan(y_taperratio_0_5))
        x_taperratio_0_5 = x_taperratio_0_5[np.logical_not(errors)].tolist()
        y_taperratio_0_5 = y_taperratio_0_5[np.logical_not(errors)].tolist()

        x_taperratio_1 = db["TAPER_RATIO_1_X"]
        y_taperratio_1 = db["TAPER_RATIO_1_Y"]
        errors = np.logical_or(np.isnan(x_taperratio_1), np.isnan(y_taperratio_1))
        x_taperratio_1 = x_taperratio_1[np.logical_not(errors)].tolist()
        y_taperratio_1 = y_taperratio_1[np.logical_not(errors)].tolist()

        k_taperratio_0 = interpolate.interp1d(x_taperratio_0, y_taperratio_0)
        k_taperratio_0_5 = interpolate.interp1d(x_taperratio_0_5, y_taperratio_0_5)
        k_taperratio_1 = interpolate.interp1d(x_taperratio_1, y_taperratio_1)

        if (
                (aspect_ratio != np.clip(aspect_ratio, min(x_taperratio_0), max(x_taperratio_0)))
                or (aspect_ratio != np.clip(aspect_ratio, min(x_taperratio_0_5), max(x_taperratio_0_5)))
                or (aspect_ratio != np.clip(aspect_ratio, min(x_taperratio_1), max(x_taperratio_1)))

        ):
            _LOGGER.warning("Aspect ratio value outside of the range in Roskam's book, value clipped")

        k_taperratio = [
            float(k_taperratio_0(np.clip(aspect_ratio, min(x_taperratio_0), max(x_taperratio_0)))),
            float(k_taperratio_0_5(np.clip(aspect_ratio, min(x_taperratio_0_5), max(x_taperratio_0_5)))),
            float(k_taperratio_1(np.clip(aspect_ratio, min(x_taperratio_1), max(x_taperratio_1)))),

        ]

        if taper_ratio != np.clip(taper_ratio, 0.0, 1.0):
            _LOGGER.warning(
                "Taper ratio value outside of the range in Roskam's book, value clipped"
            )

        Clbeta_CL_aspect_ratio = float(
            interpolate.interp1d([0.0, 0.5, 1.0], k_taperratio)(
                np.clip(taper_ratio, 0.0, 1.0)
            )
        )

        return Clbeta_CL_aspect_ratio

    @staticmethod
    def get_Clbeta_dihedral(aspect_ratio, sweep_50, taper_ratio):
        """
        Data from Roskam - Airplane Design Part VI to compute the wing geometric dihedral contribution to rolling
        moment due to sideslip found in Figure 10.24

        :param aspect_ratio: surface aspect ratio.
        :param sweep_50: mid-chord point line sweep angle.
        :param taper_ratio: surface taper ratio.
        """

        sweep_50 = sweep_50 * 180.0 / math.pi  # radians to degrees.
        sweep_50 = abs(sweep_50)

        # ----- GRAPH FOR TAPER RATIO = 0.0 -----
        file = pth.join(digit_figures.__path__[0], "10_24_0.csv")
        db = read_csv(file)

        x_sweep50_0_graph_1 = db["SWEEP_50_0_X"]
        y_sweep50_0_graph_1 = db["SWEEP_50_0_Y"]
        errors = np.logical_or(np.isnan(x_sweep50_0_graph_1), np.isnan(y_sweep50_0_graph_1))
        x_sweep50_0_graph_1 = x_sweep50_0_graph_1[np.logical_not(errors)].tolist()
        y_sweep50_0_graph_1 = y_sweep50_0_graph_1[np.logical_not(errors)].tolist()

        x_sweep50_40_graph_1 = db["SWEEP_50_40_X"]
        y_sweep50_40_graph_1 = db["SWEEP_50_40_Y"]
        errors = np.logical_or(np.isnan(x_sweep50_40_graph_1), np.isnan(y_sweep50_40_graph_1))
        x_sweep50_40_graph_1 = x_sweep50_40_graph_1[np.logical_not(errors)].tolist()
        y_sweep50_40_graph_1 = y_sweep50_40_graph_1[np.logical_not(errors)].tolist()

        x_sweep50_60_graph_1 = db["SWEEP_50_60_X"]
        y_sweep50_60_graph_1 = db["SWEEP_50_60_Y"]
        errors = np.logical_or(np.isnan(x_sweep50_60_graph_1), np.isnan(y_sweep50_60_graph_1))
        x_sweep50_60_graph_1 = x_sweep50_60_graph_1[np.logical_not(errors)].tolist()
        y_sweep50_60_graph_1 = y_sweep50_60_graph_1[np.logical_not(errors)].tolist()

        k_sweep50_0_graph_1 = interpolate.interp1d(x_sweep50_0_graph_1, y_sweep50_0_graph_1)
        k_sweep50_40_graph_1 = interpolate.interp1d(x_sweep50_40_graph_1, y_sweep50_40_graph_1)
        k_sweep50_60_graph_1 = interpolate.interp1d(x_sweep50_60_graph_1, y_sweep50_60_graph_1)

        if (
                (aspect_ratio != np.clip(aspect_ratio, min(x_sweep50_0_graph_1), max(x_sweep50_0_graph_1)))
                or (aspect_ratio != np.clip(aspect_ratio, min(x_sweep50_40_graph_1), max(x_sweep50_40_graph_1)))
                or (aspect_ratio != np.clip(aspect_ratio, min(x_sweep50_60_graph_1), max(x_sweep50_60_graph_1)))
        ):
            _LOGGER.warning("Aspect ratio value outside of the range in Roskam's book, value clipped")

        k_sweep_50_graph_1 = [
            float(k_sweep50_0_graph_1(np.clip(aspect_ratio, min(x_sweep50_0_graph_1), max(x_sweep50_0_graph_1)))),
            float(k_sweep50_40_graph_1(np.clip(aspect_ratio, min(x_sweep50_40_graph_1), max(x_sweep50_40_graph_1)))),
            float(k_sweep50_60_graph_1(np.clip(aspect_ratio, min(x_sweep50_60_graph_1), max(x_sweep50_60_graph_1)))),

        ]

        if sweep_50 != np.clip(sweep_50, 0.0, 60.0):
            _LOGGER.warning(
                "Sweep angle value outside of the range in Roskam's book (between -60 and 60 degrees), value clipped"
            )

        Clbeta_dihedral_graph_1 = float(
            interpolate.interp1d([0.0, 40.0, 60.0], k_sweep_50_graph_1)(
                np.clip(sweep_50, 0.0, 60.0)
            )
        )

        # ----- GRAPH FOR TAPER RATIO = 0.5 -----
        file = pth.join(digit_figures.__path__[0], "10_24_0_5.csv")
        db = read_csv(file)

        x_sweep50_0_graph_2 = db["SWEEP_50_0_X"]
        y_sweep50_0_graph_2 = db["SWEEP_50_0_Y"]
        errors = np.logical_or(np.isnan(x_sweep50_0_graph_2), np.isnan(y_sweep50_0_graph_2))
        x_sweep50_0_graph_2 = x_sweep50_0_graph_2[np.logical_not(errors)].tolist()
        y_sweep50_0_graph_2 = y_sweep50_0_graph_2[np.logical_not(errors)].tolist()

        x_sweep50_40_graph_2 = db["SWEEP_50_40_X"]
        y_sweep50_40_graph_2 = db["SWEEP_50_40_Y"]
        errors = np.logical_or(np.isnan(x_sweep50_40_graph_2), np.isnan(y_sweep50_40_graph_2))
        x_sweep50_40_graph_2 = x_sweep50_40_graph_2[np.logical_not(errors)].tolist()
        y_sweep50_40_graph_2 = y_sweep50_40_graph_2[np.logical_not(errors)].tolist()

        x_sweep50_60_graph_2 = db["SWEEP_50_60_X"]
        y_sweep50_60_graph_2 = db["SWEEP_50_60_Y"]
        errors = np.logical_or(np.isnan(x_sweep50_60_graph_2), np.isnan(y_sweep50_60_graph_2))
        x_sweep50_60_graph_2 = x_sweep50_60_graph_2[np.logical_not(errors)].tolist()
        y_sweep50_60_graph_2 = y_sweep50_60_graph_2[np.logical_not(errors)].tolist()

        k_sweep50_0_graph_2 = interpolate.interp1d(x_sweep50_0_graph_2, y_sweep50_0_graph_2)
        k_sweep50_40_graph_2 = interpolate.interp1d(x_sweep50_40_graph_2, y_sweep50_40_graph_2)
        k_sweep50_60_graph_2 = interpolate.interp1d(x_sweep50_60_graph_2, y_sweep50_60_graph_2)

        if (
                (aspect_ratio != np.clip(aspect_ratio, min(x_sweep50_0_graph_2), max(x_sweep50_0_graph_2)))
                or (aspect_ratio != np.clip(aspect_ratio, min(x_sweep50_40_graph_2), max(x_sweep50_40_graph_2)))
                or (aspect_ratio != np.clip(aspect_ratio, min(x_sweep50_60_graph_2), max(x_sweep50_60_graph_2)))
        ):
            _LOGGER.warning("Aspect ratio value outside of the range in Roskam's book, value clipped")

        k_sweep_50_graph_2 = [
            float(k_sweep50_0_graph_2(np.clip(aspect_ratio, min(x_sweep50_0_graph_2), max(x_sweep50_0_graph_2)))),
            float(k_sweep50_40_graph_2(np.clip(aspect_ratio, min(x_sweep50_40_graph_2), max(x_sweep50_40_graph_2)))),
            float(k_sweep50_60_graph_2(np.clip(aspect_ratio, min(x_sweep50_60_graph_2), max(x_sweep50_60_graph_2)))),
        ]

        if sweep_50 != np.clip(sweep_50, 0.0, 60.0):
            _LOGGER.warning(
                "Sweep angle value outside of the range in Roskam's book, value clipped"
            )

        Clbeta_dihedral_graph_2 = float(
            interpolate.interp1d([0.0, 40.0, 60.0], k_sweep_50_graph_2)(
                np.clip(sweep_50, 0.0, 60.0)
            )
        )

        # ----- GRAPH FOR TAPER RATIO = 1.0 -----
        file = pth.join(digit_figures.__path__[0], "10_24_1.csv")
        db = read_csv(file)

        x_sweep50_0_graph_3 = db["SWEEP_50_0_X"]
        y_sweep50_0_graph_3 = db["SWEEP_50_0_Y"]
        errors = np.logical_or(np.isnan(x_sweep50_0_graph_3), np.isnan(y_sweep50_0_graph_3))
        x_sweep50_0_graph_3 = x_sweep50_0_graph_3[np.logical_not(errors)].tolist()
        y_sweep50_0_graph_3 = y_sweep50_0_graph_3[np.logical_not(errors)].tolist()

        x_sweep50_40_graph_3 = db["SWEEP_50_40_X"]
        y_sweep50_40_graph_3 = db["SWEEP_50_40_Y"]
        errors = np.logical_or(np.isnan(x_sweep50_40_graph_3), np.isnan(y_sweep50_40_graph_3))
        x_sweep50_40_graph_3 = x_sweep50_40_graph_3[np.logical_not(errors)].tolist()
        y_sweep50_40_graph_3 = y_sweep50_40_graph_3[np.logical_not(errors)].tolist()

        x_sweep50_60_graph_3 = db["SWEEP_50_60_X"]
        y_sweep50_60_graph_3 = db["SWEEP_50_60_Y"]
        errors = np.logical_or(np.isnan(x_sweep50_60_graph_3), np.isnan(y_sweep50_60_graph_3))
        x_sweep50_60_graph_3 = x_sweep50_60_graph_3[np.logical_not(errors)].tolist()
        y_sweep50_60_graph_3 = y_sweep50_60_graph_3[np.logical_not(errors)].tolist()

        k_sweep50_0_graph_3 = interpolate.interp1d(x_sweep50_0_graph_3, y_sweep50_0_graph_3)
        k_sweep50_40_graph_3 = interpolate.interp1d(x_sweep50_40_graph_3, y_sweep50_40_graph_3)
        k_sweep50_60_graph_3 = interpolate.interp1d(x_sweep50_60_graph_3, y_sweep50_60_graph_3)

        if (
                (aspect_ratio != np.clip(aspect_ratio, min(x_sweep50_0_graph_3), max(x_sweep50_0_graph_3)))
                or (aspect_ratio != np.clip(aspect_ratio, min(x_sweep50_40_graph_3), max(x_sweep50_40_graph_3)))
                or (aspect_ratio != np.clip(aspect_ratio, min(x_sweep50_60_graph_3), max(x_sweep50_60_graph_3)))
        ):
            _LOGGER.warning("Aspect ratio value outside of the range in Roskam's book, value clipped")

        k_sweep_50_graph_3 = [
            float(k_sweep50_0_graph_3(np.clip(aspect_ratio, min(x_sweep50_0_graph_3), max(x_sweep50_0_graph_3)))),
            float(k_sweep50_40_graph_3(np.clip(aspect_ratio, min(x_sweep50_40_graph_3), max(x_sweep50_40_graph_3)))),
            float(k_sweep50_60_graph_3(np.clip(aspect_ratio, min(x_sweep50_60_graph_3), max(x_sweep50_60_graph_3)))),

        ]

        if sweep_50 != np.clip(sweep_50, 0.0, 60.0):
            _LOGGER.warning(
                "Sweep angle value outside of the range in Roskam's book, value clipped"
            )

        Clbeta_dihedral_graph_3 = float(
            interpolate.interp1d([0.0, 40.0, 60.0], k_sweep_50_graph_3)(
                np.clip(sweep_50, 0.0, 60.0)
            )
        )

        # INTERPOLATION BETWEEN THE 3 GRAPHS
        Clbeta_dihedral_graph = [
            float(Clbeta_dihedral_graph_1),
            float(Clbeta_dihedral_graph_2),
            float(Clbeta_dihedral_graph_3),
        ]
        Clbeta_dihedral = float(
            interpolate.interp1d([0.0, 0.5, 1.0], Clbeta_dihedral_graph)(
                np.clip(taper_ratio, 0.0, 1.0)
            )
        )

        return Clbeta_dihedral

    @staticmethod
    def get_k_mach_dihedral(mach, aspect_ratio, sweep_50):
        """
        Data from Roskam - Airplane Design Part VI to estimate the compressibility correction to wing dihedral found in
        Figure 10.25.
        
        :param mach: flight reference condition mach number.
        :param aspect_ratio: surface aspect ratio.
        :param sweep_50: surface mid-chord point line sweep angle in radians.
        """

        ratio_aspect_ratio_sweep50 = aspect_ratio / math.cos(sweep_50)
        mach_sweep_50 = mach * math.cos(sweep_50)

        file = pth.join(digit_figures.__path__[0], "10_25.csv")
        db = read_csv(file)

        x_ratio_2 = db["RATIO_AR_SWEEP_2_X"]
        y_ratio_2 = db["RATIO_AR_SWEEP_2_Y"]
        errors = np.logical_or(np.isnan(x_ratio_2), np.isnan(y_ratio_2))
        x_ratio_2 = x_ratio_2[np.logical_not(errors)].tolist()
        y_ratio_2 = y_ratio_2[np.logical_not(errors)].tolist()

        x_ratio_4 = db["RATIO_AR_SWEEP_4_X"]
        y_ratio_4 = db["RATIO_AR_SWEEP_4_Y"]
        errors = np.logical_or(np.isnan(x_ratio_4), np.isnan(y_ratio_4))
        x_ratio_4 = x_ratio_4[np.logical_not(errors)].tolist()
        y_ratio_4 = y_ratio_4[np.logical_not(errors)].tolist()

        x_ratio_6 = db["RATIO_AR_SWEEP_6_X"]
        y_ratio_6 = db["RATIO_AR_SWEEP_6_Y"]
        errors = np.logical_or(np.isnan(x_ratio_6), np.isnan(y_ratio_6))
        x_ratio_6 = x_ratio_6[np.logical_not(errors)].tolist()
        y_ratio_6 = y_ratio_6[np.logical_not(errors)].tolist()

        x_ratio_8 = db["RATIO_AR_SWEEP_8_X"]
        y_ratio_8 = db["RATIO_AR_SWEEP_8_Y"]
        errors = np.logical_or(np.isnan(x_ratio_8), np.isnan(y_ratio_8))
        x_ratio_8 = x_ratio_8[np.logical_not(errors)].tolist()
        y_ratio_8 = y_ratio_8[np.logical_not(errors)].tolist()

        x_ratio_10 = db["RATIO_AR_SWEEP_10_X"]
        y_ratio_10 = db["RATIO_AR_SWEEP_10_Y"]
        errors = np.logical_or(np.isnan(x_ratio_10), np.isnan(y_ratio_10))
        x_ratio_10 = x_ratio_10[np.logical_not(errors)].tolist()
        y_ratio_10 = y_ratio_10[np.logical_not(errors)].tolist()

        k_ratio_2 = interpolate.interp1d(x_ratio_2, y_ratio_2)
        k_ratio_4 = interpolate.interp1d(x_ratio_4, y_ratio_4)
        k_ratio_6 = interpolate.interp1d(x_ratio_6, y_ratio_6)
        k_ratio_8 = interpolate.interp1d(x_ratio_8, y_ratio_8)
        k_ratio_10 = interpolate.interp1d(x_ratio_10, y_ratio_10)

        if (
                (mach_sweep_50 != np.clip(mach_sweep_50, min(x_ratio_2), max(x_ratio_2)))
                or (mach_sweep_50 != np.clip(mach_sweep_50, min(x_ratio_4), max(x_ratio_4)))
                or (mach_sweep_50 != np.clip(mach_sweep_50, min(x_ratio_6), max(x_ratio_6)))
                or (mach_sweep_50 != np.clip(mach_sweep_50, min(x_ratio_8), max(x_ratio_8)))
                or (mach_sweep_50 != np.clip(mach_sweep_50, min(x_ratio_10), max(x_ratio_10)))

        ):
            _LOGGER.warning("Ratio between mach and sweep angle 50% value outside of the range in Roskam's book, "
                            "value clipped")

        k_ratio = [
            float(k_ratio_2(np.clip(mach_sweep_50, min(x_ratio_2), max(x_ratio_2)))),
            float(k_ratio_4(np.clip(mach_sweep_50, min(x_ratio_4), max(x_ratio_4)))),
            float(k_ratio_6(np.clip(mach_sweep_50, min(x_ratio_6), max(x_ratio_6)))),
            float(k_ratio_8(np.clip(mach_sweep_50, min(x_ratio_8), max(x_ratio_8)))),
            float(k_ratio_10(np.clip(mach_sweep_50, min(x_ratio_10), max(x_ratio_10)))),

        ]

        if ratio_aspect_ratio_sweep50 != np.clip(ratio_aspect_ratio_sweep50, 2.0, 10.0):
            _LOGGER.warning(
                "Ratio between aspect ratio and sweep angle outside of the range in Roskam's book, value clipped"
            )

        k_mach_dihedral = float(
            interpolate.interp1d([2.0, 4.0, 6.0, 8.0, 10.0], k_ratio)(
                np.clip(ratio_aspect_ratio_sweep50, 2.0, 10.0)
            )
        )

        return k_mach_dihedral

    @staticmethod
    def get_delta_Clbeta_twist(aspect_ratio, taper_ratio):
        """
        Data from Roskam - Airplane Design Part VI to compute the contribution of wing twist to rolling moment due to
        sideslip found in Figure 10.26.

        :param aspect_ratio: surface aspect ratio.
        :param taper_ratio: surface taper ratio.
        """

        file = pth.join(digit_figures.__path__[0], "10_26.csv")
        db = read_csv(file)

        x_taperratio_0 = db["TAPER_RATIO_0_X"]
        y_taperratio_0 = db["TAPER_RATIO_0_Y"]
        errors = np.logical_or(np.isnan(x_taperratio_0), np.isnan(y_taperratio_0))
        x_taperratio_0 = x_taperratio_0[np.logical_not(errors)].tolist()
        y_taperratio_0 = y_taperratio_0[np.logical_not(errors)].tolist()

        x_taperratio_0_4 = db["TAPER_RATIO_0.4_X"]
        y_taperratio_0_4 = db["TAPER_RATIO_0.4_Y"]
        errors = np.logical_or(np.isnan(x_taperratio_0_4), np.isnan(y_taperratio_0_4))
        x_taperratio_0_4 = x_taperratio_0_4[np.logical_not(errors)].tolist()
        y_taperratio_0_4 = y_taperratio_0_4[np.logical_not(errors)].tolist()

        x_taperratio_0_6 = db["TAPER_RATIO_0.6_X"]
        y_taperratio_0_6 = db["TAPER_RATIO_0.6_Y"]
        errors = np.logical_or(np.isnan(x_taperratio_0_6), np.isnan(y_taperratio_0_6))
        x_taperratio_0_6 = x_taperratio_0_6[np.logical_not(errors)].tolist()
        y_taperratio_0_6 = y_taperratio_0_6[np.logical_not(errors)].tolist()

        k_taperratio_0 = interpolate.interp1d(x_taperratio_0, y_taperratio_0)
        k_taperratio_0_4 = interpolate.interp1d(x_taperratio_0_4, y_taperratio_0_4)
        k_taperratio_0_6 = interpolate.interp1d(x_taperratio_0_6, y_taperratio_0_6)

        if (
                (aspect_ratio != np.clip(aspect_ratio, min(x_taperratio_0), max(x_taperratio_0)))
                or (aspect_ratio != np.clip(aspect_ratio, min(x_taperratio_0_4), max(x_taperratio_0_4)))
                or (aspect_ratio != np.clip(aspect_ratio, min(x_taperratio_0_6), max(x_taperratio_0_6)))

        ):
            _LOGGER.warning("Aspect ratio value outside of the range in Roskam's book, value clipped")

        k_taperratio = [
            float(k_taperratio_0(np.clip(aspect_ratio, min(x_taperratio_0), max(x_taperratio_0)))),
            float(k_taperratio_0_4(np.clip(aspect_ratio, min(x_taperratio_0_4), max(x_taperratio_0_4)))),
            float(k_taperratio_0_6(np.clip(aspect_ratio, min(x_taperratio_0_6), max(x_taperratio_0_6)))),

        ]

        if taper_ratio != np.clip(taper_ratio, 0.0, 0.6):
            _LOGGER.warning(
                "Taper ratio value outside of the range in Roskam's book, value clipped"
            )

        delta_Clbeta_twist = float(
            interpolate.interp1d([0.0, 0.4, 0.6], k_taperratio)(
                np.clip(taper_ratio, 0.0, 0.6)
            )
        )

        return delta_Clbeta_twist

    @staticmethod
    def get_beta_Clp_k(mach, k, sweep_25, aspect_ratio, taper_ratio):
        """
        Data from Roskam - Airplane Design Part VI to compute a surface roll damping parameter found in Figure 10.35

        :param mach: mach number in reference flight conditions
        :param k: induced drag coefficient of the surface
        :param sweep_25: quarter-chord point line sweep angle of the surface in radians.
        :param aspect_ratio: surface aspect ratio.
        :param taper_ratio: surface taper ratio.
        """
        # quotient = beta_A_k
        beta = math.sqrt(1 - mach ** 2)
        beta_A_k = beta * aspect_ratio / k
        delta_beta = math.atan(math.tan(sweep_25) / beta)

        # ----- GRAPH 1 -----
        file = pth.join(digit_figures.__path__[0], "10_35_0.csv")
        db = read_csv(file)

        x_quotient_1_5_graph_1 = db["BETA_A_K_1.5_X"]
        y_quotient_1_5_graph_1 = db["BETA_A_K_1.5_Y"]
        errors = np.logical_or(np.isnan(x_quotient_1_5_graph_1), np.isnan(y_quotient_1_5_graph_1))
        x_quotient_1_5_graph_1 = x_quotient_1_5_graph_1[np.logical_not(errors)].tolist()
        y_quotient_1_5_graph_1 = y_quotient_1_5_graph_1[np.logical_not(errors)].tolist()

        x_quotient_3_graph_1 = db["BETA_A_K_3_X"]
        y_quotient_3_graph_1 = db["BETA_A_K_3_Y"]
        errors = np.logical_or(np.isnan(x_quotient_3_graph_1), np.isnan(y_quotient_3_graph_1))
        x_quotient_3_graph_1 = x_quotient_3_graph_1[np.logical_not(errors)].tolist()
        y_quotient_3_graph_1 = y_quotient_3_graph_1[np.logical_not(errors)].tolist()

        x_quotient_4_5_graph_1 = db["BETA_A_K_4.5_X"]
        y_quotient_4_5_graph_1 = db["BETA_A_K_4.5_Y"]
        errors = np.logical_or(np.isnan(x_quotient_4_5_graph_1), np.isnan(y_quotient_4_5_graph_1))
        x_quotient_4_5_graph_1 = x_quotient_4_5_graph_1[np.logical_not(errors)].tolist()
        y_quotient_4_5_graph_1 = y_quotient_4_5_graph_1[np.logical_not(errors)].tolist()

        x_quotient_6_graph_1 = db["BETA_A_K_6_X"]
        y_quotient_6_graph_1 = db["BETA_A_K_6_Y"]
        errors = np.logical_or(np.isnan(x_quotient_6_graph_1), np.isnan(y_quotient_6_graph_1))
        x_quotient_6_graph_1 = x_quotient_6_graph_1[np.logical_not(errors)].tolist()
        y_quotient_6_graph_1 = y_quotient_6_graph_1[np.logical_not(errors)].tolist()

        x_quotient_8_graph_1 = db["BETA_A_K_8_X"]
        y_quotient_8_graph_1 = db["BETA_A_K_8_Y"]
        errors = np.logical_or(np.isnan(x_quotient_8_graph_1), np.isnan(y_quotient_8_graph_1))
        x_quotient_8_graph_1 = x_quotient_8_graph_1[np.logical_not(errors)].tolist()
        y_quotient_8_graph_1 = y_quotient_8_graph_1[np.logical_not(errors)].tolist()

        x_quotient_10_graph_1 = db["BETA_A_K_10_X"]
        y_quotient_10_graph_1 = db["BETA_A_K_10_Y"]
        errors = np.logical_or(np.isnan(x_quotient_10_graph_1), np.isnan(y_quotient_10_graph_1))
        x_quotient_10_graph_1 = x_quotient_10_graph_1[np.logical_not(errors)].tolist()
        y_quotient_10_graph_1 = y_quotient_10_graph_1[np.logical_not(errors)].tolist()

        k_quotient_1_5_graph_1 = interpolate.interp1d(x_quotient_1_5_graph_1, y_quotient_1_5_graph_1)
        k_quotient_3_graph_1 = interpolate.interp1d(x_quotient_3_graph_1, y_quotient_3_graph_1)
        k_quotient_4_5_graph_1 = interpolate.interp1d(x_quotient_4_5_graph_1, y_quotient_4_5_graph_1)
        k_quotient_6_graph_1 = interpolate.interp1d(x_quotient_6_graph_1, y_quotient_6_graph_1)
        k_quotient_8_graph_1 = interpolate.interp1d(x_quotient_8_graph_1, y_quotient_8_graph_1)
        k_quotient_10_graph_1 = interpolate.interp1d(x_quotient_10_graph_1, y_quotient_10_graph_1)

        if (
                (delta_beta != np.clip(delta_beta, min(x_quotient_1_5_graph_1), max(x_quotient_1_5_graph_1)))
                or (delta_beta != np.clip(delta_beta, min(x_quotient_3_graph_1), max(x_quotient_3_graph_1)))
                or (delta_beta != np.clip(delta_beta, min(x_quotient_4_5_graph_1), max(x_quotient_4_5_graph_1)))
                or (delta_beta != np.clip(delta_beta, min(x_quotient_6_graph_1), max(x_quotient_6_graph_1)))
                or (delta_beta != np.clip(delta_beta, min(x_quotient_8_graph_1), max(x_quotient_8_graph_1)))
                or (delta_beta != np.clip(delta_beta, min(x_quotient_10_graph_1), max(x_quotient_10_graph_1)))
        ):
            _LOGGER.warning("Value outside of the range in Roskam's book, value clipped")

        k_quotient_graph_1 = [
            float(
                k_quotient_1_5_graph_1(np.clip(delta_beta, min(x_quotient_1_5_graph_1), max(x_quotient_1_5_graph_1)))),
            float(k_quotient_3_graph_1(np.clip(delta_beta, min(x_quotient_3_graph_1), max(x_quotient_3_graph_1)))),
            float(
                k_quotient_4_5_graph_1(np.clip(delta_beta, min(x_quotient_4_5_graph_1), max(x_quotient_4_5_graph_1)))),
            float(k_quotient_6_graph_1(np.clip(delta_beta, min(x_quotient_6_graph_1), max(x_quotient_6_graph_1)))),
            float(k_quotient_8_graph_1(np.clip(delta_beta, min(x_quotient_8_graph_1), max(x_quotient_8_graph_1)))),
            float(k_quotient_10_graph_1(np.clip(delta_beta, min(x_quotient_10_graph_1), max(x_quotient_10_graph_1)))),
        ]

        if beta_A_k != np.clip(beta_A_k, 1.5, 10.0):
            _LOGGER.warning(
                "Value outside of the range in Roskam's book, value clipped"
            )

        beta_Clp_k_graph_1 = float(
            interpolate.interp1d([1.5, 3.0, 4.5, 6, 8, 10], k_quotient_graph_1)(
                np.clip(beta_A_k, 1.5, 10.0)
            )
        )

        # ----- GRAPH 2 -----
        file = pth.join(digit_figures.__path__[0], "10_35_0_25.csv")
        db = read_csv(file)

        x_quotient_1_5_graph_2 = db["BETA_A_K_1.5_X"]
        y_quotient_1_5_graph_2 = db["BETA_A_K_1.5_Y"]
        errors = np.logical_or(np.isnan(x_quotient_1_5_graph_2), np.isnan(y_quotient_1_5_graph_2))
        x_quotient_1_5_graph_2 = x_quotient_1_5_graph_2[np.logical_not(errors)].tolist()
        y_quotient_1_5_graph_2 = y_quotient_1_5_graph_2[np.logical_not(errors)].tolist()

        x_quotient_3_graph_2 = db["BETA_A_K_3_X"]
        y_quotient_3_graph_2 = db["BETA_A_K_3_Y"]
        errors = np.logical_or(np.isnan(x_quotient_3_graph_2), np.isnan(y_quotient_3_graph_2))
        x_quotient_3_graph_2 = x_quotient_3_graph_2[np.logical_not(errors)].tolist()
        y_quotient_3_graph_2 = y_quotient_3_graph_2[np.logical_not(errors)].tolist()

        x_quotient_4_5_graph_2 = db["BETA_A_K_4.5_X"]
        y_quotient_4_5_graph_2 = db["BETA_A_K_4.5_Y"]
        errors = np.logical_or(np.isnan(x_quotient_4_5_graph_2), np.isnan(y_quotient_4_5_graph_2))
        x_quotient_4_5_graph_2 = x_quotient_4_5_graph_2[np.logical_not(errors)].tolist()
        y_quotient_4_5_graph_2 = y_quotient_4_5_graph_2[np.logical_not(errors)].tolist()

        x_quotient_6_graph_2 = db["BETA_A_K_6_X"]
        y_quotient_6_graph_2 = db["BETA_A_K_6_Y"]
        errors = np.logical_or(np.isnan(x_quotient_6_graph_2), np.isnan(y_quotient_6_graph_2))
        x_quotient_6_graph_2 = x_quotient_6_graph_2[np.logical_not(errors)].tolist()
        y_quotient_6_graph_2 = y_quotient_6_graph_2[np.logical_not(errors)].tolist()

        x_quotient_8_graph_2 = db["BETA_A_K_8_X"]
        y_quotient_8_graph_2 = db["BETA_A_K_8_Y"]
        errors = np.logical_or(np.isnan(x_quotient_8_graph_2), np.isnan(y_quotient_8_graph_2))
        x_quotient_8_graph_2 = x_quotient_8_graph_2[np.logical_not(errors)].tolist()
        y_quotient_8_graph_2 = y_quotient_8_graph_2[np.logical_not(errors)].tolist()

        x_quotient_10_graph_2 = db["BETA_A_K_10_X"]
        y_quotient_10_graph_2 = db["BETA_A_K_10_Y"]
        errors = np.logical_or(np.isnan(x_quotient_10_graph_2), np.isnan(y_quotient_10_graph_2))
        x_quotient_10_graph_2 = x_quotient_10_graph_2[np.logical_not(errors)].tolist()
        y_quotient_10_graph_2 = y_quotient_10_graph_2[np.logical_not(errors)].tolist()

        k_quotient_1_5_graph_2 = interpolate.interp1d(x_quotient_1_5_graph_2, y_quotient_1_5_graph_2)
        k_quotient_3_graph_2 = interpolate.interp1d(x_quotient_3_graph_2, y_quotient_3_graph_2)
        k_quotient_4_5_graph_2 = interpolate.interp1d(x_quotient_4_5_graph_2, y_quotient_4_5_graph_2)
        k_quotient_6_graph_2 = interpolate.interp1d(x_quotient_6_graph_2, y_quotient_6_graph_2)
        k_quotient_8_graph_2 = interpolate.interp1d(x_quotient_8_graph_2, y_quotient_8_graph_2)
        k_quotient_10_graph_2 = interpolate.interp1d(x_quotient_10_graph_2, y_quotient_10_graph_2)

        if (
                (delta_beta != np.clip(delta_beta, min(x_quotient_1_5_graph_2), max(x_quotient_1_5_graph_2)))
                or (delta_beta != np.clip(delta_beta, min(x_quotient_3_graph_2), max(x_quotient_3_graph_2)))
                or (delta_beta != np.clip(delta_beta, min(x_quotient_4_5_graph_2), max(x_quotient_4_5_graph_2)))
                or (delta_beta != np.clip(delta_beta, min(x_quotient_6_graph_2), max(x_quotient_6_graph_2)))
                or (delta_beta != np.clip(delta_beta, min(x_quotient_8_graph_2), max(x_quotient_8_graph_2)))
                or (delta_beta != np.clip(delta_beta, min(x_quotient_10_graph_2), max(x_quotient_10_graph_2)))
        ):
            _LOGGER.warning("Value outside of the range in Roskam's book, value clipped")

        k_quotient_graph_2 = [
            float(
                k_quotient_1_5_graph_2(np.clip(delta_beta, min(x_quotient_1_5_graph_2), max(x_quotient_1_5_graph_2)))),
            float(k_quotient_3_graph_2(np.clip(delta_beta, min(x_quotient_3_graph_2), max(x_quotient_3_graph_2)))),
            float(
                k_quotient_4_5_graph_2(np.clip(delta_beta, min(x_quotient_4_5_graph_2), max(x_quotient_4_5_graph_2)))),
            float(k_quotient_6_graph_2(np.clip(delta_beta, min(x_quotient_6_graph_2), max(x_quotient_6_graph_2)))),
            float(k_quotient_8_graph_2(np.clip(delta_beta, min(x_quotient_8_graph_2), max(x_quotient_8_graph_2)))),
            float(k_quotient_10_graph_2(np.clip(delta_beta, min(x_quotient_10_graph_2), max(x_quotient_10_graph_2)))),

        ]

        if beta_A_k != np.clip(beta_A_k, 1.5, 10.0):
            _LOGGER.warning(
                "Value outside of the range in Roskam's book, value clipped"
            )

        beta_Clp_k_graph_2 = float(
            interpolate.interp1d([1.5, 3.0, 4.5, 6, 8, 10], k_quotient_graph_2)(
                np.clip(beta_A_k, 1.5, 10.0)
            )
        )

        # ----- GRAPH 3 -----
        file = pth.join(digit_figures.__path__[0], "10_35_0_5.csv")
        db = read_csv(file)

        x_quotient_1_5_graph_3 = db["BETA_A_K_1.5_X"]
        y_quotient_1_5_graph_3 = db["BETA_A_K_1.5_Y"]
        errors = np.logical_or(np.isnan(x_quotient_1_5_graph_3), np.isnan(y_quotient_1_5_graph_3))
        x_quotient_1_5_graph_3 = x_quotient_1_5_graph_3[np.logical_not(errors)].tolist()
        y_quotient_1_5_graph_3 = y_quotient_1_5_graph_3[np.logical_not(errors)].tolist()

        x_quotient_3_graph_3 = db["BETA_A_K_3_X"]
        y_quotient_3_graph_3 = db["BETA_A_K_3_Y"]
        errors = np.logical_or(np.isnan(x_quotient_3_graph_3), np.isnan(y_quotient_3_graph_3))
        x_quotient_3_graph_3 = x_quotient_3_graph_3[np.logical_not(errors)].tolist()
        y_quotient_3_graph_3 = y_quotient_3_graph_3[np.logical_not(errors)].tolist()

        x_quotient_4_5_graph_3 = db["BETA_A_K_4.5_X"]
        y_quotient_4_5_graph_3 = db["BETA_A_K_4.5_Y"]
        errors = np.logical_or(np.isnan(x_quotient_4_5_graph_3), np.isnan(y_quotient_4_5_graph_3))
        x_quotient_4_5_graph_3 = x_quotient_4_5_graph_3[np.logical_not(errors)].tolist()
        y_quotient_4_5_graph_3 = y_quotient_4_5_graph_3[np.logical_not(errors)].tolist()

        x_quotient_6_graph_3 = db["BETA_A_K_6_X"]
        y_quotient_6_graph_3 = db["BETA_A_K_6_Y"]
        errors = np.logical_or(np.isnan(x_quotient_6_graph_3), np.isnan(y_quotient_6_graph_3))
        x_quotient_6_graph_3 = x_quotient_6_graph_3[np.logical_not(errors)].tolist()
        y_quotient_6_graph_3 = y_quotient_6_graph_3[np.logical_not(errors)].tolist()

        x_quotient_8_graph_3 = db["BETA_A_K_8_X"]
        y_quotient_8_graph_3 = db["BETA_A_K_8_Y"]
        errors = np.logical_or(np.isnan(x_quotient_8_graph_3), np.isnan(y_quotient_8_graph_3))
        x_quotient_8_graph_3 = x_quotient_8_graph_3[np.logical_not(errors)].tolist()
        y_quotient_8_graph_3 = y_quotient_8_graph_3[np.logical_not(errors)].tolist()

        x_quotient_10_graph_3 = db["BETA_A_K_10_X"]
        y_quotient_10_graph_3 = db["BETA_A_K_10_Y"]
        errors = np.logical_or(np.isnan(x_quotient_10_graph_3), np.isnan(y_quotient_10_graph_3))
        x_quotient_10_graph_3 = x_quotient_10_graph_3[np.logical_not(errors)].tolist()
        y_quotient_10_graph_3 = y_quotient_10_graph_3[np.logical_not(errors)].tolist()

        k_quotient_1_5_graph_3 = interpolate.interp1d(x_quotient_1_5_graph_3, y_quotient_1_5_graph_3)
        k_quotient_3_graph_3 = interpolate.interp1d(x_quotient_3_graph_3, y_quotient_3_graph_3)
        k_quotient_4_5_graph_3 = interpolate.interp1d(x_quotient_4_5_graph_3, y_quotient_4_5_graph_3)
        k_quotient_6_graph_3 = interpolate.interp1d(x_quotient_6_graph_3, y_quotient_6_graph_3)
        k_quotient_8_graph_3 = interpolate.interp1d(x_quotient_8_graph_3, y_quotient_8_graph_3)
        k_quotient_10_graph_3 = interpolate.interp1d(x_quotient_10_graph_3, y_quotient_10_graph_3)

        if (
                (delta_beta != np.clip(delta_beta, min(x_quotient_1_5_graph_3), max(x_quotient_1_5_graph_3)))
                or (delta_beta != np.clip(delta_beta, min(x_quotient_3_graph_3), max(x_quotient_3_graph_3)))
                or (delta_beta != np.clip(delta_beta, min(x_quotient_4_5_graph_3), max(x_quotient_4_5_graph_3)))
                or (delta_beta != np.clip(delta_beta, min(x_quotient_6_graph_3), max(x_quotient_6_graph_3)))
                or (delta_beta != np.clip(delta_beta, min(x_quotient_8_graph_3), max(x_quotient_8_graph_3)))
                or (delta_beta != np.clip(delta_beta, min(x_quotient_10_graph_3), max(x_quotient_10_graph_3)))
        ):
            _LOGGER.warning("Value outside of the range in Roskam's book, value clipped")

        k_quotient_graph_3 = [
            float(
                k_quotient_1_5_graph_3(np.clip(delta_beta, min(x_quotient_1_5_graph_3), max(x_quotient_1_5_graph_3)))),
            float(k_quotient_3_graph_3(np.clip(delta_beta, min(x_quotient_3_graph_3), max(x_quotient_3_graph_3)))),
            float(
                k_quotient_4_5_graph_3(np.clip(delta_beta, min(x_quotient_4_5_graph_3), max(x_quotient_4_5_graph_3)))),
            float(k_quotient_6_graph_3(np.clip(delta_beta, min(x_quotient_6_graph_3), max(x_quotient_6_graph_3)))),
            float(k_quotient_8_graph_3(np.clip(delta_beta, min(x_quotient_8_graph_3), max(x_quotient_8_graph_3)))),
            float(k_quotient_10_graph_3(np.clip(delta_beta, min(x_quotient_10_graph_3), max(x_quotient_10_graph_3)))),
        ]

        if beta_A_k != np.clip(beta_A_k, 1.5, 10.0):
            _LOGGER.warning(
                "Value outside of the range in Roskam's book, value clipped"
            )

        beta_Clp_k_graph_3 = float(
            interpolate.interp1d([1.5, 3.0, 4.5, 6, 8, 10], k_quotient_graph_3)(
                np.clip(beta_A_k, 1.5, 10.0)
            )
        )

        # ----- GRAPH 4 -----
        file = pth.join(digit_figures.__path__[0], "10_35_1.csv")
        db = read_csv(file)

        x_quotient_1_5_graph_4 = db["BETA_A_K_1.5_X"]
        y_quotient_1_5_graph_4 = db["BETA_A_K_1.5_Y"]
        errors = np.logical_or(np.isnan(x_quotient_1_5_graph_4), np.isnan(y_quotient_1_5_graph_4))
        x_quotient_1_5_graph_4 = x_quotient_1_5_graph_4[np.logical_not(errors)].tolist()
        y_quotient_1_5_graph_4 = y_quotient_1_5_graph_4[np.logical_not(errors)].tolist()

        x_quotient_3_graph_4 = db["BETA_A_K_3_X"]
        y_quotient_3_graph_4 = db["BETA_A_K_3_Y"]
        errors = np.logical_or(np.isnan(x_quotient_3_graph_4), np.isnan(y_quotient_3_graph_4))
        x_quotient_3_graph_4 = x_quotient_3_graph_4[np.logical_not(errors)].tolist()
        y_quotient_3_graph_4 = y_quotient_3_graph_4[np.logical_not(errors)].tolist()

        x_quotient_4_5_graph_4 = db["BETA_A_K_4.5_X"]
        y_quotient_4_5_graph_4 = db["BETA_A_K_4.5_Y"]
        errors = np.logical_or(np.isnan(x_quotient_4_5_graph_4), np.isnan(y_quotient_4_5_graph_4))
        x_quotient_4_5_graph_4 = x_quotient_4_5_graph_4[np.logical_not(errors)].tolist()
        y_quotient_4_5_graph_4 = y_quotient_4_5_graph_4[np.logical_not(errors)].tolist()

        x_quotient_6_graph_4 = db["BETA_A_K_6_X"]
        y_quotient_6_graph_4 = db["BETA_A_K_6_Y"]
        errors = np.logical_or(np.isnan(x_quotient_6_graph_4), np.isnan(y_quotient_6_graph_4))
        x_quotient_6_graph_4 = x_quotient_6_graph_4[np.logical_not(errors)].tolist()
        y_quotient_6_graph_4 = y_quotient_6_graph_4[np.logical_not(errors)].tolist()

        x_quotient_8_graph_4 = db["BETA_A_K_8_X"]
        y_quotient_8_graph_4 = db["BETA_A_K_8_Y"]
        errors = np.logical_or(np.isnan(x_quotient_8_graph_4), np.isnan(y_quotient_8_graph_4))
        x_quotient_8_graph_4 = x_quotient_8_graph_4[np.logical_not(errors)].tolist()
        y_quotient_8_graph_4 = y_quotient_8_graph_4[np.logical_not(errors)].tolist()

        x_quotient_10_graph_4 = db["BETA_A_K_10_X"]
        y_quotient_10_graph_4 = db["BETA_A_K_10_Y"]
        errors = np.logical_or(np.isnan(x_quotient_10_graph_4), np.isnan(y_quotient_10_graph_4))
        x_quotient_10_graph_4 = x_quotient_10_graph_4[np.logical_not(errors)].tolist()
        y_quotient_10_graph_4 = y_quotient_10_graph_4[np.logical_not(errors)].tolist()

        k_quotient_1_5_graph_4 = interpolate.interp1d(x_quotient_1_5_graph_4, y_quotient_1_5_graph_4)
        k_quotient_3_graph_4 = interpolate.interp1d(x_quotient_3_graph_4, y_quotient_3_graph_4)
        k_quotient_4_5_graph_4 = interpolate.interp1d(x_quotient_4_5_graph_4, y_quotient_4_5_graph_4)
        k_quotient_6_graph_4 = interpolate.interp1d(x_quotient_6_graph_4, y_quotient_6_graph_4)
        k_quotient_8_graph_4 = interpolate.interp1d(x_quotient_8_graph_4, y_quotient_8_graph_4)
        k_quotient_10_graph_4 = interpolate.interp1d(x_quotient_10_graph_4, y_quotient_10_graph_4)

        if (
                (delta_beta != np.clip(delta_beta, min(x_quotient_1_5_graph_4), max(x_quotient_1_5_graph_4)))
                or (delta_beta != np.clip(delta_beta, min(x_quotient_3_graph_4), max(x_quotient_3_graph_4)))
                or (delta_beta != np.clip(delta_beta, min(x_quotient_4_5_graph_4), max(x_quotient_4_5_graph_4)))
                or (delta_beta != np.clip(delta_beta, min(x_quotient_6_graph_4), max(x_quotient_6_graph_4)))
                or (delta_beta != np.clip(delta_beta, min(x_quotient_8_graph_4), max(x_quotient_8_graph_4)))
                or (delta_beta != np.clip(delta_beta, min(x_quotient_10_graph_4), max(x_quotient_10_graph_4)))
        ):
            _LOGGER.warning("Value outside of the range in Roskam's book, value clipped")

        k_quotient_graph_4 = [
            float(
                k_quotient_1_5_graph_4(np.clip(delta_beta, min(x_quotient_1_5_graph_4), max(x_quotient_1_5_graph_4)))),
            float(k_quotient_3_graph_4(np.clip(delta_beta, min(x_quotient_3_graph_4), max(x_quotient_3_graph_4)))),
            float(
                k_quotient_4_5_graph_4(np.clip(delta_beta, min(x_quotient_4_5_graph_4), max(x_quotient_4_5_graph_4)))),
            float(k_quotient_6_graph_4(np.clip(delta_beta, min(x_quotient_6_graph_4), max(x_quotient_6_graph_4)))),
            float(k_quotient_8_graph_4(np.clip(delta_beta, min(x_quotient_8_graph_4), max(x_quotient_8_graph_4)))),
            float(k_quotient_10_graph_4(np.clip(delta_beta, min(x_quotient_10_graph_4), max(x_quotient_10_graph_4)))),

        ]

        if beta_A_k != np.clip(beta_A_k, 1.5, 10.0):
            _LOGGER.warning(
                "Value outside of the range in Roskam's book, value clipped"
            )

        beta_Clp_k_graph_4 = float(
            interpolate.interp1d([1.5, 3.0, 4.5, 6, 8, 10], k_quotient_graph_4)(
                np.clip(beta_A_k, 1.5, 10.0)
            )
        )

        # INTERPOLACION ENTRE LAS 3 GRÁFICAS
        beta_Clp_k_graph = [
            float(beta_Clp_k_graph_1),
            float(beta_Clp_k_graph_2),
            float(beta_Clp_k_graph_3),
            float(beta_Clp_k_graph_4),
        ]
        beta_Clp_k = float(
            interpolate.interp1d([0.0, 0.25, 0.5, 1.0], beta_Clp_k_graph)(
                np.clip(taper_ratio, 0.0, 1.0)
            )
        )

        return beta_Clp_k

    @staticmethod
    def get_Clp_CDL_CLW(aspect_ratio, sweep_25):
        """
        Data from Roskam - Airplane Design Part VI to compute the drag-due-to-lift roll damping parameter found in
        Figure 10.36.

        :param aspect_ratio: surface aspect ratio.
        :param sweep_25: quarter-chord point line sweep angle of the surface in radians.
        """

        sweep_25 = sweep_25 * 180.0 / math.pi  # radians to degrees

        file = pth.join(digit_figures.__path__[0], "10_36.csv")
        db = read_csv(file)

        x_sweep_10 = db["SWEEP_25_10_X"]
        y_sweep_10 = db["SWEEP_25_10_Y"]
        errors = np.logical_or(np.isnan(x_sweep_10), np.isnan(y_sweep_10))
        x_sweep_10 = x_sweep_10[np.logical_not(errors)].tolist()
        y_sweep_10 = y_sweep_10[np.logical_not(errors)].tolist()

        x_sweep_40 = db["SWEEP_25_40_X"]
        y_sweep_40 = db["SWEEP_25_40_Y"]
        errors = np.logical_or(np.isnan(x_sweep_40), np.isnan(y_sweep_40))
        x_sweep_40 = x_sweep_40[np.logical_not(errors)].tolist()
        y_sweep_40 = y_sweep_40[np.logical_not(errors)].tolist()

        x_sweep_50 = db["SWEEP_25_50_X"]
        y_sweep_50 = db["SWEEP_25_50_Y"]
        errors = np.logical_or(np.isnan(x_sweep_50), np.isnan(y_sweep_50))
        x_sweep_50 = x_sweep_50[np.logical_not(errors)].tolist()
        y_sweep_50 = y_sweep_50[np.logical_not(errors)].tolist()

        x_sweep_60 = db["SWEEP_25_60_X"]
        y_sweep_60 = db["SWEEP_25_60_Y"]
        errors = np.logical_or(np.isnan(x_sweep_60), np.isnan(y_sweep_60))
        x_sweep_60 = x_sweep_60[np.logical_not(errors)].tolist()
        y_sweep_60 = y_sweep_60[np.logical_not(errors)].tolist()

        x_sweep_70 = db["SWEEP_25_70_X"]
        y_sweep_70 = db["SWEEP_25_70_Y"]
        errors = np.logical_or(np.isnan(x_sweep_70), np.isnan(y_sweep_70))
        x_sweep_70 = x_sweep_70[np.logical_not(errors)].tolist()
        y_sweep_70 = y_sweep_70[np.logical_not(errors)].tolist()

        k_sweep_10 = interpolate.interp1d(x_sweep_10, y_sweep_10)
        k_sweep_40 = interpolate.interp1d(x_sweep_40, y_sweep_40)
        k_sweep_50 = interpolate.interp1d(x_sweep_50, y_sweep_50)
        k_sweep_60 = interpolate.interp1d(x_sweep_60, y_sweep_60)
        k_sweep_70 = interpolate.interp1d(x_sweep_70, y_sweep_70)

        if (
                (aspect_ratio != np.clip(aspect_ratio, min(x_sweep_10), max(x_sweep_10)))
                or (aspect_ratio != np.clip(aspect_ratio, min(x_sweep_40), max(x_sweep_40)))
                or (aspect_ratio != np.clip(aspect_ratio, min(x_sweep_50), max(x_sweep_50)))
                or (aspect_ratio != np.clip(aspect_ratio, min(x_sweep_60), max(x_sweep_60)))
                or (aspect_ratio != np.clip(aspect_ratio, min(x_sweep_70), max(x_sweep_70)))
        ):
            _LOGGER.warning("Aspect ratio value outside of the range in Roskam's book, value clipped")

        k_sweep = [
            float(k_sweep_10(np.clip(aspect_ratio, min(x_sweep_10), max(x_sweep_10)))),
            float(k_sweep_40(np.clip(aspect_ratio, min(x_sweep_40), max(x_sweep_40)))),
            float(k_sweep_50(np.clip(aspect_ratio, min(x_sweep_50), max(x_sweep_50)))),
            float(k_sweep_60(np.clip(aspect_ratio, min(x_sweep_60), max(x_sweep_60)))),
            float(k_sweep_70(np.clip(aspect_ratio, min(x_sweep_70), max(x_sweep_70)))),

        ]

        if sweep_25 != np.clip(sweep_25, 10.0, 70.0):
            _LOGGER.warning(
                "Sweep angle value outside of the range in Roskam's book, value clipped"
            )

        Clp_CDL_CLW = float(
            interpolate.interp1d([10.0, 40.0, 50.0, 60.0, 70.0], k_sweep)(
                np.clip(sweep_25, 10.0, 70.0)
            )
        )

        return Clp_CDL_CLW

    @staticmethod
    def get_delta_Cnp_twist(aspect_ratio, taper_ratio):
        """
        Data from Roskam - Airplane Design Part VI to compute the effect of wing twist on Cn_p found in Figure 10.37

        :param aspect_ratio: surface aspect ratio.
        :param taper_ratio: surface taper ratio
        """

        file = pth.join(digit_figures.__path__[0], "10_37.csv")
        db = read_csv(file)

        x_taperratio_0 = db["TAPER_RATIO_0_X"]
        y_taperratio_0 = db["TAPER_RATIO_0_Y"]
        errors = np.logical_or(np.isnan(x_taperratio_0), np.isnan(y_taperratio_0))
        x_taperratio_0 = x_taperratio_0[np.logical_not(errors)].tolist()
        y_taperratio_0 = y_taperratio_0[np.logical_not(errors)].tolist()

        x_taperratio_0_2 = db["TAPER_RATIO_0.2_X"]
        y_taperratio_0_2 = db["TAPER_RATIO_0.2_Y"]
        errors = np.logical_or(np.isnan(x_taperratio_0_2), np.isnan(y_taperratio_0_2))
        x_taperratio_0_2 = x_taperratio_0_2[np.logical_not(errors)].tolist()
        y_taperratio_0_2 = y_taperratio_0_2[np.logical_not(errors)].tolist()

        x_taperratio_0_4 = db["TAPER_RATIO_0.4_X"]
        y_taperratio_0_4 = db["TAPER_RATIO_0.4_Y"]
        errors = np.logical_or(np.isnan(x_taperratio_0_4), np.isnan(y_taperratio_0_4))
        x_taperratio_0_4 = x_taperratio_0_4[np.logical_not(errors)].tolist()
        y_taperratio_0_4 = y_taperratio_0_4[np.logical_not(errors)].tolist()

        x_taperratio_0_6 = db["TAPER_RATIO_0.6_X"]
        y_taperratio_0_6 = db["TAPER_RATIO_0.6_Y"]
        errors = np.logical_or(np.isnan(x_taperratio_0_6), np.isnan(y_taperratio_0_6))
        x_taperratio_0_6 = x_taperratio_0_6[np.logical_not(errors)].tolist()
        y_taperratio_0_6 = y_taperratio_0_6[np.logical_not(errors)].tolist()

        x_taperratio_0_8 = db["TAPER_RATIO_0.8_X"]
        y_taperratio_0_8 = db["TAPER_RATIO_0.8_Y"]
        errors = np.logical_or(np.isnan(x_taperratio_0_8), np.isnan(y_taperratio_0_8))
        x_taperratio_0_8 = x_taperratio_0_8[np.logical_not(errors)].tolist()
        y_taperratio_0_8 = y_taperratio_0_8[np.logical_not(errors)].tolist()

        x_taperratio_1 = db["TAPER_RATIO_1_X"]
        y_taperratio_1 = db["TAPER_RATIO_1_Y"]
        errors = np.logical_or(np.isnan(x_taperratio_1), np.isnan(y_taperratio_1))
        x_taperratio_1 = x_taperratio_1[np.logical_not(errors)].tolist()
        y_taperratio_1 = y_taperratio_1[np.logical_not(errors)].tolist()

        k_taperratio_0 = interpolate.interp1d(x_taperratio_0, y_taperratio_0)
        k_taperratio_0_2 = interpolate.interp1d(x_taperratio_0_2, y_taperratio_0_2)
        k_taperratio_0_4 = interpolate.interp1d(x_taperratio_0_4, y_taperratio_0_4)
        k_taperratio_0_6 = interpolate.interp1d(x_taperratio_0_6, y_taperratio_0_6)
        k_taperratio_0_8 = interpolate.interp1d(x_taperratio_0_8, y_taperratio_0_8)
        k_taperratio_1 = interpolate.interp1d(x_taperratio_1, y_taperratio_1)

        if (
                (aspect_ratio != np.clip(aspect_ratio, min(x_taperratio_0), max(x_taperratio_0)))
                or (aspect_ratio != np.clip(aspect_ratio, min(x_taperratio_0_2), max(x_taperratio_0_2)))
                or (aspect_ratio != np.clip(aspect_ratio, min(x_taperratio_0_4), max(x_taperratio_0_4)))
                or (aspect_ratio != np.clip(aspect_ratio, min(x_taperratio_0_6), max(x_taperratio_0_6)))
                or (aspect_ratio != np.clip(aspect_ratio, min(x_taperratio_0_8), max(x_taperratio_0_8)))
                or (aspect_ratio != np.clip(aspect_ratio, min(x_taperratio_1), max(x_taperratio_1)))
        ):
            _LOGGER.warning("Aspect ratio value outside of the range in Roskam's book, value clipped")

        k_taperratio = [
            float(k_taperratio_0(np.clip(aspect_ratio, min(x_taperratio_0), max(x_taperratio_0)))),
            float(k_taperratio_0_2(np.clip(aspect_ratio, min(x_taperratio_0_2), max(x_taperratio_0_2)))),
            float(k_taperratio_0_4(np.clip(aspect_ratio, min(x_taperratio_0_4), max(x_taperratio_0_4)))),
            float(k_taperratio_0_6(np.clip(aspect_ratio, min(x_taperratio_0_6), max(x_taperratio_0_6)))),
            float(k_taperratio_0_8(np.clip(aspect_ratio, min(x_taperratio_0_8), max(x_taperratio_0_8)))),
            float(k_taperratio_1(np.clip(aspect_ratio, min(x_taperratio_1), max(x_taperratio_1)))),
        ]

        if taper_ratio != np.clip(taper_ratio, 0.0, 1.0):
            _LOGGER.warning(
                "Taper ratio value outside of the range in Roskam's book, value clipped"
            )

        delta_Cnp_twist = float(
            interpolate.interp1d([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], k_taperratio)(
                np.clip(taper_ratio, 0.0, 1.0)
            )
        )

        return delta_Cnp_twist

    @staticmethod
    def get_delta_Cnp_flaps(aspect_ratio, taper_ratio, flaps_span_ratio):
        """
        Data from Roskam - Airplane Design PArt VI to compute the effect of symmetrical flap deflection on Cn_p found
        in Figure 10.38.

        :param aspect_ratio: surface aspect ratio.
        :param taper_ratio: surface taper ratio.
        :param flaps_span_ratio: ratio between the flaps span and the wing span
        """

        # ----- GRAPH 1. SPAN RATIO = 0.4 -----
        file = pth.join(digit_figures.__path__[0], "10_38_0_4.csv")
        db = read_csv(file)

        x_taperratio_0_graph_1 = db["TAPER_RATIO_0_X"]
        y_taperratio_0_graph_1 = db["TAPER_RATIO_0_Y"]
        errors = np.logical_or(np.isnan(x_taperratio_0_graph_1), np.isnan(y_taperratio_0_graph_1))
        x_taperratio_0_graph_1 = x_taperratio_0_graph_1[np.logical_not(errors)].tolist()
        y_taperratio_0_graph_1 = y_taperratio_0_graph_1[np.logical_not(errors)].tolist()

        x_taperratio_0_2_graph_1 = db["TAPER_RATIO_0.2_X"]
        y_taperratio_0_2_graph_1 = db["TAPER_RATIO_0.2_Y"]
        errors = np.logical_or(np.isnan(x_taperratio_0_2_graph_1), np.isnan(y_taperratio_0_2_graph_1))
        x_taperratio_0_2_graph_1 = x_taperratio_0_2_graph_1[np.logical_not(errors)].tolist()
        y_taperratio_0_2_graph_1 = y_taperratio_0_2_graph_1[np.logical_not(errors)].tolist()

        x_taperratio_0_4_graph_1 = db["TAPER_RATIO_0.4_X"]
        y_taperratio_0_4_graph_1 = db["TAPER_RATIO_0.4_Y"]
        errors = np.logical_or(np.isnan(x_taperratio_0_4_graph_1), np.isnan(y_taperratio_0_4_graph_1))
        x_taperratio_0_4_graph_1 = x_taperratio_0_4_graph_1[np.logical_not(errors)].tolist()
        y_taperratio_0_4_graph_1 = y_taperratio_0_4_graph_1[np.logical_not(errors)].tolist()

        x_taperratio_0_6_graph_1 = db["TAPER_RATIO_0.6_X"]
        y_taperratio_0_6_graph_1 = db["TAPER_RATIO_0.6_Y"]
        errors = np.logical_or(np.isnan(x_taperratio_0_6_graph_1), np.isnan(y_taperratio_0_6_graph_1))
        x_taperratio_0_6_graph_1 = x_taperratio_0_6_graph_1[np.logical_not(errors)].tolist()
        y_taperratio_0_6_graph_1 = y_taperratio_0_6_graph_1[np.logical_not(errors)].tolist()

        x_taperratio_1_graph_1 = db["TAPER_RATIO_1_X"]
        y_taperratio_1_graph_1 = db["TAPER_RATIO_1_Y"]
        errors = np.logical_or(np.isnan(x_taperratio_1_graph_1), np.isnan(y_taperratio_1_graph_1))
        x_taperratio_1_graph_1 = x_taperratio_1_graph_1[np.logical_not(errors)].tolist()
        y_taperratio_1_graph_1 = y_taperratio_1_graph_1[np.logical_not(errors)].tolist()

        k_taperratio_0_graph_1 = interpolate.interp1d(x_taperratio_0_graph_1, y_taperratio_0_graph_1)
        k_taperratio_0_2_graph_1 = interpolate.interp1d(x_taperratio_0_2_graph_1, y_taperratio_0_2_graph_1)
        k_taperratio_0_4_graph_1 = interpolate.interp1d(x_taperratio_0_4_graph_1, y_taperratio_0_4_graph_1)
        k_taperratio_0_6_graph_1 = interpolate.interp1d(x_taperratio_0_6_graph_1, y_taperratio_0_6_graph_1)
        k_taperratio_1_graph_1 = interpolate.interp1d(x_taperratio_1_graph_1, y_taperratio_1_graph_1)

        if (
                (aspect_ratio != np.clip(aspect_ratio, min(x_taperratio_0_graph_1), max(x_taperratio_0_graph_1)))
                or (aspect_ratio != np.clip(aspect_ratio, min(x_taperratio_0_2_graph_1), max(x_taperratio_0_2_graph_1)))
                or (aspect_ratio != np.clip(aspect_ratio, min(x_taperratio_0_4_graph_1), max(x_taperratio_0_4_graph_1)))
                or (aspect_ratio != np.clip(aspect_ratio, min(x_taperratio_0_6_graph_1), max(x_taperratio_0_6_graph_1)))
                or (aspect_ratio != np.clip(aspect_ratio, min(x_taperratio_1_graph_1), max(x_taperratio_1_graph_1)))
        ):
            _LOGGER.warning("Aspect ratio value outside of the range in Roskam's book, value clipped")

        k_taperratio_graph_1 = [
            float(k_taperratio_0_graph_1(
                np.clip(aspect_ratio, min(x_taperratio_0_graph_1), max(x_taperratio_0_graph_1)))),
            float(k_taperratio_0_2_graph_1(
                np.clip(aspect_ratio, min(x_taperratio_0_2_graph_1), max(x_taperratio_0_2_graph_1)))),
            float(k_taperratio_0_4_graph_1(
                np.clip(aspect_ratio, min(x_taperratio_0_4_graph_1), max(x_taperratio_0_4_graph_1)))),
            float(k_taperratio_0_6_graph_1(
                np.clip(aspect_ratio, min(x_taperratio_0_6_graph_1), max(x_taperratio_0_6_graph_1)))),
            float(k_taperratio_1_graph_1(
                np.clip(aspect_ratio, min(x_taperratio_1_graph_1), max(x_taperratio_1_graph_1)))),
        ]

        if taper_ratio != np.clip(taper_ratio, 0.0, 1.0):
            _LOGGER.warning(
                "Taper ratio value outside of the range in Roskam's book, value clipped"
            )

        delta_Cnp_flaps_graph_1 = float(
            interpolate.interp1d([0.0, 0.2, 0.4, 0.6, 1.0], k_taperratio_graph_1)(
                np.clip(taper_ratio, 0.0, 1.0)
            )
        )

        # ----- GRAPH 2. SPAN RATIO = 0.6 -----
        file = pth.join(digit_figures.__path__[0], "10_38_0_6.csv")
        db = read_csv(file)

        x_taperratio_0_graph_2 = db["TAPER_RATIO_0_X"]
        y_taperratio_0_graph_2 = db["TAPER_RATIO_0_Y"]
        errors = np.logical_or(np.isnan(x_taperratio_0_graph_2), np.isnan(y_taperratio_0_graph_2))
        x_taperratio_0_graph_2 = x_taperratio_0_graph_2[np.logical_not(errors)].tolist()
        y_taperratio_0_graph_2 = y_taperratio_0_graph_2[np.logical_not(errors)].tolist()

        x_taperratio_0_2_graph_2 = db["TAPER_RATIO_0.2_X"]
        y_taperratio_0_2_graph_2 = db["TAPER_RATIO_0.2_Y"]
        errors = np.logical_or(np.isnan(x_taperratio_0_2_graph_2), np.isnan(y_taperratio_0_2_graph_2))
        x_taperratio_0_2_graph_2 = x_taperratio_0_2_graph_2[np.logical_not(errors)].tolist()
        y_taperratio_0_2_graph_2 = y_taperratio_0_2_graph_2[np.logical_not(errors)].tolist()

        x_taperratio_0_4_graph_2 = db["TAPER_RATIO_0.4_X"]
        y_taperratio_0_4_graph_2 = db["TAPER_RATIO_0.4_Y"]
        errors = np.logical_or(np.isnan(x_taperratio_0_4_graph_2), np.isnan(y_taperratio_0_4_graph_2))
        x_taperratio_0_4_graph_2 = x_taperratio_0_4_graph_2[np.logical_not(errors)].tolist()
        y_taperratio_0_4_graph_2 = y_taperratio_0_4_graph_2[np.logical_not(errors)].tolist()

        x_taperratio_0_6_graph_2 = db["TAPER_RATIO_0.6_X"]
        y_taperratio_0_6_graph_2 = db["TAPER_RATIO_0.6_Y"]
        errors = np.logical_or(np.isnan(x_taperratio_0_6_graph_2), np.isnan(y_taperratio_0_6_graph_2))
        x_taperratio_0_6_graph_2 = x_taperratio_0_6_graph_2[np.logical_not(errors)].tolist()
        y_taperratio_0_6_graph_2 = y_taperratio_0_6_graph_2[np.logical_not(errors)].tolist()

        x_taperratio_1_graph_2 = db["TAPER_RATIO_1_X"]
        y_taperratio_1_graph_2 = db["TAPER_RATIO_1_Y"]
        errors = np.logical_or(np.isnan(x_taperratio_1_graph_2), np.isnan(y_taperratio_1_graph_2))
        x_taperratio_1_graph_2 = x_taperratio_1_graph_2[np.logical_not(errors)].tolist()
        y_taperratio_1_graph_2 = y_taperratio_1_graph_2[np.logical_not(errors)].tolist()

        k_taperratio_0_graph_2 = interpolate.interp1d(x_taperratio_0_graph_2, y_taperratio_0_graph_2)
        k_taperratio_0_2_graph_2 = interpolate.interp1d(x_taperratio_0_2_graph_2, y_taperratio_0_2_graph_2)
        k_taperratio_0_4_graph_2 = interpolate.interp1d(x_taperratio_0_4_graph_2, y_taperratio_0_4_graph_2)
        k_taperratio_0_6_graph_2 = interpolate.interp1d(x_taperratio_0_6_graph_2, y_taperratio_0_6_graph_2)
        k_taperratio_1_graph_2 = interpolate.interp1d(x_taperratio_1_graph_2, y_taperratio_1_graph_2)

        if (
                (aspect_ratio != np.clip(aspect_ratio, min(x_taperratio_0_graph_2), max(x_taperratio_0_graph_2)))
                or (aspect_ratio != np.clip(aspect_ratio, min(x_taperratio_0_2_graph_2), max(x_taperratio_0_2_graph_2)))
                or (aspect_ratio != np.clip(aspect_ratio, min(x_taperratio_0_4_graph_2), max(x_taperratio_0_4_graph_2)))
                or (aspect_ratio != np.clip(aspect_ratio, min(x_taperratio_0_6_graph_2), max(x_taperratio_0_6_graph_2)))
                or (aspect_ratio != np.clip(aspect_ratio, min(x_taperratio_1_graph_2), max(x_taperratio_1_graph_2)))
        ):
            _LOGGER.warning("Aspect ratio value outside of the range in Roskam's book, value clipped")

        k_taperratio_graph_2 = [
            float(k_taperratio_0_graph_2(
                np.clip(aspect_ratio, min(x_taperratio_0_graph_2), max(x_taperratio_0_graph_2)))),
            float(k_taperratio_0_2_graph_2(
                np.clip(aspect_ratio, min(x_taperratio_0_2_graph_2), max(x_taperratio_0_2_graph_2)))),
            float(k_taperratio_0_4_graph_2(
                np.clip(aspect_ratio, min(x_taperratio_0_4_graph_2), max(x_taperratio_0_4_graph_2)))),
            float(k_taperratio_0_6_graph_2(
                np.clip(aspect_ratio, min(x_taperratio_0_6_graph_2), max(x_taperratio_0_6_graph_2)))),
            float(k_taperratio_1_graph_2(
                np.clip(aspect_ratio, min(x_taperratio_1_graph_2), max(x_taperratio_1_graph_2)))),

        ]

        if taper_ratio != np.clip(taper_ratio, 0.0, 1.0):
            _LOGGER.warning(
                "Taper ratio value outside of the range in Roskam's book, value clipped"
            )

        delta_Cnp_flaps_graph_2 = float(
            interpolate.interp1d([0.0, 0.2, 0.4, 0.6, 1.0], k_taperratio_graph_2)(
                np.clip(taper_ratio, 0.0, 1.0)
            )
        )

        # ----- GRAPH 3. SPAN RATIO = 0.8 -----
        file = pth.join(digit_figures.__path__[0], "10_38_0_8.csv")
        db = read_csv(file)

        x_taperratio_0_graph_3 = db["TAPER_RATIO_0_X"]
        y_taperratio_0_graph_3 = db["TAPER_RATIO_0_Y"]
        errors = np.logical_or(np.isnan(x_taperratio_0_graph_3), np.isnan(y_taperratio_0_graph_3))
        x_taperratio_0_graph_3 = x_taperratio_0_graph_3[np.logical_not(errors)].tolist()
        y_taperratio_0_graph_3 = y_taperratio_0_graph_3[np.logical_not(errors)].tolist()

        x_taperratio_0_2_graph_3 = db["TAPER_RATIO_0.2_X"]
        y_taperratio_0_2_graph_3 = db["TAPER_RATIO_0.2_Y"]
        errors = np.logical_or(np.isnan(x_taperratio_0_2_graph_3), np.isnan(y_taperratio_0_2_graph_3))
        x_taperratio_0_2_graph_3 = x_taperratio_0_2_graph_3[np.logical_not(errors)].tolist()
        y_taperratio_0_2_graph_3 = y_taperratio_0_2_graph_3[np.logical_not(errors)].tolist()

        x_taperratio_0_4_graph_3 = db["TAPER_RATIO_0.4_X"]
        y_taperratio_0_4_graph_3 = db["TAPER_RATIO_0.4_Y"]
        errors = np.logical_or(np.isnan(x_taperratio_0_4_graph_3), np.isnan(y_taperratio_0_4_graph_3))
        x_taperratio_0_4_graph_3 = x_taperratio_0_4_graph_3[np.logical_not(errors)].tolist()
        y_taperratio_0_4_graph_3 = y_taperratio_0_4_graph_3[np.logical_not(errors)].tolist()

        x_taperratio_0_6_graph_3 = db["TAPER_RATIO_0.6_X"]
        y_taperratio_0_6_graph_3 = db["TAPER_RATIO_0.6_Y"]
        errors = np.logical_or(np.isnan(x_taperratio_0_6_graph_3), np.isnan(y_taperratio_0_6_graph_3))
        x_taperratio_0_6_graph_3 = x_taperratio_0_6_graph_3[np.logical_not(errors)].tolist()
        y_taperratio_0_6_graph_3 = y_taperratio_0_6_graph_3[np.logical_not(errors)].tolist()

        x_taperratio_1_graph_3 = db["TAPER_RATIO_1_X"]
        y_taperratio_1_graph_3 = db["TAPER_RATIO_1_Y"]
        errors = np.logical_or(np.isnan(x_taperratio_1_graph_3), np.isnan(y_taperratio_1_graph_3))
        x_taperratio_1_graph_3 = x_taperratio_1_graph_3[np.logical_not(errors)].tolist()
        y_taperratio_1_graph_3 = y_taperratio_1_graph_3[np.logical_not(errors)].tolist()

        k_taperratio_0_graph_3 = interpolate.interp1d(x_taperratio_0_graph_3, y_taperratio_0_graph_3)
        k_taperratio_0_2_graph_3 = interpolate.interp1d(x_taperratio_0_2_graph_3, y_taperratio_0_2_graph_3)
        k_taperratio_0_4_graph_3 = interpolate.interp1d(x_taperratio_0_4_graph_3, y_taperratio_0_4_graph_3)
        k_taperratio_0_6_graph_3 = interpolate.interp1d(x_taperratio_0_6_graph_3, y_taperratio_0_6_graph_3)
        k_taperratio_1_graph_3 = interpolate.interp1d(x_taperratio_1_graph_3, y_taperratio_1_graph_3)

        if (
                (aspect_ratio != np.clip(aspect_ratio, min(x_taperratio_0_graph_3), max(x_taperratio_0_graph_3)))
                or (aspect_ratio != np.clip(aspect_ratio, min(x_taperratio_0_2_graph_3), max(x_taperratio_0_2_graph_3)))
                or (aspect_ratio != np.clip(aspect_ratio, min(x_taperratio_0_4_graph_3), max(x_taperratio_0_4_graph_3)))
                or (aspect_ratio != np.clip(aspect_ratio, min(x_taperratio_0_6_graph_3), max(x_taperratio_0_6_graph_3)))
                or (aspect_ratio != np.clip(aspect_ratio, min(x_taperratio_1_graph_3), max(x_taperratio_1_graph_3)))
        ):
            _LOGGER.warning("Aspect ratio value outside of the range in Roskam's book, value clipped")

        k_taperratio_graph_3 = [
            float(k_taperratio_0_graph_3(
                np.clip(aspect_ratio, min(x_taperratio_0_graph_3), max(x_taperratio_0_graph_3)))),
            float(k_taperratio_0_2_graph_3(
                np.clip(aspect_ratio, min(x_taperratio_0_2_graph_3), max(x_taperratio_0_2_graph_3)))),
            float(k_taperratio_0_4_graph_3(
                np.clip(aspect_ratio, min(x_taperratio_0_4_graph_3), max(x_taperratio_0_4_graph_3)))),
            float(k_taperratio_0_6_graph_3(
                np.clip(aspect_ratio, min(x_taperratio_0_6_graph_3), max(x_taperratio_0_6_graph_3)))),
            float(k_taperratio_1_graph_3(
                np.clip(aspect_ratio, min(x_taperratio_1_graph_3), max(x_taperratio_1_graph_3)))),

        ]

        if taper_ratio != np.clip(taper_ratio, 0.0, 1.0):
            _LOGGER.warning(
                "Taper ratio value outside of the range in Roskam's book, value clipped"
            )

        delta_Cnp_flaps_graph_3 = float(
            interpolate.interp1d([0.0, 0.2, 0.4, 0.6, 1.0], k_taperratio_graph_3)(
                np.clip(taper_ratio, 0.0, 1.0)
            )
        )

        # INTERPOLATION BETWEEN THE RESULTS OF THE 3 GRAPHS
        delta_Cnp_flaps_graph = [
            float(delta_Cnp_flaps_graph_1),
            float(delta_Cnp_flaps_graph_2),
            float(delta_Cnp_flaps_graph_3),
        ]
        delta_Cnp_flaps = float(
            interpolate.interp1d([0.4, 0.6, 0.8], delta_Cnp_flaps_graph)(
                np.clip(flaps_span_ratio, 0.4, 0.8)
            )
        )

        return delta_Cnp_flaps


    @staticmethod
    def get_k_w(aspect_ratio) -> float:
        """
        Data from USAF DATCOM to estimate the correction constant for wing contribution to Pitch Damping presented in
        Figure 10.40 of Roskam.

        :param aspect_ratio: lifting surface aspect ratio.
        """

        file = pth.join(digit_figures.__path__[0], "10_40.csv")
        db = read_csv(file)

        x = db["X"]
        y = db["Y"]
        errors = np.logical_or(np.isnan(x), np.isnan(y))
        x = x[np.logical_not(errors)].tolist()
        y = y[np.logical_not(errors)].tolist()

        if float(aspect_ratio) != np.clip(float(aspect_ratio), min(x), max(x)):
            _LOGGER.warning("Aspect ratio value outside of the range in Roskam's book, value clipped")

        k_w = float(interpolate.interp1d(x, y)(np.clip(float(aspect_ratio), min(x), max(x))))

        return k_w


    @staticmethod
    def get_Clr_CL_mach0(aspect_ratio, taper_ratio, sweep_25):
        """
        Data from Roskam - Airplane Design Part VI to estimate the slope of the low-speed rolling momento due to yaw
        rate at zero lift found from Figure 10.41. The figure is separated into two parts (a and b).

        :param aspect_ratio: wing aspect ratio
        :param taper_ratio: wing taper ratio
        :param sweep_25: wing sweep angle at quarter-taper point line
        """

        sweep_25 = sweep_25 * 180.0 / math.pi  # radians to degrees

        # Reading data from the first part (a) relative to the wing taper ratio
        file = pth.join(digit_figures.__path__[0], "10_41a.csv")
        db = read_csv(file)

        x_0 = db["TAPER_RATIO_0_X"]
        y_0 = db["TAPER_RATIO_0_Y"]
        errors = np.logical_or(np.isnan(x_0), np.isnan(y_0))
        x_0 = x_0[np.logical_not(errors)].tolist()
        y_0 = y_0[np.logical_not(errors)].tolist()
        x_0.sort()
        y_0.sort()

        x_0_25 = db["TAPER_RATIO_0.25_X"]
        y_0_25 = db["TAPER_RATIO_0.25_Y"]
        errors = np.logical_or(np.isnan(x_0_25), np.isnan(y_0_25))
        x_0_25 = x_0_25[np.logical_not(errors)].tolist()
        y_0_25 = y_0_25[np.logical_not(errors)].tolist()
        x_0_25.sort()
        y_0_25.sort()

        x_0_5 = db["TAPER_RATIO_0.5_X"]
        y_0_5 = db["TAPER_RATIO_0.5_Y"]
        errors = np.logical_or(np.isnan(x_0_5), np.isnan(y_0_5))
        x_0_5 = x_0_5[np.logical_not(errors)].tolist()
        y_0_5 = y_0_5[np.logical_not(errors)].tolist()
        x_0_5.sort()
        y_0_5.sort()

        x_1_0 = db["TAPER_RATIO_1_X"]
        y_1_0 = db["TAPER_RATIO_1_Y"]
        errors = np.logical_or(np.isnan(x_1_0), np.isnan(y_1_0))
        x_1_0 = x_1_0[np.logical_not(errors)].tolist()
        y_1_0 = y_1_0[np.logical_not(errors)].tolist()
        x_1_0.sort()
        y_1_0.sort()

        k_taper0 = interpolate.interp1d(x_0, y_0)
        k_taper025 = interpolate.interp1d(x_0_25, y_0_25)
        k_taper05 = interpolate.interp1d(x_0_5, y_0_5)
        k_taper1 = interpolate.interp1d(x_1_0, y_1_0)

        if (
                (aspect_ratio != np.clip(aspect_ratio, min(x_1_0), max(x_1_0)))
                or (aspect_ratio != np.clip(aspect_ratio, min(x_0_5), max(x_0_5)))
                or (aspect_ratio != np.clip(aspect_ratio, min(x_0_25), max(x_0_25)))
                or (aspect_ratio != np.clip(aspect_ratio, min(x_0), max(x_0)))

        ):
            _LOGGER.warning("Aspect ratio value outside of the range in Roskam's book, value clipped")

        k_taper = [
            float(k_taper0(np.clip(aspect_ratio, min(x_1_0), max(x_1_0)))),
            float(k_taper025(np.clip(aspect_ratio, min(x_0_5), max(x_0_5)))),
            float(k_taper05(np.clip(aspect_ratio, min(x_0_25), max(x_0_25)))),
            float(k_taper1(np.clip(aspect_ratio, min(x_0), max(x_0)))),
        ]

        if taper_ratio != np.clip(taper_ratio, 0.0, 1.0):
            _LOGGER.warning(
                "taper ratio value outside of the range in Roskam's book, value clipped"
            )

        k_intermediate = float(
            interpolate.interp1d([0.0, 0.25, 0.5, 1.0], k_taper)(
                np.clip(taper_ratio, 0.0, 1.0)
            )
        )

        # Reading the second part of the figure (b) relative to the different wing sweep angles.
        file = pth.join(digit_figures.__path__[0], "10_41b.csv")
        db = read_csv(file)

        x_sw_0 = db["SWEEP_25_0_X"]
        y_sw_0 = db["SWEEP_25_0_Y"]
        errors = np.logical_or(np.isnan(x_sw_0), np.isnan(y_sw_0))
        x_sw_0 = x_sw_0[np.logical_not(errors)].tolist()
        y_sw_0 = y_sw_0[np.logical_not(errors)].tolist()
        x_sw_0.sort()
        y_sw_0.sort()

        x_sw_15 = db["SWEEP_25_15_X"]
        y_sw_15 = db["SWEEP_25_15_Y"]
        errors = np.logical_or(np.isnan(x_sw_15), np.isnan(y_sw_15))
        x_sw_15 = x_sw_15[np.logical_not(errors)].tolist()
        y_sw_15 = y_sw_15[np.logical_not(errors)].tolist()
        x_sw_15.sort()
        y_sw_15.sort()

        x_sw_30 = db["SWEEP_25_30_X"]
        y_sw_30 = db["SWEEP_25_30_Y"]
        errors = np.logical_or(np.isnan(x_sw_30), np.isnan(y_sw_30))
        x_sw_30 = x_sw_30[np.logical_not(errors)].tolist()
        y_sw_30 = y_sw_30[np.logical_not(errors)].tolist()
        x_sw_30.sort()
        y_sw_30.sort()

        x_sw_45 = db["SWEEP_25_45_X"]
        y_sw_45 = db["SWEEP_25_45_Y"]
        errors = np.logical_or(np.isnan(x_sw_45), np.isnan(y_sw_45))
        x_sw_45 = x_sw_45[np.logical_not(errors)].tolist()
        y_sw_45 = y_sw_45[np.logical_not(errors)].tolist()
        x_sw_45.sort()
        y_sw_45.sort()

        x_sw_60 = db["SWEEP_25_60_X"]
        y_sw_60 = db["SWEEP_25_60_Y"]
        errors = np.logical_or(np.isnan(x_sw_60), np.isnan(y_sw_60))
        x_sw_60 = x_sw_60[np.logical_not(errors)].tolist()
        y_sw_60 = y_sw_60[np.logical_not(errors)].tolist()
        x_sw_60.sort()
        y_sw_60.sort()

        k_sweep0 = interpolate.interp1d(x_sw_0, y_sw_0)
        k_sweep15 = interpolate.interp1d(x_sw_15, y_sw_15)
        k_sweep30 = interpolate.interp1d(x_sw_30, y_sw_30)
        k_sweep45 = interpolate.interp1d(x_sw_45, y_sw_45)
        k_sweep60 = interpolate.interp1d(x_sw_60, y_sw_60)

        if (
                (k_intermediate != np.clip(k_intermediate, min(x_sw_45), max(x_sw_45)))
                or (k_intermediate != np.clip(k_intermediate, min(x_sw_30), max(x_sw_30)))
                or (k_intermediate != np.clip(k_intermediate, min(x_sw_15), max(x_sw_15)))
                or (k_intermediate != np.clip(k_intermediate, min(x_sw_0), max(x_sw_0)))
                or (k_intermediate != np.clip(k_intermediate, min(x_sw_60), max(x_sw_60)))

        ):
            _LOGGER.warning("Intermediate value outside of the range in Roskam's book, value clipped")

        k_sweep = [
            float(k_sweep0(np.clip(k_intermediate, min(x_sw_0), max(x_sw_0)))),
            float(k_sweep15(np.clip(k_intermediate, min(x_sw_15), max(x_sw_15)))),
            float(k_sweep30(np.clip(k_intermediate, min(x_sw_30), max(x_sw_30)))),
            float(k_sweep45(np.clip(k_intermediate, min(x_sw_45), max(x_sw_45)))),
            float(k_sweep60(np.clip(k_intermediate, min(x_sw_60), max(x_sw_60)))),
        ]

        if sweep_25 != np.clip(sweep_25, 0.0, 1.0):
            _LOGGER.warning(
                "Sweep angle value outside of the range in Roskam's book, value clipped"
            )

        Clr_CL_mach0 = float(
            interpolate.interp1d([0.0, 15.0, 30.0, 45.0, 60.0], k_sweep)(
                np.clip(sweep_25, 0.0, 60.0)
            )
        )

        return Clr_CL_mach0

    @staticmethod
    def get_delta_Clr_twist(aspect_ratio, taper_ratio):
        """
        Data from Roskam - Airplane Design Part VI to estimate the effect of wing twist on Clr found from Figure 10.42.

        :param aspect_ratio: wing aspect ratio
        :param taper_ratio: wing taper ratio
        """

        file = pth.join(digit_figures.__path__[0], "10_42.csv")
        db = read_csv(file)

        x_0 = db["TAPER_RATIO_0_X"]
        y_0 = db["TAPER_RATIO_0_Y"]
        errors = np.logical_or(np.isnan(x_0), np.isnan(y_0))
        x_0 = x_0[np.logical_not(errors)].tolist()
        y_0 = y_0[np.logical_not(errors)].tolist()

        x_0_2 = db["TAPER_RATIO_0.2_X"]
        y_0_2 = db["TAPER_RATIO_0.2_Y"]
        errors = np.logical_or(np.isnan(x_0_2), np.isnan(y_0_2))
        x_0_2 = x_0_2[np.logical_not(errors)].tolist()
        y_0_2 = y_0_2[np.logical_not(errors)].tolist()

        x_0_4 = db["TAPER_RATIO_0.4_X"]
        y_0_4 = db["TAPER_RATIO_0.4_Y"]
        errors = np.logical_or(np.isnan(x_0_4), np.isnan(y_0_4))
        x_0_4 = x_0_4[np.logical_not(errors)].tolist()
        y_0_4 = y_0_4[np.logical_not(errors)].tolist()

        k_taper0 = interpolate.interp1d(x_0, y_0)
        k_taper02 = interpolate.interp1d(x_0_2, y_0_2)
        k_taper04 = interpolate.interp1d(x_0_4, y_0_4)

        if (
                (aspect_ratio != np.clip(aspect_ratio, min(x_0), max(x_0)))
                or (aspect_ratio != np.clip(aspect_ratio, min(x_0_2), max(x_0_2)))
                or (aspect_ratio != np.clip(aspect_ratio, min(x_0_4), max(x_0_4)))

        ):
            _LOGGER.warning("Aspect ratio value outside of the range in Roskam's book, value clipped")

        k_taper = [
            float(k_taper0(np.clip(aspect_ratio, min(x_0), max(x_0)))),
            float(k_taper02(np.clip(aspect_ratio, min(x_0_2), max(x_0_2)))),
            float(k_taper04(np.clip(aspect_ratio, min(x_0_4), max(x_0_4)))),
        ]

        if taper_ratio != np.clip(taper_ratio, 0.0, 0.4):
            _LOGGER.warning(
                "taper ratio value outside of the range in Roskam's book, value clipped"
            )

        delta_Clr_twist = float(
            interpolate.interp1d([0.0, 0.2, 0.4], k_taper)(
                np.clip(taper_ratio, 0.0, 0.4)
            )
        )

        return delta_Clr_twist

    @staticmethod
    def get_delta_Clr_flaps(flap_location, aspect_ratio, taper_ratio):
        """
        Data from Roskam - Airplane Design Part VI to estimate the effect of symmetric flap deflection found from
        Figure 10.43. The figure is separated into two parts (a and b).

        :param flap_location: location of inboard and outboard flaps in percent semispan y/(b/2)
        :param aspect_ratio: wing aspect ratio
        :param taper_ratio: wing taper ratio
        """

        # Reading data from the first part (a) relative to the wing taper ratio
        file = pth.join(digit_figures.__path__[0], "10_43a.csv")
        db = read_csv(file)

        x_0 = db["TAPER_RATIO_0_X"]
        y_0 = db["TAPER_RATIO_0_Y"]
        errors = np.logical_or(np.isnan(x_0), np.isnan(y_0))
        x_0 = x_0[np.logical_not(errors)].tolist()
        y_0 = y_0[np.logical_not(errors)].tolist()

        x_0_2 = db["TAPER_RATIO_0.2_X"]
        y_0_2 = db["TAPER_RATIO_0.2_Y"]
        errors = np.logical_or(np.isnan(x_0_2), np.isnan(y_0_2))
        x_0_2 = x_0_2[np.logical_not(errors)].tolist()
        y_0_2 = y_0_2[np.logical_not(errors)].tolist()

        x_1_0 = db["TAPER_RATIO_1.0_X"]
        y_1_0 = db["TAPER_RATIO_1.0_Y"]
        errors = np.logical_or(np.isnan(x_1_0), np.isnan(y_1_0))
        x_1_0 = x_1_0[np.logical_not(errors)].tolist()
        y_1_0 = y_1_0[np.logical_not(errors)].tolist()

        k_taper0 = interpolate.interp1d(x_0, y_0)
        k_taper02 = interpolate.interp1d(x_0_2, y_0_2)
        k_taper1 = interpolate.interp1d(x_1_0, y_1_0)

        if (
                (flap_location != np.clip(flap_location, min(x_1_0), max(x_1_0)))
                or (flap_location != np.clip(flap_location, min(x_0_2), max(x_0_2)))
                or (flap_location != np.clip(flap_location, min(x_0), max(x_0)))

        ):
            _LOGGER.warning("Flap location value outside of the range in Roskam's book, value clipped")

        k_taper = [
            float(k_taper0(np.clip(flap_location, min(x_0), max(x_0)))),
            float(k_taper02(np.clip(flap_location, min(x_0_2), max(x_0_2)))),
            float(k_taper1(np.clip(flap_location, min(x_1_0), max(x_1_0)))),
        ]

        if taper_ratio != np.clip(taper_ratio, 0.0, 1.0):
            _LOGGER.warning(
                "taper ratio value outside of the range in Roskam's book, value clipped"
            )

        k_intermediate = float(
            interpolate.interp1d([0.0, 0.2, 1.0], k_taper)(
                np.clip(taper_ratio, 0.0, 1.0)
            )
        )

        # Reading the second part of the figure (b) relative to the different wing sweep angles.
        file = pth.join(digit_figures.__path__[0], "10_43b.csv")
        db = read_csv(file)

        x_ar_1 = db["ASPECT_RATIO_1_X"]
        y_ar_1 = db["ASPECT_RATIO_1_Y"]
        errors = np.logical_or(np.isnan(x_ar_1), np.isnan(y_ar_1))
        x_ar_1 = x_ar_1[np.logical_not(errors)].tolist()
        y_ar_1 = y_ar_1[np.logical_not(errors)].tolist()

        x_ar_2 = db["ASPECT_RATIO_2_X"]
        y_ar_2 = db["ASPECT_RATIO_2_Y"]
        errors = np.logical_or(np.isnan(x_ar_2), np.isnan(y_ar_2))
        x_ar_2 = x_ar_2[np.logical_not(errors)].tolist()
        y_ar_2 = y_ar_2[np.logical_not(errors)].tolist()

        x_ar_3 = db["ASPECT_RATIO_3_X"]
        y_ar_3 = db["ASPECT_RATIO_3_Y"]
        errors = np.logical_or(np.isnan(x_ar_3), np.isnan(y_ar_3))
        x_ar_3 = x_ar_3[np.logical_not(errors)].tolist()
        y_ar_3 = y_ar_3[np.logical_not(errors)].tolist()

        x_ar_4 = db["ASPECT_RATIO_4_X"]
        y_ar_4 = db["ASPECT_RATIO_4_Y"]
        errors = np.logical_or(np.isnan(x_ar_4), np.isnan(y_ar_4))
        x_ar_4 = x_ar_4[np.logical_not(errors)].tolist()
        y_ar_4 = y_ar_4[np.logical_not(errors)].tolist()

        x_ar_6 = db["ASPECT_RATIO_6_X"]
        y_ar_6 = db["ASPECT_RATIO_6_Y"]
        errors = np.logical_or(np.isnan(x_ar_6), np.isnan(y_ar_6))
        x_ar_6 = x_ar_6[np.logical_not(errors)].tolist()
        y_ar_6 = y_ar_6[np.logical_not(errors)].tolist()

        x_ar_8 = db["ASPECT_RATIO_8_X"]
        y_ar_8 = db["ASPECT_RATIO_8_Y"]
        errors = np.logical_or(np.isnan(x_ar_8), np.isnan(y_ar_8))
        x_ar_8 = x_ar_8[np.logical_not(errors)].tolist()
        y_ar_8 = y_ar_8[np.logical_not(errors)].tolist()

        x_ar_10 = db["ASPECT_RATIO_10_X"]
        y_ar_10 = db["ASPECT_RATIO_10_Y"]
        errors = np.logical_or(np.isnan(x_ar_10), np.isnan(y_ar_10))
        x_ar_10 = x_ar_10[np.logical_not(errors)].tolist()
        y_ar_10 = y_ar_10[np.logical_not(errors)].tolist()

        k_ar_1 = interpolate.interp1d(x_ar_1, y_ar_1)
        k_ar_2 = interpolate.interp1d(x_ar_2, y_ar_2)
        k_ar_3 = interpolate.interp1d(x_ar_3, y_ar_3)
        k_ar_4 = interpolate.interp1d(x_ar_4, y_ar_4)
        k_ar_6 = interpolate.interp1d(x_ar_6, y_ar_6)
        k_ar_8 = interpolate.interp1d(x_ar_8, y_ar_8)
        k_ar_10 = interpolate.interp1d(x_ar_10, y_ar_10)

        if (
                (k_intermediate != np.clip(k_intermediate, min(x_ar_1), max(x_ar_1)))
                or (k_intermediate != np.clip(k_intermediate, min(x_ar_2), max(x_ar_2)))
                or (k_intermediate != np.clip(k_intermediate, min(x_ar_3), max(x_ar_3)))
                or (k_intermediate != np.clip(k_intermediate, min(x_ar_4), max(x_ar_4)))
                or (k_intermediate != np.clip(k_intermediate, min(x_ar_6), max(x_ar_6)))
                or (k_intermediate != np.clip(k_intermediate, min(x_ar_8), max(x_ar_8)))
                or (k_intermediate != np.clip(k_intermediate, min(x_ar_10), max(x_ar_10)))
        ):
            _LOGGER.warning("Intermediate value outside of the range in Roskam's book, value clipped")

        k_ar = [
            float(k_ar_1(np.clip(k_intermediate, min(x_ar_1), max(x_ar_1)))),
            float(k_ar_2(np.clip(k_intermediate, min(x_ar_2), max(x_ar_2)))),
            float(k_ar_3(np.clip(k_intermediate, min(x_ar_3), max(x_ar_3)))),
            float(k_ar_4(np.clip(k_intermediate, min(x_ar_4), max(x_ar_4)))),
            float(k_ar_6(np.clip(k_intermediate, min(x_ar_6), max(x_ar_6)))),
            float(k_ar_8(np.clip(k_intermediate, min(x_ar_8), max(x_ar_8)))),
            float(k_ar_10(np.clip(k_intermediate, min(x_ar_10), max(x_ar_10)))),

        ]

        if aspect_ratio != np.clip(aspect_ratio, 0.0, 1.0):
            _LOGGER.warning(
                "Sweep angle value outside of the range in Roskam's book, value clipped"
            )

        delta_Clr_flaps = float(
            interpolate.interp1d([1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0], k_ar)(
                np.clip(aspect_ratio, 1.0, 10.0)
            )
        )

        return delta_Clr_flaps

    @staticmethod
    def get_Cnr_CL2(aspect_ratio, sweep_25, taper_ratio, x, c):
        """
        Data from Roskam - Airplane Design Part VI to compute the lifting effect in the yaw damping derivative Cnr found
        in Figure 10.44.

        :param aspect_ratio: surface aspect ratio.
        :param sweep_25: quarter-chord point line sweep angle in radians
        :param taper_ratio: wing taper ratio.
        :param x: is the distance from the c.g. to the a.c. positive for the a.c. aft of the c.g.
        :param c: length of the mean aerodynamic chord of the surface.
        """

        ratio_x_c = x / c
        sweep_25 = sweep_25 * 180.0 / math.pi  # radians to degrees

        ###### GRAPH FOR x/c = 0.0 ######
        # Reading data from the first part (a) relative to the wing sweep angle
        file = pth.join(digit_figures.__path__[0], "10_44_0a.csv")
        db = read_csv(file)

        x_0 = db["SWEEP_25_0_X"]
        y_0 = db["SWEEP_25_0_Y"]
        errors = np.logical_or(np.isnan(x_0), np.isnan(y_0))
        x_0 = x_0[np.logical_not(errors)].tolist()
        y_0 = y_0[np.logical_not(errors)].tolist()

        x_40 = db["SWEEP_25_40_X"]
        y_40 = db["SWEEP_25_40_Y"]
        errors = np.logical_or(np.isnan(x_40), np.isnan(y_40))
        x_40 = x_40[np.logical_not(errors)].tolist()
        y_40 = y_40[np.logical_not(errors)].tolist()

        x_50 = db["SWEEP_25_50_X"]
        y_50 = db["SWEEP_25_50_Y"]
        errors = np.logical_or(np.isnan(x_50), np.isnan(y_50))
        x_50 = x_50[np.logical_not(errors)].tolist()
        y_50 = y_50[np.logical_not(errors)].tolist()

        x_60 = db["SWEEP_25_60_X"]
        y_60 = db["SWEEP_25_60_Y"]
        errors = np.logical_or(np.isnan(x_60), np.isnan(y_60))
        x_60 = x_60[np.logical_not(errors)].tolist()
        y_60 = y_60[np.logical_not(errors)].tolist()

        k_sweep0 = interpolate.interp1d(x_0, y_0)
        k_sweep40 = interpolate.interp1d(x_40, y_40)
        k_sweep50 = interpolate.interp1d(x_50, y_50)
        k_sweep60 = interpolate.interp1d(x_60, y_60)

        if (
                (aspect_ratio != np.clip(aspect_ratio, min(x_60), max(x_60)))
                or (aspect_ratio != np.clip(aspect_ratio, min(x_50), max(x_50)))
                or (aspect_ratio != np.clip(aspect_ratio, min(x_40), max(x_40)))
                or (aspect_ratio != np.clip(aspect_ratio, min(x_0), max(x_0)))

        ):
            _LOGGER.warning("Aspect ratio value outside of the range in Roskam's book, value clipped")

        k_sweep = [
            float(k_sweep0(np.clip(aspect_ratio, min(x_0), max(x_0)))),
            float(k_sweep40(np.clip(aspect_ratio, min(x_40), max(x_40)))),
            float(k_sweep50(np.clip(aspect_ratio, min(x_50), max(x_50)))),
            float(k_sweep60(np.clip(aspect_ratio, min(x_60), max(x_60)))),
        ]

        if sweep_25 != np.clip(sweep_25, 0.0, 60.0):
            _LOGGER.warning(
                "Sweep angle value outside of the range in Roskam's book, value clipped"
            )

        k_intermediate = float(
            interpolate.interp1d([0.0, 40.0, 50.0, 60.0], k_sweep)(
                np.clip(sweep_25, 0.0, 60.0)
            )
        )

        # Reading the second part of the figure (b) relative to the different taper ratio.
        file = pth.join(digit_figures.__path__[0], "10_44_0b.csv")
        db = read_csv(file)

        x_taper_0 = db["TAPER_RATIO_0_X"]
        y_taper_0 = db["TAPER_RATIO_0_Y"]
        errors = np.logical_or(np.isnan(x_taper_0), np.isnan(y_taper_0))
        x_taper_0 = x_taper_0[np.logical_not(errors)].tolist()
        y_taper_0 = y_taper_0[np.logical_not(errors)].tolist()

        x_taper_1 = db["TAPER_RATIO_1_X"]
        y_taper_1 = db["TAPER_RATIO_1_Y"]
        errors = np.logical_or(np.isnan(x_taper_1), np.isnan(y_taper_1))
        x_taper_1 = x_taper_1[np.logical_not(errors)].tolist()
        y_taper_1 = y_taper_1[np.logical_not(errors)].tolist()

        K_taper0 = interpolate.interp1d(x_taper_0, y_taper_0)
        K_taper1 = interpolate.interp1d(x_taper_1, y_taper_1)

        if (
                (k_intermediate != np.clip(k_intermediate, min(x_taper_1), max(x_taper_1)))
                or (k_intermediate != np.clip(k_intermediate, min(x_taper_0), max(x_taper_0)))
        ):
            _LOGGER.warning("Intermediate value outside of the range in Roskam's book, value clipped")

        k_taper = [
            float(K_taper0(np.clip(k_intermediate, min(x_taper_0), max(x_taper_0)))),
            float(K_taper1(np.clip(k_intermediate, min(x_taper_1), max(x_taper_1)))),
        ]

        if taper_ratio != np.clip(taper_ratio, 0.0, 1.0):
            _LOGGER.warning(
                "Sweep angle value outside of the range in Roskam's book, value clipped"
            )

        Cnr_CL2_graph_1 = float(
            interpolate.interp1d([0.0, 1.0], k_taper)(
                np.clip(taper_ratio, 0.0, 1.0)
            )
        )

        ###### GRAPH FOR x/c = 0.2 ######
        # Reading data from the first part (a) relative to the wing sweep angle
        file = pth.join(digit_figures.__path__[0], "10_44_0_2a.csv")
        db = read_csv(file)

        x_0 = db["SWEEP_25_0_X"]
        y_0 = db["SWEEP_25_0_Y"]
        errors = np.logical_or(np.isnan(x_0), np.isnan(y_0))
        x_0 = x_0[np.logical_not(errors)].tolist()
        y_0 = y_0[np.logical_not(errors)].tolist()

        x_40 = db["SWEEP_25_40_X"]
        y_40 = db["SWEEP_25_40_Y"]
        errors = np.logical_or(np.isnan(x_40), np.isnan(y_40))
        x_40 = x_40[np.logical_not(errors)].tolist()
        y_40 = y_40[np.logical_not(errors)].tolist()

        x_50 = db["SWEEP_25_50_X"]
        y_50 = db["SWEEP_25_50_Y"]
        errors = np.logical_or(np.isnan(x_50), np.isnan(y_50))
        x_50 = x_50[np.logical_not(errors)].tolist()
        y_50 = y_50[np.logical_not(errors)].tolist()

        x_60 = db["SWEEP_25_60_X"]
        y_60 = db["SWEEP_25_60_Y"]
        errors = np.logical_or(np.isnan(x_60), np.isnan(y_60))
        x_60 = x_60[np.logical_not(errors)].tolist()
        y_60 = y_60[np.logical_not(errors)].tolist()

        k_sweep0 = interpolate.interp1d(x_0, y_0)
        k_sweep40 = interpolate.interp1d(x_40, y_40)
        k_sweep50 = interpolate.interp1d(x_50, y_50)
        k_sweep60 = interpolate.interp1d(x_60, y_60)

        if (
                (aspect_ratio != np.clip(aspect_ratio, min(x_60), max(x_60)))
                or (aspect_ratio != np.clip(aspect_ratio, min(x_50), max(x_50)))
                or (aspect_ratio != np.clip(aspect_ratio, min(x_40), max(x_40)))
                or (aspect_ratio != np.clip(aspect_ratio, min(x_0), max(x_0)))

        ):
            _LOGGER.warning("Aspect ratio value outside of the range in Roskam's book, value clipped")

        k_sweep = [
            float(k_sweep0(np.clip(aspect_ratio, min(x_0), max(x_0)))),
            float(k_sweep40(np.clip(aspect_ratio, min(x_40), max(x_40)))),
            float(k_sweep50(np.clip(aspect_ratio, min(x_50), max(x_50)))),
            float(k_sweep60(np.clip(aspect_ratio, min(x_60), max(x_60)))),
        ]

        if sweep_25 != np.clip(sweep_25, 0.0, 60.0):
            _LOGGER.warning(
                "Sweep angle value outside of the range in Roskam's book, value clipped"
            )

        k_intermediate = float(
            interpolate.interp1d([0.0, 40.0, 50.0, 60.0], k_sweep)(
                np.clip(sweep_25, 0.0, 60.0)
            )
        )

        # Reading the second part of the figure (b) relative to the different taper ratio.
        file = pth.join(digit_figures.__path__[0], "10_44_0_2b.csv")
        db = read_csv(file)

        x_taper_0 = db["TAPER_RATIO_0_X"]
        y_taper_0 = db["TAPER_RATIO_0_Y"]
        errors = np.logical_or(np.isnan(x_taper_0), np.isnan(y_taper_0))
        x_taper_0 = x_taper_0[np.logical_not(errors)].tolist()
        y_taper_0 = y_taper_0[np.logical_not(errors)].tolist()

        x_taper_1 = db["TAPER_RATIO_1_X"]
        y_taper_1 = db["TAPER_RATIO_1_Y"]
        errors = np.logical_or(np.isnan(x_taper_1), np.isnan(y_taper_1))
        x_taper_1 = x_taper_1[np.logical_not(errors)].tolist()
        y_taper_1 = y_taper_1[np.logical_not(errors)].tolist()

        K_taper0 = interpolate.interp1d(x_taper_0, y_taper_0)
        K_taper1 = interpolate.interp1d(x_taper_1, y_taper_1)

        if (
                (k_intermediate != np.clip(k_intermediate, min(x_taper_1), max(x_taper_1)))
                or (k_intermediate != np.clip(k_intermediate, min(x_taper_0), max(x_taper_0)))
        ):
            _LOGGER.warning("Intermediate value outside of the range in Roskam's book, value clipped")

        k_taper = [
            float(K_taper0(np.clip(k_intermediate, min(x_taper_0), max(x_taper_0)))),
            float(K_taper1(np.clip(k_intermediate, min(x_taper_1), max(x_taper_1)))),
        ]

        if taper_ratio != np.clip(taper_ratio, 0.0, 1.0):
            _LOGGER.warning(
                "Sweep angle value outside of the range in Roskam's book, value clipped"
            )

        Cnr_CL2_graph_2 = float(
            interpolate.interp1d([0.0, 1.0], k_taper)(
                np.clip(taper_ratio, 0.0, 1.0)
            )
        )

        ###### GRAPH FOR x/c = 0.4 ######
        # Reading data from the first part (a) relative to the wing sweep angle
        file = pth.join(digit_figures.__path__[0], "10_44_0_4a.csv")
        db = read_csv(file)

        x_0 = db["SWEEP_25_0_X"]
        y_0 = db["SWEEP_25_0_Y"]
        errors = np.logical_or(np.isnan(x_0), np.isnan(y_0))
        x_0 = x_0[np.logical_not(errors)].tolist()
        y_0 = y_0[np.logical_not(errors)].tolist()

        x_40 = db["SWEEP_25_40_X"]
        y_40 = db["SWEEP_25_40_Y"]
        errors = np.logical_or(np.isnan(x_40), np.isnan(y_40))
        x_40 = x_40[np.logical_not(errors)].tolist()
        y_40 = y_40[np.logical_not(errors)].tolist()

        x_50 = db["SWEEP_25_50_X"]
        y_50 = db["SWEEP_25_50_Y"]
        errors = np.logical_or(np.isnan(x_50), np.isnan(y_50))
        x_50 = x_50[np.logical_not(errors)].tolist()
        y_50 = y_50[np.logical_not(errors)].tolist()

        x_60 = db["SWEEP_25_60_X"]
        y_60 = db["SWEEP_25_60_Y"]
        errors = np.logical_or(np.isnan(x_60), np.isnan(y_60))
        x_60 = x_60[np.logical_not(errors)].tolist()
        y_60 = y_60[np.logical_not(errors)].tolist()

        k_sweep0 = interpolate.interp1d(x_0, y_0)
        k_sweep40 = interpolate.interp1d(x_40, y_40)
        k_sweep50 = interpolate.interp1d(x_50, y_50)
        k_sweep60 = interpolate.interp1d(x_60, y_60)

        if (
                (aspect_ratio != np.clip(aspect_ratio, min(x_60), max(x_60)))
                or (aspect_ratio != np.clip(aspect_ratio, min(x_50), max(x_50)))
                or (aspect_ratio != np.clip(aspect_ratio, min(x_40), max(x_40)))
                or (aspect_ratio != np.clip(aspect_ratio, min(x_0), max(x_0)))

        ):
            _LOGGER.warning("Aspect ratio value outside of the range in Roskam's book, value clipped")

        k_sweep = [
            float(k_sweep0(np.clip(aspect_ratio, min(x_0), max(x_0)))),
            float(k_sweep40(np.clip(aspect_ratio, min(x_40), max(x_40)))),
            float(k_sweep50(np.clip(aspect_ratio, min(x_50), max(x_50)))),
            float(k_sweep60(np.clip(aspect_ratio, min(x_60), max(x_60)))),
        ]

        if sweep_25 != np.clip(sweep_25, 0.0, 60.0):
            _LOGGER.warning(
                "Sweep angle value outside of the range in Roskam's book, value clipped"
            )

        k_intermediate = float(
            interpolate.interp1d([0.0, 40.0, 50.0, 60.0], k_sweep)(
                np.clip(sweep_25, 0.0, 60.0)
            )
        )

        # Reading the second part of the figure (b) relative to the different taper ratio.
        file = pth.join(digit_figures.__path__[0], "10_44_0_4b.csv")
        db = read_csv(file)

        x_taper_0 = db["TAPER_RATIO_0_X"]
        y_taper_0 = db["TAPER_RATIO_0_Y"]
        errors = np.logical_or(np.isnan(x_taper_0), np.isnan(y_taper_0))
        x_taper_0 = x_taper_0[np.logical_not(errors)].tolist()
        y_taper_0 = y_taper_0[np.logical_not(errors)].tolist()

        x_taper_1 = db["TAPER_RATIO_1_X"]
        y_taper_1 = db["TAPER_RATIO_1_Y"]
        errors = np.logical_or(np.isnan(x_taper_1), np.isnan(y_taper_1))
        x_taper_1 = x_taper_1[np.logical_not(errors)].tolist()
        y_taper_1 = y_taper_1[np.logical_not(errors)].tolist()

        K_taper0 = interpolate.interp1d(x_taper_0, y_taper_0)
        K_taper1 = interpolate.interp1d(x_taper_1, y_taper_1)

        if (
                (k_intermediate != np.clip(k_intermediate, min(x_taper_1), max(x_taper_1)))
                or (k_intermediate != np.clip(k_intermediate, min(x_taper_0), max(x_taper_0)))
        ):
            _LOGGER.warning("Intermediate value outside of the range in Roskam's book, value clipped")

        k_taper = [
            float(K_taper0(np.clip(k_intermediate, min(x_taper_0), max(x_taper_0)))),
            float(K_taper1(np.clip(k_intermediate, min(x_taper_1), max(x_taper_1)))),
        ]

        if taper_ratio != np.clip(taper_ratio, 0.0, 1.0):
            _LOGGER.warning(
                "Sweep angle value outside of the range in Roskam's book, value clipped"
            )

        Cnr_CL2_graph_3 = float(
            interpolate.interp1d([0.0, 1.0], k_taper)(
                np.clip(taper_ratio, 0.0, 1.0)
            )
        )

        # INTERPOLATION BETWEEN THE 3 GRAPHS
        Cnr_CL2_graph = [
            float(Cnr_CL2_graph_1),
            float(Cnr_CL2_graph_2),
            float(Cnr_CL2_graph_3),
        ]

        if ratio_x_c != np.clip(ratio_x_c, 0.0, 0.4):
            _LOGGER.warning(
                "Ratio between x and c is outside of the range in Roskam's book, value clipped"
            )

        Cnr_CL2 = float(
            interpolate.interp1d([0.0, 0.2, 0.4], Cnr_CL2_graph)(
                np.clip(ratio_x_c, 0.0, 0.4)
            )
        )

        return Cnr_CL2



    @staticmethod
    def get_Cnr_CD0(aspect_ratio, sweep_25, x, c):
        """
        Data from Roskam - Airplane Design Part VI to compute the drag effect on the yaw damping Derivative Cnr found
        in Figure 10.45.

        :param aspect_ratio: surface aspect ratio.
        :param sweep_25: quarter-chord point line sweep angle in radians
        :param x: is the distance from the c.g. to the a.c. positive for the a.c. aft of the c.g.
        :param c: length of the mean aerodynamic chord of the surface.
        """
        ratio_x_c = x / c
        sweep_25 = sweep_25 * 180.0 / math.pi  # radians to degrees

        # ----- GRAPH 1. DATA FOR RATIO X_C = 0.0 -----
        file = pth.join(digit_figures.__path__[0], "10_45_0.csv")
        db = read_csv(file)

        x_sweep_0_graph_1 = db["SWEEP_25_0_X"]
        y_sweep_0_graph_1 = db["SWEEP_25_0_Y"]
        errors = np.logical_or(np.isnan(x_sweep_0_graph_1), np.isnan(y_sweep_0_graph_1))
        x_sweep_0_graph_1 = x_sweep_0_graph_1[np.logical_not(errors)].tolist()
        y_sweep_0_graph_1 = y_sweep_0_graph_1[np.logical_not(errors)].tolist()

        x_sweep_40_graph_1 = db["SWEEP_25_40_X"]
        y_sweep_40_graph_1 = db["SWEEP_25_40_Y"]
        errors = np.logical_or(np.isnan(x_sweep_40_graph_1), np.isnan(y_sweep_40_graph_1))
        x_sweep_40_graph_1 = x_sweep_40_graph_1[np.logical_not(errors)].tolist()
        y_sweep_40_graph_1 = y_sweep_40_graph_1[np.logical_not(errors)].tolist()

        x_sweep_50_graph_1 = db["SWEEP_25_50_X"]
        y_sweep_50_graph_1 = db["SWEEP_25_50_Y"]
        errors = np.logical_or(np.isnan(x_sweep_50_graph_1), np.isnan(y_sweep_50_graph_1))
        x_sweep_50_graph_1 = x_sweep_50_graph_1[np.logical_not(errors)].tolist()
        y_sweep_50_graph_1 = y_sweep_50_graph_1[np.logical_not(errors)].tolist()

        x_sweep_60_graph_1 = db["SWEEP_25_60_X"]
        y_sweep_60_graph_1 = db["SWEEP_25_60_Y"]
        errors = np.logical_or(np.isnan(x_sweep_60_graph_1), np.isnan(y_sweep_60_graph_1))
        x_sweep_60_graph_1 = x_sweep_60_graph_1[np.logical_not(errors)].tolist()
        y_sweep_60_graph_1 = y_sweep_60_graph_1[np.logical_not(errors)].tolist()

        k_sweep_0_graph_1 = interpolate.interp1d(x_sweep_0_graph_1, y_sweep_0_graph_1)
        k_sweep_40_graph_1 = interpolate.interp1d(x_sweep_40_graph_1, y_sweep_40_graph_1)
        k_sweep_50_graph_1 = interpolate.interp1d(x_sweep_50_graph_1, y_sweep_50_graph_1)
        k_sweep_60_graph_1 = interpolate.interp1d(x_sweep_60_graph_1, y_sweep_60_graph_1)
        if (
                (aspect_ratio != np.clip(aspect_ratio, min(x_sweep_0_graph_1), max(x_sweep_0_graph_1)))
                or (aspect_ratio != np.clip(aspect_ratio, min(x_sweep_40_graph_1), max(x_sweep_40_graph_1)))
                or (aspect_ratio != np.clip(aspect_ratio, min(x_sweep_50_graph_1), max(x_sweep_50_graph_1)))
                or (aspect_ratio != np.clip(aspect_ratio, min(x_sweep_60_graph_1), max(x_sweep_60_graph_1)))
        ):
            _LOGGER.warning("Aspect ratio value outside of the range in Roskam's book, value clipped")

        k_sweep_graph_1 = [
            float(k_sweep_0_graph_1(np.clip(aspect_ratio, min(x_sweep_0_graph_1), max(x_sweep_0_graph_1)))),
            float(k_sweep_40_graph_1(np.clip(aspect_ratio, min(x_sweep_40_graph_1), max(x_sweep_40_graph_1)))),
            float(k_sweep_50_graph_1(np.clip(aspect_ratio, min(x_sweep_50_graph_1), max(x_sweep_50_graph_1)))),
            float(k_sweep_60_graph_1(np.clip(aspect_ratio, min(x_sweep_60_graph_1), max(x_sweep_60_graph_1)))),
        ]

        if sweep_25 != np.clip(sweep_25, 0.0, 60.0):
            _LOGGER.warning(
                "Sweep angle value outside of the range in Roskam's book, value clipped"
            )

        Cnr_CD0_graph_1 = float(
            interpolate.interp1d([0.0, 40.0, 50.0, 60.0], k_sweep_graph_1)(
                np.clip(sweep_25, 0.0, 60.0)
            )
        )

        # ----- GRAPH 2. DATA FOR RATIO X_C = 0.2 -----
        file = pth.join(digit_figures.__path__[0], "10_45_0_2.csv")
        db = read_csv(file)

        x_sweep_0_graph_2 = db["SWEEP_25_0_X"]
        y_sweep_0_graph_2 = db["SWEEP_25_0_Y"]
        errors = np.logical_or(np.isnan(x_sweep_0_graph_2), np.isnan(y_sweep_0_graph_2))
        x_sweep_0_graph_2 = x_sweep_0_graph_2[np.logical_not(errors)].tolist()
        y_sweep_0_graph_2 = y_sweep_0_graph_2[np.logical_not(errors)].tolist()

        x_sweep_40_graph_2 = db["SWEEP_25_40_X"]
        y_sweep_40_graph_2 = db["SWEEP_25_40_Y"]
        errors = np.logical_or(np.isnan(x_sweep_40_graph_2), np.isnan(y_sweep_40_graph_2))
        x_sweep_40_graph_2 = x_sweep_40_graph_2[np.logical_not(errors)].tolist()
        y_sweep_40_graph_2 = y_sweep_40_graph_2[np.logical_not(errors)].tolist()

        x_sweep_50_graph_2 = db["SWEEP_25_50_X"]
        y_sweep_50_graph_2 = db["SWEEP_25_50_Y"]
        errors = np.logical_or(np.isnan(x_sweep_50_graph_2), np.isnan(y_sweep_50_graph_2))
        x_sweep_50_graph_2 = x_sweep_50_graph_2[np.logical_not(errors)].tolist()
        y_sweep_50_graph_2 = y_sweep_50_graph_2[np.logical_not(errors)].tolist()

        x_sweep_60_graph_2 = db["SWEEP_25_60_X"]
        y_sweep_60_graph_2 = db["SWEEP_25_60_Y"]
        errors = np.logical_or(np.isnan(x_sweep_60_graph_2), np.isnan(y_sweep_60_graph_2))
        x_sweep_60_graph_2 = x_sweep_60_graph_2[np.logical_not(errors)].tolist()
        y_sweep_60_graph_2 = y_sweep_60_graph_2[np.logical_not(errors)].tolist()

        k_sweep_0_graph_2 = interpolate.interp1d(x_sweep_0_graph_2, y_sweep_0_graph_2)
        k_sweep_40_graph_2 = interpolate.interp1d(x_sweep_40_graph_2, y_sweep_40_graph_2)
        k_sweep_50_graph_2 = interpolate.interp1d(x_sweep_50_graph_2, y_sweep_50_graph_2)
        k_sweep_60_graph_2 = interpolate.interp1d(x_sweep_60_graph_2, y_sweep_60_graph_2)

        if (
                (aspect_ratio != np.clip(aspect_ratio, min(x_sweep_0_graph_2), max(x_sweep_0_graph_2)))
                or (aspect_ratio != np.clip(aspect_ratio, min(x_sweep_40_graph_2), max(x_sweep_40_graph_2)))
                or (aspect_ratio != np.clip(aspect_ratio, min(x_sweep_50_graph_2), max(x_sweep_50_graph_2)))
                or (aspect_ratio != np.clip(aspect_ratio, min(x_sweep_60_graph_2), max(x_sweep_60_graph_2)))
        ):
            _LOGGER.warning("Aspect ratio value outside of the range in Roskam's book, value clipped")

        k_sweep_graph_2 = [
            float(k_sweep_0_graph_2(np.clip(aspect_ratio, min(x_sweep_0_graph_2), max(x_sweep_0_graph_2)))),
            float(k_sweep_40_graph_2(np.clip(aspect_ratio, min(x_sweep_40_graph_2), max(x_sweep_40_graph_2)))),
            float(k_sweep_50_graph_2(np.clip(aspect_ratio, min(x_sweep_50_graph_2), max(x_sweep_50_graph_2)))),
            float(k_sweep_60_graph_2(np.clip(aspect_ratio, min(x_sweep_60_graph_2), max(x_sweep_60_graph_2)))),
        ]

        if sweep_25 != np.clip(sweep_25, 0.0, 60.0):
            _LOGGER.warning(
                "Sweep angle value outside of the range in Roskam's book, value clipped"
            )

        Cnr_CD0_graph_2 = float(
            interpolate.interp1d([0.0, 40.0, 50.0, 60.0], k_sweep_graph_2)(
                np.clip(sweep_25, 0.0, 60.0)
            )
        )

        # ----- GRAPH 3. DATA FOR RATIO X_C = 0.4 -----
        file = pth.join(digit_figures.__path__[0], "10_45_0_4.csv")
        db = read_csv(file)

        x_sweep_0_graph_3 = db["SWEEP_25_0_X"]
        y_sweep_0_graph_3 = db["SWEEP_25_0_Y"]
        errors = np.logical_or(np.isnan(x_sweep_0_graph_3), np.isnan(y_sweep_0_graph_3))
        x_sweep_0_graph_3 = x_sweep_0_graph_3[np.logical_not(errors)].tolist()
        y_sweep_0_graph_3 = y_sweep_0_graph_3[np.logical_not(errors)].tolist()

        x_sweep_40_graph_3 = db["SWEEP_25_40_X"]
        y_sweep_40_graph_3 = db["SWEEP_25_40_Y"]
        errors = np.logical_or(np.isnan(x_sweep_40_graph_3), np.isnan(y_sweep_40_graph_3))
        x_sweep_40_graph_3 = x_sweep_40_graph_3[np.logical_not(errors)].tolist()
        y_sweep_40_graph_3 = y_sweep_40_graph_3[np.logical_not(errors)].tolist()

        x_sweep_50_graph_3 = db["SWEEP_25_50_X"]
        y_sweep_50_graph_3 = db["SWEEP_25_50_Y"]
        errors = np.logical_or(np.isnan(x_sweep_50_graph_3), np.isnan(y_sweep_50_graph_3))
        x_sweep_50_graph_3 = x_sweep_50_graph_3[np.logical_not(errors)].tolist()
        y_sweep_50_graph_3 = y_sweep_50_graph_3[np.logical_not(errors)].tolist()

        x_sweep_60_graph_3 = db["SWEEP_25_60_X"]
        y_sweep_60_graph_3 = db["SWEEP_25_60_Y"]
        errors = np.logical_or(np.isnan(x_sweep_60_graph_3), np.isnan(y_sweep_60_graph_3))
        x_sweep_60_graph_3 = x_sweep_60_graph_3[np.logical_not(errors)].tolist()
        y_sweep_60_graph_3 = y_sweep_60_graph_3[np.logical_not(errors)].tolist()

        k_sweep_0_graph_3 = interpolate.interp1d(x_sweep_0_graph_3, y_sweep_0_graph_3)
        k_sweep_40_graph_3 = interpolate.interp1d(x_sweep_40_graph_3, y_sweep_40_graph_3)
        k_sweep_50_graph_3 = interpolate.interp1d(x_sweep_50_graph_3, y_sweep_50_graph_3)
        k_sweep_60_graph_3 = interpolate.interp1d(x_sweep_60_graph_3, y_sweep_60_graph_3)

        if (
                (aspect_ratio != np.clip(aspect_ratio, min(x_sweep_0_graph_3), max(x_sweep_0_graph_3)))
                or (aspect_ratio != np.clip(aspect_ratio, min(x_sweep_40_graph_3), max(x_sweep_40_graph_3)))
                or (aspect_ratio != np.clip(aspect_ratio, min(x_sweep_50_graph_3), max(x_sweep_50_graph_3)))
                or (aspect_ratio != np.clip(aspect_ratio, min(x_sweep_60_graph_3), max(x_sweep_60_graph_3)))
        ):
            _LOGGER.warning("Aspect ratio value outside of the range in Roskam's book, value clipped")

        k_sweep_graph_3 = [
            float(k_sweep_0_graph_3(np.clip(aspect_ratio, min(x_sweep_0_graph_3), max(x_sweep_0_graph_3)))),
            float(k_sweep_40_graph_3(np.clip(aspect_ratio, min(x_sweep_40_graph_3), max(x_sweep_40_graph_3)))),
            float(k_sweep_50_graph_3(np.clip(aspect_ratio, min(x_sweep_50_graph_3), max(x_sweep_50_graph_3)))),
            float(k_sweep_60_graph_3(np.clip(aspect_ratio, min(x_sweep_60_graph_3), max(x_sweep_60_graph_3)))),
        ]

        if sweep_25 != np.clip(sweep_25, 0.0, 60.0):
            _LOGGER.warning(
                "Sweep angle value outside of the range in Roskam's book, value clipped"
            )

        Cnr_CD0_graph_3 = float(
            interpolate.interp1d([0.0, 40.0, 50.0, 60.0], k_sweep_graph_3)(
                np.clip(sweep_25, 0.0, 60.0)
            )
        )

        # INTERPOLATION BETWEEN THE RESULTS OF THE 3 GRAPHS
        Cnr_CD0_graph = [
            float(Cnr_CD0_graph_1),
            float(Cnr_CD0_graph_2),
            float(Cnr_CD0_graph_3),
        ]
        Cnr_CD0 = float(
            interpolate.interp1d([0.0, 0.2, 0.4], Cnr_CD0_graph)(
                np.clip(ratio_x_c, 0.0, 0.4)
            )
        )

        return Cnr_CD0
