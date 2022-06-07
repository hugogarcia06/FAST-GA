"""Test module for aerodynamics groups."""
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

import pytest

from platform import system
import numpy as np

from .dummy_engines import ENGINE_WRAPPER_BE76 as ENGINE_WRAPPER

from .test_functions import (
    xfoil_path,
    compute_reynolds,
    cd0_high_speed,
    cd0_low_speed,
    polar,
    airfoil_slope_wt_xfoil,
    airfoil_slope_xfoil,
    comp_high_speed,
    comp_low_speed,
    hinge_moment_2d,
    hinge_moment_3d,
    hinge_moments,
    high_lift,
    extreme_cl,
    wing_extreme_cl_clean,
    htp_extreme_cl_clean,
    l_d_max,
    cnbeta,
    slipstream_openvsp_cruise,
    slipstream_openvsp_low_speed,
    compute_mach_interpolation_roskam,
    cl_alpha_vt,
    cy_delta_r,
    cm_alpha_fus,
    high_speed_connection,
    low_speed_connection,
    v_n_diagram,
    load_factor,
    propeller,
    non_equilibrated_cl_cd_polar,
    equilibrated_cl_cd_polar,
)

XML_FILE = "beechcraft_76.xml"
SKIP_STEPS = False  # avoid some tests to accelerate validation process (intermediary VLM/OpenVSP)


def test_compute_reynolds():
    """Tests high and low speed reynolds calculation."""
    compute_reynolds(
        XML_FILE,
        mach_high_speed=0.2488,
        reynolds_high_speed=4629639,
        mach_low_speed=0.1179,
        reynolds_low_speed=2746999,
    )


def test_cd0_high_speed():
    """Tests drag coefficient @ high speed."""
    cd0_high_speed(
        XML_FILE,
        ENGINE_WRAPPER,
        cd0_wing=0.00541,
        cd0_fus=0.00490,
        cd0_ht=0.00119,
        cd0_vt=0.00066,
        cd0_nac=0.00209,
        cd0_lg=0.0,
        cd0_other=0.00205,
        cd0_total=0.02040,
    )


def test_cd0_low_speed():
    """Tests drag coefficient @ low speed."""
    cd0_low_speed(
        XML_FILE,
        ENGINE_WRAPPER,
        cd0_wing=0.00587,
        cd0_fus=0.00543,
        cd0_ht=0.00129,
        cd0_vt=0.00074,
        cd0_nac=0.00229,
        cd0_lg=0.01459,
        cd0_other=0.00205,
        cd0_total=0.04036,
    )


@pytest.mark.skipif(
    system() != "Windows" and xfoil_path is None or SKIP_STEPS,
    reason="No XFOIL executable available",
)
def test_polar():
    """Tests polar execution (XFOIL) @ high and low speed."""
    polar(
        XML_FILE,
        mach_high_speed=0.245,
        reynolds_high_speed=4571770 * 1.549,
        mach_low_speed=0.1179,
        reynolds_low_speed=2746999 * 1.549,
        cdp_1_high_speed=0.0046,
        cl_max_2d=1.6965,
        cdp_1_low_speed=0.0049,
    )


@pytest.mark.skipif(
    system() != "Windows" and xfoil_path is None or SKIP_STEPS,
    reason="No XFOIL executable available (or skipped)",
)
def test_airfoil_slope():
    """Tests polar execution (XFOIL) @ low speed."""
    airfoil_slope_xfoil(
        XML_FILE,
        wing_airfoil_file="naca63_415.af",
        htp_airfoil_file="naca0012.af",
        vtp_airfoil_file="naca0012.af",
        cl_alpha_wing=6.4975,
        cl_alpha_htp=6.3321,
        cl_alpha_vtp=6.3321,
    )


def test_airfoil_slope_wt_xfoil():
    """Tests polar reading @ low speed."""
    airfoil_slope_wt_xfoil(
        XML_FILE,
        wing_airfoil_file="naca63_415.af",
        htp_airfoil_file="naca0012.af",
        vtp_airfoil_file="naca0012.af",
    )


@pytest.mark.skipif(
    system() != "Windows" or SKIP_STEPS,
    reason="No XFOIL executable available: VLM basic function not computed with "
    "empty result folder (or skipped)",
)
def test_vlm_comp_high_speed():
    """Tests vlm components @ high speed."""
    comp_high_speed(
        XML_FILE,
        use_openvsp=False,
        cl0_wing=0.0894,
        cl_alpha_wing=4.820,
        cm0=-0.0247,
        coeff_k_wing=0.0522,
        cl0_htp=-0.0058,
        cl_alpha_htp=0.5068,
        cl_alpha_htp_isolated=0.8223,
        coeff_k_htp=0.4252,
        cl_alpha_vector=np.array([4.8, 4.8, 4.86, 4.94, 5.03, 5.14]),
        mach_vector=np.array([0.0, 0.15, 0.21, 0.27, 0.33, 0.39]),
    )


@pytest.mark.skipif(
    system() != "Windows" or SKIP_STEPS,
    reason="No XFOIL executable available: VLM basic function not computed with "
    "empty result folder (or skipped)",
)
def test_vlm_comp_low_speed():
    """Tests vlm components @ low speed."""
    y_vector_wing = np.array(
        [
            0.09981667,
            0.29945,
            0.49908333,
            0.84051074,
            1.32373222,
            1.8069537,
            2.29017518,
            2.77339666,
            3.25661814,
            3.73983962,
            4.11154845,
            4.37174463,
            4.63194081,
            4.89213699,
            5.15233317,
            5.41252935,
            5.67272554,
        ]
    )
    cl_vector_wing = np.array(
        [
            0.0992114,
            0.09916514,
            0.09906688,
            0.09886668,
            0.09832802,
            0.09747888,
            0.09626168,
            0.09457869,
            0.09227964,
            0.08915983,
            0.08571362,
            0.08229423,
            0.07814104,
            0.07282649,
            0.06570522,
            0.05554471,
            0.03926555,
        ]
    )
    chord_vector_wing = np.array(
        [
            1.45415954,
            1.45415954,
            1.45415954,
            1.45415954,
            1.45415954,
            1.45415954,
            1.45415954,
            1.45415954,
            1.45415954,
            1.45415954,
            1.45415954,
            1.45415954,
            1.45415954,
            1.45415954,
            1.45415954,
            1.45415954,
            1.45415954,
        ]
    )
    y_vector_htp = np.array(
        [
            0.05551452,
            0.16654356,
            0.2775726,
            0.38860163,
            0.49963067,
            0.61065971,
            0.72168875,
            0.83271779,
            0.94374682,
            1.05477586,
            1.1658049,
            1.27683394,
            1.38786298,
            1.49889201,
            1.60992105,
            1.72095009,
            1.83197913,
        ]
    )
    cl_vector_htp = np.array(
        [
            0.09855168,
            0.09833078,
            0.09788402,
            0.09720113,
            0.09626606,
            0.09505592,
            0.09353963,
            0.09167573,
            0.08940928,
            0.08666708,
            0.08335027,
            0.07932218,
            0.07438688,
            0.06824747,
            0.06041279,
            0.04994064,
            0.03442474,
        ]
    )
    comp_low_speed(
        XML_FILE,
        use_openvsp=False,
        cl0_wing=0.0872,
        cl_alpha_wing=4.701,
        cm0=-0.0241,
        coeff_k_wing=0.0500,
        cl0_htp=-0.0055,
        cl_alpha_htp=0.5019,
        cl_alpha_htp_isolated=0.8020,
        coeff_k_htp=0.4287,
        y_vector_wing=y_vector_wing,
        cl_vector_wing=cl_vector_wing,
        chord_vector_wing=chord_vector_wing,
        cl_ref_htp=0.0820,
        y_vector_htp=y_vector_htp,
        cl_vector_htp=cl_vector_htp,
    )


@pytest.mark.skipif(
    system() != "Windows" or SKIP_STEPS, reason="OPENVSP is windows dependent platform (or skipped)"
)
def test_openvsp_comp_high_speed():
    """Tests openvsp components @ high speed."""
    comp_high_speed(
        XML_FILE,
        use_openvsp=True,
        cl0_wing=0.1170,
        cl_alpha_wing=4.591,
        cm0=-0.0264,
        coeff_k_wing=0.0483,
        cl0_htp=-0.0046,
        cl_alpha_htp=0.5433,
        cl_alpha_htp_isolated=0.8438,
        coeff_k_htp=0.6684,
        cl_alpha_vector=np.array([5.06, 5.06, 5.10, 5.15, 5.22, 5.29]),
        mach_vector=np.array([0.0, 0.15, 0.21, 0.27, 0.33, 0.38]),
    )


@pytest.mark.skipif(
    system() != "Windows" or SKIP_STEPS, reason="OPENVSP is windows dependent platform (or skipped)"
)
def test_openvsp_comp_low_speed():
    """Tests openvsp components @ low speed."""
    y_vector_wing = np.array(
        [
            0.04278,
            0.12834,
            0.21389,
            0.29945,
            0.38501,
            0.47056,
            0.55612,
            0.68047,
            0.84409,
            1.00863,
            1.174,
            1.34009,
            1.50681,
            1.67405,
            1.84172,
            2.00971,
            2.17792,
            2.34624,
            2.51456,
            2.68279,
            2.85082,
            3.01853,
            3.18584,
            3.35264,
            3.51882,
            3.68429,
            3.84895,
            4.0127,
            4.17545,
            4.3371,
            4.49758,
            4.65679,
            4.81464,
            4.97107,
            5.12599,
            5.27933,
            5.43102,
            5.58099,
            5.72918,
        ]
    )
    cl_vector_wing = np.array(
        [
            0.12775493,
            0.127785,
            0.12768477,
            0.12756449,
            0.12749433,
            0.12745424,
            0.12734398,
            0.12666241,
            0.12616125,
            0.12662232,
            0.12676264,
            0.12664236,
            0.12611114,
            0.12582047,
            0.12587058,
            0.12554984,
            0.12503866,
            0.12455755,
            0.12389602,
            0.12322447,
            0.1222422,
            0.12147042,
            0.12058839,
            0.11969633,
            0.11841337,
            0.11723063,
            0.11562693,
            0.11409339,
            0.11226918,
            0.11052515,
            0.10843032,
            0.10615507,
            0.10331852,
            0.10026146,
            0.09581119,
            0.0902283,
            0.08231002,
            0.07021209,
            0.05284199,
        ]
    )
    chord_vector_wing = np.array(
        [
            1.44645,
            1.44645,
            1.44645,
            1.44645,
            1.44645,
            1.44645,
            1.44645,
            1.44645,
            1.44645,
            1.44645,
            1.44645,
            1.44645,
            1.44645,
            1.44645,
            1.44645,
            1.44645,
            1.44645,
            1.44645,
            1.44645,
            1.44645,
            1.44645,
            1.44645,
            1.44645,
            1.44645,
            1.44645,
            1.44645,
            1.44645,
            1.44645,
            1.44645,
            1.44645,
            1.44645,
            1.44645,
            1.44645,
            1.44645,
            1.44645,
            1.44645,
            1.44645,
            1.44645,
            1.44645,
        ]
    )
    y_vector_htp = np.array(
        [
            0.03932,
            0.11797,
            0.19661,
            0.27526,
            0.35391,
            0.43255,
            0.5112,
            0.58984,
            0.66849,
            0.74713,
            0.82578,
            0.90442,
            0.98307,
            1.06172,
            1.14036,
            1.21901,
            1.29765,
            1.3763,
            1.45494,
            1.53359,
            1.61223,
            1.69088,
            1.76953,
            1.84817,
        ]
    )
    cl_vector_htp = np.array(
        [
            0.10558592,
            0.10618191,
            0.10586742,
            0.10557932,
            0.10526483,
            0.10469962,
            0.10383751,
            0.10288744,
            0.10180321,
            0.10075857,
            0.09939723,
            0.09789734,
            0.09558374,
            0.09371877,
            0.09145355,
            0.08914873,
            0.08635569,
            0.08325035,
            0.07887824,
            0.07424443,
            0.06829986,
            0.06080702,
            0.05132166,
            0.04379143,
        ]
    )
    comp_low_speed(
        XML_FILE,
        use_openvsp=True,
        cl0_wing=0.1147,
        cl_alpha_wing=4.510,
        cm0=-0.0258,
        coeff_k_wing=0.0483,
        cl0_htp=-0.0044,
        cl_alpha_htp=0.5401,
        cl_alpha_htp_isolated=0.8318,
        coeff_k_htp=0.6648,
        y_vector_wing=y_vector_wing,
        cl_vector_wing=cl_vector_wing,
        chord_vector_wing=chord_vector_wing,
        cl_ref_htp=0.0897,
        y_vector_htp=y_vector_htp,
        cl_vector_htp=cl_vector_htp,
    )


def test_2d_hinge_moment():
    """Tests tail hinge-moments."""
    hinge_moment_2d(XML_FILE, ch_alpha_2d=-0.3998, ch_delta_2d=-0.6146)


def test_3d_hinge_moment():
    """Tests tail hinge-moments."""
    hinge_moment_3d(XML_FILE, ch_alpha=-0.2625, ch_delta=-0.6822)


def test_all_hinge_moment():
    """Tests tail hinge-moments full computation."""
    hinge_moments(XML_FILE, ch_alpha=-0.2625, ch_delta=-0.6822)


def test_high_lift():
    """Tests high-lift contribution."""
    high_lift(
        XML_FILE,
        delta_cl0_landing=0.5037,
        delta_cl0_landing_2d=1.0673,
        delta_clmax_landing=0.3613,
        delta_cm_landing=-0.1552,
        delta_cm_landing_2d=-0.2401,
        delta_cd_landing=0.005,
        delta_cd_landing_2d=0.0086,
        delta_cl0_takeoff=0.1930,
        delta_cl0_takeoff_2d=0.4090,
        delta_clmax_takeoff=0.0740,
        delta_cm_takeoff=-0.05949,
        delta_cm_takeoff_2d=-0.0920,
        delta_cd_takeoff=0.0004,
        delta_cd_takeoff_2d=0.0006,
        cl_delta_elev=0.5115,
        cd_delta_elev=0.0680,
    )


@pytest.mark.skipif(
    system() != "Windows",
    reason="No XFOIL executable available: not computed with empty result folder",
)
def test_extreme_cl_wing_clean():
    """Tests maximum/minimum cl component with default result cl=f(y) curve."""
    wing_extreme_cl_clean(
        XML_FILE,
        cl_max_clean_wing=1.50,
        cl_min_clean_wing=-1.20,
    )


@pytest.mark.skipif(
    system() != "Windows",
    reason="No XFOIL executable available: not computed with empty result folder",
)
def test_extreme_cl_htp_clean():
    """Tests maximum/minimum cl component with default result cl=f(y) curve."""
    htp_extreme_cl_clean(
        XML_FILE,
        cl_max_clean_htp=0.30,
        cl_min_clean_htp=-0.30,
        alpha_max_clean_htp=30.39,
        alpha_min_clean_htp=-30.36,
    )


@pytest.mark.skipif(
    system() != "Windows",
    reason="No XFOIL executable available: not computed with empty result folder",
)
def test_extreme_cl():
    """Tests maximum/minimum cl component with default result cl=f(y) curve."""
    extreme_cl(
        XML_FILE,
        cl_max_takeoff_wing=1.45,
        cl_max_landing_wing=1.73,
    )


def test_l_d_max():
    """Tests best lift/drag component."""
    l_d_max(XML_FILE, l_d_max_=15.422, optimal_cl=0.6475, optimal_cd=0.0419, optimal_alpha=4.92)


def test_cnbeta():
    """Tests cn beta fuselage."""
    cnbeta(XML_FILE, cn_beta_fus=-0.0557)


@pytest.mark.skipif(
    system() != "Windows" or SKIP_STEPS, reason="OPENVSP is windows dependent platform (or skipped)"
)
def test_slipstream_openvsp_cruise():
    """Compute slipstream @ high speed."""
    y_vector_prop_on = np.array(
        [
            0.04,
            0.13,
            0.21,
            0.3,
            0.39,
            0.47,
            0.56,
            0.68,
            0.84,
            1.01,
            1.17,
            1.34,
            1.51,
            1.67,
            1.84,
            2.01,
            2.18,
            2.35,
            2.51,
            2.68,
            2.85,
            3.02,
            3.19,
            3.35,
            3.52,
            3.68,
            3.85,
            4.01,
            4.18,
            4.34,
            4.5,
            4.66,
            4.81,
            4.97,
            5.13,
            5.28,
            5.43,
            5.58,
            5.73,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )
    cl_vector_prop_on = np.array(
        [
            1.44114,
            1.42469,
            1.424,
            1.4236,
            1.42298,
            1.4217,
            1.42592,
            1.41841,
            1.41474,
            1.42406,
            1.46203,
            1.49892,
            1.51803,
            1.51493,
            1.49662,
            1.41779,
            1.38589,
            1.36438,
            1.34718,
            1.34213,
            1.30898,
            1.3053,
            1.29663,
            1.28695,
            1.26843,
            1.25268,
            1.23072,
            1.21016,
            1.18169,
            1.15659,
            1.12409,
            1.09145,
            1.04817,
            1.00535,
            0.94375,
            0.87738,
            0.78817,
            0.67273,
            0.6258,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )
    slipstream_openvsp_cruise(
        XML_FILE,
        ENGINE_WRAPPER,
        y_vector_prop_on=y_vector_prop_on,
        cl_vector_prop_on=cl_vector_prop_on,
        ct=0.04436,
        delta_cl=0.00635,
    )


@pytest.mark.skipif(
    system() != "Windows" or SKIP_STEPS, reason="OPENVSP is windows dependent platform (or skipped)"
)
def test_slipstream_openvsp_low_speed():
    """Compute slipstream @ low speed."""
    y_vector_prop_on = np.array(
        [
            0.04,
            0.13,
            0.21,
            0.3,
            0.39,
            0.47,
            0.56,
            0.68,
            0.84,
            1.01,
            1.17,
            1.34,
            1.51,
            1.67,
            1.84,
            2.01,
            2.18,
            2.35,
            2.51,
            2.68,
            2.85,
            3.02,
            3.19,
            3.35,
            3.52,
            3.68,
            3.85,
            4.01,
            4.18,
            4.34,
            4.5,
            4.66,
            4.81,
            4.97,
            5.13,
            5.28,
            5.43,
            5.58,
            5.73,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )
    cl_vector_prop_on = np.array(
        [
            1.46684,
            1.4263,
            1.42597,
            1.42597,
            1.42592,
            1.42528,
            1.4305,
            1.42393,
            1.42232,
            1.43533,
            1.53227,
            1.64675,
            1.68906,
            1.69715,
            1.66992,
            1.5344,
            1.48763,
            1.45131,
            1.4174,
            1.39078,
            1.2942,
            1.29499,
            1.28963,
            1.28212,
            1.26494,
            1.25025,
            1.22919,
            1.20921,
            1.18114,
            1.15665,
            1.125,
            1.09314,
            1.05066,
            1.00871,
            0.94801,
            0.88231,
            0.79402,
            0.679,
            0.63579,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
    )
    slipstream_openvsp_low_speed(
        XML_FILE,
        ENGINE_WRAPPER,
        y_vector_prop_on=y_vector_prop_on,
        cl_vector_prop_on=cl_vector_prop_on,
        ct=0.03487,
        delta_cl=0.02241,
    )


def test_compute_mach_interpolation_roskam():
    """Tests computation of the mach interpolation vector using Roskam's approach."""
    compute_mach_interpolation_roskam(
        XML_FILE,
        cl_alpha_vector=np.array([5.33, 5.35, 5.42, 5.54, 5.72, 5.96]),
        mach_vector=np.array([0.0, 0.08, 0.15, 0.23, 0.31, 0.39]),
    )


def test_non_equilibrated_cl_cd_polar():
    """Tests computation of the non equilibrated cl/cd polar computation."""
    non_equilibrated_cl_cd_polar(
        XML_FILE,
        cl_polar_ls_=np.array(
            [0.25, 0.33, 0.4, 0.48, 0.56, 0.64, 0.72, 0.8, 0.88, 0.96, 1.04, 1.12, 1.2, 1.28, 1.36]
        ),
        cd_polar_ls_=np.array(
            [
                0.044,
                0.0463,
                0.0492,
                0.0528,
                0.057,
                0.0618,
                0.0673,
                0.0734,
                0.0801,
                0.0875,
                0.0955,
                0.1041,
                0.1134,
                0.1233,
                0.1339,
            ]
        ),
        cl_polar_cruise_=np.array(
            [
                0.25,
                0.33,
                0.41,
                0.49,
                0.57,
                0.66,
                0.74,
                0.82,
                0.9,
                0.98,
                1.06,
                1.14,
                1.22,
                1.31,
                1.39,
            ]
        ),
        cd_polar_cruise_=np.array(
            [
                0.0241,
                0.0265,
                0.0295,
                0.0332,
                0.0375,
                0.0425,
                0.0482,
                0.0545,
                0.0615,
                0.0691,
                0.0774,
                0.0864,
                0.096,
                0.1063,
                0.1172,
            ]
        ),
    )


def test_equilibrated_cl_cd_polar():
    """Tests computation of the non equilibrated cl/cd polar computation."""
    equilibrated_cl_cd_polar(
        XML_FILE,
        cl_polar_ls_=np.array(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.59, 0.69, 0.79, 0.88, 0.98, 1.08, 1.17, 1.26]
        ),
        cd_polar_ls_=np.array(
            [
                0.042,
                0.0439,
                0.0469,
                0.0509,
                0.056,
                0.0622,
                0.0695,
                0.0778,
                0.0872,
                0.0976,
                0.1091,
                0.1216,
                0.1352,
            ]
        ),
        cl_polar_cruise_=np.array(
            [0.03, 0.14, 0.24, 0.34, 0.45, 0.55, 0.65, 0.75, 0.86, 0.96, 1.06, 1.16, 1.26]
        ),
        cd_polar_cruise_=np.array(
            [
                0.0214,
                0.0225,
                0.0249,
                0.0285,
                0.0333,
                0.0392,
                0.0464,
                0.0547,
                0.0641,
                0.0747,
                0.0865,
                0.0994,
                0.1134,
            ]
        ),
    )


def test_cl_alpha_vt():
    """Tests Cl alpha vt."""
    cl_alpha_vt(XML_FILE, cl_alpha_vt_ls=2.6812, k_ar_effective=1.8630, cl_alpha_vt_cruise=2.7321)


def test_cy_delta_r():
    """Tests cy delta of the rudder."""
    cy_delta_r(XML_FILE, cy_delta_r_=1.8882, cy_delta_r_cruise=1.9241)


def test_cm_alpha_fus():
    """Tests cy delta of the rudder."""
    cm_alpha_fus(XML_FILE, cm_alpha_fus_=-0.2018)


@pytest.mark.skipif(
    system() != "Windows" or SKIP_STEPS, reason="OPENVSP is windows dependent platform (or skipped)"
)
def test_high_speed_connection_openvsp():
    """Tests high speed components connection."""
    high_speed_connection(XML_FILE, ENGINE_WRAPPER, use_openvsp=True)


@pytest.mark.skipif(SKIP_STEPS, reason="Skip test because already performed on Cirrus")
def test_high_speed_connection_vlm():
    """Tests high speed components connection."""
    high_speed_connection(XML_FILE, ENGINE_WRAPPER, use_openvsp=False)


@pytest.mark.skipif(
    system() != "Windows" or SKIP_STEPS, reason="OPENVSP is windows dependent platform (or skipped)"
)
def test_low_speed_connection_openvsp():
    """Tests low speed components connection."""
    low_speed_connection(XML_FILE, ENGINE_WRAPPER, use_openvsp=True)


@pytest.mark.skipif(SKIP_STEPS, reason="Skip test because already performed on Cirrus")
def test_low_speed_connection_vlm():
    """Tests low speed components connection."""
    low_speed_connection(XML_FILE, ENGINE_WRAPPER, use_openvsp=False)


@pytest.mark.skipif(SKIP_STEPS, reason="Skip test because already performed on Cirrus")
def test_v_n_diagram():
    # load all inputs
    velocity_vect = np.array(
        [
            34.688,
            45.763,
            67.62,
            56.42,
            0.0,
            0.0,
            77.998,
            77.998,
            77.998,
            109.139,
            109.139,
            109.139,
            109.139,
            98.225,
            77.998,
            0.0,
            30.871,
            43.659,
            55.568,
        ]
    )
    load_factor_vect = np.array(
        [
            1.0,
            -1.0,
            3.8,
            -1.52,
            0.0,
            0.0,
            -1.52,
            3.8,
            -1.52,
            3.8,
            0.0,
            2.805,
            -0.805,
            0.0,
            0.0,
            0.0,
            1.0,
            2.0,
            2.0,
        ]
    )
    v_n_diagram(
        XML_FILE, ENGINE_WRAPPER, velocity_vect=velocity_vect, load_factor_vect=load_factor_vect
    )


def test_load_factor():
    # load all inputs
    load_factor(
        XML_FILE,
        ENGINE_WRAPPER,
        load_factor_ultimate=5.7,
        load_factor_ultimate_mtow=5.7,
        load_factor_ultimate_mzfw=5.7,
        vh=102.09,
        va=67.62,
        vc=77.998,
        vd=109.139,
    )


@pytest.mark.skipif(
    system() != "Windows" and xfoil_path is None,
    reason="No XFOIL executable available",
)
def test_propeller():
    thrust_SL = np.array(
        [
            118.28404737,
            343.04728219,
            567.810517,
            792.57375182,
            1017.33698664,
            1242.10022145,
            1466.86345627,
            1691.62669108,
            1916.3899259,
            2141.15316071,
            2365.91639553,
            2590.67963035,
            2815.44286516,
            3040.20609998,
            3264.96933479,
            3489.73256961,
            3714.49580442,
            3939.25903924,
            4164.02227405,
            4388.78550887,
            4613.54874369,
            4838.3119785,
            5063.07521332,
            5287.83844813,
            5512.60168295,
            5737.36491776,
            5962.12815258,
            6186.89138739,
            6411.65462221,
            6636.41785703,
        ]
    )
    thrust_SL_limit = np.array(
        [
            4208.4616783,
            4505.40962551,
            4789.77591159,
            5058.48426464,
            5306.73819484,
            5543.13586359,
            5794.74759939,
            6060.43907064,
            6338.4260971,
            6636.41785703,
        ]
    )
    efficiency_SL = np.array(
        [
            [
                0.05822727,
                0.13977398,
                0.18159523,
                0.19849126,
                0.2018458,
                0.1990279,
                0.19295559,
                0.18574411,
                0.1781099,
                0.17047836,
                0.16317614,
                0.15608156,
                0.14922972,
                0.14273312,
                0.13636042,
                0.1301907,
                0.12361888,
                0.11615645,
                0.10360319,
                0.09637868,
                0.09637868,
                0.09637868,
                0.09637868,
                0.09637868,
                0.09637868,
                0.09637868,
                0.09637868,
                0.09637868,
                0.09637868,
                0.09637868,
            ],
            [
                0.16440952,
                0.34718785,
                0.43041829,
                0.4636416,
                0.47226257,
                0.4698733,
                0.46045339,
                0.44883606,
                0.43531374,
                0.42188159,
                0.40813311,
                0.39476957,
                0.38164161,
                0.36868179,
                0.35626102,
                0.34406412,
                0.33189893,
                0.31952329,
                0.30582993,
                0.28635943,
                0.26619418,
                0.26619418,
                0.26619418,
                0.26619418,
                0.26619418,
                0.26619418,
                0.26619418,
                0.26619418,
                0.26619418,
                0.26619418,
            ],
            [
                0.24784447,
                0.47373779,
                0.56486852,
                0.60220775,
                0.61475448,
                0.61476364,
                0.60799142,
                0.5977694,
                0.58543972,
                0.57223204,
                0.55854393,
                0.54485208,
                0.53113312,
                0.51753348,
                0.5039672,
                0.4905188,
                0.4774016,
                0.46415346,
                0.45053412,
                0.43546808,
                0.41555585,
                0.38462898,
                0.38462898,
                0.38462898,
                0.38462898,
                0.38462898,
                0.38462898,
                0.38462898,
                0.38462898,
                0.38462898,
            ],
            [
                0.30608854,
                0.54429969,
                0.63616786,
                0.67504266,
                0.69043073,
                0.6929904,
                0.69014147,
                0.6827371,
                0.67307863,
                0.66239424,
                0.65074542,
                0.63888842,
                0.6265992,
                0.6143684,
                0.60191776,
                0.58944273,
                0.57698492,
                0.56462733,
                0.55200457,
                0.53874821,
                0.52423645,
                0.50503998,
                0.46943816,
                0.46943816,
                0.46943816,
                0.46943816,
                0.46943816,
                0.46943816,
                0.46943816,
                0.46943816,
            ],
            [
                0.34007528,
                0.58054599,
                0.67376707,
                0.71356005,
                0.73165884,
                0.73727781,
                0.73684619,
                0.73268604,
                0.72597082,
                0.71786017,
                0.70872569,
                0.69906156,
                0.68881949,
                0.67841664,
                0.66773992,
                0.65688926,
                0.64579144,
                0.63476471,
                0.62374823,
                0.61226421,
                0.59997916,
                0.58634737,
                0.56873011,
                0.53566428,
                0.5315517,
                0.5315517,
                0.5315517,
                0.5315517,
                0.5315517,
                0.5315517,
            ],
            [
                0.34307711,
                0.59743354,
                0.68946849,
                0.73294957,
                0.75358771,
                0.76278853,
                0.76519866,
                0.7635913,
                0.75924055,
                0.75331617,
                0.74643723,
                0.73881224,
                0.73061866,
                0.72206102,
                0.71313835,
                0.70394178,
                0.69452939,
                0.68501267,
                0.6752966,
                0.66543289,
                0.65516064,
                0.64401444,
                0.63164348,
                0.61538335,
                0.58596082,
                0.57801417,
                0.57801417,
                0.57801417,
                0.57801417,
                0.57801417,
            ],
            [
                0.34112256,
                0.59956891,
                0.69591324,
                0.74168853,
                0.76453618,
                0.77660405,
                0.78191473,
                0.78259502,
                0.78106457,
                0.777386,
                0.772183,
                0.766279,
                0.759598,
                0.75272135,
                0.7453902,
                0.73775181,
                0.72985757,
                0.72171251,
                0.71344196,
                0.70489371,
                0.69620055,
                0.68691217,
                0.67678873,
                0.66564501,
                0.6507914,
                0.62568955,
                0.61369045,
                0.61369045,
                0.61369045,
                0.61369045,
            ],
            [
                0.34829032,
                0.59717235,
                0.69564344,
                0.74376438,
                0.76999397,
                0.7840045,
                0.79149927,
                0.79475601,
                0.79488539,
                0.7933805,
                0.79022314,
                0.78593621,
                0.78095263,
                0.77503085,
                0.76905952,
                0.76271836,
                0.75612639,
                0.74927534,
                0.74223533,
                0.73499466,
                0.72745938,
                0.71972347,
                0.71143357,
                0.70209462,
                0.69206076,
                0.67846608,
                0.6572271,
                0.64146696,
                0.64146696,
                0.64146696,
            ],
            [
                0.32471476,
                0.58635818,
                0.69039558,
                0.74192999,
                0.7712092,
                0.78798684,
                0.79738905,
                0.80228425,
                0.80432569,
                0.80415267,
                0.80281659,
                0.80005516,
                0.7963294,
                0.79217418,
                0.78707586,
                0.78177645,
                0.77622345,
                0.77040122,
                0.76451095,
                0.7582906,
                0.75198211,
                0.74523106,
                0.7382862,
                0.7309323,
                0.72253913,
                0.71336896,
                0.70163365,
                0.68447983,
                0.663251,
                0.663251,
            ],
            [
                0.32060259,
                0.57825861,
                0.68342211,
                0.73802561,
                0.76980819,
                0.78909635,
                0.80050975,
                0.80718087,
                0.81063591,
                0.81174742,
                0.81157086,
                0.81028252,
                0.80785159,
                0.80471234,
                0.80107565,
                0.79676752,
                0.79205362,
                0.78712525,
                0.78197697,
                0.77675869,
                0.77131429,
                0.76570256,
                0.75976232,
                0.75338828,
                0.74681263,
                0.73940135,
                0.73114866,
                0.72130738,
                0.70767664,
                0.68050973,
            ],
        ]
    )
    thrust_CL = np.array(
        [
            93.15819825,
            272.74322548,
            452.3282527,
            631.91327993,
            811.49830716,
            991.08333439,
            1170.66836162,
            1350.25338884,
            1529.83841607,
            1709.4234433,
            1889.00847053,
            2068.59349776,
            2248.17852498,
            2427.76355221,
            2607.34857944,
            2786.93360667,
            2966.5186339,
            3146.10366112,
            3325.68868835,
            3505.27371558,
            3684.85874281,
            3864.44377003,
            4044.02879726,
            4223.61382449,
            4403.19885172,
            4582.78387895,
            4762.36890617,
            4941.9539334,
            5121.53896063,
            5301.12398786,
        ]
    )
    thrust_CL_limit = np.array(
        [
            3332.22612057,
            3568.5270316,
            3795.79754353,
            4010.15603254,
            4211.1313009,
            4403.77876683,
            4607.16086781,
            4825.00322804,
            5052.98188671,
            5301.12398786,
        ]
    )
    efficiency_CL = np.array(
        [
            [
                0.05832829,
                0.14079049,
                0.18250237,
                0.19889311,
                0.20185194,
                0.19864187,
                0.19232217,
                0.18489622,
                0.17713393,
                0.169383,
                0.16202158,
                0.15487315,
                0.14799255,
                0.14147145,
                0.1350509,
                0.12884379,
                0.12218142,
                0.1143114,
                0.098278,
                0.09629643,
                0.09629643,
                0.09629643,
                0.09629643,
                0.09629643,
                0.09629643,
                0.09629643,
                0.09629643,
                0.09629643,
                0.09629643,
                0.09629643,
            ],
            [
                0.16464549,
                0.34937694,
                0.4320869,
                0.46456366,
                0.47241919,
                0.46932065,
                0.45942554,
                0.44737059,
                0.43355694,
                0.41984293,
                0.40593461,
                0.39239829,
                0.37916171,
                0.36607983,
                0.35356898,
                0.34131591,
                0.32908306,
                0.31649795,
                0.30220592,
                0.28047326,
                0.26582644,
                0.26582644,
                0.26582644,
                0.26582644,
                0.26582644,
                0.26582644,
                0.26582644,
                0.26582644,
                0.26582644,
                0.26582644,
            ],
            [
                0.24807387,
                0.47626515,
                0.56688916,
                0.60334754,
                0.6150433,
                0.61437775,
                0.60704808,
                0.5963657,
                0.58367032,
                0.5701879,
                0.55623785,
                0.54233945,
                0.52843778,
                0.51468738,
                0.50095422,
                0.48741274,
                0.47419021,
                0.46079011,
                0.44699676,
                0.43163168,
                0.40985175,
                0.38410368,
                0.38410368,
                0.38410368,
                0.38410368,
                0.38410368,
                0.38410368,
                0.38410368,
                0.38410368,
                0.38410368,
            ],
            [
                0.30685727,
                0.54665378,
                0.63797837,
                0.67605774,
                0.69082869,
                0.69276381,
                0.68945124,
                0.68159086,
                0.67158595,
                0.66056794,
                0.64868949,
                0.63657521,
                0.62409671,
                0.61166259,
                0.59908618,
                0.58639865,
                0.57391603,
                0.56144183,
                0.54859472,
                0.53520903,
                0.52011367,
                0.50012129,
                0.46878648,
                0.46878648,
                0.46878648,
                0.46878648,
                0.46878648,
                0.46878648,
                0.46878648,
                0.46878648,
            ],
            [
                0.340129,
                0.58270009,
                0.67554724,
                0.7145794,
                0.7321444,
                0.7371614,
                0.73633602,
                0.73175949,
                0.72473221,
                0.71632368,
                0.70692595,
                0.69705051,
                0.68658939,
                0.67602641,
                0.66515926,
                0.65418825,
                0.64290923,
                0.63181933,
                0.62065987,
                0.60898019,
                0.59660921,
                0.58251464,
                0.56399548,
                0.53084885,
                0.53084885,
                0.53084885,
                0.53084885,
                0.53084885,
                0.53084885,
                0.53084885,
            ],
            [
                0.34184802,
                0.59950129,
                0.69104455,
                0.73395353,
                0.75405815,
                0.76278581,
                0.7648174,
                0.7628961,
                0.75819446,
                0.75201648,
                0.74490195,
                0.73705107,
                0.72867309,
                0.71993475,
                0.71084771,
                0.70150699,
                0.69196096,
                0.68227681,
                0.67252741,
                0.6625356,
                0.65207525,
                0.64079986,
                0.62818434,
                0.61097824,
                0.57760451,
                0.57729546,
                0.57729546,
                0.57729546,
                0.57729546,
                0.57729546,
            ],
            [
                0.3412724,
                0.60122211,
                0.69728334,
                0.74265334,
                0.76500517,
                0.77666731,
                0.78165439,
                0.78201356,
                0.78021451,
                0.77626877,
                0.7708136,
                0.76475612,
                0.75787127,
                0.75083939,
                0.74335507,
                0.73557937,
                0.72755363,
                0.71928625,
                0.71088322,
                0.70226482,
                0.69347577,
                0.68404099,
                0.67383429,
                0.66250047,
                0.64697365,
                0.62053704,
                0.61292161,
                0.61292161,
                0.61292161,
                0.61292161,
            ],
            [
                0.35036934,
                0.59905036,
                0.69698069,
                0.74466696,
                0.77044277,
                0.78410506,
                0.7912957,
                0.79426072,
                0.79414541,
                0.7924214,
                0.78904834,
                0.78457615,
                0.77938224,
                0.77333485,
                0.76721963,
                0.76076506,
                0.7540438,
                0.74707494,
                0.73994021,
                0.73259243,
                0.72498673,
                0.7171197,
                0.70878265,
                0.69934165,
                0.68926762,
                0.67541168,
                0.65337175,
                0.64065639,
                0.64065639,
                0.64065639,
            ],
            [
                0.32469377,
                0.5876176,
                0.69155958,
                0.74273127,
                0.7716199,
                0.78808525,
                0.79719517,
                0.8018334,
                0.80365341,
                0.80326908,
                0.80174261,
                0.79880096,
                0.79492425,
                0.7906093,
                0.78537555,
                0.77997202,
                0.77430827,
                0.76838767,
                0.76239203,
                0.75607923,
                0.74968677,
                0.7428594,
                0.73586644,
                0.72842889,
                0.72003172,
                0.71087646,
                0.6990703,
                0.68166696,
                0.66238912,
                0.66238912,
            ],
            [
                0.3145431,
                0.57867382,
                0.68424655,
                0.73867142,
                0.77014463,
                0.78915803,
                0.80029181,
                0.80674024,
                0.80998128,
                0.81092258,
                0.81055898,
                0.80911526,
                0.80654143,
                0.80325933,
                0.79950918,
                0.79507275,
                0.79024699,
                0.78523908,
                0.77999356,
                0.77469879,
                0.76916701,
                0.7634865,
                0.75749943,
                0.75109724,
                0.7444907,
                0.73707485,
                0.72888744,
                0.71917273,
                0.70566043,
                0.67963045,
            ],
        ]
    )
    speed = np.array(
        [
            5.0,
            15.41925926,
            25.83851852,
            36.25777778,
            46.67703704,
            57.0962963,
            67.51555556,
            77.93481481,
            88.35407407,
            98.77333333,
        ]
    )
    propeller(
        XML_FILE,
        thrust_SL=thrust_SL,
        thrust_SL_limit=thrust_SL_limit,
        efficiency_SL=efficiency_SL,
        thrust_CL=thrust_CL,
        thrust_CL_limit=thrust_CL_limit,
        efficiency_CL=efficiency_CL,
        speed=speed,
    )
