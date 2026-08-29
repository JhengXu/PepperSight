#!/usr/bin/env python3
"""Prepare and finalize pixel-preserving single-pepper transparent PNGs.

Workflow:
  1. `prepare` detects/boxes every verified pepper, writes local crop jobs.
  2. The Objective-C `vision_mask --manifest` tool performs Apple Vision
     foreground matting on each tight crop without regenerating source pixels.
  3. `finalize` validates, trims, pads, writes RGBA PNGs and QC artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from detect_peppers import detect, diagnostic_image, load_oriented


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
SPECIAL_PAD_STEM = "2b55121f261ea04965cf3f9ca0f8b0fb"
SECOND_PAD_STEM = "dcb35000f647f77db0db77e6c00b2309"

# A visual review of all 52 source-level contact sheets found a small set of
# crops where the foreground model grouped a cast tabletop shadow with the
# pepper.  Giving Vision a little less table context removes most of those
# shadows without changing any source pixels.  Values are deliberately stored
# per instance because a single global crop size can either retain a shadow or
# clip a long stem depending on the photo.
CROP_PAD_RATIO_OVERRIDES: dict[tuple[str, int], float] = {
    ("7d046aa3e9596c2c4f315932410cd304", 19): 0.30,
    ("b2df15cdb194251b4a44e8f776e244f4", 6): 0.36,
    ("1a2cd6fc9a4b57d759e275f1d97c38ee", 5): 0.08,
    ("1a2cd6fc9a4b57d759e275f1d97c38ee", 6): 0.18,
    ("1a2cd6fc9a4b57d759e275f1d97c38ee", 9): 0.08,
    ("1a2cd6fc9a4b57d759e275f1d97c38ee", 12): 0.08,
    ("1a2cd6fc9a4b57d759e275f1d97c38ee", 14): 0.08,
    ("1f530e7b997d3585012893d78a6b10bf", 7): 0.18,
    ("1f530e7b997d3585012893d78a6b10bf", 8): 0.12,
    ("1f530e7b997d3585012893d78a6b10bf", 10): 0.12,
    ("2a5106dd81c134835e18f7d60e165164", 10): 0.08,
    ("2a5106dd81c134835e18f7d60e165164", 13): 0.08,
    ("2a5106dd81c134835e18f7d60e165164", 15): 0.18,
    ("2b49ff4440bcd8a70b72064ba522bfb8", 5): 0.12,
    ("2b49ff4440bcd8a70b72064ba522bfb8", 10): 0.12,
    ("2b49ff4440bcd8a70b72064ba522bfb8", 13): 0.18,
    ("2b49ff4440bcd8a70b72064ba522bfb8", 15): 0.08,
    ("318675d4a55b47d3cb48b404d6e9648a", 9): 0.08,
    ("318675d4a55b47d3cb48b404d6e9648a", 11): 0.08,
    ("347452807a17d731a437c0120593e6bd", 7): 0.08,
    ("347452807a17d731a437c0120593e6bd", 9): 0.08,
    ("347452807a17d731a437c0120593e6bd", 11): 0.08,
    ("347452807a17d731a437c0120593e6bd", 14): 0.08,
    ("68f205a13735c5b0868347fd4413d243", 8): 0.12,
    ("68f205a13735c5b0868347fd4413d243", 12): 0.08,
    ("68f205a13735c5b0868347fd4413d243", 14): 0.12,
    ("cb45b866ba653632a88bd98b13bc196f", 4): 0.08,
    ("cb45b866ba653632a88bd98b13bc196f", 7): 0.08,
    ("cb45b866ba653632a88bd98b13bc196f", 8): 0.08,
    ("dd4b21ec65ed7d2a5dfbbebf46d4c7b1", 9): 0.36,
    ("dd4b21ec65ed7d2a5dfbbebf46d4c7b1", 15): 0.12,
    ("e03e654719aa1c71d85821916c56512f", 1): 0.12,
    ("e03e654719aa1c71d85821916c56512f", 12): 0.08,
    ("e03e654719aa1c71d85821916c56512f", 15): 0.08,
    ("fcc30567945e238d28b5809cb352d248", 9): 0.08,
    ("fcc30567945e238d28b5809cb352d248", 10): 0.08,
    ("fcc30567945e238d28b5809cb352d248", 13): 0.18,
    ("23ad86284c816ad7459086f1e5c8b956", 2): 0.08,
    ("2e9a6e439e83108cba0f2e09c33e2999", 10): 0.08,
    ("3179421ea0e459ae0d6c8a4540733233", 8): 0.12,
    ("520915dad4498d8250496c407f302557", 15): 0.08,
    ("791540b058f888736ddc589427165ed3", 5): 0.08,
    ("791540b058f888736ddc589427165ed3", 10): 0.12,
    ("929776721a77b477d113532559a6ec64", 5): 0.08,
    ("a23c6f930daf7409f5954ac2c8580cf8", 5): 0.08,
    ("aaaa7bf305d2981673060a9865ea9e87", 3): 0.08,
    ("aaaa7bf305d2981673060a9865ea9e87", 15): 0.12,
    ("ff74ebd29b1d399cd7da5770b767b143", 5): 0.08,
}

# Source-coordinate edge curves are used only where a shadow is inseparable
# from the pepper even after the crop refinement above.  Each curve keeps one
# side unchanged, provides a 2 px protection band, then feathers alpha over
# 4 px.  Coordinates refer to the EXIF-corrected source image, so the edits are
# reproducible independently of a crop's local origin.
MANUAL_EDGE_CURVES_SOURCE: dict[tuple[str, int], list[dict]] = {
    ("dd4b21ec65ed7d2a5dfbbebf46d4c7b1", 15): [
        {
            "axis": "x_from_y",
            "keep": "greater",
            "protect": 0,
            "feather": 3,
            "points": [
                (3809, 2455), (3814, 2465), (3811, 2485), (3810, 2505),
                (3809, 2525), (3810, 2545), (3812, 2565), (3816, 2585),
                (3824, 2605), (3830, 2625), (3841, 2645), (3844, 2665),
                (3850, 2685), (3855, 2705), (3862, 2725), (3870, 2745),
                (3875, 2765), (3881, 2785), (3889, 2805), (3892, 2825),
                (3895, 2845), (3897, 2865), (3898, 2885), (3899, 2905),
                (3901, 2925), (3903, 2945), (3905, 2965), (3906, 2985),
                (3907, 3005), (3910, 3025), (3912, 3045), (3916, 3065),
                (3923, 3085), (3931, 3105), (3950, 3125), (3962, 3135),
                (3972, 3145), (3980, 3155),
            ],
        },
        {
            "axis": "y_from_x",
            "keep": "less",
            "protect": 0,
            "feather": 3,
            "points": [
                (3940, 3125), (3952, 3135), (3962, 3145), (3972, 3151),
                (3982, 3156), (3992, 3159), (4004, 3159), (4014, 3157),
                (4022, 3151), (4030, 3141), (4036, 3125),
            ],
        },
    ],
    ("e03e654719aa1c71d85821916c56512f", 15): [
        {
            "axis": "x_from_y",
            "keep": "greater",
            "protect": 0,
            "feather": 3,
            "points": [
                (3705, 2548), (3699, 2558), (3691, 2568), (3689, 2578),
                (3695, 2588), (3697, 2598), (3697, 2608), (3698, 2618),
                (3699, 2628), (3701, 2638), (3703, 2648), (3705, 2658),
                (3707, 2668), (3709, 2678), (3712, 2688), (3714, 2698),
                (3716, 2708), (3718, 2718), (3720, 2728), (3722, 2738),
                (3724, 2748), (3727, 2758), (3729, 2768), (3731, 2778),
                (3733, 2788), (3735, 2798), (3737, 2808), (3739, 2818),
                (3742, 2828), (3745, 2838), (3747, 2848), (3750, 2858),
                (3753, 2868), (3756, 2878), (3759, 2888), (3762, 2898),
                (3764, 2908), (3766, 2918), (3768, 2928), (3770, 2938),
                (3772, 2948), (3774, 2958), (3775, 2968), (3777, 2978),
                (3778, 2988), (3780, 2998), (3781, 3008), (3783, 3018),
                (3785, 3028), (3787, 3038), (3788, 3048), (3786, 3058),
                (3783, 3068), (3780, 3078), (3777, 3088), (3776, 3098),
                (3775, 3108), (3775, 3118), (3777, 3128), (3785, 3138),
                (3792, 3148),
            ],
        },
        {
            "axis": "y_from_x",
            "keep": "less",
            "protect": 0,
            "feather": 3,
            "points": [
                (3775, 3132), (3779, 3137), (3783, 3141), (3787, 3145),
                (3792, 3148), (3797, 3150), (3802, 3150), (3807, 3148),
                (3812, 3144), (3816, 3138), (3819, 3131),
            ],
        },
    ],
    ("1a2cd6fc9a4b57d759e275f1d97c38ee", 12): [
        {
            "axis": "x_from_y",
            "keep": "greater",
            "protect": 0,
            "feather": 3,
            "points": [
                (2173, 2858), (2173, 2878), (2171, 2898), (2169, 2918),
                (2165, 2938), (2160, 2958), (2152, 2978), (2148, 2998),
                (2145, 3018), (2142, 3038), (2145, 3058), (2151, 3078),
                (2158, 3098), (2166, 3118), (2175, 3138), (2184, 3158),
                (2191, 3178), (2203, 3198), (2215, 3218), (2230, 3238),
                (2245, 3258), (2254, 3278), (2281, 3298), (2305, 3318),
                (2323, 3338),
            ],
        },
        {
            "axis": "y_from_x",
            "keep": "less",
            "protect": 0,
            "feather": 3,
            "points": [
                (2253, 3294), (2268, 3306), (2283, 3317), (2303, 3328),
                (2323, 3337), (2343, 3344), (2363, 3349), (2383, 3352),
                (2403, 3354), (2421, 3353), (2431, 3346),
            ],
        },
    ],
    ("2a5106dd81c134835e18f7d60e165164", 13): [
        {
            "axis": "x_from_y",
            "keep": "greater",
            "protect": 0,
            "feather": 3,
            "points": [
                (2222, 2669), (2197, 2689), (2187, 2709), (2179, 2729),
                (2175, 2749), (2175, 2769), (2175, 2789), (2180, 2809),
                (2185, 2829), (2195, 2849), (2210, 2869), (2225, 2889),
                (2230, 2909), (2234, 2929), (2237, 2949), (2239, 2969),
                (2240, 2989), (2241, 3009), (2242, 3029), (2243, 3049),
                (2244, 3069), (2243, 3089), (2243, 3109), (2245, 3129),
                (2249, 3149), (2253, 3169), (2257, 3189), (2263, 3209),
                (2269, 3229), (2275, 3249), (2280, 3269), (2285, 3289),
                (2293, 3309), (2301, 3329), (2310, 3349), (2317, 3369),
                (2325, 3389), (2337, 3409), (2350, 3429), (2365, 3449),
                (2385, 3469), (2408, 3489), (2430, 3509), (2455, 3529),
            ],
        },
        {
            "axis": "y_from_x",
            "keep": "less",
            "protect": 0,
            "feather": 3,
            "points": [
                (2415, 3504), (2430, 3516), (2445, 3527),
                (2460, 3533), (2475, 3536), (2485, 3532),
            ],
        },
    ],
    ("2b49ff4440bcd8a70b72064ba522bfb8", 5): [
        {
            "axis": "y_from_x",
            "keep": "less",
            "protect": 0,
            "feather": 3,
            "points": [
                (3119, 1668), (3139, 1671), (3164, 1668), (3184, 1660),
                (3199, 1648), (3214, 1637), (3234, 1626), (3254, 1616),
                (3264, 1615),
            ],
        }
    ],
    ("2b49ff4440bcd8a70b72064ba522bfb8", 15): [
        {
            "axis": "x_from_y",
            "keep": "greater",
            "protect": 0,
            "feather": 3,
            "points": [
                (3635, 2548), (3638, 2568), (3637, 2588), (3640, 2613),
                (3646, 2638), (3656, 2663), (3665, 2688), (3674, 2713),
                (3684, 2738), (3697, 2763), (3711, 2788), (3726, 2813),
                (3742, 2838), (3757, 2863), (3772, 2888), (3791, 2913),
                (3811, 2938), (3833, 2963), (3846, 2983), (3858, 2998),
                (3870, 3008),
            ],
        }
    ],
    ("68f205a13735c5b0868347fd4413d243", 14): [
        {
            "axis": "x_from_y",
            "keep": "greater",
            "protect": 0,
            "feather": 3,
            "points": [
                (3525, 2888), (3542, 2908), (3557, 2928), (3571, 2948),
                (3583, 2968), (3595, 2988), (3605, 3008), (3613, 3028),
                (3620, 3048), (3625, 3068), (3628, 3088), (3629, 3108),
                (3630, 3128), (3633, 3148), (3631, 3168), (3626, 3188),
                (3619, 3208), (3609, 3228), (3596, 3241), (3582, 3247),
                (3569, 3252), (3562, 3255),
            ],
        },
        {
            "axis": "y_from_x",
            "keep": "less",
            "protect": 0,
            "feather": 3,
            "points": [
                (3546, 3247), (3553, 3251), (3562, 3254), (3574, 3256),
                (3587, 3257), (3601, 3257), (3615, 3255), (3630, 3252),
                (3645, 3247), (3660, 3238), (3669, 3230),
            ],
        },
    ],
    ("cb45b866ba653632a88bd98b13bc196f", 4): [
        {
            "axis": "x_from_y",
            "keep": "greater",
            "protect": 0,
            "feather": 3,
            "points": [
                (3116, 1181), (3118, 1191), (3122, 1201), (3127, 1216),
                (3133, 1236), (3140, 1256), (3148, 1276), (3156, 1296),
                (3165, 1316), (3174, 1336), (3183, 1356), (3192, 1376),
                (3201, 1396), (3209, 1416), (3217, 1436), (3225, 1456),
                (3232, 1476), (3238, 1496), (3242, 1516), (3244, 1536),
                (3243, 1556), (3237, 1576), (3226, 1593), (3214, 1601),
                (3203, 1611), (3194, 1621), (3188, 1630), (3186, 1635),
            ],
        },
        {
            "axis": "y_from_x",
            "keep": "less",
            "protect": 0,
            "feather": 3,
            "points": [
                (3168, 1633), (3179, 1634), (3186, 1635), (3199, 1635),
                (3214, 1633), (3229, 1629), (3244, 1623), (3258, 1614),
                (3270, 1602), (3279, 1589),
            ],
        },
    ],
    ("1f530e7b997d3585012893d78a6b10bf", 7): [
        {
            "axis": "x_from_y",
            "keep": "greater",
            "protect": 0,
            "feather": 3,
            "points": [
                (1296, 1939), (1294, 1966), (1298, 1992),
                (1311, 2016), (1327, 2039), (1340, 2059),
            ],
        },
        {
            "axis": "x_from_y",
            "keep": "less",
            "protect": 0,
            "feather": 3,
            "points": [
                (1364, 1919), (1379, 1944), (1391, 1979), (1398, 2019),
                (1403, 2064), (1407, 2109), (1408, 2154), (1405, 2199),
                (1398, 2239), (1385, 2279), (1376, 2319), (1371, 2359),
                (1362, 2394), (1349, 2424), (1338, 2442),
            ],
        },
    ],
    ("b2df15cdb194251b4a44e8f776e244f4", 6): [
        {
            "axis": "y_from_x",
            "keep": "less",
            "points": [
                (355, 1862), (370, 1883), (390, 1904), (413, 1921),
                (437, 1931), (463, 1934), (489, 1929), (514, 1916),
                (535, 1896), (551, 1871), (561, 1844),
            ],
        }
    ],
    ("23ad86284c816ad7459086f1e5c8b956", 2): [
        {
            "axis": "x_from_y",
            "keep": "less",
            "points": [
                (1358, 1430), (1358, 1434), (1357, 1438), (1355, 1442),
                (1353, 1446), (1351, 1450), (1349, 1454), (1347, 1458),
                (1343, 1478), (1333, 1503), (1319, 1528),
                (1304, 1553), (1290, 1578), (1274, 1603), (1260, 1628),
                (1248, 1653), (1236, 1678), (1225, 1703), (1214, 1728),
                (1203, 1753), (1192, 1778), (1181, 1803), (1170, 1828),
                (1159, 1853), (1148, 1878), (1138, 1903),
            ],
        },
        {
            "axis": "x_from_y",
            "keep": "less",
            "protect": 0,
            "feather": 2,
            "points": [
                (1346, 1478), (1337, 1498), (1327, 1518), (1316, 1538),
                (1304, 1553), (1294, 1568), (1284, 1583), (1273, 1598),
                (1260, 1618), (1249, 1628), (1240, 1643), (1235, 1658),
                (1228, 1678), (1220, 1698), (1215, 1718), (1208, 1738),
                (1200, 1758), (1195, 1778),
            ],
        },
    ],
    ("2e9a6e439e83108cba0f2e09c33e2999", 10): [
        {
            "axis": "x_from_y",
            "keep": "greater",
            "points": [
                (3387, 2046), (3387, 2054), (3387, 2062), (3388, 2070),
                (3388, 2076), (3389, 2082), (3390, 2085), (3396, 2110),
                (3406, 2135), (3418, 2160),
                (3432, 2185), (3446, 2210), (3460, 2235), (3476, 2260),
                (3491, 2285), (3507, 2310), (3523, 2335), (3540, 2360),
                (3557, 2385), (3575, 2410), (3589, 2425),
            ],
        },
        {
            "axis": "x_from_y",
            "keep": "greater",
            "protect": 0,
            "feather": 2,
            "points": [
                (3387, 2046), (3387, 2054), (3387, 2062), (3388, 2070),
                (3388, 2076), (3389, 2082), (3390, 2085), (3396, 2110),
                (3406, 2135), (3418, 2160),
                (3432, 2185), (3446, 2210), (3462, 2235), (3486, 2260),
                (3498, 2280), (3507, 2295), (3518, 2310), (3536, 2335),
                (3552, 2360), (3566, 2385), (3579, 2410), (3589, 2425),
            ],
        },
    ],
    ("791540b058f888736ddc589427165ed3", 5): [
        {
            "axis": "y_from_x",
            "keep": "less",
            "protect": 0,
            "feather": 3,
            "points": [
                (3593, 1524), (3689, 1524), (3700, 1539), (3717, 1554),
                (3738, 1571), (3762, 1589), (3790, 1607), (3819, 1624),
                (3849, 1639), (3878, 1652), (3904, 1664), (3921, 1677),
                (3929, 1690), (3927, 1703), (3994, 1771),
            ],
        }
    ],
    ("791540b058f888736ddc589427165ed3", 10): [
        {
            "axis": "y_from_x",
            "keep": "less",
            "protect": 0,
            "feather": 3,
            "points": [
                (3681, 2352), (3735, 2352), (3746, 2352), (3758, 2355),
                (3770, 2357), (3782, 2359), (3793, 2362), (3802, 2366),
                (3992, 2469),
            ],
        }
    ],
    ("aaaa7bf305d2981673060a9865ea9e87", 15): [
        {
            "axis": "x_from_y",
            "keep": "greater",
            "protect": 0,
            "feather": 3,
            "points": [
                (3863, 2570), (3863, 2591), (3860, 2604), (3853, 2618),
                (3844, 2633), (3835, 2652), (3830, 2675), (3829, 2700),
                (3829, 2730), (3832, 2760), (3836, 2790), (3842, 2820),
                (3849, 2850), (3858, 2880), (3868, 2910), (3879, 2940),
                (3891, 2970), (3904, 3000), (3910, 3020), (3910, 3035),
                (3765, 3104),
            ],
        }
    ],
    ("aaaa7bf305d2981673060a9865ea9e87", 3): [
        {
            "axis": "y_from_x",
            "keep": "less",
            "protect": 0,
            "feather": 3,
            "points": [
                (2234, 1406), (2241, 1411), (2249, 1417), (2260, 1423),
                (2271, 1428), (2282, 1429), (2292, 1425), (2301, 1418),
                (2305, 1408),
            ],
        }
    ],
    ("fcc30567945e238d28b5809cb352d248", 10): [
        {
            "axis": "x_from_y",
            "keep": "greater",
            "protect": 0,
            "feather": 3,
            "points": [
                (3360, 1778), (3360, 1798), (3343, 1828), (3324, 1853),
                (3324, 1878), (3337, 1903), (3354, 1928), (3373, 1953),
                (3390, 1978), (3404, 2003), (3416, 2028), (3426, 2053),
                (3441, 2078), (3452, 2103), (3460, 2128), (3466, 2153),
                (3471, 2178), (3470, 2203), (3470, 2228), (3471, 2253),
                (3474, 2278), (3479, 2303), (3488, 2313), (3492, 2318),
                (3272, 2363),
            ],
        }
    ],
}

MANUAL_CUT_POLYGONS_SOURCE: dict[tuple[str, int], list[dict]] = {
    ("fcc30567945e238d28b5809cb352d248", 13): [
        {
            "blur_radius": 1.5,
            "points": [
                (2936, 2746), (2940, 2752), (2946, 2758), (2950, 2764),
                (2950, 2770), (2947, 2776), (2946, 2783), (2948, 2793),
                (2954, 2803), (2959, 2813), (2963, 2823), (2968, 2833),
                (2974, 2843), (2981, 2853), (2989, 2863), (2996, 2873),
                (3003, 2883), (3010, 2893), (3017, 2903), (3023, 2913),
                (3028, 2923), (3034, 2933), (3041, 2943), (3049, 2953),
                (3056, 2963), (3060, 2973), (3063, 2983), (3067, 2993),
                (3072, 3003), (3077, 3013), (3081, 3023), (3086, 3033),
                (3091, 3043), (3096, 3053), (3101, 3063), (3105, 3073),
                (3109, 3083), (3113, 3093), (3117, 3103), (3121, 3113),
                (3125, 3123), (3129, 3133), (3133, 3143), (3136, 3153),
                (3140, 3163), (3145, 3173), (3151, 3183), (3158, 3191),
                (3165, 3196), (3172, 3198), (3178, 3195), (3181, 3190),
                (3230, 3248), (2884, 3248), (2884, 2746),
            ],
        },
    ],
    ("fcc30567945e238d28b5809cb352d248", 9): [
        {
            "blur_radius": 1.5,
            "points": [
                (2587, 2047), (2591, 2057), (2600, 2072), (2618, 2092),
                (2633, 2112), (2650, 2132), (2666, 2152), (2679, 2172),
                (2692, 2192), (2706, 2212), (2721, 2232), (2735, 2252),
                (2749, 2272), (2766, 2292), (2782, 2312), (2796, 2332),
                (2809, 2352), (2816, 2367), (2820, 2377), (2831, 2387),
                (2537, 2433), (2537, 2047),
            ],
        },
    ],
    ("1f530e7b997d3585012893d78a6b10bf", 8): [
        {
            "blur_radius": 1.5,
            "points": [
                (2012, 2030), (2018, 2048), (2029, 2058), (2031, 2080),
                (2028, 2105), (2021, 2135), (2014, 2165), (2007, 2195),
                (2000, 2225), (1993, 2255), (1989, 2285), (1989, 2300),
                (2098, 2300), (2098, 2000), (2012, 2000),
            ],
        },
        {
            "blur_radius": 1.5,
            "points": [
                (1859, 2295), (1874, 2300), (1882, 2320), (1884, 2340),
                (1886, 2352), (1892, 2365), (1899, 2378), (1909, 2389),
                (1924, 2400), (1944, 2404), (1964, 2405), (1984, 2402),
                (2004, 2396), (2019, 2382), (2032, 2367), (2042, 2350),
                (2049, 2330), (2059, 2315), (2098, 2300), (2098, 2474),
                (1824, 2474), (1824, 2300),
            ],
        },
    ],
    ("2b49ff4440bcd8a70b72064ba522bfb8", 13): [
        {
            "blur_radius": 1.5,
            "points": [
                (2052, 2924), (2058, 2944), (2068, 2974), (2080, 3014),
                (2093, 3049), (2114, 3084), (2138, 3114), (2166, 3139),
                (2193, 3159), (2218, 3174), (2258, 3186), (2288, 3193),
                (2318, 3200), (2348, 3203), (2378, 3203), (2408, 3203),
                (2438, 3202), (2458, 3196), (1972, 3261), (1972, 2924),
            ],
        },
    ],
    ("347452807a17d731a437c0120593e6bd", 7): [
        {
            "blur_radius": 1.5,
            "points": [
                (1325, 3484), (1338, 3493), (1347, 3497), (1360, 3495),
                (1374, 3495), (1387, 3498), (1396, 3504), (1406, 3517),
                (1418, 3534), (1514, 3501), (1514, 3456), (1290, 3456),
                (1290, 3478),
            ],
        },
        {
            "blur_radius": 1.5,
            "points": [
                (1394, 3496), (1405, 3521), (1418, 3556), (1434, 3606),
                (1448, 3656), (1461, 3706), (1468, 3746), (1466, 3766),
                (1458, 3788), (1444, 3806), (1436, 3818), (1436, 3836),
                (1427, 3856), (1514, 3856), (1514, 3456), (1402, 3456),
            ],
        },
    ],
    ("347452807a17d731a437c0120593e6bd", 9): [
        {
            "blur_radius": 1.5,
            "points": [
                (2616, 3194), (2628, 3190), (2640, 3185), (2651, 3180),
                (2662, 3175), (2673, 3171), (2684, 3170), (2695, 3173),
                (2703, 3178), (2713, 3186), (2726, 3200), (2814, 3167),
                (2814, 3112), (2508, 3112), (2508, 3170),
            ],
        },
        {
            "blur_radius": 1.5,
            "points": [
                (2623, 3160), (2636, 3184), (2636, 3207), (2636, 3242),
                (2641, 3282), (2649, 3322), (2655, 3362), (2658, 3412),
                (2656, 3457), (2647, 3497), (2633, 3532), (2615, 3567),
                (2599, 3612), (2584, 3657), (2576, 3697), (2570, 3712),
                (2560, 3732), (2563, 3754), (2576, 3774), (2590, 3792),
                (2508, 3804), (2508, 3112), (2623, 3112),
            ],
        },
    ],
    ("347452807a17d731a437c0120593e6bd", 11): [
        {
            "blur_radius": 1.5,
            "points": [
                (946, 4245), (968, 4253), (1000, 4253), (1015, 4258),
                (1025, 4265), (1030, 4273), (1038, 4283), (1120, 4269),
                (1120, 4211), (870, 4211), (870, 4239),
            ],
        },
        {
            "blur_radius": 1.5,
            "points": [
                (1024, 4269), (1016, 4283), (1006, 4291), (996, 4296),
                (986, 4306), (976, 4321), (972, 4341), (973, 4361),
                (978, 4381), (986, 4401), (996, 4421), (1006, 4441),
                (1014, 4461), (1025, 4481), (1037, 4501), (1048, 4521),
                (1058, 4536), (1063, 4556), (1066, 4576), (1066, 4596),
                (1063, 4616), (1057, 4636), (1052, 4649), (1060, 4659),
                (1120, 4659), (1120, 4211), (1040, 4211),
            ],
        },
    ],
    ("347452807a17d731a437c0120593e6bd", 14): [
        {
            "blur_radius": 1.5,
            "points": [
                (2834, 3960), (2856, 3976), (2844, 3985), (2832, 4000),
                (2822, 4020), (2818, 4045), (2816, 4070), (2814, 4090),
                (2813, 4110), (2819, 4130), (2832, 4145), (2842, 4158),
                (2764, 4165), (2764, 3940), (2834, 3940),
            ],
        },
        {
            "blur_radius": 1.5,
            "points": [
                (2860, 4085), (2860, 4105), (2864, 4123), (2872, 4145),
                (2884, 4170), (2902, 4200), (2922, 4230), (2938, 4260),
                (2949, 4290), (2959, 4320), (2969, 4355), (2977, 4390),
                (2981, 4420), (2984, 4428), (2994, 4430), (3029, 4430),
                (3029, 3940), (2864, 3940),
            ],
        },
        {
            "blur_radius": 1.5,
            "points": [
                (2894, 4430), (2904, 4438), (2914, 4446), (2924, 4452),
                (2939, 4456), (2954, 4456), (2966, 4452), (2974, 4446),
                (2980, 4438), (2984, 4428), (2994, 4420), (3029, 4420),
                (3029, 4509), (2764, 4509), (2764, 4430),
            ],
        },
    ],
    ("dd4b21ec65ed7d2a5dfbbebf46d4c7b1", 9): [
        {
            "blur_radius": 1.5,
            "points": [
                (2808, 1693), (2798, 1714), (2792, 1741), (2791, 1771),
                (2794, 1804), (2799, 1839), (2806, 1874), (2816, 1909),
                (2828, 1944), (2842, 1979), (2860, 2012), (2880, 2039),
                (2899, 2063), (2921, 2082), (2947, 2099), (2973, 2109),
                (3000, 2108), (3023, 2101), (3040, 2092), (3203, 2289),
                (2618, 2289), (2618, 1693),
            ],
        }
    ],
}


def clamp_box(box: tuple[int, int, int, int], size: tuple[int, int]) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    width, height = size
    return max(0, x0), max(0, y0), min(width, x1), min(height, y1)


def crop_box_for_item(
    source_stem: str,
    index: int,
    bbox: tuple[int, int, int, int],
    source_size: tuple[int, int],
) -> tuple[tuple[int, int, int, int], str]:
    x0, y0, x1, y1 = bbox
    width = x1 - x0
    height = y1 - y0
    # Dark/charred tips and green stems can extend well beyond the conservative
    # red seed.  Forty percent leaves enough source context for a complete mask
    # while the verified photos still keep neighboring peppers outside the crop.
    override_ratio = CROP_PAD_RATIO_OVERRIDES.get((source_stem, index))
    if override_ratio is None:
        pad = max(22, round(max(width, height) * 0.40))
    else:
        pad = max(10, round(max(width, height) * override_ratio))
    crop_box = clamp_box((x0 - pad, y0 - pad, x1 + pad, y1 + pad), source_size)
    guard = ""
    if source_stem == SPECIAL_PAD_STEM and x0 < 250 and 2500 < y0 < 2900:
        # Visually verified crop around the row-3 left pepper that touches a
        # bright red tabletop pad.  The finalize step masks the pad side of the
        # visible boundary while preserving the pepper.
        crop_box = (140, 2540, 560, 3420)
        guard = "red_pad_left_boundary"
    if source_stem == SECOND_PAD_STEM and x0 < 250 and 2700 < y0 < 3000:
        crop_box = (145, 2680, 570, 3460)
        guard = "red_pad_left_boundary_dcb"
    return crop_box, guard


def iter_sources(input_root: Path, only: list[str]) -> list[Path]:
    sources = sorted(path for path in input_root.glob("*/*") if path.suffix.lower() in IMAGE_SUFFIXES)
    if only:
        sources = [path for path in sources if any(token in path.stem for token in only)]
    return sources


def prepare(args: argparse.Namespace) -> int:
    output_root: Path = args.output_root
    work_root: Path = args.work_root
    annotated_root = output_root / "质检" / "标注原图"
    work_root.mkdir(parents=True, exist_ok=True)
    annotated_root.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    jobs: list[tuple[str, str]] = []
    detection_summaries: list[dict] = []

    for source_path in iter_sources(args.input_root, args.only):
        source_image, items, metadata = detect(source_path)
        if metadata["count_status"] != "ok":
            raise RuntimeError(
                f"Candidate shortfall for {source_path}: {metadata['selected']}/{metadata['verified_count']}"
            )
        group = source_path.parent.name
        stem = source_path.stem
        annotated_dir = annotated_root / group
        annotated_dir.mkdir(parents=True, exist_ok=True)
        diagnostic_image(source_image, items).save(annotated_dir / f"{stem}_编号.jpg", quality=92)

        crop_dir = work_root / "crops" / group / stem
        mask_dir = work_root / "masks" / group / stem
        final_dir = output_root / "成品" / group / stem
        crop_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        final_dir.mkdir(parents=True, exist_ok=True)

        for index, item in enumerate(items, start=1):
            crop_box, guard = crop_box_for_item(stem, index, item.bbox, source_image.size)
            filename = f"{stem}__pepper_{index:03d}.png"
            crop_path = crop_dir / filename
            mask_path = mask_dir / filename
            final_path = final_dir / filename
            source_image.crop(crop_box).save(crop_path, format="PNG")
            jobs.append((str(crop_path), str(mask_path)))
            records.append(
                {
                    "group": group,
                    "source": str(source_path),
                    "source_stem": stem,
                    "source_size": list(source_image.size),
                    "orientation_applied": metadata["orientation_applied"],
                    "index": index,
                    "detected_bbox_xyxy": list(item.bbox),
                    "crop_box_xyxy": list(crop_box),
                    "guard": guard,
                    "refinement": (
                        f"tight_crop_p{round(CROP_PAD_RATIO_OVERRIDES[(stem, index)] * 100):02d}"
                        if (stem, index) in CROP_PAD_RATIO_OVERRIDES
                        else ""
                    ),
                    "crop_path": str(crop_path),
                    "mask_path": str(mask_path),
                    "output_path": str(final_path),
                    "detector_score": round(item.score, 5),
                    "wood_fraction": round(item.wood_fraction, 5),
                }
            )
        detection_summaries.append(metadata)
        print(f"prepared {group}/{source_path.name}: {len(items)}")

    manifest_path = work_root / "work_manifest.json"
    manifest_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_root / "质检" / "detection_summary.json").write_text(
        json.dumps(detection_summaries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    jobs_path = work_root / "vision_jobs.tsv"
    jobs_path.write_text("".join(f"{source}\t{target}\n" for source, target in jobs), encoding="utf-8")
    print(f"prepared_total={len(records)}")
    print(f"jobs_manifest={jobs_path}")
    return 0


def apply_special_guard(rgba: np.ndarray, record: dict) -> None:
    guard_name = record.get("guard")
    if guard_name not in {"red_pad_left_boundary", "red_pad_left_boundary_dcb"}:
        return
    crop_x0, crop_y0, _, _ = record["crop_box_xyxy"]
    height, width = rgba.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width]
    source_y = yy + crop_y0
    source_x = xx + crop_x0
    # The pad's visible right edge slopes slightly left toward the bottom.
    if guard_name == "red_pad_left_boundary":
        pad_right = 198.0 - 0.042 * (source_y - 2680.0)
    else:
        pad_right = 190.0 - 0.046 * (source_y - 2700.0)
    guard = source_x < pad_right
    rgba[guard, 3] = 0
    rgba[guard, :3] = 0


def apply_manual_edge_curves(rgba: np.ndarray, record: dict) -> bool:
    """Lower alpha outside visually verified source-coordinate edges."""
    key = (record["source_stem"], record["index"])
    specs = MANUAL_EDGE_CURVES_SOURCE.get(key, [])
    polygon_specs = MANUAL_CUT_POLYGONS_SOURCE.get(key, [])
    if not specs and not polygon_specs:
        return False

    crop_x0, crop_y0, _, _ = record["crop_box_xyxy"]
    height, width = rgba.shape[:2]
    source_x = np.arange(width, dtype=np.float64) + crop_x0
    source_y = np.arange(height, dtype=np.float64) + crop_y0
    alpha = rgba[:, :, 3]

    for spec in specs:
        points = np.asarray(spec["points"], dtype=np.float64)
        protect = float(spec.get("protect", 2.0))
        feather = float(spec.get("feather", 4.0))
        if spec["axis"] == "y_from_x":
            points = points[np.argsort(points[:, 0])]
            dependent = source_y[:, None]
            cutoff = np.interp(source_x, points[:, 0], points[:, 1])[None, :]
            domain_1d = (source_x >= points[0, 0]) & (source_x <= points[-1, 0])
            domain = np.broadcast_to(domain_1d[None, :], (height, width))
        elif spec["axis"] == "x_from_y":
            points = points[np.argsort(points[:, 1])]
            dependent = source_x[None, :]
            cutoff = np.interp(source_y, points[:, 1], points[:, 0])[:, None]
            domain_1d = (source_y >= points[0, 1]) & (source_y <= points[-1, 1])
            domain = np.broadcast_to(domain_1d[:, None], (height, width))
        else:
            raise ValueError(f"Unknown manual curve axis: {spec['axis']}")

        signed = dependent - cutoff
        if spec["keep"] == "less":
            factor = np.clip((protect + feather - signed) / feather, 0.0, 1.0)
        elif spec["keep"] == "greater":
            factor = np.clip((signed + protect + feather) / feather, 0.0, 1.0)
        else:
            raise ValueError(f"Unknown manual curve keep side: {spec['keep']}")
        matte = np.rint(factor * 255.0).astype(np.uint8)
        alpha[domain] = np.minimum(alpha[domain], matte[domain])

    for spec in polygon_specs:
        local_points = [
            (round(x - crop_x0), round(y - crop_y0))
            for x, y in spec["points"]
        ]
        matte_image = Image.new("L", (width, height), 255)
        ImageDraw.Draw(matte_image).polygon(local_points, fill=0)
        blur_radius = float(spec.get("blur_radius", 1.5))
        if blur_radius > 0:
            matte_image = matte_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        matte = np.asarray(matte_image, dtype=np.float64)
        alpha[:] = np.rint(alpha.astype(np.float64) * matte / 255.0).astype(np.uint8)

    rgba[:, :, 3] = alpha
    rgba[alpha == 0, :3] = 0
    return True


def clean_alpha_components(rgba: np.ndarray, record: dict) -> tuple[np.ndarray, int, list[str]]:
    alpha = rgba[:, :, 3]
    binary = alpha >= 10
    label_count, labels = cv2.connectedComponents(binary.astype(np.uint8), connectivity=8)
    count = label_count - 1
    flags: list[str] = []
    if count == 0:
        return rgba, 0, ["empty_alpha"]

    crop_x0, crop_y0, _, _ = record["crop_box_xyxy"]
    seed_x0, seed_y0, seed_x1, seed_y1 = record["detected_bbox_xyxy"]
    local_seed = (
        max(0, seed_x0 - crop_x0),
        max(0, seed_y0 - crop_y0),
        min(alpha.shape[1], seed_x1 - crop_x0),
        min(alpha.shape[0], seed_y1 - crop_y0),
    )
    sx0, sy0, sx1, sy1 = local_seed
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    overlaps = np.bincount(labels[sy0:sy1, sx0:sx1].ravel(), minlength=count + 1)
    overlaps[0] = 0
    primary = int(overlaps.argmax()) if overlaps.max(initial=0) > 0 else int(sizes[1:].argmax() + 1)
    primary_size = max(1, int(sizes[primary]))

    keep = {primary}
    seed_width = max(1, sx1 - sx0)
    seed_height = max(1, sy1 - sy0)
    seed_pad = round(max(seed_width, seed_height) * 0.45)
    near_seed = (
        max(0, sx0 - seed_pad),
        max(0, sy0 - seed_pad),
        min(alpha.shape[1], sx1 + seed_pad),
        min(alpha.shape[0], sy1 + seed_pad),
    )
    for label_id in range(1, count + 1):
        if label_id == primary:
            continue
        ys, xs = np.nonzero(labels == label_id)
        if len(xs) == 0:
            continue
        touches_border = xs.min() <= 1 or ys.min() <= 1 or xs.max() >= alpha.shape[1] - 2 or ys.max() >= alpha.shape[0] - 2
        nx0, ny0, nx1, ny1 = near_seed
        close_to_seed = xs.max() >= nx0 and xs.min() < nx1 and ys.max() >= ny0 and ys.min() < ny1
        if sizes[label_id] >= primary_size * 0.012 and not touches_border and close_to_seed:
            keep.add(label_id)

    keep_mask = np.isin(labels, list(keep))
    rgba[~keep_mask, 3] = 0
    rgba[~keep_mask, :3] = 0
    if count > len(keep):
        flags.append(f"removed_components:{count - len(keep)}")
    return rgba, len(keep), flags


def trim_and_pad(rgba: np.ndarray) -> tuple[Image.Image, dict, list[str]]:
    alpha = rgba[:, :, 3]
    flags: list[str] = []
    ys, xs = np.nonzero(alpha >= 4)
    if len(xs) == 0:
        raise RuntimeError("Mask has no non-transparent pixels")
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    object_width = x1 - x0
    object_height = y1 - y0
    margin = max(8, round(max(object_width, object_height) * 0.06))

    if np.any(alpha[:2] >= 16) or np.any(alpha[-2:] >= 16) or np.any(alpha[:, :2] >= 16) or np.any(alpha[:, -2:] >= 16):
        flags.append("alpha_touches_crop_edge")
    crop = Image.fromarray(rgba, mode="RGBA").crop((x0, y0, x1, y1))
    canvas = Image.new("RGBA", (object_width + margin * 2, object_height + margin * 2), (0, 0, 0, 0))
    canvas.alpha_composite(crop, (margin, margin))
    final_array = np.asarray(canvas).copy()
    final_array[final_array[:, :, 3] == 0, :3] = 0
    canvas = Image.fromarray(final_array, mode="RGBA")
    alpha_pixels = int(np.count_nonzero(alpha >= 128))
    metrics = {
        "raw_alpha_bbox_xyxy": [x0, y0, x1, y1],
        "object_size": [object_width, object_height],
        "output_size": list(canvas.size),
        "transparent_margin": margin,
        "alpha_pixels_ge_128": alpha_pixels,
        "alpha_coverage_in_crop": round(alpha_pixels / max(1, alpha.shape[0] * alpha.shape[1]), 6),
    }
    if metrics["alpha_coverage_in_crop"] < 0.002:
        flags.append("very_low_alpha_coverage")
    if metrics["alpha_coverage_in_crop"] > 0.75:
        flags.append("very_high_alpha_coverage")
    return canvas, metrics, flags


def checkerboard(size: tuple[int, int], tile: int = 12) -> Image.Image:
    width, height = size
    yy, xx = np.mgrid[0:height, 0:width]
    pattern = ((xx // tile + yy // tile) % 2).astype(np.uint8)
    colors = np.where(pattern[:, :, None] == 0, np.array([238, 238, 238]), np.array([210, 210, 210])).astype(np.uint8)
    return Image.fromarray(colors, mode="RGB").convert("RGBA")


def make_contact_sheet(records: list[dict], destination: Path) -> None:
    columns = 5
    tile_width, tile_height = 230, 300
    rows = math.ceil(len(records) / columns)
    sheet = Image.new("RGB", (columns * tile_width, rows * tile_height), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=14)
    for position, record in enumerate(records):
        row, column = divmod(position, columns)
        tx, ty = column * tile_width, row * tile_height
        background = checkerboard((tile_width - 12, tile_height - 34), tile=14)
        with Image.open(record["output_path"]) as source:
            pepper = source.convert("RGBA")
        pepper.thumbnail((tile_width - 34, tile_height - 54), Image.Resampling.LANCZOS)
        px = (background.width - pepper.width) // 2
        py = (background.height - pepper.height) // 2
        background.alpha_composite(pepper, (px, py))
        sheet.paste(background.convert("RGB"), (tx + 6, ty + 25))
        draw.rectangle((tx + 6, ty + 5, tx + 46, ty + 23), fill=(0, 0, 0))
        draw.text((tx + 10, ty + 8), f"{record['index']:03d}", fill=(255, 255, 255), font=font)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, quality=92)


def expand_edge_jobs(args: argparse.Namespace) -> int:
    manifest_path = args.work_root / "work_manifest.json"
    records: list[dict] = json.loads(manifest_path.read_text(encoding="utf-8"))
    sources: dict[str, Image.Image] = {}
    retry_jobs: list[tuple[str, str]] = []

    for record in records:
        mask_path = Path(record["mask_path"])
        if not mask_path.exists():
            continue
        with Image.open(mask_path) as source:
            rgba = np.asarray(source.convert("RGBA")).copy()
        apply_special_guard(rgba, record)
        apply_manual_edge_curves(rgba, record)
        rgba, _, _ = clean_alpha_components(rgba, record)
        alpha = rgba[:, :, 3]
        top = bool(np.any(alpha[:2] >= 16))
        bottom = bool(np.any(alpha[-2:] >= 16))
        left = bool(np.any(alpha[:, :2] >= 16))
        right = bool(np.any(alpha[:, -2:] >= 16))
        if not any((top, bottom, left, right)):
            continue

        source_path = record["source"]
        if source_path not in sources:
            sources[source_path], _ = load_oriented(Path(source_path))
        source_image = sources[source_path]
        x0, y0, x1, y1 = record["crop_box_xyxy"]
        grow_x = max(30, round((x1 - x0) * 0.38))
        grow_y = max(30, round((y1 - y0) * 0.38))
        # Expand the touched side strongly and the other sides slightly so the
        # matting model still sees balanced background context.
        new_box = clamp_box(
            (
                x0 - (grow_x if left else max(8, grow_x // 6)),
                y0 - (grow_y if top else max(8, grow_y // 6)),
                x1 + (grow_x if right else max(8, grow_x // 6)),
                y1 + (grow_y if bottom else max(8, grow_y // 6)),
            ),
            source_image.size,
        )
        if tuple(record["crop_box_xyxy"]) == new_box:
            continue
        record["crop_box_xyxy"] = list(new_box)
        crop_path = Path(record["crop_path"])
        source_image.crop(new_box).save(crop_path, format="PNG")
        retry_jobs.append((str(crop_path), record["mask_path"]))

    for image in sources.values():
        image.close()
    manifest_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    retry_path = args.work_root / "vision_retry_jobs.tsv"
    retry_path.write_text("".join(f"{source}\t{target}\n" for source, target in retry_jobs), encoding="utf-8")
    print(f"retry_total={len(retry_jobs)}")
    print(f"retry_manifest={retry_path}")
    return 0


def finalize(args: argparse.Namespace) -> int:
    records: list[dict] = json.loads((args.work_root / "work_manifest.json").read_text(encoding="utf-8"))
    completed: list[dict] = []
    failures: list[str] = []
    for position, record in enumerate(records, start=1):
        mask_path = Path(record["mask_path"])
        if not mask_path.exists():
            failures.append(f"missing mask: {mask_path}")
            continue
        with Image.open(mask_path) as source:
            rgba = np.asarray(source.convert("RGBA")).copy()
        apply_special_guard(rgba, record)
        manually_refined = apply_manual_edge_curves(rgba, record)
        rgba, component_count, component_flags = clean_alpha_components(rgba, record)
        try:
            final_image, metrics, trim_flags = trim_and_pad(rgba)
        except RuntimeError as error:
            failures.append(f"{record['source_stem']} #{record['index']}: {error}")
            continue
        output_path = Path(record["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        final_image.save(output_path, format="PNG", optimize=True)
        record.update(metrics)
        record["alpha_component_count"] = component_count
        record["qa_flags"] = component_flags + trim_flags
        if manually_refined:
            record["qa_flags"].append("manual_edge_refinement")
        blocking_flags = [
            flag
            for flag in record["qa_flags"]
            if not flag.startswith("removed_components:") and flag != "manual_edge_refinement"
        ]
        # Removing a separate non-seed component is expected when an expanded
        # crop catches a neighboring pepper or tabletop debris; keep it as an
        # audit note, but do not treat the cleaned final PNG as failed QC.
        record["qc_status"] = "review" if blocking_flags else "ok"
        completed.append(record)
        if position == 1 or position % 50 == 0 or position == len(records):
            print(f"finalized {position}/{len(records)}")

    if failures:
        raise RuntimeError("; ".join(failures[:20]))

    by_source: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in completed:
        by_source[(record["group"], record["source_stem"])].append(record)
    contact_root = args.output_root / "质检" / "联系表"
    for (group, stem), source_records in by_source.items():
        source_records.sort(key=lambda item: item["index"])
        make_contact_sheet(source_records, contact_root / group / f"{stem}_联系表.jpg")

    csv_path = args.output_root / "manifest.csv"
    fields = [
        "group",
        "source",
        "source_stem",
        "orientation_applied",
        "index",
        "refinement",
        "detected_bbox_xyxy",
        "crop_box_xyxy",
        "output_path",
        "output_size",
        "object_size",
        "transparent_margin",
        "alpha_pixels_ge_128",
        "alpha_coverage_in_crop",
        "alpha_component_count",
        "detector_score",
        "wood_fraction",
        "qc_status",
        "qa_flags",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for record in completed:
            row = dict(record)
            for field in ("detected_bbox_xyxy", "crop_box_xyxy", "output_size", "object_size", "qa_flags"):
                row[field] = json.dumps(row[field], ensure_ascii=False)
            writer.writerow(row)

    group_counts = Counter(record["group"] for record in completed)
    review_records = [record for record in completed if record["qc_status"] == "review"]
    summary = {
        "source_images": len(by_source),
        "output_pngs": len(completed),
        "group_counts": dict(sorted(group_counts.items())),
        "review_count": len(review_records),
        "review_outputs": [record["output_path"] for record in review_records],
        "manifest": str(csv_path),
        "method": (
            "coarse red-body localization + per-crop Apple Vision foreground matting + "
            "visually verified local alpha refinement; original pixels preserved"
        ),
    }
    (args.output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("input_root", type=Path)
    prepare_parser.add_argument("output_root", type=Path)
    prepare_parser.add_argument("work_root", type=Path)
    prepare_parser.add_argument("--only", action="append", default=[])
    prepare_parser.set_defaults(func=prepare)

    finalize_parser = subparsers.add_parser("finalize")
    finalize_parser.add_argument("output_root", type=Path)
    finalize_parser.add_argument("work_root", type=Path)
    finalize_parser.set_defaults(func=finalize)

    retry_parser = subparsers.add_parser("expand-edge-jobs")
    retry_parser.add_argument("work_root", type=Path)
    retry_parser.set_defaults(func=expand_edge_jobs)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
