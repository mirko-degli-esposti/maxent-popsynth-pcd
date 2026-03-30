"""
attr_meta.py
------------
Attribute metadata, domain sizes, anchor marginals and
conditional probability tables (CPTs) for the Syn-ISTAT benchmark.

Syn-ISTAT is an ISTAT-inspired synthetic benchmark (K=15).
CPTs are constructed synthetically to reflect known structural
patterns of the Italian demographic system; they are NOT
derived from official ISTAT microdata.
"""

import numpy as np

# ------------------------------------------------------------------ #
#  Attribute definitions                                               #
# ------------------------------------------------------------------ #

_ATTR_DEFS = [
    ("sex",               ["F", "M"]),
    ("age",               ["0-24", "25-49", "50-66", "67+"]),
    ("marital",           ["NeverMarried", "Married", "Separated",
                           "Divorced", "Widowed"]),
    ("education",         ["LessThanHS", "HighSchool",
                           "SomeCollege", "Bachelor+"]),
    ("employment",        ["Employed", "Unemployed", "NotInLF"]),
    ("income",            ["None", "Low", "Medium",
                           "UpperMedium", "High"]),
    ("household_size",    ["1", "2", "3-4", "5+"]),
    ("has_children",      ["No", "Yes"]),
    ("residence_area",    ["Urban", "Suburban", "Rural"]),
    ("car_access",        ["NoCar", "SharedCar", "OwnCar"]),
    ("main_transport",    ["Car", "PublicTransport", "Walking",
                           "Bike", "Mixed"]),
    ("commute_time",      ["None", "<15", "15-45", "45+"]),
    ("diet_type",         ["Omnivore", "ReducedMeat", "VegetarianLike"]),
    ("alcohol_use",       ["Never", "Occasional", "Weekly", "Frequent"]),
    ("physical_activity", ["Sedentary", "Low", "Moderate", "High"]),
]

ATTR_NAMES_SYNTH   = [name for name, _ in _ATTR_DEFS]
DOMAIN_SIZES_SYNTH = np.array([len(vals) for _, vals in _ATTR_DEFS],
                               dtype=np.int32)

ATTR_META = {
    name: {
        'idx':        idx,
        'vals':       vals,
        'val_to_int': {v: i for i, v in enumerate(vals)},
    }
    for idx, (name, vals) in enumerate(_ATTR_DEFS)
}

K_SYNTH = len(ATTR_NAMES_SYNTH)

# ------------------------------------------------------------------ #
#  Anchor marginals                                                     #
# ------------------------------------------------------------------ #

marginals = {
    "sex":            {"F": 0.51,  "M": 0.49},
    "age":            {"0-24": 0.24, "25-49": 0.34,
                       "50-66": 0.24, "67+": 0.18},
    "education":      {"LessThanHS": 0.14, "HighSchool": 0.33,
                       "SomeCollege": 0.28, "Bachelor+": 0.25},
    "household_size": {"1": 0.28, "2": 0.34, "3-4": 0.30, "5+": 0.08},
    "residence_area": {"Urban": 0.42, "Suburban": 0.36, "Rural": 0.22},
    "car_access":     {"NoCar": 0.18, "SharedCar": 0.27, "OwnCar": 0.55},
    # implied marginals (derived from binary/ternary tables)
    "marital":        {"NeverMarried": 0.346, "Married": 0.426,
                       "Separated": 0.038,   "Divorced": 0.112,
                       "Widowed": 0.078},
    "employment":     {"Employed": 0.529, "Unemployed": 0.078,
                       "NotInLF": 0.393},
    "income":         {"None": 0.127, "Low": 0.251, "Medium": 0.315,
                       "UpperMedium": 0.205, "High": 0.103},
    "has_children":   {"No": 0.621, "Yes": 0.379},
    "main_transport": {"Car": 0.489, "PublicTransport": 0.211,
                       "Walking": 0.103, "Bike": 0.065, "Mixed": 0.132},
    "commute_time":   {"None": 0.371, "<15": 0.192,
                       "15-45": 0.319, "45+": 0.119},
    "diet_type":      {"Omnivore": 0.759, "ReducedMeat": 0.178,
                       "VegetarianLike": 0.063},
    "alcohol_use":    {"Never": 0.266, "Occasional": 0.336,
                       "Weekly": 0.280, "Frequent": 0.118},
    "physical_activity": {"Sedentary": 0.236, "Low": 0.290,
                          "Moderate": 0.298, "High": 0.175},
}

# ------------------------------------------------------------------ #
#  Binary CPTs                                                          #
# ------------------------------------------------------------------ #

# B1: P(marital | age)
age_marital = {
    "0-24":  {"NeverMarried": 0.88, "Married": 0.10, "Separated": 0.005,
               "Divorced": 0.01,  "Widowed": 0.005},
    "25-49": {"NeverMarried": 0.27, "Married": 0.55, "Separated": 0.05,
               "Divorced": 0.11,  "Widowed": 0.02},
    "50-66": {"NeverMarried": 0.12, "Married": 0.58, "Separated": 0.06,
               "Divorced": 0.18,  "Widowed": 0.06},
    "67+":   {"NeverMarried": 0.08, "Married": 0.42, "Separated": 0.03,
               "Divorced": 0.16,  "Widowed": 0.31},
}

# B2: P(employment | age)
age_employment = {
    "0-24":  {"Employed": 0.38, "Unemployed": 0.12, "NotInLF": 0.50},
    "25-49": {"Employed": 0.74, "Unemployed": 0.08, "NotInLF": 0.18},
    "50-66": {"Employed": 0.63, "Unemployed": 0.07, "NotInLF": 0.30},
    "67+":   {"Employed": 0.08, "Unemployed": 0.02, "NotInLF": 0.90},
}

# B3: P(employment | education)
education_employment = {
    "LessThanHS":  {"Employed": 0.39, "Unemployed": 0.11, "NotInLF": 0.50},
    "HighSchool":  {"Employed": 0.56, "Unemployed": 0.09, "NotInLF": 0.35},
    "SomeCollege": {"Employed": 0.61, "Unemployed": 0.08, "NotInLF": 0.31},
    "Bachelor+":   {"Employed": 0.71, "Unemployed": 0.05, "NotInLF": 0.24},
}

# B4: P(income | employment)
employment_income = {
    "Employed":   {"None": 0.02, "Low": 0.18, "Medium": 0.38,
                   "UpperMedium": 0.28, "High": 0.14},
    "Unemployed": {"None": 0.24, "Low": 0.46, "Medium": 0.23,
                   "UpperMedium": 0.06, "High": 0.01},
    "NotInLF":    {"None": 0.25, "Low": 0.31, "Medium": 0.25,
                   "UpperMedium": 0.13, "High": 0.06},
}

# B5: P(has_children | household_size)
household_children = {
    "1":   {"No": 0.98, "Yes": 0.02},
    "2":   {"No": 0.73, "Yes": 0.27},
    "3-4": {"No": 0.29, "Yes": 0.71},
    "5+":  {"No": 0.14, "Yes": 0.86},
}

# B6: P(main_transport | residence_area)
area_transport = {
    "Urban":    {"Car": 0.29, "PublicTransport": 0.34, "Walking": 0.14,
                 "Bike": 0.09, "Mixed": 0.14},
    "Suburban": {"Car": 0.58, "PublicTransport": 0.16, "Walking": 0.08,
                 "Bike": 0.05, "Mixed": 0.13},
    "Rural":    {"Car": 0.72, "PublicTransport": 0.05, "Walking": 0.07,
                 "Bike": 0.04, "Mixed": 0.12},
}

# B7: P(main_transport | car_access)
car_transport = {
    "NoCar":     {"Car": 0.02, "PublicTransport": 0.42, "Walking": 0.24,
                  "Bike": 0.12, "Mixed": 0.20},
    "SharedCar": {"Car": 0.41, "PublicTransport": 0.20, "Walking": 0.08,
                  "Bike": 0.05, "Mixed": 0.26},
    "OwnCar":    {"Car": 0.68, "PublicTransport": 0.10, "Walking": 0.05,
                  "Bike": 0.04, "Mixed": 0.13},
}

# B8: P(alcohol_use | age)
age_alcohol = {
    "0-24":  {"Never": 0.37, "Occasional": 0.36, "Weekly": 0.20,
               "Frequent": 0.07},
    "25-49": {"Never": 0.18, "Occasional": 0.31, "Weekly": 0.35,
               "Frequent": 0.16},
    "50-66": {"Never": 0.20, "Occasional": 0.33, "Weekly": 0.32,
               "Frequent": 0.15},
    "67+":   {"Never": 0.38, "Occasional": 0.36, "Weekly": 0.20,
               "Frequent": 0.06},
}

# B9: P(employment | sex)
sex_employment = {
    "F": {"Employed": 0.48, "Unemployed": 0.09, "NotInLF": 0.43},
    "M": {"Employed": 0.62, "Unemployed": 0.07, "NotInLF": 0.31},
}

# B10: P(income | sex)
sex_income = {
    "F": {"None": 0.17, "Low": 0.28, "Medium": 0.31,
           "UpperMedium": 0.16, "High": 0.08},
    "M": {"None": 0.07, "Low": 0.21, "Medium": 0.32,
           "UpperMedium": 0.26, "High": 0.14},
}

# B11: P(commute_time | employment)
employment_commute = {
    "Employed":   {"None": 0.05, "<15": 0.22, "15-45": 0.52, "45+": 0.21},
    "Unemployed": {"None": 0.60, "<15": 0.18, "15-45": 0.17, "45+": 0.05},
    "NotInLF":    {"None": 0.72, "<15": 0.16, "15-45": 0.10, "45+": 0.02},
}

# B12: P(diet_type | age)
age_diet = {
    "0-24":  {"Omnivore": 0.68, "ReducedMeat": 0.22, "VegetarianLike": 0.10},
    "25-49": {"Omnivore": 0.72, "ReducedMeat": 0.21, "VegetarianLike": 0.07},
    "50-66": {"Omnivore": 0.81, "ReducedMeat": 0.15, "VegetarianLike": 0.04},
    "67+":   {"Omnivore": 0.87, "ReducedMeat": 0.10, "VegetarianLike": 0.03},
}

# B13: P(physical_activity | age)
age_activity = {
    "0-24":  {"Sedentary": 0.12, "Low": 0.21, "Moderate": 0.36, "High": 0.31},
    "25-49": {"Sedentary": 0.20, "Low": 0.27, "Moderate": 0.33, "High": 0.20},
    "50-66": {"Sedentary": 0.28, "Low": 0.34, "Moderate": 0.28, "High": 0.10},
    "67+":   {"Sedentary": 0.40, "Low": 0.37, "Moderate": 0.18, "High": 0.05},
}

# ------------------------------------------------------------------ #
#  Ternary CPTs                                                         #
# ------------------------------------------------------------------ #

# T1: P(income | education, employment)
triple_eei = {
    "LessThanHS": {
        "Employed":   {"None": 0.03, "Low": 0.32, "Medium": 0.40,
                       "UpperMedium": 0.18, "High": 0.07},
        "Unemployed": {"None": 0.30, "Low": 0.48, "Medium": 0.18,
                       "UpperMedium": 0.03, "High": 0.01},
        "NotInLF":    {"None": 0.30, "Low": 0.38, "Medium": 0.22,
                       "UpperMedium": 0.08, "High": 0.02},
    },
    "HighSchool": {
        "Employed":   {"None": 0.02, "Low": 0.21, "Medium": 0.43,
                       "UpperMedium": 0.24, "High": 0.10},
        "Unemployed": {"None": 0.24, "Low": 0.47, "Medium": 0.22,
                       "UpperMedium": 0.06, "High": 0.01},
        "NotInLF":    {"None": 0.24, "Low": 0.33, "Medium": 0.26,
                       "UpperMedium": 0.12, "High": 0.05},
    },
    "SomeCollege": {
        "Employed":   {"None": 0.01, "Low": 0.16, "Medium": 0.39,
                       "UpperMedium": 0.29, "High": 0.15},
        "Unemployed": {"None": 0.22, "Low": 0.44, "Medium": 0.25,
                       "UpperMedium": 0.07, "High": 0.02},
        "NotInLF":    {"None": 0.22, "Low": 0.29, "Medium": 0.27,
                       "UpperMedium": 0.15, "High": 0.07},
    },
    "Bachelor+": {
        "Employed":   {"None": 0.01, "Low": 0.08, "Medium": 0.28,
                       "UpperMedium": 0.37, "High": 0.26},
        "Unemployed": {"None": 0.18, "Low": 0.39, "Medium": 0.28,
                       "UpperMedium": 0.11, "High": 0.04},
        "NotInLF":    {"None": 0.18, "Low": 0.21, "Medium": 0.25,
                       "UpperMedium": 0.20, "High": 0.16},
    },
}

# T2: P(main_transport | residence_area, car_access)
triple_rct = {
    "Urban": {
        "NoCar":     {"Car": 0.01, "PublicTransport": 0.48, "Walking": 0.25,
                      "Bike": 0.10, "Mixed": 0.16},
        "SharedCar": {"Car": 0.33, "PublicTransport": 0.24, "Walking": 0.10,
                      "Bike": 0.07, "Mixed": 0.26},
        "OwnCar":    {"Car": 0.56, "PublicTransport": 0.15, "Walking": 0.08,
                      "Bike": 0.05, "Mixed": 0.16},
    },
    "Suburban": {
        "NoCar":     {"Car": 0.03, "PublicTransport": 0.43, "Walking": 0.22,
                      "Bike": 0.10, "Mixed": 0.22},
        "SharedCar": {"Car": 0.46, "PublicTransport": 0.18, "Walking": 0.07,
                      "Bike": 0.04, "Mixed": 0.25},
        "OwnCar":    {"Car": 0.72, "PublicTransport": 0.08, "Walking": 0.04,
                      "Bike": 0.03, "Mixed": 0.13},
    },
    "Rural": {
        "NoCar":     {"Car": 0.04, "PublicTransport": 0.22, "Walking": 0.30,
                      "Bike": 0.10, "Mixed": 0.34},
        "SharedCar": {"Car": 0.58, "PublicTransport": 0.08, "Walking": 0.07,
                      "Bike": 0.03, "Mixed": 0.24},
        "OwnCar":    {"Car": 0.84, "PublicTransport": 0.03, "Walking": 0.03,
                      "Bike": 0.02, "Mixed": 0.08},
    },
}

# T3: P(employment | sex, age)
triple_sae = {
    "F": {
        "0-24":  {"Employed": 0.33, "Unemployed": 0.14, "NotInLF": 0.53},
        "25-49": {"Employed": 0.61, "Unemployed": 0.10, "NotInLF": 0.29},
        "50-66": {"Employed": 0.52, "Unemployed": 0.07, "NotInLF": 0.41},
        "67+":   {"Employed": 0.05, "Unemployed": 0.01, "NotInLF": 0.94},
    },
    "M": {
        "0-24":  {"Employed": 0.43, "Unemployed": 0.10, "NotInLF": 0.47},
        "25-49": {"Employed": 0.87, "Unemployed": 0.06, "NotInLF": 0.07},
        "50-66": {"Employed": 0.74, "Unemployed": 0.07, "NotInLF": 0.19},
        "67+":   {"Employed": 0.11, "Unemployed": 0.02, "NotInLF": 0.87},
    },
}
