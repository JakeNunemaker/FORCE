# -*- coding: utf-8 -*-
"""
Calculate FCR based on ATB framework

@author: mshields
"""

from __future__ import division
import numpy as np

# INPUTS
tax_rate = 0.257    # Effective tax rate (federal + local, ATB = 25.7%)
interest_dur_construct = 0.04  # interest rate for construction loan (NOMINAL, ATB = 3.7)
DF = 0.685      # Debt fraction (ATB = 68.5%)
IR = .015      # interest rate on debt (REAL, ATB = 1.5% (calculated from IR = 4% nominal))
RROE = .10     # rate of return on equity (NOMINAl, ATB = 16%)
i = .025       # historical inflation rate (ATB = 2.5%)
M = 5           # number of years in accelerated depreciation schedule (ATB = 5)
FD = np.array([.2,.32,.192,.115,.115,.0576])  # capital depreciation per year (ATB = .2, .32, 19.2, 11.5, 11.5, .0576)
t = 30        # length of time to pay off assets (ATB = 25).  From Wiser: t = 30 in 2035

#Spend schedule for calculating construction financing costs
spend_sched = {0: 0.4,
               1: 0.4,
               2: 0.2,
               3: 0.0,
               4: 0.0,
               5: 0.0}   # Different from report ({.2, .4, .4, 0, 0})

def calc_FCR(tax_rate=tax_rate,
             interest_dur_construct=interest_dur_construct,
             DF = DF,
             IR = IR,
             RROE = RROE,
             i = i,
             M = M,
             FD = FD,
             t = t,
             spend_sched = spend_sched,
             WACC = False):


    # Calculate contruction finance factor
    ConFinFactor = 0
    for ky, val in spend_sched.items():
        ConFinFactor += val * (1 + (1 - tax_rate) * ((1 + interest_dur_construct) ** (ky + 0.5) - 1))

    ### Project financing
    # Weighted average cost of capital
    if WACC:
        WACC = WACC
    else:
        WACC = np.round((1 + ((1-DF)*((1+RROE)-1)) + (DF*((1+IR)*(i+1)-1)*(1-tax_rate)))/(i+1),5)   # Real WACC

    # Calculate multiplier for taxes and depreciation (PVD)
    WACC_nom = WACC*(i+1)
    y = np.linspace(1.0,M+1,M+1)
    f = WACC_nom**-y
    PVD = np.sum(FD*f)

    # Calculate Project Finance Factor
    ProFinFactor = (1-tax_rate*PVD)/(1-tax_rate)
    # Calculate Captial Recovery Factor (CRF) and Fixed Charge Rate (FCR)
    CRF = (WACC-1)/(1 - WACC**(-t))
    FCR = CRF*ProFinFactor

    return (WACC-1), PVD, CRF, FCR

### run
if __name__ == "__main__":
    wacc, pvd, crf, fcr = calc_FCR()
    print(wacc, pvd, crf, fcr)