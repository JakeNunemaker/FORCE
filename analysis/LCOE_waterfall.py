"""
Take LCOE trajectories and create waterfall plot
"""

import numpy as np
import pandas as pd
import waterfall_chart
import matplotlib.pyplot as plt

if __name__ == '__main__':
    df = pd.read_csv('results/floating_data_out.csv')

    year1 = 2023
    year2 = 2030

    def lcoe_calc(c, o, a, f):
        lcoe = 1000 * (f * c + o) / a
        return lcoe

    def delta_lcoe_calc(df, lcoe_i, capex, opex, aep, fcr, ind):  

        if ind == 'Min capex':
            d_capex = df[(df['Year']==year2)]['Min capex'].to_numpy()[0] 
            lcoe_f = lcoe_calc(d_capex, opex, aep, fcr)
            d_lcoe = lcoe_i - lcoe_f
            ind_f = d_capex
        elif ind == 'Opex':
            d_opex = df[(df['Year'] == year2)]['Opex'].to_numpy()[0]
            lcoe_f = lcoe_calc(capex, d_opex, aep, fcr)
            d_lcoe = lcoe_i - lcoe_f
            ind_f = d_opex
        elif ind == 'AEP':
            d_aep = df[(df['Year'] == year2)]['AEP'].to_numpy()[0]
            lcoe_f = lcoe_calc(capex, opex, d_aep, fcr)
            d_lcoe = lcoe_i - lcoe_f
            ind_f = d_aep
        elif ind == 'FCR':
            d_fcr = df[(df['Year'] == year2)]['FCR'].to_numpy()[0]
            lcoe_f = lcoe_calc(capex, opex, aep, d_fcr)
            d_lcoe = lcoe_i - lcoe_f
            ind_f = d_fcr

        return d_lcoe, lcoe_f, ind_f



        # return d_lcoe

    delta_lcoe = {}

    capex = df[(df['Year']==year1)]['Min capex'].to_numpy()[0] 
    opex = df[(df['Year']==year1)]['Opex'].to_numpy()[0]
    aep = df[(df['Year']==year1)]['AEP'].to_numpy()[0]
    fcr = df[(df['Year'] == year1)]['FCR'].to_numpy()[0]

    lcoe_0 = lcoe_calc(capex, opex, aep, fcr)
    
    delta_lcoe['Capex'], lcoe_dcapex, dcapex = delta_lcoe_calc(df, lcoe_0, capex, opex, aep, fcr, 'Min capex')
    delta_lcoe['Opex'], lcoe_dopex, dopex = delta_lcoe_calc(df, lcoe_dcapex, dcapex, opex, aep, fcr,'Opex')
    delta_lcoe['AEP'], lcoe_daep, daep = delta_lcoe_calc(df, lcoe_dopex, dcapex, dopex, aep, fcr,'AEP')
    delta_lcoe['FCR'], lcoe_dfcr, dfcr = delta_lcoe_calc(df, lcoe_daep,dcapex, dopex, daep, fcr, 'FCR')

    print('final lcoe', lcoe_dfcr)

    baseline_lcoe = df[(df['Year'] == year1)]['Min LCOE'].to_numpy()[0]
    waterfall = [baseline_lcoe,
        -delta_lcoe['Capex'],
        -delta_lcoe['Opex'],
        -delta_lcoe['AEP'],
        -delta_lcoe['FCR']
    ]

    waterfall_percent =  [baseline_lcoe ,
        -delta_lcoe['Capex'],
        -delta_lcoe['Opex'],
        -delta_lcoe['AEP'],
        -delta_lcoe['FCR']
    ] / (.01 * baseline_lcoe)
    
    

    waterfall_chart.plot(['Baseline \nPlant','CapEx', 'OpEx', 'Energy \nProduction', 'Lifetime \nextension'], waterfall,
        net_label=str(year2) + '\nConceptual \nPlant',
        blue_color='grey',
        green_color='grey',
        red_color='darkgreen',
        y_lab='LCOE (2020$/MWh)',
        rotation_value=0
        )

    plt.ylim([0, 225])
    
    plt.savefig('results/waterfall.png', dpi=300, bbox_inches='tight')

    waterfall_chart.plot(['Baseline \nPlant','CapEx', 'OpEx', 'Energy \nProduction', 'Lifetime \nextension'], waterfall_percent,
        net_label= str(year2) + '\nConceptual \nPlant',
        blue_color='grey',
        green_color='grey',
        red_color='darkgreen',
        y_lab='Percent of baseline LCOE',
        rotation_value=0
        )

    plt.ylim([0, 110])
    
    plt.savefig('results/waterfall_percent.png', dpi=300, bbox_inches='tight')