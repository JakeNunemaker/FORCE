"""
Take LCOE trajectories and create waterfall plot
"""

import numpy as np
import pandas as pd
import waterfall_chart
import matplotlib.pyplot as plt

if __name__ == '__main__':
    df = pd.read_csv('results/floating_data_out.csv')
    print(df)
    # print(df['Min capex'], df['Opex'], df['AEP'])

    year1 = 2023
    year2 = 2035    

    def lcoe_calc(c, o, a, f):
        lcoe = 1000 * (f * c + o) / a
        return lcoe

    def delta_lcoe_calc(df, ind):        

        capex = df[(df['Year']==year1)]['Min capex'].to_numpy()[0] 
        opex = df[(df['Year']==year1)]['Opex'].to_numpy()[0]
        aep = df[(df['Year']==year1)]['AEP'].to_numpy()[0]
        fcr = df[(df['Year'] == year1)]['FCR'].to_numpy()[0]

        if ind == 'Min capex':
            d_capex = df[(df['Year']==year2)]['Min capex'].to_numpy()[0] 
            d_fcr = df[(df['Year']==year2)]['FCR'].to_numpy()[0]
            d_lcoe = lcoe_calc(capex, opex, aep, fcr) - lcoe_calc(d_capex, opex, aep, d_fcr)
        elif ind == 'Opex':
            d_opex = df[(df['Year'] == year2)]['Opex'].to_numpy()[0]
            d_lcoe = lcoe_calc(capex, opex, aep, fcr) - lcoe_calc(capex, d_opex, aep, fcr)
        elif ind == 'AEP':
            d_aep = df[(df['Year'] == year2)]['AEP'].to_numpy()[0]
            d_lcoe = lcoe_calc(capex, opex, aep, fcr) - lcoe_calc(capex, opex, d_aep, fcr)
        elif ind == 'FCR':
            d_fcr = df[(df['Year'] == year2)]['FCR'].to_numpy()[0]
            d_lcoe = lcoe_calc(capex, opex, aep, fcr) - lcoe_calc(capex, opex, aep, d_fcr)
        return d_lcoe

    delta_lcoe = {}

    delta_lcoe['Capex'] = delta_lcoe_calc(df, 'Min capex')
    delta_lcoe['Opex'] = delta_lcoe_calc(df, 'Opex')
    delta_lcoe['AEP'] = delta_lcoe_calc(df, 'AEP')
    # delta_lcoe['FCR'] = delta_lcoe_calc(df, 'FCR')

    waterfall = [np.ceil(df[(df['Year'] == year1)]['Min LCOE'].to_numpy()[0]),
        -delta_lcoe['Capex'],
        -delta_lcoe['Opex'],
        -delta_lcoe['AEP'],
    # -delta_lcoe['FCR']
    ]
    

    waterfall_chart.plot(['Baseline \nPlant','CapEx*', 'OpEx', 'Energy \nProduction'], waterfall,
        net_label='2035 Conceptual \nPlant',
        blue_color='grey',
        green_color='grey',
        red_color='darkgreen',
        y_lab='LCOE (2020$/MWh)',
        rotation_value=0
        )
        
    plt.ylim([0, 225])
    
    plt.savefig('results/waterfall.png', dpi=300, bbox_inches='tight')

