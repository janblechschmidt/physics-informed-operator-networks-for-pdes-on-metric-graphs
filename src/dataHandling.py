import jax.numpy as jnp
import os
import glob
import pickle


def extract_data(data):
    LOAD_SAMPLES = None
    RES_DATA_INFLOW = [data['RES_INFLOW0'][0:LOAD_SAMPLES],
                data['RES_INFLOW1'][0:LOAD_SAMPLES],
                data['RES_INFLOW2'][0:LOAD_SAMPLES]]
    INIT_DATA_INFLOW = [data['INIT_INFLOW0'][0:LOAD_SAMPLES],
                        data['INIT_INFLOW1']]
    BC_DATA_INFLOW = [data['BC_INFLOW0'][0:LOAD_SAMPLES],
               data['BC_INFLOW1'],
               data['BC_INFLOW2'][0:LOAD_SAMPLES]]
    
    RES_DATA_INNER = [data['RES_INNER0'][0:LOAD_SAMPLES],
                data['RES_INNER1'][0:LOAD_SAMPLES],
                data['RES_INNER2'][0:LOAD_SAMPLES]]
    INIT_DATA_INNER = [data['INIT_INNER0'][0:LOAD_SAMPLES],
                 data['INIT_INNER1']]
    BC_DATA_INNER = [data['BC_INNER0'][0:LOAD_SAMPLES],
               data['BC_INNER1'],
               data['BC_INNER2'][0:LOAD_SAMPLES]]
    
    RES_DATA_OUTFLOW = [data['RES_OUTFLOW0'][0:LOAD_SAMPLES],
                data['RES_OUTFLOW1'][0:LOAD_SAMPLES],
                data['RES_OUTFLOW2'][0:LOAD_SAMPLES]]
    INIT_DATA_OUTFLOW = [data['INIT_OUTFLOW0'][0:LOAD_SAMPLES],
                 data['INIT_OUTFLOW1']]
    BC_DATA_OUTFLOW = [data['BC_OUTFLOW0'][0:LOAD_SAMPLES],
               data['BC_OUTFLOW1'],
               data['BC_OUTFLOW2'][0:LOAD_SAMPLES]]
    
    return [RES_DATA_INFLOW, INIT_DATA_INFLOW, BC_DATA_INFLOW,
            RES_DATA_INNER, INIT_DATA_INNER, BC_DATA_INNER,
            RES_DATA_OUTFLOW, INIT_DATA_OUTFLOW, BC_DATA_OUTFLOW]

def loadData(DATA_PATH, idx_2, idx_4, idx_6):
    # To be memory efficient, positional data of the sensor locations is only stored once
    exclude_list = [(1,1), (2,1), (4,1), (5,1), (7,1), (8,1)]
    
    INIT_DATA = True
    for i in idx_2:
        data = jnp.load(os.path.join(DATA_PATH, 'example_2', f'run_{i:05}.npz'))
        if INIT_DATA:
            DATA = extract_data(data)
            INIT_DATA = False
        else:
            # Append data
            this_data = extract_data(data)
            for j in range(len(DATA)):
                for k in range(len(DATA[j])):
                    if (j, k) not in exclude_list:
                        DATA[j][k] = jnp.vstack([DATA[j][k], this_data[j][k]])
    
    for i in idx_4:
        data = jnp.load(os.path.join(DATA_PATH, 'example_4', f'run_{i:05}.npz'))
        if INIT_DATA:
            DATA = extract_data(data)
            INIT_DATA = False
        else:
            # Append data
            this_data = extract_data(data)
            for j in range(len(DATA)):
                for k in range(len(DATA[j])):
                    if (j, k) not in exclude_list:
                        DATA[j][k] = jnp.vstack([DATA[j][k], this_data[j][k]])
    
    for i in idx_6:
        data = jnp.load(os.path.join(DATA_PATH, 'example_6', f'run_{i:05}.npz'))
        if INIT_DATA:
            DATA = extract_data(data)
            INIT_DATA = False
        else:
            # Append data
            this_data = extract_data(data)
            for j in range(len(DATA)):
                for k in range(len(DATA[j])):
                    if (j, k) not in exclude_list:
                        DATA[j][k] = jnp.vstack([DATA[j][k], this_data[j][k]])

    return DATA

def load_final_model(n = 100, mode = 0, prefix='.'):
    ret = []
    for t in ['inflow', 'inner', 'outflow']:
        ff = glob.glob(f'{prefix}/final_params/params_{t}_mode_{mode}_width_{n}_*_FF_[0-9]*.pkl')[0]
        print('Loaded ', ff)
        ret.append(ff) 
    return [pickle.load(open(f, 'rb' )) for f in ret]

def load_best_model(n = 100, mode = 0, prefix='.'):
    ret = []
    for t in ['inflow', 'inner', 'outflow']:
        ff = glob.glob(f'{prefix}/final_params/params_{t}_mode_{mode}_width_{n}_*_FF_best*.pkl')[0]
        print('Loaded ', ff)
        ret.append(ff) 
    return [pickle.load(open(f, 'rb' )) for f in ret]
