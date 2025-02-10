import numpy as np
import xarray as xr
import random

def fixed_bias(ds_time_slice, bias, main_var='Tb'):
    tmp_ds_time_slice = ds_time_slice.copy()
    tmp_ds_time_slice[main_var] = tmp_ds_time_slice[main_var]+bias
    return tmp_ds_time_slice

def incrementing_bias(ds_time_slice, bias_rate, main_var='Tb'):
    tmp_ds_time_slice = ds_time_slice.copy()
    num_time_steps = ds_time_slice['time'].shape[0]
    increments = np.arange(num_time_steps) * bias_rate
    increments_da = xr.DataArray(increments, dims=['time'], coords={'time': ds_time_slice['time']})
    tmp_ds_time_slice[main_var] = tmp_ds_time_slice[main_var]+increments_da
    return tmp_ds_time_slice

def adversarial_perturbation(ds_time_slice, epsilon, main_var='Tb'):
    tmp_ds_time_slice = ds_time_slice.copy()
    perturbation = epsilon * np.sign(np.random.randn(*ds_time_slice[main_var].shape))
    perturbation_da = xr.DataArray(perturbation, dims=ds_time_slice[main_var].dims, coords=ds_time_slice[main_var].coords)
    tmp_ds_time_slice[main_var] = ds_time_slice[main_var]+perturbation_da
    return tmp_ds_time_slice

def sensor_spoofing(ds_time_slice, sr, er, sc, ec,main_var='Tb'):
    min_temp = ds_time_slice[main_var].min().values.item()
    max_temp = ds_time_slice[main_var].max().values.item()
    fake_value = random.randint(min_temp, max_temp)
    tmp_ds_time_slice = ds_time_slice.copy() 
    tmp_ds_time_slice[main_var][:, sr:er, sc:ec] = fake_value
    return tmp_ds_time_slice

def rnd_sensor_spoofing(ds_time_slice, w_min, w_max, h_min, h_max, main_var='Tb'):
    tmp_ds_time_slice = ds_time_slice.copy(deep=True)
    w = random.randint(w_min, w_max)
    h = random.randint(h_min, h_max)
    sr = random.randint(0, ds_time_slice.lat.values.shape[0])
    er = sr+h 
    sc = random.randint(0, ds_time_slice.lon.values.shape[0])
    ec = sc+w
    return sensor_spoofing(tmp_ds_time_slice, sr, er, sc, ec, main_var)

def data_block_jamming(ds_time_slice, fraction, main_var='Tb'):
    tmp_ds = ds_time_slice.copy()
    arr = np.array(tmp_ds[main_var])
    mask = np.random.rand(*arr.shape) < fraction
    arr[mask] = np.nan
    tmp_ds[main_var] = xr.DataArray(arr, dims=tmp_ds[main_var].dims, coords=tmp_ds[main_var].coords)
    return tmp_ds

def temporal_shifting(ds_time_slice,shift, main_var='Tb'):
    tmp_ds_time_slice = ds_time_slice.copy() 
    shifted_ds = np.roll(tmp_ds_time_slice[main_var], shift, axis=0)
    tmp_ds_time_slice[main_var] =  xr.DataArray(shifted_ds, dims=tmp_ds_time_slice.dims, coords=tmp_ds_time_slice.coords)
    return tmp_ds_time_slice

def spatial_shifting(ds_time_slice,shift_rows,shift_cols, main_var='Tb'):
    tmp_ds_time_slice = ds_time_slice.copy() 
    shifted_ds = np.roll(np.roll(tmp_ds_time_slice[main_var], shift_rows, axis=0), shift_cols, axis=1)
    tmp_ds_time_slice[main_var] = xr.DataArray(shifted_ds, dims=tmp_ds_time_slice.dims, coords=tmp_ds_time_slice.coords)
    return tmp_ds_time_slice

def rnd_spatial_shifting(ds_time_slice, shift_max, main_var='Tb'):
    shift_rows = random.randint(10, shift_max)
    shift_cols = random.randint(10, shift_max)
    return spatial_shifting(ds_time_slice, shift_rows, shift_cols, main_var)

def miscalibration_error_injection(ds_time_slice, factor, main_var='Tb'):
    tmp_ds_time_slice = ds_time_slice.copy() 
    tmp_ds_time_slice[main_var] = ds_time_slice[main_var]*factor
    return tmp_ds_time_slice

def replay_attack(ds_time_slice, replay_period, main_var='Tb'):
    tmp_ds_time_slice = ds_time_slice.copy() 
    data = ds_time_slice[main_var].copy()
    total_time = data.shape[0]
    if total_time < 2 * replay_period:
        raise ValueError("Not enough time steps for a replay attack with the given replay_period.")
    
    replay_start = np.random.randint(0, total_time - replay_period)
    # Choose a random insertion index after the replay segment such that:
    # insertion_index >= replay_start + replay_period and insertion_index + replay_period <= total_time.
    insertion_low = replay_start + replay_period
    insertion_high = total_time - replay_period + 1 
    if insertion_low >= insertion_high:
        raise ValueError("Not enough time steps to insert replayed data after the replay segment.")
    insertion_index = np.random.randint(insertion_low, insertion_high)
    
    data.values[insertion_index:insertion_index+replay_period, :, :] = \
        data.values[replay_start:replay_start+replay_period, :, :].copy()
    
    tmp_ds_time_slice[main_var] = xr.DataArray(data.values, dims=data.dims, coords=data.coords)
    return tmp_ds_time_slice

def backdoor_trigger(
    ds_time_slice, 
    trigger_strength=0.5,  # Amplitude of the trigger signal
    trigger_start=10,       # Time step where the trigger starts
    trigger_duration=10,     # Number of time steps the trigger lasts
    main_var='Tb'
):
    tmp_ds = ds_time_slice.copy()
    trigger_signal = trigger_strength * np.sin(
        2 * np.pi * np.arange(trigger_duration) / trigger_duration
    )
    trigger_signal = trigger_signal[:, np.newaxis, np.newaxis] 
    tmp_ds[main_var][trigger_start:trigger_start+trigger_duration, :, :] += trigger_signal
    
    return tmp_ds



