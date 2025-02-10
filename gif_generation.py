import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import imageio
import io
from tqdm import tqdm

def generate_image(ds, ts_val):
    with xr.set_options(keep_attrs=False):
        temp_ts = ds["Tb"].sel(time=ts_val, method="nearest").where(ds["Tb"].sel(time=ts_val) != -9999)
    temp_ts = temp_ts.compute()

    fig = plt.figure(figsize=(8, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.9, 0.1], wspace=0.1)
    ax = fig.add_subplot(gs[0], projection=ccrs.PlateCarree())
    cax = fig.add_subplot(gs[1])

    lon_min, lon_max = ds['lon'].min().item(), ds['lon'].max().item()
    lat_min, lat_max = ds['lat'].min().item(), ds['lat'].max().item()
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)

    temp_plot = ax.pcolormesh(ds["lon"], ds["lat"], temp_ts,
                              transform=ccrs.PlateCarree(), cmap="jet")
    plt.colorbar(temp_plot, cax=cax, label="Temperature (K)")
    ax.set_title(f"Temperature at {np.datetime_as_string(ts_val.values, unit='h')}")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches=None, pad_inches=0.1)
    buf.seek(0)
    
    plt.close(fig)
    return imageio.imread(buf)

def generate_gif(ds, t1, time_window, save_file=False, gif_filename=None):
    time_slice = ds.time[t1:t1+time_window] 
    images = []
    for time_step in tqdm(time_slice, desc="Generating GIF"):
        try:
            image = generate_image(ds, time_step)
            if images and image.shape != images[0].shape:
                image = imageio.core.util.Image(image)  
                image = np.array(image.resize(images[0].shape[:2][::-1]))  
            images.append(image)
        except Exception as e:
            print(f"Skipping timestep {time_step}: {e}")
            continue

    fig, ax = plt.subplots()
    im = ax.imshow(images[0], animated=True)
    ax.axis('off')

    def update(frame):
        im.set_array(images[frame])
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=len(images), blit=True)
    
    # Save GIF
    if save_file and gif_filename:
        imageio.mimsave(gif_filename, images, duration=0.2)
        print(f"GIF saved as {gif_filename}")
    
    plt.close(fig)
    return ani