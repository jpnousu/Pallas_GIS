# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 10:05:11 2022

@author: janousu
"""

# basic packages
import os
import glob
import urllib
from numpy import newaxis
import xarray as xr
import numpy as np
import pandas as pd
import warnings

# rasterio
import rasterio
import rasterio.plot
from rasterio import features
from rasterio.windows import from_bounds
from rasterio.plot import show
from rasterio.enums import Resampling
from rasterio.mask import mask
from rasterio.fill import fillnodata
from rasterio.warp import reproject, Resampling, calculate_default_transform

# gis
from pysheds.grid import Grid
from scipy import ndimage
import geopandas as gpd
from rasterio.crs import CRS

# plotting
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import colors


def dem_from_mml(out_fd, subset, apikey, layer='korkeusmalli_2m', form='image/tiff', scalefactor=None, plot=True, save_in='asc'):

    '''Downloads a raster from MML database and writes it to dirpath folder in local memory

        Parameters:
        subset = boundary coordinates [minx, miny, maxx, maxy] (list)
        layer = the layer wanted to fetch e.g. 'korkeusmalli_2m' or 'korkeusmalli_10m' (str)
        form = form of the raster e.g 'image/tiff' (str)
        plot = whether or not to plot the created raster, True/False
        '''

    # The base url for maanmittauslaitos
    url = 'https://avoin-karttakuva.maanmittauslaitos.fi/ortokuvat-ja-korkeusmallit/wcs/v2?'

    # Defining the latter url code
    params = dict(service='service=WCS',
                  version='version=2.0.1',
                  request='request=GetCoverage',
                  CoverageID=f'CoverageID={layer}',
                  SUBSET=f'SUBSET=E({subset[0]},{subset[2]})&SUBSET=N({subset[1]},{subset[3]})',
                  outformat=f'format={form}',
                  compression='geotiff:compression=LZW',
                  api=f'api-key={apikey}')
    
    # Add scalefactor only if it's not None
    if scalefactor is not None:
        params['scalefactor'] = f'SCALEFACTOR={scalefactor}'
        
    if not os.path.exists(out_fd):
        # Create a new directory because it does not exist
        os.makedirs(out_fd)
    
    par_url = ''
    for par in params.keys():
        par_url += params[par] + '&'
    par_url = par_url[0:-1]
    new_url = (url + par_url)

    # Putting the whole url together
    r = urllib.request.urlretrieve(new_url)

    # Open the file with the url:
    raster = rasterio.open(r[0])

    del r
    if scalefactor is not None:
        res = int(2/scalefactor) # !! WATCHOUT 2 IS HARD CODED
    else:
        res = 2
    layer = f'korkeusmalli_{res}m'

    if save_in=='tif':
        out_fp = os.path.join(out_fd, layer) + '.tif'
    elif save_in=='asc':
        out_fp = os.path.join(out_fd, layer) + '.asc'
        
    # Copy the metadata
    out_meta = raster.meta.copy()

    # Update the metadata
    out_meta.update({"driver": "GTiff",
                     "height": raster.height,
                     "width": raster.width,
                     "transform": raster.meta['transform'],
                     "crs": raster.meta['crs'],
                     "nodata":-9999,
                         }
                    )
    if save_in=='asc':
            out_meta.update({"driver": "AAIGrid"})
        
    # Manipulating the data for writing purpose
    raster_dem = raster.read(1)
    raster_dem = raster_dem[newaxis, :, :]

    # Write the raster to disk
    with rasterio.open(out_fp, "w", **out_meta) as dest:
        dest.write(raster_dem)

    raster_dem = rasterio.open(out_fp)

    if plot==True:
        show(raster_dem)
    
    return raster_dem, out_fp

def dem_from_puhti_2m(fp, subset, out_fp, plot=True, save_in='geotiff'):
    '''
    fp = file path to be downloaded
    subset = cropping to coordinate box
    out_fp = file path to be saved
    plot = show the raster
    '''
    with rasterio.open(fp) as src:
        data = src.read(1, window=from_bounds(subset[0], subset[1], subset[2], subset[3], src.transform))
        profile = src.profile

    out_meta = profile.copy()

    out_fd = os.path.dirname(out_fp)
    if not os.path.exists(out_fd):
        # Create a new directory because it does not exist
        os.makedirs(out_fd)
    
    new_affine = rasterio.Affine(out_meta['transform'][0], 
                                 out_meta['transform'][1], 
                                 subset[0], 
                                 out_meta['transform'][3], 
                                 out_meta['transform'][4], 
                                 subset[3])
    # Update the metadata
    out_meta.update({"driver": "GTiff",
                    "height": data.shape[0],
                    "width": data.shape[1],
                    "transform": new_affine,
                    "crs": profile['crs']
                        }
                    )
    if save_in=='asc':
        out_meta.update({"driver": "AAIGrid"})
    
    with rasterio.Env():
        # Write an array as a raster band to a new 8-bit file. For
        # the new file's profile, we start with the profile of the source
        # And then change the band count to 1, set the
        # dtype to uint8, and specify LZW compression.
        with rasterio.open(out_fp, 'w', **out_meta) as dst:
            src = dst.write(data, 1)
            if plot==True:
                plt.imshow(data)
                #show(src)
            # At the end of the ``with rasterio.Env()`` block, context
            # manager exits and all drivers are de-registered
    
    raster_dem = rasterio.open(out_fp)
    if plot==True:
        show(raster_dem)
        
    return raster_dem, out_fp    


def resample_raster2(fp, out_fp, scale_factor=0.125, resampling_method='bilinear', plot=True, save_in='geotiff'):
    '''
    fp = file path to be downloaded
    out_fp = file path to be saved
    scaling factor (e.g. if 2m to be resampled to 16m scale_factor=0.125)   
    '''

    if resampling_method == 'bilinear':
        resample_as = Resampling.bilinear
    if resampling_method == 'nearest':
        resample_as = Resampling.nearest

    with rasterio.open(fp) as dataset:
        
        # resample data to target shape
        data = dataset.read(1, 
                out_shape=(dataset.count,int(dataset.height / scale_factor),int(dataset.width / scale_factor)),
                resampling=resample_as
                )

        # scale image transform
        transform = dataset.transform * dataset.transform.scale(
            (dataset.width / data.shape[-1]),
            (dataset.height / data.shape[-2])
        )    
        out_meta = dataset.profile.copy()

        out_meta.update({"driver": "GTiff",
                 "height": data.shape[0],
                  "width": data.shape[1],
                  "transform": transform,
                        }
                    )
        if save_in=='asc':
            out_meta.update({"driver": "AAIGrid"})
        
        with rasterio.open(out_fp, 'w', **out_meta) as dst:
            src = dst.write(data, 1)
            
    raster = rasterio.open(out_fp)
    if plot==True:
        show(raster)
    return raster, out_fp

def resample_raster(fp, out_fp, scale_factor=0.125, resampling_method='bilinear', plot=True, save_in='geotiff'):
    '''
    fp = file path to be downloaded
    out_fp = file path to be saved
    scaling factor (e.g. if 2m to be resampled to 16m scale_factor=0.125)   
    '''

    if resampling_method == 'bilinear':
        resample_as = Resampling.bilinear
    if resampling_method == 'nearest':
        resample_as = Resampling.nearest

    with rasterio.open(fp) as dataset:
        
        # resample data to target shape
        data = dataset.read(1, 
                out_shape=(dataset.count,int(dataset.height * scale_factor),int(dataset.width * scale_factor)),
                resampling=resample_as
                )

        # scale image transform
        transform = dataset.transform * dataset.transform.scale(
            (dataset.width / data.shape[-1]),
            (dataset.height / data.shape[-2])
        )    
        out_meta = dataset.profile.copy()

        out_meta.update({"driver": "GTiff",
                 "height": data.shape[0],
                  "width": data.shape[1],
                  "transform": transform,
                        }
                    )
        if save_in=='asc':
            out_meta.update({"driver": "AAIGrid"})
        
        with rasterio.open(out_fp, 'w', **out_meta) as dst:
            src = dst.write(data, 1)
            
    raster = rasterio.open(out_fp)
    if plot==True:
        show(raster)
    return raster, out_fp

def open_raster_with_subset(fp, out_fp, subset, plot=True, save_in='asc'):
    
    with rasterio.open(fp) as dataset:
        data = dataset.read(1, window=from_bounds(subset[0], subset[1], subset[2], subset[3], dataset.transform))
        data = data.astype(float)
        out_meta = dataset.profile.copy()
        
    new_affine = rasterio.Affine(out_meta['transform'][0], 
                                out_meta['transform'][1], 
                                subset[0], 
                                out_meta['transform'][3], 
                                out_meta['transform'][4], 
                                subset[3])   

    out_meta.update({"height": data.shape[0],
                      "width": data.shape[1],
                      "transform": new_affine,
                    }
                    )

    if save_in=='asc':
        out_meta.update({"driver": "AAIGrid"})  
        
    with rasterio.open(out_fp, 'w', **out_meta) as dst:
        src = dst.write(data, 1)
            
    raster = rasterio.open(out_fp)
    if plot==True:
        show(raster)

def fill_layer_na_with_layer(priority_layer, secondary_layer, out_fp, save_in='geotiff'):

    with rasterio.open(priority_layer) as src1:
        data1 = src1.read(1)
        data1 = data1.astype(float)
        meta1 = src1.meta.copy()
        nodata1 = meta1['nodata']
        
    with rasterio.open(secondary_layer) as src2:
        data2 = src2.read(1)
        data2 = data2.astype(float)
        meta2 = src2.meta.copy()
        nodata2 = meta2['nodata']
        
    data1[data1 == nodata1] = data2[data1 == nodata1]
    data1 = data1.astype(int)

    out_meta = meta2.copy()
    if save_in=='asc':
        out_meta.update({"driver": "AAIGrid"})  
        
    with rasterio.open(out_fp, 'w+', **out_meta) as out:
            src = out.write(data1, 1)


def resample_raster_set(fd, file, out_fd, scale_factor=0.5, plot=True, save_in='asc'):
    '''
    fd = file directory path to be downloaded
    out_fd = directory path to be saved
    scaling factor (e.g. if 2m to be resampled to 16m scale_factor=0.125)   
    '''

    if file=='*.asc':
        p = os.path.join(fd, file)
    elif file=='*.tif':
        p = os.path.join(fd, file)
    else:
        p = os.path.join(fd, file)

    if not os.path.exists(out_fd):
        # Create a new directory because it does not exist
        os.makedirs(out_fd)

    for file in glob.glob(p):
        out_fn = file.rpartition('/')[-1][:-4]
        if save_in == 'geotiff':
            out_fp = os.path.join(out_fd, out_fn) + '.tif'
        elif save_in == 'asc':
            out_fp = os.path.join(out_fd, out_fn) + '.asc'
    
        with rasterio.open(file) as dataset:
        
            # resample data to target shape
            data = dataset.read(1, 
                    out_shape=(dataset.count,int(dataset.height * scale_factor),int(dataset.width * scale_factor)),
                    resampling=Resampling.bilinear
                    )

            # scale image transform
            transform = dataset.transform * dataset.transform.scale(
                (dataset.width / data.shape[-1]),
                (dataset.height / data.shape[-2])
                )    
            out_meta = dataset.profile.copy()

            out_meta.update({"driver": "GTiff",
                             "height": data.shape[0],
                              "width": data.shape[1],
                              "transform": transform,
                            }
                            )
            if save_in=='asc':
                out_meta.update({"driver": "AAIGrid"})
        
            with rasterio.open(out_fp, 'w', **out_meta) as dst:
                src = dst.write(data, 1)
            
            raster = rasterio.open(out_fp)
            if plot==True:
                show(raster, vmax=100)

def orto_from_mml(outpath, subset, layer='ortokuva_vari', form='image/tiff', scalefactor=0.01, plot=False):

    '''Downloads a raster from MML database and writes it to dirpath folder in local memory

        Parameters:
        subset = boundary coordinates [minx, miny, maxx, maxy] (list)
        layer = the layer wanted to fetch e.g. 'korkeusmalli_2m' or 'korkeusmalli_10m' (str)
        form = form of the raster e.g 'image/tiff' (str)
        plot = whether or not to plot the created raster, True/False
        cmap = colormap for plotting (str - default = 'terrain')
        '''


    # The base url for maanmittauslaitos
    url = 'https://beta-karttakuva.maanmittauslaitos.fi/ortokuvat-ja-korkeusmallit/wcs/v1?'
    # Defining the latter url code
    params = dict(service='service=WCS',
                  version='version=2.0.1',
                  request='request=GetCoverage',
                  CoverageID=f'CoverageID={layer}',
                  SUBSET=f'SUBSET=E({subset[0]},{subset[2]})&SUBSET=N({subset[1]},{subset[3]})',
                  outformat=f'format={form}')


    par_url = ''
    for par in params.keys():
        par_url += params[par] + '&'
    par_url = par_url[0:-1]
    new_url = (url + par_url)

    # Putting the whole url together
    r = urllib.request.urlretrieve(new_url)

    # Open the file with the url:
    raster = rasterio.open(r[0])

    del r
    res = int(2/scalefactor)
    layer = f'korkeusmalli_{res}m'
    out_fp = os.path.join(outpath, layer) + '.tif'

    # Copy the metadata
    out_meta = raster.meta.copy()

    # Update the metadata
    out_meta.update({"driver": "GTiff",
                     "height": raster.height,
                     "width": raster.width,
                     "transform": raster.meta['transform'],
                     "crs": raster.meta['crs']
                         }
                    )

    # Manipulating the data for writing purpose
    raster_dem = raster.read(1)
    raster_dem = raster_dem[newaxis, :, :]

    # Write the raster to disk
    with rasterio.open(out_fp, "w", **out_meta) as dest:
        dest.write(raster_dem)

    raster_dem = rasterio.open(out_fp)

    return raster_dem, out_fp

def vmi_from_puhti(fd, subset, out_fd, layer='all', interpolate=None, use_center=None, max_search_distance=2, resample=False, plot=True, save_in='geotiff'):
    '''
    Downloads VMI data (all layers 'all' or one given layer e.g. 'kasvupaikka_vmi1x_1721.img') 
    from given file directory (fd) for a given subset area
    Option to interpolate over the VMI classified 'nonland' grid-cells with a given max_search_distance (1)
    Option to assign the 'nonland' grid-cells as zeros and then interpolate only those values (2)
    Option to resample the original 16m resolution rasters with a given scaling factor (e.g. 0.5 -> 32m)
    Saving in output file directory (out_fd) as 'geotiff' or 'ascii'
    '''
    
    if layer=='all':
        p = os.path.join(fd, '*.img')
    else:
        p = os.path.join(fd, layer)

    if not os.path.exists(out_fd):
        os.makedirs(out_fd)

    # file count so that mask will be only saved simulataneously with first file
    i = 0
    for file in glob.glob(p):
        print(file)
        out_fn = file.rpartition('/')[-1][:-4]
        mask_fn = 'nonland'
        if save_in == 'geotiff':
            out_fp = os.path.join(out_fd, out_fn) + '.tif'
            mask_fp = os.path.join(out_fd, mask_fn) + '.tif'
        elif save_in == 'asc':
            out_fp = os.path.join(out_fd, out_fn) + '.asc'
            mask_fp = os.path.join(out_fd, mask_fn) + '.asc'

        with rasterio.open(file) as src:
            data = src.read(1, window=from_bounds(subset[0], subset[1], subset[2], subset[3], src.transform))
            profile = src.profile
            out_meta = src.profile.copy()

            new_affine = rasterio.Affine(out_meta['transform'][0], 
                                         out_meta['transform'][1], 
                                         subset[0], 
                                         out_meta['transform'][3], 
                                         out_meta['transform'][4], 
                                         subset[3])

            # save nonland mask at first loop iteration
            if i == 0:
                data_mask = np.zeros(shape=data.shape)
                data_mask[data == 32767] = 32767
            
            if len(data.flatten()[data.flatten() == 32766]) > 0:
                print('*** Data has', len(data.flatten()[data.flatten() == 32766]), 'nan values (=32766) ***')
                #print('--> converted to 0 ***')
            if len(data.flatten()[data.flatten() == 32767]) > 0:
                print('*** Data has', len(data.flatten()[data.flatten() == 32767]), 'non land values (=32767) ***')
                #print('--> converted to 0 ***')

            # Update the metadata for geotiff
            out_meta.update({"driver": "GTiff",
                             "height": data.shape[0],
                             "width": data.shape[1],
                              "transform": new_affine,
                              "nodata": 32767,
                              "crs": CRS.from_epsg(3067)})
            
            if save_in == 'asc':
                out_meta.update({"driver": "AAIGrid"})
                
            with rasterio.Env():
                with rasterio.open(out_fp, 'w', **out_meta, force_cellsize=True) as dst:
                    src = dst.write(data, 1)
                            # if save_mask:
                if i == 0:
                    with rasterio.open(mask_fp, 'w', **out_meta, force_cellsize=True) as dst_mask:
                        src_mask = dst_mask.write(data_mask, 1)  
                    
            if interpolate == 1:
                with rasterio.open(out_fp, 'r+') as src_new:
                        data = src_new.read(1)
                        data_filled = fillnodata(data, 
                                                 mask=src_new.read_masks(1), 
                                                 max_search_distance=max_search_distance, 
                                                 smoothing_iterations=0)
                        print('*** non land interpolated ***')
                if i == 0:
                    with rasterio.open(mask_fp) as src_mask_interp:
                        mask_interp = src_mask_interp.read(1)
                        mask_interp_fill = fillnodata(mask_interp, 
                                                      mask=src_mask_interp.read_masks(1), 
                                                      max_search_distance=max_search_distance, smoothing_iterations=0)
                        
                with rasterio.Env():
                    with rasterio.open(out_fp, 'w', **out_meta, force_cellsize=True) as dst_new:
                        src_new = dst_new.write(data_filled, 1)
                    if i == 0:
                        with rasterio.open(mask_fp, 'w', **out_meta, force_cellsize=True) as dst_mi:
                            src_mi = dst_mi.write(mask_interp_fill, 1)

            elif interpolate == 2:
                with rasterio.open(out_fp, 'r+') as src_new:
                        data = src_new.read(1)
                        data_filled = interpolate_over_mask(data=data, mask=data_mask, use_center=use_center)
        
                with rasterio.Env():
                    with rasterio.open(out_fp, 'w', **out_meta, force_cellsize=False) as dst_new:
                        src_new = dst_new.write(data_filled, 1)         

            elif interpolate == 3: # does not interpolate but assigns non_land as zeros
                with rasterio.open(out_fp, 'r+') as src_new:
                        data = src_new.read(1)
                        data[data_mask == 32767] = 0.0
        
                with rasterio.Env():
                    with rasterio.open(out_fp, 'w', **out_meta, force_cellsize=False) as dst_new:
                        src_new = dst_new.write(data, 1)    
                        
            if resample:
                with rasterio.open(out_fp) as src_new:
                    # resample data to target shape
                    print(f'*** resampled with a scalefactor of {resample} ***')
                    new_width = int((subset[2]-subset[0])/(16/resample))
                    new_height = int((subset[3]-subset[1])/(16/resample))
                    data_resampled = src_new.read(1,
                                    out_shape=(src_new.count,int(new_height),int(new_width)),
                                    resampling=Resampling.bilinear
                                    )
                      
                    out_meta = src_new.profile.copy()

                    new_affine = rasterio.Affine(out_meta['transform'][0]/resample, 
                                                 out_meta['transform'][1], 
                                                 subset[0], 
                                                 out_meta['transform'][3], 
                                                 out_meta['transform'][4]/resample, 
                                                 subset[3])
                    # Update the metadata for geotiff
                    out_meta.update({"height": data_resampled.shape[0],
                                    "width": data_resampled.shape[1],
                                    "transform": new_affine})

                if i == 0:
                    with rasterio.open(mask_fp) as src_mask_new:
                        data_mask_resampled = src_mask_new.read(1,
                                        out_shape=(src_mask_new.count,int(new_height),int(new_width)),
                                        resampling=Resampling.bilinear
                                        )
                with rasterio.Env():
                    with rasterio.open(out_fp, 'w', **out_meta, force_cellsize=True) as dst:
                        src_new = dst.write(data_resampled, 1)
                    if i == 0:
                        with rasterio.open(mask_fp, 'w', **out_meta, force_cellsize=True) as dst_mask:
                            src_mask_new = dst_mask.write(data_mask_resampled, 1)

        i += 1
                                
        if plot==True:
            raster = rasterio.open(out_fp)
            show(raster, vmax=100)


def interpolate_over_mask(data, mask, use_center=None):
    """
    Interpolates data raster only over a given mask.
    Optionally includes the center cell in the interpolation.
    
    Parameters:
        data (np.ndarray): 2D array of data to be interpolated.
        mask (np.ndarray): 2D mask array where values > 1 indicate regions to interpolate.
        use_center (bool): Whether to include the center cell (0 in masked region) in interpolation.
        
    Returns:
        np.ndarray: Interpolated data array.
    """
    
    i_mask = np.where(mask > 1) # = x,y
    temp_data = data.copy()
    temp_data[mask > 0] = 0
    a_e = i_mask[0]-1 # x
    a_w = i_mask[0]+1 # x
    a_n = i_mask[1]-1 # y
    a_s = i_mask[1]+1 # y
    a_e[a_e < 0] = 0
    a_w[a_w >= mask.shape[0]] = mask.shape[0]-1
    a_n[a_n <= 0] = 0
    a_s[a_s >= mask.shape[1]] = mask.shape[1]-1

    east = tuple((a_e, i_mask[1]))
    west = tuple((a_w, i_mask[1]))
    north = tuple((i_mask[0], a_n))
    south = tuple((i_mask[0], a_s))
    new_data = temp_data.copy()
    
    if use_center == True:
        print('center cell USED in interpolation')
        new_data[mask > 1] = np.mean([temp_data[i_mask], temp_data[east], temp_data[west], temp_data[north], temp_data[south]], axis=0)
        
    elif use_center == False:
        print('center cell NOT USED in interpolation')
        new_data[mask > 1] = np.mean([temp_data[east], temp_data[west], temp_data[north], temp_data[south]], axis=0)

    return new_data


def needlemass_to_lai(in_fd, in_ff, out_fd, species, save_in='asc', plot=True):

    # specific leaf area (m2/kg) for converting leaf mass to leaf area
    # SLA = {'pine': 5.54, 'spruce': 5.65, 'decid': 18.46}  # m2/kg, Kellomäki et al. 2001 Atm. Env.
    SLA = {'pine': 6.8, 'spruce': 4.7, 'decid': 14.0}  # Härkönen et al. 2015 BER 20, 181-195

    species_translations = {'spruce': 'kuusi', 
                           'pine': 'manty',
                           'decid': 'lehtip'}
    asked_species = species_translations[species]
    p = os.path.join(in_fd, f'bm*{asked_species}*neulaset*.{in_ff}')
    in_fn = glob.glob(p)[0]
    
    bm_raster = rasterio.open(in_fn)
    bm_data = bm_raster.read(1)
    
    LAI = bm_data * 1e-3 * SLA[species] # 1e-3 converts 10kg/ha to kg/m2

    if not os.path.exists(out_fd):
        # Create a new directory because it does not exist
        os.makedirs(out_fd)
    
    out_meta = bm_raster.meta.copy()

    print(bm_raster.meta)

    if save_in == 'geotiff':
        out_fn = os.path.join(out_fd, f'LAI_{species}') + '.tif'
        out_meta.update({"driver": "GTiff"})        
    elif save_in == 'asc':
        out_fn = os.path.join(out_fd, f'LAI_{species}') + '.asc'
        out_meta.update({"driver": "AAIGrid"})

    with rasterio.open(out_fn, 'w+', **out_meta) as out:
            src = out.write(LAI, 1)
    if plot==True:
        raster = rasterio.open(out_fn)
        show(raster)


def soilmap_from_puhti(soilmap, subset, out_fd, ref_raster, soildepth='surface', plot=True, save_in='geotiff'):
    '''
    Soilmap
    '''
    if soilmap == 200:
        soilfile = r'/projappl/project_2000908/geodata/soil/mp200k_maalajit.shp'
    elif soilmap == 20:
        soilfile = r'/projappl/project_2000908/geodata/soil/mp20k_maalajit.shp'

    if not os.path.exists(out_fd):
        # Create a new directory because it does not exist
        os.makedirs(out_fd)

    soil = gpd.read_file(soilfile, include_fields=["PINTAMAALA", "PINTAMAA_1", "POHJAMAALA", "POHJAMAA_1", "geometry"], bbox=subset)
    soil.PINTAMAALA = soil.PINTAMAALA.astype("float64")
    soil.POHJAMAALA = soil.POHJAMAALA.astype("float64")
    
    if save_in == 'geotiff':
        out_fn = os.path.join(out_fd, f'{soildepth}{soilmap}') + '.tif'
    elif save_in == 'asc':
        out_fn = os.path.join(out_fd, f'{soildepth}{soilmap}') + '.asc'

    rst = rasterio.open(ref_raster)
    meta = rst.meta.copy()
    meta.update(compress='lzw')
    if save_in == 'geotiff':
        meta.update({"driver": "GTiff"})        
    elif save_in == 'asc':
        meta.update({"driver": "AAIGrid"})
        
    with rasterio.open(out_fn, 'w+', **meta) as out:
        out_arr = out.read(1)

        # this is where we create a generator of geom, value pairs to use in rasterizing
        if soildepth=='surface':
            shapes = ((geom,value) for geom, value in zip(soil.geometry, soil.PINTAMAALA))
        if soildepth=='bottom':
            shapes = ((geom,value) for geom, value in zip(soil.geometry, soil.POHJAMAALA))

        burned = features.rasterize(shapes=shapes, fill=0, out=out_arr, transform=out.transform)
        burned[burned == 0] = -9999
        #for key in mpk.keys():
        #    burned[burned == key] = mpk[key]
        out.write_band(1, burned)
        if plot==True:
            raster = rasterio.open(out_fn)
            show(raster)

def rasterize_shapefile(shapefile, burn_field, out_fp, ref_raster, subset=None, plot=True, save_in='geotiff'):

    fields = [burn_field, 'geometry']
    shape = gpd.read_file(shapefile, include_fields=fields, bbox=subset)

    rst = rasterio.open(ref_raster)
    meta = rst.meta.copy()
    meta.update(compress='lzw')
    meta.update({"nodata": -9999})

    shape[burn_field] = shape[burn_field].astype("float64")

    if save_in == 'geotiff':
        meta.update({"driver": "GTiff"})        
    elif save_in == 'asc':
        meta.update({"driver": "AAIGrid"})
        
    with rasterio.open(out_fp, 'w+', **meta) as out:
        out_arr = out.read(1)

        # this is where we create a generator of geom, value pairs to use in rasterizing
        shapes = ((geom,value) for geom, value in zip(shape.geometry, shape[burn_field]))

        burned = features.rasterize(shapes=shapes, fill=-9999, out=out_arr, transform=out.transform)
        burned[burned == 0] = -9999
        #for key in mpk.keys():
        #    burned[burned == key] = mpk[key]
        out.write_band(1, burned)
        if plot==True:
            raster = rasterio.open(out_fp)
            show(raster)
            

def maastolayer_to_raster(in_fn, out_fd, layer, ref_raster, save_in='asc', plot=True):
    '''
    processing maastotietokanta (gpkg) layer to raster
    '''
    if not os.path.exists(out_fd):
        os.makedirs(out_fd)
        
    if save_in == 'geotiff':
        out_fn = os.path.join(out_fd, f'{layer}') + '.tif'
    elif save_in == 'asc':
        out_fn = os.path.join(out_fd, f'{layer}') + '.asc'
    
    rst = rasterio.open(ref_raster)
    meta = rst.meta.copy()
    meta.update(compress='lzw')

    if save_in == 'geotiff':
        meta.update({"driver": "GTiff"})        
    elif save_in == 'asc':
        meta.update({"driver": "AAIGrid"})
    
    subset=list(rst.bounds[:])
    data = gpd.read_file(in_fn, layer=layer, include_fields=["kohdeluokka", "geometry"], bbox=subset)
    with rasterio.open(out_fn, 'w+', **meta) as out:
        out_arr = out.read(1)

        # this is where we create a generator of geom, value pairs to use in rasterizing
        shapes = ((geom,value) for geom, value in zip(data.geometry, data.kohdeluokka))

        burned = features.rasterize(shapes=shapes, fill=-9999, out=out_arr, transform=out.transform)
        burned[burned == 0] = -9999
        out.write_band(1, burned)
        
        if plot==True:
            raster = rasterio.open(out_fn)
            show(raster)

def burn_water_dem(dem_fp, stream_fp, lake_fp=None, k=0.1, H=1, save_in='asc', plot=True):
    '''
    function to burn streams to dem (similar to that of whitebox)
    dem_fp
    stream_fp
    k = parameter
    H = parameter
    save_in
    '''

    fp = dem_fp[:-4]
    out_fp = str(fp + '_burned_streams')
    
    out_fd = os.path.dirname(out_fp)
    if not os.path.exists(out_fd):
        # Create a new directory because it does not exist
        os.makedirs(out_fd)
        
    if save_in == 'geotiff':
        out_fp = out_fp + '.tif'
    elif save_in == 'asc':
        out_fp = out_fp + '.asc'
        
    with rasterio.open(dem_fp) as dem_ras:
        dem_arr = dem_ras.read(1)
        out_meta = dem_ras.meta.copy()
        reso = out_meta['transform'][0]
    with rasterio.open(stream_fp) as stream_ras:
        stream_arr = stream_ras.read(1)

    if lake_fp:
        with rasterio.open(lake_fp) as lake_ras:
            lake_arr = lake_ras.read(1)
            
    # finding vectors of each stream cell
    stream_vectors_temp = np.where(stream_arr > 0)
    # finding vectors of each lake cell
    if lake_fp:
        lake_vectors_temp = np.where(lake_arr > 0)

    if lake_fp:
        stream_vectors = np.zeros(shape=[len(stream_vectors_temp[0])+len(lake_vectors_temp[0]),2])
    else:
        stream_vectors = np.zeros(shape=[len(stream_vectors_temp[0]),2])

    for i in range(len(stream_vectors_temp[0])):
        stream_vectors[i] = stream_vectors_temp[0][i], stream_vectors_temp[1][i]
    if lake_fp:
        for i2 in range(len(lake_vectors_temp[0])):
            new_i = len(stream_vectors_temp[0]) + i2
            stream_vectors[new_i] = lake_vectors_temp[0][i2], lake_vectors_temp[1][i2]
    
    dist_to_stream = np.zeros(dem_arr.shape)
    # looping through raster to find min distance to stream vectors
    for row in range(dem_arr.shape[0]):
        for col in range(dem_arr.shape[1]):
            centroid = [row, col]
            dist_to_stream[row,col] = np.min(np.linalg.norm(centroid - stream_vectors, axis=1))*reso

    new_dem = dem_arr - (reso / (reso + dist_to_stream))**k * H

    with rasterio.Env():
        with rasterio.open(out_fp, 'w', **out_meta, force_cellsize=True) as dst:
            src = dst.write(new_dem, 1)
    if plot==True:
        raster = rasterio.open(out_fp)
        show(raster)
        raster.close()
        
    return new_dem, out_fp


def read_AsciiGrid(fname, setnans=True):
    """
    reads AsciiGrid format in fixed format as below:
        ncols         750
        nrows         375
        xllcorner     350000
        yllcorner     6696000
        cellsize      16
        NODATA_value  -9999
        -9999 -9999 -9999 -9999 -9999
        -9999 4.694741 5.537514 4.551162
        -9999 4.759177 5.588773 4.767114
    IN:
        fname - filename (incl. path)
    OUT:
        data - 2D numpy array
        info - 6 first lines as list of strings
        (xloc,yloc) - lower left corner coordinates (tuple)
        cellsize - cellsize (in meters?)
        nodata - value of nodata in 'data'
    Samuli Launiainen Luke 7.9.2016
    """
    import numpy as np

    fid = open(fname, 'r')
    info = fid.readlines()[0:6]
    fid.close()

    # print info
    # conversion to float is needed for non-integers read from file...
    xloc = float(info[2].split(' ')[-1])
    yloc = float(info[3].split(' ')[-1])
    cellsize = float(info[4].split(' ')[-1])
    nodata = float(info[5].split(' ')[-1])

    # read rest to 2D numpy array
    data = np.loadtxt(fname, skiprows=6)

    if setnans is True:
        data[data == nodata] = np.NaN
        nodata = np.NaN

    data = np.array(data, ndmin=2)

    return data, info, (xloc, yloc), cellsize, nodata


def write_AsciiGrid(fname, data, info, fmt='%.18e'):
    """ writes AsciiGrid format txt file
    IN:
        fname - filename
        data - data (numpy array)
        info - info-rows (list, 6rows)
        fmt - output formulation coding

    Samuli Launiainen Luke 7.9.2016
    """
    import numpy as np

    # replace nans with nodatavalue according to info
    nodata = int(info[-1].split(' ')[-1])
    data[np.isnan(data)] = nodata
    # write info
    fid = open(fname, 'w')
    fid.writelines(info)
    fid.close()

    # write data
    fid = open(fname, 'a')
    np.savetxt(fid, data, fmt=fmt, delimiter=' ')
    fid.close()


def delineate_catchment_from_dem(dem_path, catchment_name, out_fd, outlet_file, 
                                 clip_catchment=False, snap=True, routing='d8', 
                                 plot_catchment=True, fill_holes=True):

    print('')
    print('*** Delineating', catchment_name, 'catchment ***')
    outlets = pd.read_csv(outlet_file, sep=';', encoding = "ISO-8859-1")
    #outlet_x = float(outlets.loc[outlets['stream'] == catchment_name, 'lon'])
    #outlet_y = float(outlets.loc[outlets['stream'] == catchment_name, 'lat'])

    outlet_x = outlets.loc[outlets['stream'] == catchment_name, 'lon'].values[0]
    outlet_y = outlets.loc[outlets['stream'] == catchment_name, 'lat'].values[0]

    outpath = os.path.join(out_fd)
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    warnings.simplefilter("ignore", UserWarning)

    #raster = xr.open_rasterio(dem_path)
    grid = Grid.from_raster(dem_path)
    dem = grid.read_raster(dem_path)

    pit_filled_dem = grid.fill_pits(dem)
    flooded_dem = grid.fill_depressions(pit_filled_dem)
    inflated_dem = grid.resolve_flats(flooded_dem)

    dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
    fdir = grid.flowdir(inflated_dem, dirmap=dirmap, routing=routing)
    acc = grid.accumulation(fdir, dirmap=dirmap, routing=routing)
    aspect = grid.flowdir(inflated_dem, dirmap=dirmap, routing=routing)
    slope = grid.cell_slopes(inflated_dem, fdir, routing=routing)

    eps = np.finfo(float).eps

    twi = np.log((acc+1) / (np.tan(slope) + eps))

    # Snap pour point to high accumulation cell
    if snap == True:
        x_snap, y_snap = grid.snap_to_mask(acc > 100, (outlet_x, outlet_y))
    else:
        x_snap, y_snap = outlet_x, outlet_y

    # Delineate the catchment
    catch_full = grid.catchment(x=x_snap, y=y_snap, fdir=fdir, dirmap=dirmap,
                       xytype='coordinate', routing=routing)

    if clip_catchment == True:
    # Crop and plot the catchment
    # ---------------------------
    # Clip the bounding box to the catchment
        grid.clip_to(catch_full)
        clipped_catch = grid.view(catch_full)

    else:
        clipped_catch = catch_full

    #cmask_temp = np.ones(shape=clipped_catch.shape)
    #cmask_temp[clipped_catch == False] = 0
    #cmask_temp = ndimage.binary_fill_holes(cmask_temp).astype(int)
    #cmask = np.ones(shape=cmask_temp.shape)
    #cmask[cmask_temp == 0] = int(-9999)

    info = {'ncols':clipped_catch.shape[0],
        'nrows':clipped_catch.shape[1],
        'xllcorner':clipped_catch.bbox[0],
        'yllcorner':clipped_catch.bbox[1],
        'cellsize':clipped_catch.affine[0],
        'NODATA_value':-9999}

    subset = [clipped_catch.bbox[0], clipped_catch.bbox[1], clipped_catch.bbox[2], clipped_catch.bbox[3]]
    
    #fname=os.path.join(outpath, f'cmask_{routing}_{catchment_name}.asc')
    #write_AsciiGrid_new(fname, cmask, info)

    grid.to_ascii(dem, os.path.join(outpath, f'orig_dem_{catchment_name}.asc'), nodata=-9999)
    grid.to_ascii(clipped_catch, os.path.join(outpath, f'cmask_{routing}_{catchment_name}.asc'), nodata=-9999)
    grid.to_ascii(inflated_dem, os.path.join(outpath, f'inflated_dem_{catchment_name}.asc'), nodata=-9999)
    grid.to_ascii(fdir, os.path.join(outpath, f'fdir_{routing}_{catchment_name}.asc'), nodata=-9999)
    grid.to_ascii(acc, os.path.join(outpath, f'acc_{routing}_{catchment_name}.asc'), nodata=-9999)
    grid.to_ascii(slope, os.path.join(outpath, f'slope_{routing}_{catchment_name}.asc'), nodata=-9999)
    grid.to_ascii(aspect, os.path.join(outpath, f'aspect_{routing}_{catchment_name}.asc'), nodata=-9999)
    grid.to_ascii(twi, os.path.join(outpath, f'twi_{routing}_{catchment_name}.asc'), nodata=-9999)
    print('***', catchment_name, 'catchment is delineated and DEM derivatives are saved ***')

    if fill_holes==True:
        in_fn = os.path.join(outpath, f'cmask_{routing}_{catchment_name}.asc')
        filled_cmask, cmask_fp = fill_cmask_holes(in_fn, plot=False)
        filled_cmask[filled_cmask == 0] = -9999

    cmask_fill_grid = Grid.from_ascii(cmask_fp)
    cmask_fill_raster = grid.read_ascii(cmask_fp)
    
    if plot_catchment == True:
        # Plot the catchment
        
        fig, ax = plt.subplots(figsize=(8,6))
        fig.patch.set_alpha(0)

        plt.grid('on', zorder=0)
        ax.imshow(inflated_dem, extent=inflated_dem.extent, cmap='terrain', zorder=1)
        #ax.imshow(np.where(catch_full, catch_full, np.nan), extent=dem.extent,
        #       zorder=1, cmap='Greys_r', alpha=0.5)
        ax.imshow(np.where(cmask_fill_raster, cmask_fill_raster, np.nan), extent=inflated_dem.extent,
               zorder=1, cmap='Greys_r', alpha=0.3)       
        
        #ax.imshow(cmask, alpha=0.3)
        #raster.plot(ax=ax, vmin=200, vmax=600)
        #ax.imshow(acc, extent=clipped_catch.extent, zorder=2,
        #       cmap='cubehelix',
        #       norm=colors.LogNorm(1, acc.max()),
        #       interpolation='bilinear', alpha=0.1)
        plt.ylabel('Latitude')
        plt.xlabel('Longitude')
        plt.title(f'Delineated {catchment_name} catchment', size=14)

    return subset

def dem_derivatives(dem_path, out_fd, routing='d8'):

    outpath = os.path.join(out_fd)
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    warnings.simplefilter("ignore", UserWarning)

    #raster = xr.open_rasterio(dem_path)
    grid = Grid.from_raster(dem_path)
    dem = grid.read_raster(dem_path)

    pit_filled_dem = grid.fill_pits(dem)
    flooded_dem = grid.fill_depressions(pit_filled_dem)
    inflated_dem = grid.resolve_flats(flooded_dem)

    dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
    fdir = grid.flowdir(inflated_dem, dirmap=dirmap, routing=routing)
    acc = grid.accumulation(fdir, dirmap=dirmap, routing=routing)
    aspect = grid.flowdir(inflated_dem, dirmap=dirmap, routing=routing)
    slope = grid.cell_slopes(inflated_dem, fdir, routing=routing)

    eps = np.finfo(float).eps

    twi = np.log((acc+1) / (np.tan(slope) + eps))

    info = {'ncols':dem.shape[0],
        'nrows':dem.shape[1],
        'xllcorner':dem.bbox[0],
        'yllcorner':dem.bbox[1],
        'cellsize':dem.affine[0],
        'NODATA_value':-9999}
    

    grid.to_ascii(dem, os.path.join(outpath, f'orig_dem.asc'), nodata=-9999)
    grid.to_ascii(inflated_dem, os.path.join(outpath, f'inflated_dem.asc'), nodata=-9999)
    grid.to_ascii(fdir, os.path.join(outpath, f'fdir_{routing}.asc'), nodata=-9999)
    grid.to_ascii(acc, os.path.join(outpath, f'acc_{routing}.asc'), nodata=-9999)
    grid.to_ascii(slope, os.path.join(outpath, f'slope_{routing}.asc'), nodata=-9999)
    grid.to_ascii(aspect, os.path.join(outpath, f'aspect_{routing}.asc'), nodata=-9999)
    grid.to_ascii(twi, os.path.join(outpath, f'twi_{routing}.asc'), nodata=-9999)
    

def fill_cmask_holes(fp, fmt='%i', plot=True):
    '''
    for float fmt='%.18e'
    for int fmt='%i'
    '''
    # new filename
    fn = fp[0:-4]+'_fill.asc'

    # reading the old file for information
    old_cmask = read_AsciiGrid(fp)
    
    # taking the np.array to be edited
    arr = old_cmask[0]
    arr[np.isnan(arr)] = 0
    
    # filling the holes 
    new_arr = ndimage.binary_fill_holes(arr).astype(int)
    
    # assigning zeros to -9999 and plotting
    new_arr = np.where(new_arr==0, int(-9999), int(1))
    #print(new_arr)
    if plot==True:
        plt.imshow(np.where(new_arr==-9999, np.nan, 1))    
        plt.show()
    write_AsciiGrid(fn, new_arr, old_cmask[1])

    return new_arr, fn
    # writing the new cmask file
    #fid = open(fn, 'w')
    #for i in range(len(old_cmask[1])):
    #    fid.write(old_cmask[1][i]) 
    #fid.close()
    #fid = open(fn, 'a')
    #np.savetxt(fid, new_arr, fmt=fmt, delimiter=' ')
    #np.savetxt(fid, new_arr.astype(int), fmt=fmt, delimiter=' ')

def common_cmask(files):
    lens = len(files)
    i = 0
    for file in files:
        cnum = int(file.split('/')[-1].split('_')[2])
        temp_data = read_AsciiGrid(file)
        if i == 0:
            info = temp_data[1]
            out_fn = os.path.join(os.path.dirname(file), 'cmask_all.asc')
            final_data = temp_data[0].copy() # new ascii
            final_data[np.isfinite(final_data)] = cnum
        if (cnum != 0) & (i != 0):
            final_data[(temp_data[0] == 1) & (np.isnan(final_data))] = cnum
        else:
            whole = temp_data[0].copy()
        i += 1
        if i == lens:
            final_data[(whole == 1) & (np.isnan(final_data))] = 0
    
        del temp_data

    write_AsciiGrid(out_fn, final_data, info)
    masked_array = np.ma.array(final_data, mask=(final_data == -9999))
    plt.imshow(masked_array, vmin=0, vmax=14); plt.colorbar()

    return final_data
            

def reproj_match(infile, match, outfile, resampling_method='bilinear', save_in='asc'):
    """Reproject a file to match the shape and projection of existing raster. 
    
    Parameters
    ----------
    infile : (string) path to input file to reproject
    match : (string) path to raster with desired shape and projection 
    outfile : (string) path to output file tif
    """

    if resampling_method == 'bilinear':
        resample_as = Resampling.bilinear
    if resampling_method == 'nearest':
        resample_as = Resampling.nearest
    
    # open input
    with rasterio.open(infile) as src:
        src_transform = src.transform
        
        # open input to match
        with rasterio.open(match) as match:
            dst_crs = match.crs
            
            # calculate the output transform matrix
            dst_transform, dst_width, dst_height = calculate_default_transform(
                src.crs,     # input CRS
                dst_crs,     # output CRS
                match.width,   # input width
                match.height,  # input height 
                *match.bounds,  # unpacks input outer boundaries (left, bottom, right, top)
            )

        # set properties for output
        dst_kwargs = src.meta.copy()
        dst_kwargs.update({"crs": dst_crs,
                           "transform": dst_transform,
                           "width": dst_width,
                           "height": dst_height,
                           "nodata": 0})
        if save_in=='asc':
            dst_kwargs.update({"driver": "AAIGrid"})
            
        print("Coregistered to shape:", dst_height,dst_width,'\n Affine',dst_transform)
        # open output
        with rasterio.open(outfile, "w", **dst_kwargs) as dst:
            # iterate through bands and write using reproject function
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=resample_as)



def mask_raster_with_shapefile(raster_path, shapefile_path, output_path, nodata_value=None):
    """
    Mask a raster file using a shapefile, setting NoData values outside the shapefile.
    
    Parameters:
    - raster_path: str, path to the input raster file.
    - shapefile_path: str, path to the input shapefile.
    - output_path: str, path to the output masked raster file.
    - nodata_value: float or int, optional NoData value to use. If None, will use the raster's NoData value.
    """
    
    # Read the shapefile
    shapefile = gpd.read_file(shapefile_path)
    
    # Open the raster file
    with rasterio.open(raster_path) as src:
        # Read the raster's profile
        profile = src.profile
        
        nodata_value_old = profile.get('nodata')  # Get existing NoData value
        nodata_value_new = nodata_value  # Get existing NoData value
        
        # Mask the raster with the shapefile
        out_image, out_transform = mask(src, shapefile.geometry, crop=True)
        
        # Set NoData values outside the mask
        if nodata_value is not None:
            out_image = np.where(out_image == nodata_value_old, nodata_value, out_image)
            profile['nodata'] = nodata_value
            
        # Update the profile with the new transform and dimensions
        profile.update({
            'height': out_image.shape[1],
            'width': out_image.shape[2],
            'transform': out_transform
        })
        
        # Write the masked raster to a new file
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(out_image)


