import os
import numpy as np
import datetime

import matplotlib
import matplotlib.pyplot as plt
import geopandas as gpd

from skimage.filters import threshold_otsu
from shapely.wkt import loads
from shapely.geometry import Polygon

from eolearn.core import EOTask, EOPatch, LinearWorkflow, Dependency, FeatureType
from eolearn.io import S2L1CWCSInput
from eolearn.core import LoadFromDisk, SaveToDisk
# cloud detection
from eolearn.mask import AddCloudMaskTask, get_s2_pixel_cloud_detector
from eolearn.mask import AddValidDataMaskTask
# filtering of scenes
from eolearn.features import SimpleFilterTask
# burning the vectorised polygon to raster
from eolearn.geometry import VectorToRaster

# Sentinel Hub
from sentinelhub import BBox, CRS

with open('myID.txt') as f:
    WMS_INSTANCE = f.readline()


def shp2wkt(shapefile):
    tmp = gpd.GeoDataFrame.from_file(shapefile)
    tmp.to_crs(epsg=4326, inplace=True)
    wkt = tmp.geometry.values[0].to_wkt()

    with open(shapefile.replace('shp', 'wkt'), "w") as text_file:
        text_file.write(wkt)


# The polygon of the dam is written in wkt format (CRS=WGS84)
lake = 'Karaoun'
wkt_file = '/DATA/OBS2CO/vrac/shape/'+lake+'/'+lake+'.wkt'
wkt_file = 'eth_afar_lakes.wkt'

if not os.path.isfile(wkt_file):
    shp2wkt(wkt_file.replace('wkt', 'shp'))
# wkt_file = 'theewaterskloof_dam_nominal.wkt'
with open(wkt_file, 'r') as f:
    dam_wkt = f.read()

dam_nominal = loads(dam_wkt)

# inflate the BBOX
inflate_bbox = 5
minx, miny, maxx, maxy = dam_nominal.bounds
delx = maxx - minx
dely = maxy - miny

minx = minx - delx * inflate_bbox
maxx = maxx + delx * inflate_bbox
miny = miny - dely * inflate_bbox
maxy = maxy + dely * inflate_bbox

dam_bbox = BBox(bbox=[minx, miny, maxx, maxy], crs=CRS.WGS84)

input_task = S2L1CWCSInput('BANDS-S2-L1C', resx='20m', resy='20m', maxcc=1.,
                           time_difference=datetime.timedelta(hours=2), instance_id=WMS_INSTANCE)
add_ndwi = S2L1CWCSInput('NDWI', instance_id=WMS_INSTANCE)

gdf = gpd.GeoDataFrame(crs={'init': 'epsg:4326'}, geometry=[dam_nominal])
gdf.plot()
add_nominal_water = VectorToRaster((FeatureType.MASK_TIMELESS, 'NOMINAL_WATER'), gdf, 1,
                                   (FeatureType.MASK, 'IS_DATA'), np.uint8)

cloud_classifier = get_s2_pixel_cloud_detector(average_over=2, dilation_size=1, all_bands=False)
cloud_det = AddCloudMaskTask(cloud_classifier, 'BANDS-S2CLOUDLESS', cm_size_y='60m', cm_size_x='60m',
                             cmask_feature='CLM', cprobs_feature='CLP', instance_id=WMS_INSTANCE)


class ValidDataPredicate:
    def __call__(self, eopatch):
        return np.logical_and(eopatch.mask['IS_DATA'].astype(np.bool),
                              np.logical_not(eopatch.mask['CLM'].astype(np.bool)))


add_valmask = AddValidDataMaskTask(predicate=ValidDataPredicate())


def coverage(array):
    return 1.0 - np.count_nonzero(array) / np.size(array)


class AddValidDataCoverage(EOTask):
    def execute(self, eopatch):
        vld = eopatch.get_feature(FeatureType.MASK, 'VALID_DATA')

        cvrg = np.apply_along_axis(coverage, 1, np.reshape(vld, (vld.shape[0], vld.shape[1] * vld.shape[2])))

        eopatch.add_feature(FeatureType.SCALAR, 'COVERAGE', cvrg[:, np.newaxis])
        return eopatch


add_coverage = AddValidDataCoverage()


class ValidDataCoveragePredicate:
    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, array):
        coverage = 1.0 - np.count_nonzero(array) / np.size(array)
        return coverage < self.threshold


remove_cloudy_scenes = SimpleFilterTask((FeatureType.MASK, 'VALID_DATA'), ValidDataCoveragePredicate(1.))


def water_detection(ndwi):
    """
    Very simple water detector based on Otsu thresholding method of NDWI.
    """
    otsu_thr = 1.0
    if len(np.unique(ndwi)) > 1:
        otsu_thr = threshold_otsu(ndwi)

    return ndwi > otsu_thr


class WaterDetector(EOTask):
    def execute(self, eopatch):
        water_masks = np.asarray([water_detection(ndwi[..., 0]) for ndwi in eopatch.data['NDWI']])

        # we're only interested in the water within the dam borders
        water_masks = water_masks[..., np.newaxis] * eopatch.mask_timeless['NOMINAL_WATER']

        water_levels = np.asarray(
            [np.count_nonzero(mask) / np.count_nonzero(eopatch.mask_timeless['NOMINAL_WATER']) for mask in water_masks])

        eopatch.add_feature(FeatureType.MASK, 'WATER_MASK', water_masks)
        eopatch.add_feature(FeatureType.SCALAR, 'WATER_LEVEL', water_levels[..., np.newaxis])

        return eopatch


water_det = WaterDetector()

workflow = LinearWorkflow(input_task, add_ndwi, cloud_det, add_nominal_water, add_valmask,
                                           add_coverage, remove_cloudy_scenes, water_det)

time_interval = ['2018-08-01', '2018-08-31']
result = workflow.execute({input_task: {'bbox': dam_bbox, 'time_interval': time_interval}, })

patch = list(result.values())[-1]

from skimage.filters import sobel
from skimage.morphology import disk
from skimage.morphology import erosion, dilation, opening, closing, white_tophat


def plot_rgb_w_water(eopatch, idx):
    ratio = np.abs(eopatch.bbox.max_x - eopatch.bbox.min_x) / np.abs(eopatch.bbox.max_y - eopatch.bbox.min_y)
    fig, ax = plt.subplots(figsize=(ratio * 10, 10))

    ax.imshow(eopatch.data['BANDS-S2-L1C'][idx][..., [3, 2, 1]])

    observed = closing(eopatch.mask['WATER_MASK'][idx, ..., 0], disk(1))
    nominal = sobel(eopatch.mask_timeless['NOMINAL_WATER'][..., 0])
    observed = sobel(observed)
    nominal = np.ma.masked_where(nominal == False, nominal)
    observed = np.ma.masked_where(observed == False, observed)

    ax.imshow(nominal, cmap=plt.cm.Reds)
    ax.imshow(observed, cmap=plt.cm.Blues)
    ax.axis('off')


plot_rgb_w_water(patch, 0)


def plot_water_levels(eopatch, max_coverage=1.0):
    fig, ax = plt.subplots(figsize=(20, 7))

    dates = np.asarray(eopatch.timestamp)
    ax.plot(dates[eopatch.scalar['COVERAGE'][..., 0] < max_coverage],
            eopatch.scalar['WATER_LEVEL'][eopatch.scalar['COVERAGE'][..., 0] < max_coverage],
            'bo-', alpha=0.7)
    ax.plot(dates[eopatch.scalar['COVERAGE'][..., 0] < max_coverage],
            eopatch.scalar['COVERAGE'][eopatch.scalar['COVERAGE'][..., 0] < max_coverage],
            '--', color='gray', alpha=0.7)
    ax.set_ylim(0.0, 1.1)
    ax.set_xlabel('Date')
    ax.set_ylabel('Water level')
    ax.set_title(lake + ' Water Levels')
    ax.grid(axis='y')
    return ax


ax = plot_water_levels(patch, .05)

import imageio, os
from wand.image import Image


def make_gif(eopatch, project_dir, filename, fps, scale=None):
    """
    Generates a GIF animation from an EOPatch.
    """
    fout = os.path.join(project_dir, filename)

    with imageio.get_writer(fout, mode='I', fps=fps) as writer:
        for image in eopatch:
            writer.append_data(np.array(image[..., [3, 2, 1]], dtype=np.uint8))
    if scale != None:
        xsize = eopatch[0].shape[1] * scale
        ysize = eopatch[0].shape[0] * scale

        with Image(filename=fout) as img:
            img.resize(xsize, ysize)
            img.save(filename=fout)


make_gif(eopatch=patch.data['BANDS-S2-L1C'] * 2.5 * 255, project_dir='.', filename='eopatch_clean.gif', fps=3,
         scale=None)
