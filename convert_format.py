from skimage import io
import os
import osgeo.ogr as ogr
import osgeo.osr as osr
from osgeo import gdal

def main():
    root_dir = r'G:\My Drive\Digital_Agriculture\Liheng\manuscript\3rd_round_revision\figures'
    save_dir = r'G:\My Drive\Digital_Agriculture\Liheng\manuscript\3rd_round_revision\figures_jepg'
    if not os.path.exists(save_dir):
        os.makedirs((save_dir))
    for file in os.listdir(root_dir):
        if not file.endswith('.png'):
            continue
        img = io.imread(os.path.join(root_dir, file))
        io.imsave(os.path.join(root_dir, file).replace('png', 'jepg'), img)

def generate_shapefile():
    root_dir = r'X:\Morocco\rescaled'
    save_dir = r'X:\Morocco\shp'
    for file in os.listdir(root_dir):
        if not file.endswith('tif'):
            continue
        reference_path = os.path.join(root_dir, file)
        shape_path = os.path.join(save_dir, file.replace('tif', 'shp'))
        wkt = buffer_bbox(gdal.Open(reference_path))
        driver = ogr.GetDriverByName("ESRI Shapefile")
        data_source = driver.CreateDataSource(shape_path)
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(32630)
        layer = data_source.CreateLayer(file.replace('tif', 'shp'), srs, ogr.wkbPolygon)
        # field_name = ogr.FieldDefn("Name", ogr.OFTString)
        # field_name.SetWidth(14)
        # layer.CreateField(field_name)
        # field_name = ogr.FieldDefn("data", ogr.OFTString)
        # field_name.SetWidth(14)
        # layer.CreateField(field_name)
        feature = ogr.Feature(layer.GetLayerDefn())
        # feature.SetField("Name", "test")
        # feature.SetField("data", "1.2")
        polygon = ogr.CreateGeometryFromWkt(wkt)
        feature.SetGeometry(polygon)
        layer.CreateFeature(feature)

        feature = None
        data_source = None


def buffer_bbox(img):
    """
    Buffers the geom by buff and then calculates the bounding box.
    Returns a Geometry of the bounding box
    """
    img_geotrans = img.GetGeoTransform()
    lon1 = img_geotrans[0]
    w_e_pixel_resolution = img_geotrans[1]
    lat1 = img_geotrans[3]
    n_s_pixel_resolution = img_geotrans[5]
    lon2 = lon1 + img.RasterXSize
    lat2 = lat1 - img.RasterYSize
    wkt = """POLYGON((
        %s %s,
        %s %s,
        %s %s,
        %s %s,
        %s %s
    ))""" % (lon1, lat1, lon1, lat2, lon2, lat2, lon2, lat1, lon1, lat1)
    wkt = wkt.replace('\n', '')
    return wkt

if __name__ == '__main__':
    # main()
    generate_shapefile()
