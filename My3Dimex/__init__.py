from . import vectors, triangle, handlers

versionMag = 1
versionMin = 0
versionRev = 0

if __name__ == 'My3Dimex':
    print(f'3Dimex version: {versionMag}.{versionMin}.{versionRev}')

# Custom Importer/Exporter for 3D models (STL and OBJ/STL, respectively)
# Also supports conversion between STL and DataFrame
