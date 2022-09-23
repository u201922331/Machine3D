from My3Dimex.handlers import STL, Wavefront
from My3Dimex.vectors import Vec3
from My3Dimex.triangle import Triangle
import pandas as pd


def main():
    stlCube = STL.read('models/cube.stl')
    print(stlCube)

    stlMonkey = STL.read('models/suzanne.stl')
    stlMonkey.triangles.append(Triangle.generate(Vec3(1.0, -5.0, 0.0),
                                                 Vec3(0.0, 5.0, 0.0),
                                                 Vec3(-5.0, -5.0, 0.0),
                                                 0))
    stlMonkey.write('models/SuzanneOut.stl')
    dfMonkey = stlMonkey.toDataFrame()
    print('=' * 20 + '\n' + str(dfMonkey))

    stlMonke2: pd.DataFrame = STL.fromDataFrame(dfMonkey)
    print('=' * 20 + '\n' + str(stlMonke2))

    objCube = Wavefront.read('models/cube.obj')
    print('='*20+'\n' + str(objCube))
    stlCube2 = objCube.toSTL()
    print('='*20+'\n' + str(stlCube2))


if __name__ == '__main__':
    main()
