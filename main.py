from My3Dimex import handlers
from My3Dimex.vectors import Vec3
from My3Dimex.triangle import Triangle
import pandas as pd


def main():
    stlCube = handlers.STL.read('models/cube.stl')
    print(stlCube)
    stlMonkey = handlers.STL.read('models/suzanne.stl')
    stlMonkey.triangles.append(Triangle.generate(Vec3(-1.0, -1.0, 0.0),
                                                 Vec3(0.0, 1.0, 0.0),
                                                 Vec3(1.0, -1.0, 0.0),
                                                 0))
    stlMonkey.write('models/SuzanneOut.stl')
    dfMonkey = stlMonkey.toDataFrame()
    print(dfMonkey)

    stlMonke2: pd.DataFrame = handlers.STL.fromDataFrame(dfMonkey)
    print(stlMonke2)


if __name__ == '__main__':
    main()
