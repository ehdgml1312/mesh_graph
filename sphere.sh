#!/bin/sh
for i in Afterthought-1 HLN-12-9 MMRR-21-20
SphericalRemesh -s i/lh.sphere.vtk -r ico/ico7icosphere_6.vtk -p lh.x.txt lh.y.txt lh.z.txt lh.curv.txt lh.sulc.txt lh.iH.txt lh.thickness.txt --outputProperty lh.6



