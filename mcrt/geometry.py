from __future__ import division
import numpy as np

class Mesh( object ):
    def __init__(self):
        pass

    def points():
        doc = "The points property."
        def fget(self):
            return self._points
        def fset(self, *coordgrids):
            self._points = np.meshgrid(coordgrids)
        def fdel(self):
            del self._points
        return locals()
    points = property(**points())

# class Point( object ):
#     def __init__(self, index, mesh):
#         self._index = index
#         self._mesh = mesh

#     @property
#     def index(self):
#         return self._index

#     @property
#     def coordinates(self):
#         return self._mesh(self.index)

#     @property
#     def boundary(self):
#         """Boundary of a point is empty"""
#         return set()

class Shape( object ):
    def __init__(self, vertices, sign):
        """vertices = list of vertex indices"""
        self.vertices = vertices
        self._sign = sign

    @property
    def sign(self):
        return self._sign

    @property
    def boundary(self):
        # return a list of shapes which bound the shape.
        vertices = self.vertices
        shapeclass = Shape
        boundary_shapes = []
        for i in range(len(vertices)-1):
            verts = vertices[:i]+vertices[i+1:] 
            sgn = (-1)**i
            boundary_element = shapeclass(verts, sign)
            boundary_shapes.append(boundary_element)
        return boundary_shapes


