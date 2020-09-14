#ifndef POLYGON_CLASS_H
#define POLYGON_CLASS_H

#include "parameters.hpp"
#include <stdio.h>

class polygon_class {
protected:
  struct vertexStruct {
    int index{0};
    std::pair<float, float> coordinates;
    bool ear{false};
    vertexStruct *next{nullptr};
    vertexStruct *prev{nullptr};
    vertexStruct(int index_in, std::pair<float, float> coordinates_in)
        : index(index_in), coordinates(coordinates_in) {}
  };

  typedef std::vector<std::vector<vertexStruct *>> trianglesType;
  std::vector<vertexStruct *> toBeDeletedVertices;

  trianglesType triangles;

  vertexStruct *vertices{nullptr};
  int number_of_vertices{0};
  float domainArea{0};
  bool error{false};

  void add(vertexStruct *p);
  void clearVertices(vertexStruct *p);
  void clearVertices();
  bool simpleLoop();

  v_points trianglePoints(vertexStruct *v1, vertexStruct *v2, vertexStruct *v3);

  v_points flatTrianglePoints(vertexStruct *v1, vertexStruct *v2,
                              vertexStruct *v3);

  float areaTwice(vertexStruct *v1, vertexStruct *v2, vertexStruct *v3);

  bool left(vertexStruct *v1, vertexStruct *v2, vertexStruct *v3);

  bool leftOn(vertexStruct *v1, vertexStruct *v2, vertexStruct *v3);

  bool collinear(vertexStruct *v1, vertexStruct *v2, vertexStruct *v3);

  bool intersectProp(vertexStruct *v1, vertexStruct *v2, vertexStruct *v3,
                     vertexStruct *v4);

  bool between(vertexStruct *v1, vertexStruct *v2, vertexStruct *v3);

  bool intersect(vertexStruct *v1, vertexStruct *v2, vertexStruct *v3,
                 vertexStruct *v4);

  bool diagonalIE(vertexStruct *v1, vertexStruct *v2);

  bool inCone(vertexStruct *v1, vertexStruct *v2);

  bool diagonal(vertexStruct *v1, vertexStruct *v2);

  void triangulate();
  void reOrientPoly();
  void earInit();
  float areaPolyTwice();
  bool line(vertexStruct *v1, vertexStruct *v2, float &dxdy, float &x0);

  polygon_class() {}
  ~polygon_class() { clearVertices(); }

  void vectorToVertex(v_points &vector);

  virtual v_points getInsidePoints() = 0;

public:
  bool getError();
};

class polygonBlob_class : public polygon_class {
public:
  polygonBlob_class(v_points &contour) {
    vectorToVertex(contour);
    domainArea = abs(areaPolyTwice()) / 2.f;
    triangulate();
  }

  v_points getInsidePoints() override;
};

#endif // POLYGON_CLASS_H
