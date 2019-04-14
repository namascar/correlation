#include "polygon_class.h"

void polygon_class::add( vertexStruct *p )
{
    if ( !vertices )
    {
        vertices                = p;
        vertices->next          = vertices;
        vertices->prev          = vertices;
    }
    else
    {
        p->next                         = vertices;
        p->prev                         = vertices->prev;
        vertices->prev                  = p;
        p->prev->next                   = p;
    }
}

void polygon_class::clearVertices()
{
    for ( auto toBeDeleted : toBeDeletedVertices )
    {
        delete toBeDeleted;
        toBeDeleted = nullptr;
    }
}

void polygon_class::clearVertices( vertexStruct* head )
{
    vertexStruct *index = head;

    index->prev->next = nullptr;

    do
    {
        vertexStruct *toBeDeleted = index;
        index = index->next;
        delete toBeDeleted;
        number_of_vertices--;

    } while ( index );
}

void polygon_class::earInit()
{
    vertexStruct *v0 , *v1 , *v2;

    v1 = vertices;

    do
    {
        v2 = v1->next;
        v0 = v1->prev;
        v1->ear = diagonal( v0 , v2 );

        v1 = v1->next;

    } while ( v1 != vertices );
}

float polygon_class::areaTwice(  vertexStruct *v1 ,
                                 vertexStruct *v2 ,
                                 vertexStruct *v3 )
{
    return      ( v2->coordinates.first  - v1->coordinates.first  )             // By cross product
            *   ( v3->coordinates.second - v1->coordinates.second )

            -   ( v3->coordinates.first  - v1->coordinates.first  )
            *   ( v2->coordinates.second - v1->coordinates.second );
}

void polygon_class::vectorToVertex( v_points &vector )
{
    int i = 0;
    for ( auto v : vector )
    {
        vertexStruct *other = new vertexStruct( i++ , v );
        add( other );
    }
    number_of_vertices = i;
}

float polygon_class::areaPolyTwice(  )
{
    float sum { 0.f };

    vertexStruct *a = vertices->next;

    do
    {
        sum += areaTwice( vertices , a , a->next );
        a = a->next;

    } while ( a->next != vertices );

    return sum;
}

void polygon_class::reOrientPoly()
{
    vertexStruct *index = vertices;

    do
    {
        vertexStruct *next = index->next;

        vertexStruct *temp = index->prev;
        index->prev = index->next;
        index->next = temp;

        index = next;

    } while ( index != vertices );
}

bool polygon_class::left( vertexStruct *v1 ,
                          vertexStruct *v2 ,
                          vertexStruct *v3 )
{
    return  areaTwice( v1 , v2 , v3 ) > 0.f;
}

bool polygon_class::leftOn( vertexStruct *v1 ,
                            vertexStruct *v2 ,
                            vertexStruct *v3 )
{
    return  areaTwice( v1 , v2 , v3 ) >= 0.f;
}

bool polygon_class::collinear(  vertexStruct *v1 ,
                                vertexStruct *v2 ,
                                vertexStruct *v3 )
{
    return  areaTwice( v1 , v2 , v3 ) == 0.f;
}

bool polygon_class::intersectProp( vertexStruct *v1 ,
                                   vertexStruct *v2 ,
                                   vertexStruct *v3 ,
                                   vertexStruct *v4 )
{
    if ( collinear( v1 , v2 , v3 ) ||
         collinear( v1 , v2 , v4 ) ||
         collinear( v2 , v4 , v1 ) ||
         collinear( v3 , v4 , v2 ) ) return false;

    return ( !left( v1 , v2 , v3 ) ^ !left( v1 , v2 , v4 ) ) &&
           ( !left( v3 , v4 , v1 ) ^ !left( v3 , v4 , v2 ) );
}

bool polygon_class::between(    vertexStruct *v1 ,
                                vertexStruct *v2 ,
                                vertexStruct *v3 )
{
    if ( !collinear( v1 , v2 , v3 ) ) return false;

    if ( v1->coordinates.first != v2->coordinates.first )
    {
        return ( ( v1->coordinates.first <= v3->coordinates.first ) &&
                 ( v3->coordinates.first <= v2->coordinates.first ) ) ||

                ( ( v1->coordinates.first >= v3->coordinates.first ) &&
                  ( v3->coordinates.first >= v2->coordinates.first ) ) ;
    }
    else
    {
        return ( ( v1->coordinates.second <= v3->coordinates.second ) &&
                 ( v3->coordinates.second <= v2->coordinates.second ) ) ||

                ( ( v1->coordinates.second >= v3->coordinates.second ) &&
                  ( v3->coordinates.second >= v2->coordinates.second ) ) ;
    }
}

bool polygon_class::intersect(  vertexStruct *v1 ,
                                vertexStruct *v2 ,
                                vertexStruct *v3 ,
                                vertexStruct *v4 )
{
    if ( intersectProp( v1 , v2 , v3 , v4 ) )
    {
        return true;
    }
    else if ( between( v1 , v2 , v3 ) ||
              between( v1 , v2 , v4 ) ||
              between( v3 , v4 , v1 ) ||
              between( v3 , v4 , v2 ) )
    {
        return true;
    }
    else
    {
        return false;
    }
}

bool polygon_class::diagonalIE( vertexStruct *v1 ,
                                vertexStruct *v2 )
{
    vertexStruct *c , *c1;

    c = vertices;

    do
    {
        c1 = c->next;

        if(  ( c != v1 ) && ( c1 != v1 ) &&
             ( c != v2 ) && ( c1 != v2 ) &&
             intersect( v1 , v2 , c , c1 ) )
        {
            return false;
        }

        c = c->next;

    } while ( c != vertices );

    return true;
}

bool polygon_class::inCone( vertexStruct *v1 ,
                            vertexStruct *v2 )
{
    vertexStruct *a0 , *a1;

    a1 = v1->next;
    a0 = v1->prev;

    if ( leftOn( v1 , a1 , a0 ) )
    {
        return left( v1 , v2 , a0 ) &&
               left( v2 , v1 , a1 );
    }
    else
    {
        return !( leftOn( v1 , v2 , a1 ) &&
                  leftOn( v2 , v1 , a0 )    );
    }
}

bool polygon_class::diagonal( vertexStruct *v1 ,
                              vertexStruct *v2 )
{
    return inCone( v1 , v2 ) && inCone( v2 , v1 ) && diagonalIE( v1 , v2 );
}

bool polygon_class::getError( )
{
    return error;
}

bool polygon_class::simpleLoop()
{
    if ( number_of_vertices < 4 )
    {
        return true;
    }

    vertexStruct *outerLeft = vertices;
    vertexStruct *outerRight;

    do
    {
        outerRight = outerLeft->next;
        vertexStruct *innerLeft = outerRight->next;
        vertexStruct *innerRight;
        do
        {
            innerRight = innerLeft->next;
            if ( intersect( outerLeft , outerRight , innerLeft , innerRight ) )
            {
                return false;
            }

            innerLeft = innerRight;

        } while( innerLeft != vertices && innerLeft != outerLeft->prev );

        outerLeft = outerRight;

    } while ( outerLeft != vertices->prev->prev );

    return true;
}

void polygon_class::triangulate()
{
    error = !simpleLoop();
    if ( error )
    {
        clearVertices( vertices );
        return;
    }

    if ( areaPolyTwice() < 0 )
    {
        reOrientPoly();
    }

    vertexStruct *v0 , *v1 , *v2 , *v3 , *v4;

    earInit();

    triangles.clear();

    while ( number_of_vertices > 3 )            // Each step of outer loop removes one ear
    {
        v2 = vertices;

        do                                      // Inner loop searches for ears
        {
            if ( v2->ear )                      // Ear found
            {
                v3 = v2->next;                  // Fill variable
                v4 = v3->next;
                v1 = v2->prev;
                v0 = v1->prev;

                std::vector < vertexStruct* > triangle { v1 , v2 , v3 };
                triangles.push_back( triangle );

                toBeDeletedVertices.push_back( v2 );

                v1->ear = diagonal( v0 , v3 );  //Update earity of diagonal endpoints
                v3->ear = diagonal( v1 , v4 );

                v1->next = v3;                  // Cut off ear v2
                v3->prev = v1;
                vertices = v3;                  // In case head was v2
                number_of_vertices--;

                break;
            }

            v2 = v2->next;

        } while ( v2 != vertices );
    }

    v2 = vertices;
    v1 = v2->prev;
    v3 = v2->next;

    std::vector < vertexStruct* > triangle { v1 , v2 , v3 };
    triangles.push_back( triangle );
}

v_points polygon_class::trianglePoints( vertexStruct *v1 ,
                                        vertexStruct *v2 ,
                                        vertexStruct *v3 )
{
    if ( v2->coordinates.second == v1->coordinates.second )     // if one of the sides is already horizontal - call flat Triange
    {
        flatTrianglePoints( v1 , v2 , v3 );
    }

    if ( v3->coordinates.second == v1->coordinates.second )
    {
        flatTrianglePoints( v3 , v1 , v2 );
    }

    if ( v3->coordinates.second == v2->coordinates.second )
    {
        flatTrianglePoints( v2 , v3 , v1 );
    }

    vertexStruct *ymax , *ymid , *ymin;                          // Order vertices by y
    if ( v1->coordinates.second > v2->coordinates.second )
    {
        if ( v2->coordinates.second > v3->coordinates.second )
        {
            ymax = v1;
            ymid = v2;
            ymin = v3;
        }
        else if ( v3->coordinates.second > v1->coordinates.second )
        {
            ymax = v3;
            ymid = v1;
            ymin = v2;
        }
        else
        {
            ymax = v1;
            ymid = v3;
            ymin = v2;
        }
    }
    else
    {
        if ( v1->coordinates.second > v3->coordinates.second )
        {
            ymax = v2;
            ymid = v1;
            ymin = v3;
        }
        else if ( v3->coordinates.second > v2->coordinates.second )
        {
            ymax = v3;
            ymid = v2;
            ymin = v1;
        }
        else
        {
            ymax = v2;
            ymid = v3;
            ymin = v1;
        }
    }

    float dxdy;
    float x0;
    v_points all;
    if ( line( ymin , ymax , dxdy, x0 ) )               // Returm an empty vector
    {
        return all;
    }

    float newY = ymid->coordinates.second;
    float newX = dxdy * newY + x0;

    std::pair< float , float > newCoordinates( newX , newY );
    vertexStruct *ynew = new vertexStruct( -1 , newCoordinates );

    v_points upperTrianglePoints =  flatTrianglePoints( ymid , ynew , ymax );
    v_points lowerTrianglePoints =  flatTrianglePoints( ymid , ynew , ymin );

    delete ynew;

    all.reserve( upperTrianglePoints.size() + lowerTrianglePoints.size() );
    all.insert( all.end() , upperTrianglePoints.begin() , upperTrianglePoints.end() );
    all.insert( all.end() , lowerTrianglePoints.begin() , lowerTrianglePoints.end() );

    return all;
}

v_points polygon_class::flatTrianglePoints( vertexStruct *v1 ,  // v1 nad v2 have same y coordinate
                                            vertexStruct *v2 ,
                                            vertexStruct *v3 )
{
    v_points all;

    int dy = floor( v3->coordinates.second ) - floor( v1->coordinates.second );
    int dx = floor( v2->coordinates.first  ) - floor( v1->coordinates.first  );

    if ( dx == 0 || dy == 0 )
    {
        return all;
    }

    vertexStruct *vxSmall, *vxBig;
    if ( dx > 0 )
    {
        vxSmall = v1;
        vxBig   = v2;
    }
    else
    {
        vxSmall = v2;
        vxBig   = v1;
    }

    float dxdySmall , dxdyBig , x0Small , x0Big;                    // compute bounding lines.
    line( vxSmall , v3 , dxdySmall , x0Small );
    line( vxBig   , v3 , dxdyBig   , x0Big   );

    int jInit , jEnd;
    if ( dy > 0 )
    {
        jInit = ceil( v1->coordinates.second );
        jEnd  = ceil( v3->coordinates.second );
    }
    else
    {
        jInit = ceil( v3->coordinates.second );
        jEnd  = ceil( v1->coordinates.second );
    }

    all.reserve( ( ( abs ( dx ) + 1 ) * ( abs ( dy ) + 1 ) ) / 2 );

    for ( int j = jInit ; j < jEnd ; ++j )
    {
        int iInit = ceil ( dxdySmall * (float) j + x0Small );
        int iEnd  = ceil ( dxdyBig   * (float) j + x0Big   );

        for ( int i = iInit ; i < iEnd ; ++i )
        {
            all.push_back( std::make_pair( i , j ) );
        }
    }

    return all;
}

bool polygon_class::line( vertexStruct *v1 ,
                          vertexStruct *v2 ,
                          float &dxdy ,
                          float &x0 )
{
    float denominator = v2->coordinates.second - v1->coordinates.second;
    bool error = true;

    if ( denominator != 0 )
    {
        dxdy = ( v2->coordinates.first - v1->coordinates.first ) /
                denominator;
        x0   = v1->coordinates.first - dxdy * v1->coordinates.second;
        error = false;
    }
    return error;
}

v_points polygonBlob_class::getInsidePoints( )
{
    v_points all;
    all.reserve( (int) ( domainArea * 1.2f ) );

    for ( auto triangle : triangles )
    {
        v_points contribution = trianglePoints( triangle[ 0 ] , triangle[ 1 ] , triangle[ 2 ] );
        all.insert( all.end() , contribution.begin() , contribution.end() );
    }

    return all;
}




