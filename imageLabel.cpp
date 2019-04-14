/****************************************************************************
**
**  This software was developed by Javier Gonzalez on Feb 2018 from
**  Qt examples
**
**
****************************************************************************/

#include "imageLabel.h"

ImageLabel::ImageLabel( QWidget *parent ) : QLabel( parent )
{
    selecting           = false;
    selecting_ic        = false;
    correlating         = false;
    adjusting_domain    = false;
    error               = false;

    // Initializes the center so that it is outside the
    // image. This way, the first mousePress will
    // define a new domain and not modify an existing one.
    center_x = - min_blob_segment_squared * 2.f;
    center_y = - min_blob_segment_squared * 2.f;
    right_x  = - min_blob_segment_squared * 2.f;
    right_y  = - min_blob_segment_squared * 2.f;
    left_x   = - min_blob_segment_squared * 2.f;
    left_y   = - min_blob_segment_squared * 2.f;
}

void ImageLabel::mousePressEvent( QMouseEvent *event )
{
    // We are selecting/refining initial conditions
    if ( selecting_ic && selecting && event->button() == Qt::LeftButton )
    {
        ic_und_x = event->x();
        ic_und_y = event->y();

        int number_of_model_parameters = ModelClass::get_number_of_model_parameters( model );

        for ( int i = 0; i < number_of_model_parameters; ++i )
        {
            prev_model_parameters[ i ] = model_parameters[ i ];
        }

        switch ( model )
        {
            case fm_U:
                break;

            case fm_UV:
                break;

            case fm_UVQ:
                if( domain == domain_annular )
                {
                    float dx = ic_und_x - def_center_x;
                    float dy = ic_und_y - def_center_y;

                    float r = sqrt ( dx * dx + dy * dy );
                    float ro = def_outside_radius;
                    float ri = def_inside_radius;

                    if ( r < ro + ( ro - ri ) / 2.f && r > ri + ( ro - ri ) / 2.f )
                    {
                        UVUxUyVxVyIC = UVUxUyVxVyIC_Q;
                    }
                    else
                    {
                        UVUxUyVxVyIC = UVUxUyVxVyIC_Center;
                    }
                }
                else
                {
                    if (ic_und_x > def_hi_left_x && ic_und_x < def_lo_right_x &&
                            ic_und_y > def_hi_left_y && ic_und_y < def_lo_right_y )
                    {
                        UVUxUyVxVyIC = UVUxUyVxVyIC_Center;
                    }
                    else if ( abs( ic_und_x - def_center_x ) > abs( ic_und_y - def_center_y ) )
                    {
                        UVUxUyVxVyIC = UVUxUyVxVyIC_DU;
                    }
                    else
                    {
                        UVUxUyVxVyIC = UVUxUyVxVyIC_DV;
                    }
                }
                break;

            case fm_UVUxUyVxVy:
                if(domain == domain_annular)
                {
                    float dx = ic_und_x - def_center_x;
                    float dy = ic_und_y - def_center_y;

                    float r = sqrt ( dx * dx + dy * dy );
                    float ro = def_outside_radius;
                    float ri = def_inside_radius;

                    if ( r < ro + ( ro - ri ) / 2.f && r > ri + ( ro - ri ) / 2.f )
                    {
                        UVUxUyVxVyIC = UVUxUyVxVyIC_Q;
                    }
                    else
                    {
                        UVUxUyVxVyIC = UVUxUyVxVyIC_Center;
                    }
                }
                else
                {
                    if (ic_und_x > def_hi_left_x && ic_und_x < def_lo_right_x &&
                            ic_und_y > def_hi_left_y && ic_und_y < def_lo_right_y )
                    {
                        UVUxUyVxVyIC = UVUxUyVxVyIC_Center;
                    }
                    else if ( abs(ic_und_x - def_center_x) > abs(ic_und_y - def_center_y))
                    {
                        UVUxUyVxVyIC = UVUxUyVxVyIC_DU;
                    }
                    else
                    {
                        UVUxUyVxVyIC = UVUxUyVxVyIC_DV;
                    }
                }
                break;

            default:
                assert(false);
                break;
        }

        //connect(def_imageLabel, SIGNAL( stop_correlation_display() ), this, SLOT( stop_correlation_display() ));
        emit stop_correlation_display();
        emit new_initial_conditions();

    }//selecting initial_conditions bracket

    // We are selecting a domain
    else if ( selecting && !mirroring && event->button() == Qt::LeftButton )
    {
        //  Check if we are adjusting the position of the intial domain or creating a new one
        //      Clicking inside an existing domain modifies it, clicking outside creates a new
        //      one.
        adjusting_domain = false;
        float dx = (float) event->x() - center_x;
        float dy = (float) event->y() - center_y;

        float r = sqrt ( dx * dx + dy * dy );

        switch ( domain )
        {
            case domain_annular:
                {
                    if ( r < inside_radius / 2 )
                    {
                        adjusting_domain = true;
                        annularIC = annularIC_Center;
                    }
                    else if ( r < ( inside_radius + outside_radius ) / 2 )
                    {
                        adjusting_domain = true;
                        annularIC = annularIC_ri;
                    }
                    else if ( r < outside_radius + ( outside_radius - inside_radius ) / 2 )
                    {
                        adjusting_domain = true;
                        annularIC = annularIC_ro;
                    }
                }
                break;

            case domain_rectangular:
                {
                    if ( r < 0.5f * std::min( (int) abs( right_x - left_x ) , (int) abs( right_y - left_y )  ) )
                    {
                        adjusting_domain = true;
                    }
                }
                break;

            case domain_blob:
                {
                    if ( r < 0.5f * std::min( (int) abs( blob_x_max - blob_x_min ) , (int) abs( blob_y_max - blob_y_min )  ) )
                    {
                        adjusting_domain = true;
                    }
                }
                break;

            default:
                assert( false );
                break;
        }

        // Save the old domain to modify it as we move the mouse
        if( adjusting_domain )
        {
            adjust_domain_left_x = (float) event->x();
            adjust_domain_left_y = (float) event->y();

            prev_center_x = center_x;
            prev_center_y = center_y;

            prev_inside_radius  = inside_radius;
            prev_outside_radius = outside_radius;

            prev_left_x  = left_x;
            prev_left_y  = left_y;
            prev_right_x = right_x;
            prev_right_y = right_y;

            if ( domain == domain_blob )
            {
                prev_xy_blob.clear();

                for ( int i = 0; i < (int)xy_blob.size(); ++i )
                {
                    prev_xy_blob.push_back( xy_blob[i] );
                }
            }
        }
        else // Make a new domain selection if clicking away from the center
        {
            adjusting_domain = false;

            switch (domain)
            {
                case domain_annular:                
                    left_x   = (float)event->x();
                    left_y   = (float)event->y();

                    right_x  = (float)event->x();
                    right_y  = (float)event->y();

                    center_x = (float)event->x();
                    center_y = (float)event->y();

                    inside_radius = 0.f;
                    outside_radius = 0.f;

                    // Calls mirrorSelectionAnnular on the def_imageLabel
                    emit valueChanged_selectingAnnular( center_x, center_y,
                                                        inside_radius, outside_radius, ri_by_ro,
                                                        clickScale_x, clickScale_y);
                    break;

                case domain_rectangular:
                    left_x = (float)event->x();
                    left_y = (float)event->y();

                    right_x = (float)event->x();
                    right_y = (float)event->y();

                    // Calls mirrorSelectionRectangular on the def_imageLabel
                    emit valueChanged_selectingRectangular( center_x, center_y, right_x, right_y,
                                                            left_x, left_y, clickScale_x, clickScale_y);

                    break;

                case domain_blob:
                 {
                    xy_blob.clear();
                    std::pair<float,float> xy_pair = std::make_pair( (float)event->x() , (float)event->y() );
                    xy_blob.push_back( xy_pair );

                    // Calls mirrorSelectionBlob on the def_imageLabel
                    emit valueChanged_selectingBlob( center_x, center_y,
                                                     xy_blob, clickScale_x, clickScale_y);

                    break;
                  }
                default:
                    Q_ASSERT("Unavailable domain option");

                    break;
            }//switch bracket

            //connect(und_imageLabel, SIGNAL( stop_correlation_display() ), this, SLOT( stop_correlation_display() ));
        } //make new domain selection bracket
        emit stop_correlation_display();
    }// selecting domains
}

void ImageLabel::mouseMoveEvent(QMouseEvent *event)
{
    // We are selecting initial conditions
    if ( selecting_ic && selecting )
    {
        ic_def_x = event->x();
        ic_def_y = event->y();

        switch (model)
        {
        case fm_U:

            model_parameters[0] = prev_model_parameters[0] + (ic_def_x - ic_und_x)/ realScale_x;

            break;

        case fm_UV:

            model_parameters[0] = prev_model_parameters[0] + (ic_def_x - ic_und_x) / realScale_x;
            model_parameters[1] = prev_model_parameters[1] + (ic_def_y - ic_und_y) / realScale_y;

            break;

        case fm_UVQ:

            switch (UVUxUyVxVyIC)
            {
            case UVUxUyVxVyIC_Center:

                model_parameters[0] = prev_model_parameters[0] + (ic_def_x - ic_und_x) / realScale_x;
                model_parameters[1] = prev_model_parameters[1] + (ic_def_y - ic_und_y) / realScale_y;

                break;

            case UVUxUyVxVyIC_DU:

                model_parameters[2] = prev_model_parameters[2] + (ic_def_y - ic_und_y) * (ic_und_x > def_center_x ? -1.f : 1.f ) * 0.001f;

                break;

            case UVUxUyVxVyIC_DV:

                model_parameters[2] = prev_model_parameters[2] + (ic_def_x - ic_und_x) * (ic_und_y > def_center_y ? -1.f : 1.f ) * 0.001f;

                break;

            case UVUxUyVxVyIC_Q:
            {

                float angle = atan2( ic_def_y - def_center_y , ic_def_x - def_center_x )
                            - atan2( ic_und_y - def_center_y , ic_und_x - def_center_x );

                model_parameters[2] = prev_model_parameters[2] + sin( (float)angle );

                break;
            }

            default:

                break;

            } //UVQ switch
        break;

        case fm_UVUxUyVxVy:

            switch (UVUxUyVxVyIC)
            {
            case UVUxUyVxVyIC_Center:

                model_parameters[0] = prev_model_parameters[0] + (ic_def_x - ic_und_x) / realScale_x;
                model_parameters[1] = prev_model_parameters[1] + (ic_def_y - ic_und_y) / realScale_y;

                break;

            case UVUxUyVxVyIC_DU:

                model_parameters[2] = prev_model_parameters[2] + (ic_def_x - ic_und_x) / (ic_und_x - def_center_x);
                model_parameters[3] = prev_model_parameters[3] + (ic_def_y - ic_und_y) * (ic_und_x > def_center_x ? -1.f : 1.f ) * 0.001f;

                break;

            case UVUxUyVxVyIC_DV:

                model_parameters[4] = prev_model_parameters[4] + (ic_def_x - ic_und_x) * (ic_und_y > def_center_y ? -1.f : 1.f ) * 0.001f;
                model_parameters[5] = prev_model_parameters[5] + (ic_def_y - ic_und_y) / (ic_und_y - def_center_y);

                break;

            case UVUxUyVxVyIC_Q:
            {

                float angle = atan2( ic_def_y - def_center_y , ic_def_x - def_center_x )
                            - atan2( ic_und_y - def_center_y , ic_und_x - def_center_x );

                model_parameters[2] = prev_model_parameters[2] + cos( (float)angle ) - 1.f;
                model_parameters[3] = prev_model_parameters[3] - sin( (float)angle );
                model_parameters[4] = prev_model_parameters[4] + sin( (float)angle );
                model_parameters[5] = prev_model_parameters[5] + cos( (float)angle ) - 1.f;

                break;
            }

            default:

                break;

            } //UVUxUyVxVy switch
        break;

        default:

                Q_ASSERT("Unavailable domain option");

                break;
        }
        emit new_initial_conditions();
    }
    // We are selecting a domain - creating a new one or moving it
    else if ( selecting && !mirroring )
    {
        if ( adjusting_domain ) // moving existing domain
        {
            adjust_domain_right_x = (float) event->x();
            adjust_domain_right_y = (float) event->y();

            switch( domain )
            {
                case domain_annular:

                    switch ( annularIC )
                    {
                        case annularIC_Center:

                        center_x = prev_center_x + adjust_domain_right_x - adjust_domain_left_x;
                        center_y = prev_center_y + adjust_domain_right_y - adjust_domain_left_y;
                            break;

                        case annularIC_ri:
                        case annularIC_ro:
                            {
                                float old_dx = center_x - adjust_domain_left_x;
                                float new_dx = center_x - adjust_domain_right_x;
                                float old_dy = center_y - adjust_domain_left_y;
                                float new_dy = center_y - adjust_domain_right_y;

                                float old_d = old_dx * old_dx + old_dy * old_dy;
                                float new_d = new_dx * new_dx + new_dy * new_dy;

                                if ( annularIC == annularIC_ri )
                                {
                                     inside_radius = prev_inside_radius *  sqrt( new_d / old_d );
                                }
                                else
                                {
                                    outside_radius = prev_outside_radius * sqrt( new_d / old_d );
                                }
                                ri_by_ro = inside_radius / outside_radius;
                            }
                            break;

                        default:
                            assert( false );
                            break;
                    }

                    // Calls mirrorSelectionAnnular on the def_imageLabel
                    emit valueChanged_selectingAnnular( center_x, center_y,
                                                        inside_radius, outside_radius, ri_by_ro,
                                                        clickScale_x, clickScale_y);

                    break;

                case domain_rectangular:

                    center_x = prev_center_x + adjust_domain_right_x - adjust_domain_left_x;
                    center_y = prev_center_y + adjust_domain_right_y - adjust_domain_left_y;

                    left_x   = prev_left_x   + adjust_domain_right_x - adjust_domain_left_x;
                    left_y   = prev_left_y   + adjust_domain_right_y - adjust_domain_left_y;
                    right_x  = prev_right_x  + adjust_domain_right_x - adjust_domain_left_x;
                    right_y  = prev_right_y  + adjust_domain_right_y - adjust_domain_left_y;

                    // Calls mirrorSelectionRectangular on the def_imageLabel
                    emit valueChanged_selectingRectangular( center_x, center_y, right_x, right_y,
                                                            left_x, left_y, clickScale_x, clickScale_y);

                    break;

                case domain_blob:

                    center_x = prev_center_x + adjust_domain_right_x - adjust_domain_left_x;
                    center_y = prev_center_y + adjust_domain_right_y - adjust_domain_left_y;

                    for ( int i = 0; i < (int) xy_blob.size(); ++i )
                    {
                        xy_blob[ i ].first   = prev_xy_blob[i].first  + adjust_domain_right_x - adjust_domain_left_x;
                        xy_blob[ i ].second  = prev_xy_blob[i].second + adjust_domain_right_y - adjust_domain_left_y;
                    }

                    // Calls mirrorSelectionBlob on the def_imageLabel
                    emit valueChanged_selectingBlob( center_x , center_y,
                                                     xy_blob, clickScale_x, clickScale_y);

                    break;

                default:

                    assert(false);

                    break;

            }//switch domain bracket
        }// if adjusting_domain bracket
        else // Creating new domain
            switch ( domain )
            {
                case domain_annular:

                    right_x = (float) event->x();
                    right_y = (float) event->y();

                    center_x       = makeAnnularXc( right_x , right_y , left_x , left_y );
                    center_y       = makeAnnularYc( right_x , right_y , left_x , left_y );

                    inside_radius  = makeAnnularRi( right_x , right_y , left_x , left_y , ri_by_ro);
                    outside_radius = makeAnnularRo( right_x , right_y , left_x , left_y );

                    // Calls mirrorSelectionAnnular on the def_imageLabel
                    emit valueChanged_selectingAnnular( center_x, center_y,
                                                        inside_radius, outside_radius, ri_by_ro,
                                                        clickScale_x, clickScale_y);

                    break;

                case domain_rectangular:

                    right_x = (float) event->x();
                    right_y = (float) event->y();

                    center_x = makeRectangularXYc( left_x, right_x );
                    center_y = makeRectangularXYc( left_y, right_y );

                    // Calls mirrorSelectionRectangular on the def_imageLabel
                    emit valueChanged_selectingRectangular( center_x, center_y, right_x, right_y,
                                                            left_x, left_y, clickScale_x, clickScale_y);

                    break;

                case domain_blob:

                {
                    float x1 = (float) event->x();
                    float y1 = (float) event->y();
                    float x0 = xy_blob.back().first;
                    float y0 = xy_blob.back().second;

                    float dx = x0 - x1;
                    float dy = y0 - y1;
                    float dr = dx * dx + dy * dy;

                    if ( dr >= min_blob_segment_squared )
                      xy_blob.push_back( std::make_pair( x1 , y1 ) );

                    // Define center_x to be able to draw blob
                    center_x   = xy_blob[ 0 ].first;
                    blob_x_max = xy_blob[ 0 ].first;
                    blob_x_min = xy_blob[ 0 ].first;

                    center_y   = xy_blob[ 0 ].second;
                    blob_y_max = xy_blob[ 0 ].second;
                    blob_y_min = xy_blob[ 0 ].second;

                    for ( int i = 1; i < (int) xy_blob.size(); ++i )
                    {
                        center_x  += xy_blob[ i ].first;
                        center_y  += xy_blob[ i ].second;

                        if ( xy_blob[ i ].first  > blob_x_max ) blob_x_max = xy_blob[ i ].first;
                        if ( xy_blob[ i ].first  < blob_x_min ) blob_x_min = xy_blob[ i ].first;
                        if ( xy_blob[ i ].second > blob_y_max ) blob_y_max = xy_blob[ i ].second;
                        if ( xy_blob[ i ].second < blob_y_min ) blob_y_min = xy_blob[ i ].second;
                    }
                    center_x  /= (float) xy_blob.size();
                    center_y  /= (float) xy_blob.size();

                    // Calls mirrorSelectionBlob on the def_imageLabel
                    emit valueChanged_selectingBlob(center_x, center_y, xy_blob, clickScale_x, clickScale_y);

                    break;
                }

                default:

                    assert(false);

                    break;

            }//switch bracket
    }//We are selecting a domain bracket
    update();
}

void ImageLabel::mouseReleaseEvent(QMouseEvent* event)
{
    // We are selecting initial conditions
    if ( selecting_ic && selecting && event->button() == Qt::LeftButton )
    {
        update();
    }
    // We are selecting a domain
    else if (selecting && !mirroring && event->button() == Qt::LeftButton)
    {
        switch( domain )
        {
        case domain_rectangular:

#if DEBUG_DOMAIN_SELECTION
            printf("%15s %30s: center x = %6.2f  center_y = %6.2f  left_x = %6.2f  right_x = %6.2f  left_y = %6.2f  right_y = %6.2f  scale_x = %6.2f  scale_y = %6.2f\n",
                   "und_imageLabel","mouseRelease", center_x, center_y,
                   left_x, right_x, left_y, right_y,
                   clickScale_x, clickScale_y);
#endif

            // Calls enableRectangularCorrelate in MainApp
            emit valueChanged_rectangularSelected( center_x , center_y ,
                                                   right_x , right_y , left_x , left_y ,
                                                   clickScale_x , clickScale_y );

            break;

        case domain_annular:

            // Calls enableAnnularCorrelate in MainApp
            emit valueChanged_annularSelected( center_x , center_y ,
                                               inside_radius , outside_radius ,
                                               ri_by_ro , clickScale_x , clickScale_y );

            break;

        case domain_blob:

            // Calls enableBlobCorrelate in MainApp
            emit valueChanged_blobSelected( xy_blob , clickScale_x , clickScale_y );

            break;

        default:

            assert(false);
            break;
        }

        update();

        clickScale_x = realScale_x;
        clickScale_y = realScale_y;
    }
}

void ImageLabel::GUIupdated()
{
    // Command the Mirroring action
    switch (domain)
    {
        case domain_annular:

            // Calls mirrorSelectionAnnular on the def_imageLabel
            emit valueChanged_selectingAnnular(center_x, center_y,
                                            inside_radius, outside_radius, ri_by_ro,
                                            clickScale_x, clickScale_y);

            break;

        case domain_rectangular:

#if DEBUG_DOMAIN_SELECTION
        printf("%15s %30s: center x = %6.2f  center_y = %6.2f  left_x = %6.2f  right_x = %6.2f  left_y = %6.2f  right_y = %6.2f  scale_x = %6.2f  scale_y = %6.2f\n",
               "und_imageLabel","GUIupdated", center_x, center_y,
               left_x, right_x, left_y, right_y,
               clickScale_x, clickScale_y);
#endif

            // Calls mirrorSelectionRectangular on the def_imageLabel
            emit valueChanged_selectingRectangular(center_x, center_y, right_x, right_y,
                                                       left_x, left_y, clickScale_x, clickScale_y);

            break;

        case domain_blob:

            // Calls mirrorSelectionBlob on the def_imageLabel
            emit valueChanged_selectingBlob(center_x, center_y, xy_blob, clickScale_x, clickScale_y);

            break;

        default:

            assert(false);

            break;
    }

    update();
}

void ImageLabel::paintEvent( QPaintEvent *event )
{
    QLabel::paintEvent( event );

    if ( selecting || correlating )
    {
        QPainter painter( this );
        painter.setRenderHint(QPainter::Antialiasing);
        painter.setBackground(Qt::transparent);

        if ( selecting && !suppress_selection_display)
        {
            switch (domain)
            {
            case domain_rectangular:
                drawOutline_rectangular(painter);
                break;

            case domain_annular:
                drawOutline_annular(painter);
                break;

            case domain_blob:
                if( !xy_blob.empty() ) drawOutline_blob(painter);
                break;

            default:
                assert(false);
                break;
            }
        }

        if( correlating )
        {
            paint_inside_points(painter);
            paint_contour_points(painter);
        }
    }

    // After updating und_imageLabel, update def_imageLabel
    if (!mirroring) emit update_mirror();
}

void ImageLabel::applyModelRectangular()
{
    if(mirroring)
    {
        switch (model)
        {
            case fm_U:

                def_hi_left_x   = left_x   + model_parameters[0] * realScale_x;
                def_lo_right_x  = right_x  + model_parameters[0] * realScale_x;
                def_hi_left_y   = left_y;
                def_lo_right_y  = right_y;

                def_lo_left_x   = left_x   + model_parameters[0] * realScale_x;
                def_hi_right_x  = right_x  + model_parameters[0] * realScale_x;
                def_lo_left_y   = right_y;
                def_hi_right_y  = left_y;

                break;

            case fm_UV:

                def_hi_left_x   = left_x   + model_parameters[0] * realScale_x;
                def_lo_right_x  = right_x  + model_parameters[0] * realScale_x;
                def_hi_left_y   = left_y   + model_parameters[1] * realScale_y;
                def_lo_right_y  = right_y  + model_parameters[1] * realScale_y;

                def_lo_left_x   = left_x   + model_parameters[0] * realScale_x;
                def_hi_right_x  = right_x  + model_parameters[0] * realScale_x;
                def_lo_left_y   = right_y  + model_parameters[1] * realScale_y;
                def_hi_right_y  = left_y   + model_parameters[1] * realScale_y;

                break;

        case fm_UVQ:

            def_hi_left_x   = left_x   + model_parameters[0] * realScale_x + ( left_y  - center_y ) * model_parameters[2];
            def_lo_right_x  = right_x  + model_parameters[0] * realScale_x + ( right_y - center_y ) * model_parameters[2];
            def_hi_left_y   = left_y   + model_parameters[1] * realScale_y + ( left_x  - center_x ) * model_parameters[2];
            def_lo_right_y  = right_y  + model_parameters[1] * realScale_y + ( right_x - center_x ) * model_parameters[2];

            def_lo_left_x   = left_x   + model_parameters[0] * realScale_x + ( right_y - center_y ) * model_parameters[2];
            def_hi_right_x  = right_x  + model_parameters[0] * realScale_x + ( left_y  - center_y ) * model_parameters[2];
            def_lo_left_y   = right_y  + model_parameters[1] * realScale_y + ( left_x  - center_x ) * model_parameters[2];
            def_hi_right_y  = left_y   + model_parameters[1] * realScale_y + ( right_x - center_x ) * model_parameters[2];

            break;

        case fm_UVUxUyVxVy:

                def_hi_left_x   = left_x   + model_parameters[0] * realScale_x + ( left_x  - center_x ) * model_parameters[2] + ( left_y  - center_y ) * model_parameters[3];
                def_lo_right_x  = right_x  + model_parameters[0] * realScale_x + ( right_x - center_x ) * model_parameters[2] + ( right_y - center_y ) * model_parameters[3];
                def_hi_left_y   = left_y   + model_parameters[1] * realScale_y + ( left_x  - center_x ) * model_parameters[4] + ( left_y  - center_y ) * model_parameters[5];
                def_lo_right_y  = right_y  + model_parameters[1] * realScale_y + ( right_x - center_x ) * model_parameters[4] + ( right_y - center_y ) * model_parameters[5];

                def_lo_left_x   = left_x   + model_parameters[0] * realScale_x + ( left_x  - center_x ) * model_parameters[2] + ( right_y - center_y ) * model_parameters[3];
                def_hi_right_x  = right_x  + model_parameters[0] * realScale_x + ( right_x - center_x ) * model_parameters[2] + ( left_y  - center_y ) * model_parameters[3];
                def_lo_left_y   = right_y  + model_parameters[1] * realScale_y + ( left_x  - center_x ) * model_parameters[4] + ( right_y - center_y ) * model_parameters[5];
                def_hi_right_y  = left_y   + model_parameters[1] * realScale_y + ( right_x - center_x ) * model_parameters[4] + ( left_y  - center_y ) * model_parameters[5];

                break;

            default:

                assert(false);

                break;
        }

        def_center_x = center_x + model_parameters[0] * realScale_x;
        def_center_y = center_y + model_parameters[1] * realScale_y;
    }
    else
    {
        def_hi_left_x   = left_x;
        def_lo_right_x  = right_x;
        def_hi_left_y   = left_y;
        def_lo_right_y  = right_y;

        def_lo_left_x   = left_x;
        def_hi_right_x  = right_x;
        def_lo_left_y   = right_y;
        def_hi_right_y  = left_y;

        def_center_x    = center_x;
        def_center_y    = center_y;
    }

}

void ImageLabel::applyModelAnnular()
{
    if( mirroring )
    {
        switch (model)
        {
            case fm_U:

                def_center_x = center_x + model_parameters[0] * realScale_x;
                def_center_y = center_y;

                def_outside_radius = outside_radius;
                def_ri_by_ro = ri_by_ro;
                def_inside_radius  = def_outside_radius * def_ri_by_ro;

                def_q = 0;

                break;

            case fm_UV:

                def_center_x = center_x + model_parameters[0] * realScale_x;
                def_center_y = center_y + model_parameters[1] * realScale_y;

                def_outside_radius = outside_radius;
                def_ri_by_ro = ri_by_ro;
                def_inside_radius  = def_outside_radius * def_ri_by_ro;

                def_q = 0;

                break;

            case fm_UVQ:

                def_center_x = center_x + model_parameters[0] * realScale_x;
                def_center_y = center_y + model_parameters[1] * realScale_y;

                def_outside_radius = outside_radius;
                def_ri_by_ro = ri_by_ro;
                def_inside_radius  = def_outside_radius * def_ri_by_ro;

                def_q = model_parameters[2];

                break;

            case fm_UVUxUyVxVy:

                def_center_x = center_x + model_parameters[0] * realScale_x;
                def_center_y = center_y + model_parameters[1] * realScale_y;

                def_outside_radius = outside_radius;
                def_ri_by_ro = ri_by_ro;
                def_inside_radius  = def_outside_radius * def_ri_by_ro;

                def_q = best_rotation_UVUxUyVxVy( model_parameters );

                break;

            default:

                assert(false);

                break;
        }
    }
    else
    {
        def_center_x = center_x;
        def_center_y = center_y;

        def_outside_radius = outside_radius;
        def_ri_by_ro = ri_by_ro;
        def_inside_radius  = def_outside_radius * def_ri_by_ro;

        def_q = 0;
    }
}

void ImageLabel::applyModelBlob()
{
    if(mirroring)
    {
        switch (model)
        {
        case fm_U:

            def_u    = model_parameters[0] * realScale_x;
            def_v    = 0.f;
            def_dudx = 0.f;
            def_dudy = 0.f;
            def_dvdx = 0.f;
            def_dvdy = 0.f;

            // Define imaginary rectangular boundaries for def_imageLabel scaling
            // during user def ic
            def_hi_left_x   = left_x   + model_parameters[0] * realScale_x;
            def_lo_right_x  = right_x  + model_parameters[0] * realScale_x;
            def_hi_left_y   = left_y;
            def_lo_right_y  = right_y;

            def_lo_left_x   = left_x   + model_parameters[0] * realScale_x;
            def_hi_right_x  = right_x  + model_parameters[0] * realScale_x;
            def_lo_left_y   = right_y;
            def_hi_right_y  = left_y;

            break;

        case fm_UV:

            def_u    = model_parameters[0] * realScale_x;
            def_v    = model_parameters[1] * realScale_y;
            def_dudx = 0.f;
            def_dudy = 0.f;
            def_dvdx = 0.f;
            def_dvdy = 0.f;

            // Define imaginary rectangular boundaries for def_imageLabel scaling
            // during user def ic
            def_hi_left_x   = left_x   + model_parameters[0] * realScale_x;
            def_lo_right_x  = right_x  + model_parameters[0] * realScale_x;
            def_hi_left_y   = left_y   + model_parameters[1] * realScale_y;
            def_lo_right_y  = right_y  + model_parameters[1] * realScale_y;

            def_lo_left_x   = left_x   + model_parameters[0] * realScale_x;
            def_hi_right_x  = right_x  + model_parameters[0] * realScale_x;
            def_lo_left_y   = right_y  + model_parameters[1] * realScale_y;
            def_hi_right_y  = left_y   + model_parameters[1] * realScale_y;

            break;

        case fm_UVUxUyVxVy:

            def_u    = model_parameters[0] * realScale_x;
            def_v    = model_parameters[1] * realScale_y;
            def_dudx = model_parameters[2];
            def_dudy = model_parameters[3];
            def_dvdx = model_parameters[4];
            def_dvdy = model_parameters[5];

            // Define imaginary rectangular boundaries for def_imageLabel scaling
            // during user def ic
            def_hi_left_x   = left_x   + model_parameters[0] * realScale_x + ( left_x  - center_x ) * model_parameters[2] + ( left_y  - center_y ) * model_parameters[3];
            def_lo_right_x  = right_x  + model_parameters[0] * realScale_x + ( right_x - center_x ) * model_parameters[2] + ( right_y - center_y ) * model_parameters[3];
            def_hi_left_y   = left_y   + model_parameters[1] * realScale_y + ( left_x  - center_x ) * model_parameters[4] + ( left_y  - center_y ) * model_parameters[5];
            def_lo_right_y  = right_y  + model_parameters[1] * realScale_y + ( right_x - center_x ) * model_parameters[4] + ( right_y - center_y ) * model_parameters[5];

            def_lo_left_x   = left_x   + model_parameters[0] * realScale_x + ( left_x  - center_x ) * model_parameters[2] + ( right_y - center_y ) * model_parameters[3];
            def_hi_right_x  = right_x  + model_parameters[0] * realScale_x + ( right_x - center_x ) * model_parameters[2] + ( left_y  - center_y ) * model_parameters[3];
            def_lo_left_y   = right_y  + model_parameters[1] * realScale_y + ( left_x  - center_x ) * model_parameters[4] + ( right_y - center_y ) * model_parameters[5];
            def_hi_right_y  = left_y   + model_parameters[1] * realScale_y + ( right_x - center_x ) * model_parameters[4] + ( left_y  - center_y ) * model_parameters[5];

            break;

        default:

            assert(false);

            break;
    }

        def_center_x = center_x + def_u;
        def_center_y = center_y + def_v;
    }
    else
    {
        def_u    = 0.f;
        def_v    = 0.f;
        def_dudx = 0.f;
        def_dudy = 0.f;
        def_dvdx = 0.f;
        def_dvdy = 0.f;

        def_center_x = center_x;
        def_center_y = center_y;
    }
}

void ImageLabel::drawOutline_rectangular(QPainter & painter)
{
    applyModelRectangular();

    float dx   = ( def_hi_right_x - def_hi_left_x  ) / (float)h_subdivisions;
    float dy   = ( def_hi_right_y - def_lo_right_y ) / (float)v_subdivisions;
    float dvdx = ( def_hi_right_y - def_hi_left_y  ) / (float)h_subdivisions;
    float dudy = ( def_hi_right_x - def_lo_right_x ) / (float)v_subdivisions;

    QColor orange( 0xFF, 0xA0, 0x00 );
    QPen pen(orange);
    pen.setStyle(Qt::DashLine);
    painter.setPen(pen);

    for ( int i = 0; i <= h_subdivisions; i++)
    {
        painter.drawLine( def_hi_left_x + dx * i, def_hi_left_y + dvdx * i, def_lo_left_x + dx * i, def_lo_left_y + dvdx * i);
            for ( int j = 0; j <= v_subdivisions; j++)
                    painter.drawLine(def_lo_left_x + dudy * j, def_lo_left_y + dy * j, def_lo_right_x + dudy * j, def_lo_right_y + dy * j);
    }
    painter.drawPoint(def_center_x,def_center_y);
}

void ImageLabel::drawOutline_annular(QPainter & painter)
{
    applyModelAnnular();

    float dr = (def_outside_radius - def_inside_radius) / (float)r_subdivisions;
    float da = 2.f * PI / (float)a_subdivisions;

    QColor orange( 0xFF, 0xA0, 0x00 );
    QPen pen(orange);
    pen.setStyle(Qt::DashLine);
    painter.setPen(pen);

    for ( int ri = 0; ri <= r_subdivisions; ++ri)
    {
        float r = 2 * (def_inside_radius + ri * dr);
        painter.drawEllipse(
                    def_center_x - r/2,
                    def_center_y - (r * realScale_y / realScale_x)/2,
                    r ,
                    r * realScale_y / realScale_x
                    );
        for ( int ai = 1; ai < a_subdivisions; ++ai)
                painter.drawLine(
                            def_center_x + def_inside_radius  * cos(def_q + ai * da ) ,
                            def_center_y + def_inside_radius  * sin(def_q + ai * da ) * realScale_y / realScale_x,
                            def_center_x + def_outside_radius * cos(def_q + ai * da ) ,
                            def_center_y + def_outside_radius * sin(def_q + ai * da ) * realScale_y / realScale_x
                            );
    }
    painter.drawLine(def_center_x + def_inside_radius        * cos(def_q),
                     def_center_y + def_inside_radius        * sin(def_q) * realScale_y / realScale_x,
                     def_center_x + def_outside_radius * 1.1 * cos(def_q),
                     def_center_y + def_outside_radius * 1.1 * sin(def_q) * realScale_y / realScale_x );

    painter.drawPoint(def_center_x,def_center_y);
}

void ImageLabel::drawOutline_blob(QPainter & painter)
{
    applyModelBlob();

    QColor orange( 0xFF, 0xA0, 0x00 );
    QPen pen(orange);
    pen.setStyle(Qt::DashLine);
    painter.setPen(pen);

    int size = (int)xy_blob.size();

    if (size)
    {
        for ( int i = 1; i < size; ++i )
        {
            float x0 = xy_blob[i-1].first;
            float y0 = xy_blob[i-1].second;
            float x1 = xy_blob[i  ].first;
            float y1 = xy_blob[i  ].second;

            painter.drawLine(   x0 + def_u + def_dudx * (x0 - center_x) + def_dudy * (y0 - center_y),
                                y0 + def_v + def_dvdx * (x0 - center_x) + def_dvdy * (y0 - center_y),
                                x1 + def_u + def_dudx * (x1 - center_x) + def_dudy * (y1 - center_y),
                                y1 + def_v + def_dvdx * (x1 - center_x) + def_dvdy * (y1 - center_y) );
        }

        float x0 = xy_blob[size-1].first;
        float y0 = xy_blob[size-1].second;
        float x1 = xy_blob[0     ].first;
        float y1 = xy_blob[0     ].second;

        painter.drawLine(   x0 + def_u + def_dudx * (x0 - center_x) + def_dudy * (y0 - center_y),
                            y0 + def_v + def_dvdx * (x0 - center_x) + def_dvdy * (y0 - center_y),
                            x1 + def_u + def_dudx * (x1 - center_x) + def_dudy * (y1 - center_y),
                            y1 + def_v + def_dvdx * (x1 - center_x) + def_dvdy * (y1 - center_y) );

        painter.drawPoint(def_center_x,def_center_y);
    }
}

void ImageLabel::paint_inside_points(QPainter & painter)
{
    QColor transparent_red       ( 0xFF, 0x00, 0x00, std::min(255, (int)(50.f * realScale_x * 2.f ) ) );
    QColor transparent_green     ( 0x00, 0xFF, 0x00, std::min(255, (int)(50.f * realScale_x * 2.f ) ) );
    QColor transparent_red_past  ( 0xFF, 0x00, 0x00, std::min(128, (int)(25.f * realScale_x * 2.f ) ) );
    QColor transparent_green_past( 0x00, 0xFF, 0x00, std::min(128, (int)(25.f * realScale_x * 2.f ) ) );

    QPen pen;

    int n_points = (int)inside_points.size();
    if (n_points)
    {
        for ( int p = 0; p < n_points; ++p)
        {
            if( p == n_points - 1 )
            {
                if ( inside_points_error[p] )
                    pen.setColor( transparent_red );
                else
                    pen.setColor( transparent_green );
            }
            else
            {
                if ( inside_points_error[p] )
                    pen.setColor( transparent_red_past );
                else
                    pen.setColor( transparent_green_past );
            }

            painter.setPen(pen);

            int size = (int)inside_points[p].size();
            if (size)
               for ( int i = 1; i < size; ++i )
                   painter.drawPoint( inside_points[p][i].first, inside_points[p][i].second);
        }
    }
}

void ImageLabel::paint_contour_points(QPainter & painter)
{
    QPen pen;

    QColor transparent_red  ( 0xFF, 0x00, 0x00, 170);//std::min(255, (int)(50.f * realScale_x * 2.f ) ) );
    QColor transparent_green( 0x00, 0xFF, 0x00, 170);//std::min(255, (int)(50.f * realScale_x * 2.f ) ) );

    int n_contours =(int)contour_points.size();
    if( n_contours )
    {
        for ( int c = 0; c < n_contours; ++c)
        {
            if( c == n_contours - 1 )
            {
                if ( contour_points_error[c] )
                    pen.setColor( Qt::red );
                else
                    pen.setColor( Qt::green );
            }
            else
            {
                if ( contour_points_error[c] )
                    pen.setColor( transparent_red );
                else
                    pen.setColor( transparent_green );
            }

            painter.setPen(pen);

            int size = (int)contour_points[c].size();
            if (size)
            {
                for ( int i = 0; i < size - 1; ++i )
                painter.drawLine( contour_points[c][i       ].first, contour_points[c][i       ].second,
                                  contour_points[c][i    + 1].first, contour_points[c][i    + 1].second);

                painter.drawLine( contour_points[c][size - 1].first, contour_points[c][size - 1].second,
                                  contour_points[c][0       ].first, contour_points[c][0       ].second);
            }
        }
    }

}

void ImageLabel::set_inside_points(v_points inside_points_vector_in, bool error_in)
{
   error = error_in;

   v_points temp;

   for(unsigned int i = 0; i < inside_points_vector_in.size(); ++i )
   {
       temp.push_back( std::make_pair( inside_points_vector_in[i].first * realScale_x , inside_points_vector_in[i].second * realScale_y )  );
   }

   inside_points.push_back(temp);
   inside_points_error.push_back(error_in);

   update();
}

void ImageLabel::set_contour_points(v_points contour_points_vector_in, bool error_in)
{
   error = error_in;

   v_points temp;

   for(unsigned int i = 0; i < contour_points_vector_in.size(); ++i )
   {
       temp.push_back( std::make_pair( contour_points_vector_in[i].first * realScale_x , contour_points_vector_in[i].second * realScale_y )  );
   }

   contour_points.push_back(temp);
   contour_points_error.push_back(error_in);

   update();
}

void ImageLabel::scaleSquare()
{
    left_x  = left_x  * realScale_x / clickScale_x;
    left_y  = left_y  * realScale_y / clickScale_y;
    right_x = right_x * realScale_x / clickScale_x;
    right_y = right_y * realScale_y / clickScale_y;

    center_x *= realScale_x / clickScale_x;
    center_y *= realScale_y / clickScale_y;
    inside_radius *= realScale_x / clickScale_x;
    outside_radius *= realScale_x / clickScale_x;

    unsigned int xy_blob_size = xy_blob.size();

    if (xy_blob_size)
    {
        for ( unsigned int i = 0; i < xy_blob_size; ++i )
        {
            xy_blob[i].first  *= realScale_x / clickScale_x;
            xy_blob[i].second *= realScale_y / clickScale_y;
        }
    }

    int n_contours = (int)contour_points.size();
    if( n_contours )
    {
        for ( int c = 0; c < n_contours; ++c)
        {
            int contour_size = (int)contour_points[c].size();

            if (contour_size)
            {
                for ( int i = 0; i < contour_size; ++i )
                {
                    contour_points[c][i].first  *= realScale_x / clickScale_x;
                    contour_points[c][i].second *= realScale_y / clickScale_y;
                }
            }
        }
    }

    int n_points = (int)inside_points.size();
    if( n_points )
    {
        for ( int p = 0; p < n_points; ++p)
        {
            int points_size = (int)inside_points[p].size();

            if (points_size)
            {
                for ( int i = 0; i < points_size; ++i )
                {
                    inside_points[p][i].first  *= realScale_x / clickScale_x;
                    inside_points[p][i].second *= realScale_y / clickScale_y;
                }
            }
        }
    }

    clickScale_x = realScale_x;
    clickScale_y = realScale_y;

    update();
}

// Come from emit valueChanged_selecting();
void ImageLabel::mirrorSelectionRectangular(float center_x_in, float center_y_in, float right_x_in, float right_y_in,
                                 float left_x_in, float left_y_in, float clickScale_x_in, float clickScale_y_in)
{
    left_x  = left_x_in;
    left_y  = left_y_in;
    right_x = right_x_in;
    right_y = right_y_in;

    center_x = center_x_in;
    center_y = center_y_in;

    clickScale_x = clickScale_x_in;
    clickScale_y = clickScale_y_in;

    def_center_x = center_x;
    def_center_y = center_y;

#if DEBUG_DOMAIN_SELECTION
//    printf("%15s %30s: center x = %6.2f  center_y = %6.2f  left_x = %6.2f  right_x = %6.2f  left_y = %6.2f  right_y = %6.2f  scale_x = %6.2f  scale_y = %6.2f\n",
//           "def_imageLabel","mirrorSelection", center_x, center_y,
//           left_x, right_x, left_y, right_y,
//           clickScale_x, clickScale_y);
#endif

    update();
}

// Comes from emit valueChanged_selecting();
void ImageLabel::mirrorSelectionAnnular(float center_x_in, float center_y_in,
                                 float r_inside_in, float r_outside_in, float ri_by_ro_in,
                                 float clickScale_x_in, float clickScale_y_in)
{
    center_x       = center_x_in;
    center_y       = center_y_in;
    inside_radius  = r_inside_in;
    outside_radius = r_outside_in;
    ri_by_ro       = ri_by_ro_in;

    clickScale_x = clickScale_x_in;
    clickScale_y = clickScale_y_in;

    def_center_x = center_x;
    def_center_y = center_y;

    update();
}

// Comes from emit valueChanged_selecting();
void ImageLabel::mirrorSelectionBlob(float center_x_in, float center_y_in, v_points xy_blob_in,
                                     float clickScale_x_in, float clickScale_y_in)
{
    center_x     = center_x_in;
    center_y     = center_y_in;

    clickScale_x = clickScale_x_in;
    clickScale_y = clickScale_y_in;

    def_center_x = center_x;
    def_center_y = center_y;

    xy_blob = xy_blob_in;

    // To enable a click zone to user-select centration
    if(!xy_blob.empty())
    {
        left_x  = xy_blob[0].first;
        right_x = xy_blob[0].first;
        left_y  = xy_blob[0].second;
        right_y = xy_blob[0].second;

        for ( int i = 1; i < (int)xy_blob.size(); ++i )
        {
            left_x  = std::min(left_x , xy_blob[i].first);
            right_x = std::max(right_x, xy_blob[i].first);
            left_y  = std::min(left_y , xy_blob[i].second);
            right_y = std::max(right_y, xy_blob[i].second);
        }
    }

    update();
}

void ImageLabel::clear_inside_points()
{
    inside_points.clear();
    inside_points_error.clear();
}

void ImageLabel::clear_contour_points()
{
    contour_points.clear();
    contour_points_error.clear();
}

