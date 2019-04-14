#include "model_class.hpp"


ModelClass::ModelClass()
{
    error_status = false;
    number_of_points = 0;
}

ModelClass_U::ModelClass_U()
{
    fittingModel = fm_U;
    number_of_model_parameters = get_number_of_model_parameters(fittingModel);
}

ModelClass_UV::ModelClass_UV()
{
    fittingModel = fm_UV;
    number_of_model_parameters = get_number_of_model_parameters(fittingModel);
}

ModelClass_UVUxUyVxVy::ModelClass_UVUxUyVxVy()
{
    fittingModel = fm_UVUxUyVxVy;
    number_of_model_parameters = get_number_of_model_parameters(fittingModel);
}

ModelClass_UVQ::ModelClass_UVQ()
{
    fittingModel = fm_UVQ;
    number_of_model_parameters = get_number_of_model_parameters(fittingModel);
}

ModelClass::~ModelClass(){}

void ModelClass::set_points(
                                int    number_of_points_in,
                                float *xy_positions_in,
                                float *def_xy_positions_in,
                                float *model_parameters_in,
                                float  und_x_center_in,
                                float  und_y_center_in,
                                float *dTxydp_in )
{
    error_status = false;
    error_code = error_none;

    number_of_points = number_of_points_in;
    xy_positions = xy_positions_in;
    def_xy_positions = def_xy_positions_in;
    model_parameters = model_parameters_in;
    und_x_center = und_x_center_in;
    und_y_center = und_y_center_in;
    dTxydp = dTxydp_in;
}

void ModelClass::compute_model(){}

void ModelClass_U::compute_model()
{
    auto start_time = std::chrono::system_clock::now();

    float u  = model_parameters[ 0 ];

    for ( int ipoint = 0; ipoint < number_of_points; ++ipoint )
    {
        float x = xy_positions[ ipoint * 2 + 0 ];
        float y = xy_positions[ ipoint * 2 + 1 ];

        def_xy_positions[ ipoint * 2 + 0 ] =  x + u;
        def_xy_positions[ ipoint * 2 + 1 ] =  y;

        dTxydp[ipoint * ( number_of_model_parameters * 2 )                              + 0 ] = 1;
        dTxydp[ipoint * ( number_of_model_parameters * 2 ) + number_of_model_parameters + 0 ] = 0;
    }

    auto end_time = std::chrono::system_clock::now();
    time_duration_model = (float)std::chrono::duration_cast<std::chrono::microseconds>( end_time - start_time ).count() / 1000000.f;

    debug_model();
}

void ModelClass_UV::compute_model()
{
    auto start_time = std::chrono::system_clock::now();

    float u  = model_parameters[ 0 ];
    float v  = model_parameters[ 1 ];

    for ( int ipoint = 0; ipoint < number_of_points; ++ipoint )
    {
        float x = xy_positions[ ipoint * 2 + 0 ];
        float y = xy_positions[ ipoint * 2 + 1 ];

        def_xy_positions[ ipoint * 2 + 0 ] =  x + u;
        def_xy_positions[ ipoint * 2 + 1 ] =  y + v;

        dTxydp [ ipoint * ( number_of_model_parameters * 2 )                              + 0 ] = 1;
        dTxydp [ ipoint * ( number_of_model_parameters * 2 )                              + 1 ] = 0;

        dTxydp [ ipoint * ( number_of_model_parameters * 2 ) + number_of_model_parameters + 0 ] = 0;
        dTxydp [ ipoint * ( number_of_model_parameters * 2 ) + number_of_model_parameters + 1 ] = 1;
    }

    auto end_time = std::chrono::system_clock::now();
    time_duration_model = (float)std::chrono::duration_cast<std::chrono::microseconds>( end_time - start_time ).count() / 1000000.f;

    debug_model();
}

void ModelClass_UVQ::compute_model()
{
    auto start_time = std::chrono::system_clock::now();

    float u  =   model_parameters[ 0 ];
    float v  =   model_parameters[ 1 ];

    float vx =   model_parameters[ 2 ];


    def_x_center = und_x_center + u;
    def_y_center = und_y_center + v;

    for ( int ipoint = 0; ipoint < number_of_points; ++ipoint )
    {
        float x = xy_positions[ ipoint * 2 + 0 ];
        float y = xy_positions[ ipoint * 2 + 1 ];

        float dx = x - und_x_center;
        float dy = y - und_y_center;

        def_xy_positions[ ipoint * 2 + 0 ] =  x + u   -             vx * dy;
        def_xy_positions[ ipoint * 2 + 1 ] =  y + v   +   vx * dx          ;

        dTxydp [ ipoint * ( number_of_model_parameters * 2 )                              + 0] = 1;
        dTxydp [ ipoint * ( number_of_model_parameters * 2 )                              + 1] = 0;
        dTxydp [ ipoint * ( number_of_model_parameters * 2 )                              + 2] = - dy;


        dTxydp [ ipoint * ( number_of_model_parameters * 2 ) + number_of_model_parameters + 0] = 0;
        dTxydp [ ipoint * ( number_of_model_parameters * 2 ) + number_of_model_parameters + 1] = 1;
        dTxydp [ ipoint * ( number_of_model_parameters * 2 ) + number_of_model_parameters + 2] = dx;

    }

    auto end_time = std::chrono::system_clock::now();
    time_duration_model = (float)std::chrono::duration_cast<std::chrono::microseconds>( end_time - start_time ).count() / 1000000.f;

   debug_model();
}

void ModelClass_UVUxUyVxVy::compute_model()
{
    auto start_time = std::chrono::system_clock::now();

    float u  = model_parameters[ 0 ];
    float v  = model_parameters[ 1 ];
    float ux = model_parameters[ 2 ];
    float uy = model_parameters[ 3 ];
    float vx = model_parameters[ 4 ];
    float vy = model_parameters[ 5 ];

    def_x_center = und_x_center + u;
    def_y_center = und_y_center + v;

    for ( int ipoint = 0; ipoint < number_of_points; ++ipoint )
    {
        float x = xy_positions[ ipoint * 2 + 0 ];
        float y = xy_positions[ ipoint * 2 + 1 ];

        float dx = x - und_x_center;
        float dy = y - und_y_center;

        def_xy_positions[ ipoint * 2 + 0 ] =  x + u   +   ux * dx + uy * dy;
        def_xy_positions[ ipoint * 2 + 1 ] =  y + v   +   vx * dx + vy * dy;

        dTxydp [ ipoint * ( number_of_model_parameters * 2 )                              + 0 ] = 1;
        dTxydp [ ipoint * ( number_of_model_parameters * 2 )                              + 1 ] = 0;
        dTxydp [ ipoint * ( number_of_model_parameters * 2 )                              + 2 ] = dx;
        dTxydp [ ipoint * ( number_of_model_parameters * 2 )                              + 3 ] = dy;
        dTxydp [ ipoint * ( number_of_model_parameters * 2 )                              + 4 ] = 0;
        dTxydp [ ipoint * ( number_of_model_parameters * 2 )                              + 5 ] = 0;

        dTxydp [ ipoint * ( number_of_model_parameters * 2 ) + number_of_model_parameters + 0 ] = 0;
        dTxydp [ ipoint * ( number_of_model_parameters * 2 ) + number_of_model_parameters + 1 ] = 1;
        dTxydp [ ipoint * ( number_of_model_parameters * 2 ) + number_of_model_parameters + 2 ] = 0;
        dTxydp [ ipoint * ( number_of_model_parameters * 2 ) + number_of_model_parameters + 3 ] = 0;
        dTxydp [ ipoint * ( number_of_model_parameters * 2 ) + number_of_model_parameters + 4 ] = dx;
        dTxydp [ ipoint * ( number_of_model_parameters * 2 ) + number_of_model_parameters + 5 ] = dy;
    }

    auto end_time = std::chrono::system_clock::now();
    time_duration_model = (float)std::chrono::duration_cast<std::chrono::microseconds>( end_time - start_time ).count() / 1000000.f;

    debug_model();
}

bool ModelClass::get_error_status()
{
    return error_status;
}

void ModelClass::set_error_status( bool error_status_in )
{ 
    error_status = error_status_in;
}

errorEnum ModelClass::get_error_code()
{
    return error_code;
}

void ModelClass::set_error_code( errorEnum error_code_in )
{
    error_code = error_code_in;
}

int ModelClass::get_number_of_model_parameters( fittingModelEnum fittingModel_in )
{
    switch(fittingModel_in)
    {
        case fm_U:          return 1;
        case fm_UV:         return 2;
        case fm_UVQ:        return 3;
        case fm_UVUxUyVxVy: return 6;
        default: assert( false );
    }
    return -1;
}

ModelClass* ModelClass::new_ModelClass( const fittingModelEnum &fittingModel_in,
                                         const int number_of_threads )
{
    switch (fittingModel_in)
    {
        case fm_U:
        {
            #if DEBUG_MODEL
                std::cout << "+++++++++++++++++++++" << std::endl;
                std::cout << "Model U" << std::endl;
                std::cout << "+++++++++++++++++++++" << std::endl;
            #endif

            ModelClass_U *U = new ModelClass_U[ number_of_threads ];

            return U;
        }

        case fm_UV: 
        {
            #if DEBUG_MODEL
                std::cout << "+++++++++++++++++++++" << std::endl;
                std::cout << "Model UV" << std::endl;
                std::cout << "+++++++++++++++++++++" << std::endl;
            #endif

            ModelClass_UV *UV = new ModelClass_UV[ number_of_threads ];

            return UV;
        }

        case fm_UVQ:
        {
            #if DEBUG_MODEL
                std::cout << "+++++++++++++++++++++" << std::endl;
                std::cout << "Model UVQ" << std::endl;
                std::cout << "+++++++++++++++++++++" << std::endl;
            #endif

            ModelClass_UVQ *UVQ = new ModelClass_UVQ[ number_of_threads ];

            return UVQ;
        }

        case fm_UVUxUyVxVy:  
        {
            #if DEBUG_MODEL
                std::cout << "+++++++++++++++++++++" << std::endl;
                std::cout << "Model UVUxUyVxVy" << std::endl; 
                std::cout << "+++++++++++++++++++++" << std::endl;
            #endif

            ModelClass_UVUxUyVxVy *UVUxUyVxVy = new ModelClass_UVUxUyVxVy[ number_of_threads ];

            return UVUxUyVxVy;
        }

        default: 
            assert(false);

        return nullptr;

    }
}
int ModelClass::get_thread_id()
{
    return thread_id;
}

float ModelClass::get_time()
{
    return time_duration_model;
}

void ModelClass::set_thread_id( int thread_id_in )
{
    thread_id = thread_id_in;
}

void ModelClass::set_und_x_center( float und_x_center_in )
{
    und_x_center = und_x_center_in;
}

void ModelClass::set_und_y_center( float und_y_center_in )
{
    und_y_center = und_y_center_in;
}

void ModelClass::debug_model()
{
    #if DEBUG_MODEL

        std::cout << "Model Results " << std::endl;
        std::cout << "und xy points for thread " << thread_id << " #points = " << number_of_points <<  std::endl;
        std::cout << "und_x0 = " << und_x_center << " und_y0 = " << und_y_center <<  std::endl;
        std::cout << "def_x0 = " << def_x_center << " def_y0 = " << def_y_center <<  std::endl;

        std::cout << "Model Parameters = ";
        for ( int p = 0; p < number_of_model_parameters; ++p )
            std::cout << model_parameters[p] << " ";
        std::cout << std::endl;

        for ( int ipoint = 0; ipoint < number_of_points; ++ipoint )
        {
            std::cout << "point " << ipoint <<
                ", und_xy (" << xy_positions[ipoint * 2 + 0] <<
                "," << xy_positions[ipoint * 2 + 1] << ")" <<
                ", def_xy (" << def_xy_positions[ipoint * 2 + 0] <<
                 "," << def_xy_positions[ipoint * 2 + 1] << ")" << ", dTxdp = ";

            for ( int p = 0; p < number_of_model_parameters; ++p )
                std::cout << dTxydp [ ipoint * (number_of_model_parameters * 2) + p] << " ";
            std::cout << ", dTydp = ";
            for ( int p = 0; p < number_of_model_parameters; ++p )
                std::cout << dTxydp [ ipoint * (number_of_model_parameters * 2) + number_of_model_parameters + p] << " ";
            std::cout << std::endl;
        }

        std::cout << std::endl;

    #endif

#if DEBUG_MODEL_INPUTS

    std::cout << "Model Type " << fittingModel <<  std::endl;
    std::cout << "und xy points for thread " << thread_id << " #points = " << number_of_points <<  std::endl;
    std::cout << "und_x0 = " << und_x_center << " und_y0 = " << und_y_center <<  std::endl;
    std::cout << "def_x0 = " << def_x_center << " def_y0 = " << def_y_center <<  std::endl;

    std::cout << "Model Parameters = ";
    for ( int p = 0; p < number_of_model_parameters; ++p )
        std::cout << model_parameters[p] << " ";
    std::cout << std::endl;

    std::cout << std::endl;

#endif
}
