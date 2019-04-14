#ifndef MODEL_CLASS_H
#define MODEL_CLASS_H

//local includes
#include "defines.hpp"
#include "enums.hpp"

//timing
#include <chrono>
#include <iostream>
#include <math.h>
#include <assert.h>

class ModelClass
{
protected:
	int thread_id;
	bool error_status;
	errorEnum error_code;

	fittingModelEnum fittingModel;	
	int number_of_model_parameters;
    float *model_parameters        = nullptr;

    int number_of_points;

    float *xy_positions            = nullptr;
    float *def_xy_positions        = nullptr;
    float *dTxydp                  = nullptr;

    float und_x_center;
    float def_x_center;
    float und_y_center;
    float def_y_center;

    //timing variables
    float time_duration_model;

    void debug_model();

public:		
	ModelClass();
    ~ModelClass();

	int get_thread_id();
    float get_time();
    void set_thread_id( int thread_id_in );
    void set_und_x_center( float und_x_center_in );
    void set_und_y_center( float und_y_center_in );

    void set_points(            int    number_of_points_in,
                                float *xy_positions_in,
                                float *def_xy_positions_in,
                                float *model_parameters_in,
                                float  und_x_center_in,
                                float  und_y_center_in,
                                float *dTxydp_in );

    virtual void compute_model() = 0;

	bool get_error_status();
    void set_error_status( bool error_status_in );
	errorEnum get_error_code();
    void set_error_code( errorEnum error_code_in );

    static int get_number_of_model_parameters( fittingModelEnum fittingModel_in );

    static ModelClass* new_ModelClass( const fittingModelEnum &fittingModel_in,
                                       const int number_of_threads );
};

class ModelClass_U : public ModelClass
{
private:
	void compute_model();

public:
	ModelClass_U();
};

class ModelClass_UV : public ModelClass
{
private:
	void compute_model();

public:
	ModelClass_UV();
};

class ModelClass_UVQ : public ModelClass
{
private:
    void compute_model();

public:
    ModelClass_UVQ();
};

class ModelClass_UVUxUyVxVy : public ModelClass
{
private:
	void compute_model();

public:
	ModelClass_UVUxUyVxVy();
};

#endif


