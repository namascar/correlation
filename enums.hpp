#ifndef ENUMS_HPP
#define ENUMS_HPP


//----------------------------------------------------------------------
//
//   enums
//
//----------------------------------------------------------------------

enum interpolationModelEnum {   im_nearest,
                                im_bilinear,
                                im_bicubic,
                                im_NUMBER_OF_ITEMS };

enum fittingModelEnum {         fm_U,
                                fm_UV,
                                fm_UVQ,
                                fm_UVUxUyVxVy,
                                fm_NUMBER_OF_ITEMS };

enum errorEnum {                error_none,
                                error_model_out_of_image,
                                error_interpolation_out_of_image,
                                error_correlation_max_iters_reached,
                                error_bad_domain,
                                error_cuSolver,
                                error_cuda,
                                error_multiThread,
                                error_NUMBER_OF_ITEMS };

enum colorEnum {                color_monochrome,
                                color_color,
                                color_NUMBER_OF_ITEMS };

enum updateEnum{                update_forward,
                                update_backward,
                                update_NUMBER_OF_ITEMS };

enum initialGuessEnum {         ic_Null,
                                ic_Auto,
                                ic_User,
                                ic_NUMBER_OF_ITEMS };

enum domainEnum {               domain_rectangular,
                                domain_annular,
                                domain_blob,
                                domain_NUMBER_OF_ITEMS };

enum intersectionEnum {         intersection_no,
                                intersection_yes,
                                intersection_colliear,
                                intersetion_NUMBER_OF_ITEMS };

enum UVUxUyVxVyInitialGuess {   UVUxUyVxVyIC_Center,
                                UVUxUyVxVyIC_DU,
                                UVUxUyVxVyIC_DV,
                                UVUxUyVxVyIC_Q,
                                UVUxUyVxVyIC_NUMBER_OF_ITEMS };

enum annularInitialGuess {      annularIC_Q,
                                annularIC_Center,
                                annularIC_ri,
                                annularIC_ro,
                                annularIC_NUMBER_OF_ITEMS };

enum deformationDescriptionEnum {def_strict_Lagrangian,
                                 def_Lagrangian,
                                 def_Eulerian,                                 
                                 def_NUMBER_OF_ITEMS };

enum errorHandlingModeEnum {    errorMode_stopAll,
                                errorMode_stopFrame,
                                errorMode_continue,
                                errorMode_NUMBER_OF_ITEMS };

enum referenceImageEnum {       refImage_First,
                                refImage_Previous,
                                refImage_NUMBER_OF_ITEMS };

enum processorEnum      {       processor_CPU,
                                processor_GPU,
                                processor_NUMBER_OF_ITEMS };

enum ImageType          {       imageType_und,
                                imageType_def,
                                imageType_nxt,
                                imageType_NUMBER_OF_ITEMS };

enum parameterTypeEnum {        parType_tentative,
                                parType_lastGood,
                                parType_saved,
                                parType_NUMBER_OF_ITEMS };

#endif // ENUMS_HPP
