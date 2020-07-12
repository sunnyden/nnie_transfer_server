//
// Created by sunny on 19/1/2020.
//

#ifndef HISILICON_NET_H
#define HISILICON_NET_H
/*16Byte align*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <map>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <signal.h>
#include <mpi_sys.h>
#include <mpi_nnie.h>
#include <hi_comm_svp.h>
#include <hi_nnie.h>
#include <vector>
#include "mat.h"


#define NNIE_ALIGN_16 16
#define NNIE_ALIGN16(u32Num) ((u32Num + NNIE_ALIGN_16-1) / NNIE_ALIGN_16*NNIE_ALIGN_16)
/*32Byte align*/
#define NNIE_ALIGN_32 32
#define NNIE_ALIGN32(u32Num) ((u32Num + NNIE_ALIGN_32-1) / NNIE_ALIGN_32*NNIE_ALIGN_32)

#define NNIE_CONVERT_64BIT_ADDR(Type,Addr) (Type*)(HI_UL)(Addr)
#define COORDI_NUM                     4        /*num of coordinates*/
#define PROPOSAL_WIDTH                 6        /*the width of each proposal array*/
#define QUANT_BASE                     4096     /*the basic quantity*/
#define NNIE_MAX_SOFTWARE_MEM_NUM      4
#define NNIE_SSD_REPORT_NODE_NUM       12
#define NNIE_SSD_PRIORBOX_NUM          6
#define NNIE_SSD_SOFTMAX_NUM           6
#define NNIE_SSD_ASPECT_RATIO_NUM      6
#define NNIE_YOLOV1_WIDTH_GRID_NUM     7
#define NNIE_YOLOV1_HEIGHT_GRID_NUM    7
#define NNIE_EACH_SEG_STEP_ADDR_NUM    2
#define NNIE_MAX_CLASS_NUM             30
#define NNIE_MAX_ROI_NUM_OF_CLASS      50
#define NNIE_REPORT_NAME_LENGTH        64

typedef struct BoxInfo {
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int label;
    float prob;
    bool track = false;
    float *landmarks;
    float distance;
} BoxInfo;

typedef struct hiNNIE_MODEL_S
{
    SVP_NNIE_MODEL_S    stModel;
    SVP_MEM_INFO_S      stModelBuf;//store Model file
}NNIE_MODEL_S;


/*each seg input and output memory*/
typedef struct hiNNIE_SEG_DATA_S
{
    SVP_SRC_BLOB_S astSrc[SVP_NNIE_MAX_INPUT_NUM];
    SVP_DST_BLOB_S astDst[SVP_NNIE_MAX_OUTPUT_NUM];
}NNIE_SEG_DATA_S;

/*each seg input and output data memory size*/
typedef struct hiNNIE_BLOB_SIZE_S
{
    HI_U32 au32SrcSize[SVP_NNIE_MAX_INPUT_NUM];
    HI_U32 au32DstSize[SVP_NNIE_MAX_OUTPUT_NUM];
}NNIE_BLOB_SIZE_S;

/*NNIE Execution parameters */
typedef struct hiNNIE_PARAM_S
{
    SVP_NNIE_MODEL_S*    pstModel;
    HI_U32 u32TmpBufSize;
    HI_U32 au32TaskBufSize[SVP_NNIE_MAX_NET_SEG_NUM];
    SVP_MEM_INFO_S      stTaskBuf;
    SVP_MEM_INFO_S      stTmpBuf;
    SVP_MEM_INFO_S      stStepBuf;//store Lstm step info
    NNIE_SEG_DATA_S astSegData[SVP_NNIE_MAX_NET_SEG_NUM];//each seg's input and output blob
    SVP_NNIE_FORWARD_CTRL_S astForwardCtrl[SVP_NNIE_MAX_NET_SEG_NUM];
    SVP_NNIE_FORWARD_WITHBBOX_CTRL_S astForwardWithBboxCtrl[SVP_NNIE_MAX_NET_SEG_NUM];
}NNIE_PARAM_S;

/*NNIE input or output data index*/
typedef struct hiNNIE_DATA_INDEX_S
{
    HI_U32 u32SegIdx;
    HI_U32 u32NodeIdx;
}NNIE_DATA_INDEX_S;

/*this struct is used to indicate the input data from which seg's input or report node*/
typedef NNIE_DATA_INDEX_S  NNIE_INPUT_DATA_INDEX_S;
/*this struct is used to indicate which seg will be executed*/
typedef NNIE_DATA_INDEX_S  NNIE_PROCESS_SEG_INDEX_S;

typedef enum hiNNIE_NET_TYPE_E
{
    NNIE_ALEXNET_FASTER_RCNN       =  0x0,  /*FasterRcnn Alexnet*/
    NNIE_VGG16_FASTER_RCNN         =  0x1,  /*FasterRcnn Vgg16*/
    NNIE_PVANET_FASTER_RCNN        =  0x2, /*pavenet fasterRcnn*/

    NNIE_NET_TYPE_BUTT
}NNIE_NET_TYPE_E;


/*NNIE configuration parameter*/
typedef struct hiNNIE_CFG_S
{
    HI_CHAR *pszPic;
    HI_U32 u32MaxInputNum;
    HI_U32 u32MaxRoiNum;
    HI_U64 au64StepVirAddr[NNIE_EACH_SEG_STEP_ADDR_NUM*SVP_NNIE_MAX_NET_SEG_NUM];//virtual addr of LSTM's or RNN's step buffer
    SVP_NNIE_ID_E	aenNnieCoreId[SVP_NNIE_MAX_NET_SEG_NUM];
}NNIE_CFG_S;

/**
 * Implementation notice:
 * The stride is how many bytes is necessary for a row.
 * The value of a stride seems to has a minimum of 16.
 * For input, the stride is commonly equal to the width, since each pixel only consume a byte to store.
 * For output, when operating the pointer, keep in mind to divide the size of "int"(or something else if possible),
 * since the output data is commonly consuming 4 byte signed int for a single unit. (You also need to divide 4096 to
 * convert it into the final float result)
 */
typedef struct hiNNIE_LAYER_S{
    int width;
    int height;
    int channel;
    int segment_id;
    int layer_id;
    SVP_BLOB_TYPE_E data_type;
    HI_U32 *u32Stride;           /*Stride, a line bytes num*/
    HI_U64 *u64VirAddr;          /*virtual addr*/
    HI_U64 *u64PhyAddr;          /*physical addr*/
}NNIE_LAYER_S;

typedef struct{
    int id;
    int segment_id;
    int layer_id;
    int width;
    int height;
    int channel;
    int is_input;
    char name[SVP_NNIE_NODE_NAME_LEN] = {0};
}NNIE_LAYER_EXPORT;

namespace nnie{
    class Net {
    public:
        Net(const char* wk_model_path,NNIE_CFG_S nnie_cfg);
        Net(const char* wk_model_path);
        Net(void* wk_data, long length);
        //Net(void* wk_model_pointer,long len);
        void set_input(const char* layer_name,Mat<uint8_t>& data);
        void set_input(int id,Mat<uint8_t>& data);
        void set_input(const char* layer_name,VIDEO_FRAME_INFO_S frame);
        Mat<float> extract(const char* layer_name);
        Mat<float> extract(int id);
        void forward(const char* input_layer_name);
        void forward(int id);
        std::map<std::string,NNIE_LAYER_S> input_layer,output_layer;
        std::vector<NNIE_LAYER_S> input_layer_vec,output_layer_vec;
        ~Net();
    private:
        NNIE_MODEL_S nnie_model{};
        NNIE_PARAM_S nnie_param{};
        NNIE_CFG_S   nnie_config{};
        static HI_S32 NNIE_ParamDeinit(NNIE_PARAM_S *pstNnieParam);
        static HI_S32 NNIE_ParamInit(NNIE_CFG_S *pstNnieCfg, NNIE_PARAM_S *pstNnieParam);
        static HI_S32 NNIE_UnloadModel(NNIE_MODEL_S *pstNnieModel);
        static HI_S32 NNIE_LoadModel(void* data, long len,NNIE_MODEL_S *pstNnieModel);
        static HI_S32 NNIE_LoadModel(const char *pszModelFile, NNIE_MODEL_S *pstNnieModel);
        static HI_S32 NNIE_Forward(NNIE_PARAM_S *pstNnieParam,
                                   NNIE_INPUT_DATA_INDEX_S* pstInputDataIdx,
                                   NNIE_PROCESS_SEG_INDEX_S* pstProcSegIdx,HI_BOOL bInstant);
        static HI_S32 NNIE_FillForwardInfo(NNIE_CFG_S *pstNnieCfg, NNIE_PARAM_S *pstNnieParam);
        static HI_S32 NNIE_GetTaskAndBlobBufSize(NNIE_CFG_S *pstNnieCfg,
                                                 NNIE_PARAM_S *pstNnieParam,HI_U32*pu32TotalTaskBufSize, HI_U32*pu32TmpBufSize,
                                                 NNIE_BLOB_SIZE_S astBlobSize[],HI_U32*pu32TotalSize);
        static void NNIE_GetBlobMemSize(SVP_NNIE_NODE_S astNnieNode[], HI_U32 u32NodeNum,
                                        HI_U32 u32TotalStep,SVP_BLOB_S astBlob[], HI_U32 u32Align, HI_U32* pu32TotalSize,HI_U32 au32BlobSize[]);
        void nnie_model_info_init();


    };
}



#endif //HISILICON_NET_H
