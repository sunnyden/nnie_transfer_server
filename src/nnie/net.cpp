//
// Created by sunny on 19/1/2020.
//

#include <iostream>
#include <hi_comm_svp.h>
#include <hi_ive.h>
#include <mpi_ive.h>

#include "nnie/net.h"

nnie::Net::Net(const char *wk_model_path,NNIE_CFG_S nnie_cfg) {
    memset(&this->nnie_model,0,sizeof(NNIE_MODEL_S));
    memset(&this->nnie_param,0,sizeof(NNIE_PARAM_S));
    this->nnie_config = nnie_cfg;
    NNIE_LoadModel(wk_model_path,&this->nnie_model);
    this->nnie_param.pstModel = &this->nnie_model.stModel;
    NNIE_ParamInit(&this->nnie_config, &nnie_param);
    this->nnie_model_info_init();
}

nnie::Net::Net(const char *wk_model_path) {
    memset(&this->nnie_model,0,sizeof(NNIE_MODEL_S));
    memset(&this->nnie_param,0,sizeof(NNIE_PARAM_S));
    NNIE_CFG_S nnie_cfg;
    nnie_cfg.u32MaxInputNum = 1; //max input image num in each batch
    nnie_cfg.u32MaxRoiNum = 0;
    nnie_cfg.aenNnieCoreId[0] = SVP_NNIE_ID_0; //set NNIE core for 0-th Seg
    this->nnie_config = nnie_cfg;
    NNIE_LoadModel(wk_model_path,&this->nnie_model);
    this->nnie_param.pstModel = &this->nnie_model.stModel;
    NNIE_ParamInit(&this->nnie_config, &nnie_param);
    this->nnie_model_info_init();
}

nnie::Net::~Net() {
    //printf("Deinit Stage1\n");
    NNIE_ParamDeinit(&nnie_param);
    //printf("Deinit Stage2\n");
    NNIE_UnloadModel(&nnie_model);
    //printf("Deinit Done\n");
}

HI_S32 nnie::Net::NNIE_ParamDeinit(NNIE_PARAM_S *pstNnieParam) {
    if(0!=pstNnieParam->stTaskBuf.u64PhyAddr && 0!=pstNnieParam->stTaskBuf.u64VirAddr)
    {
        HI_MPI_SYS_MmzFree(pstNnieParam->stTaskBuf.u64PhyAddr,(void*)pstNnieParam->stTaskBuf.u64VirAddr);
        pstNnieParam->stTaskBuf.u64PhyAddr = 0;
        pstNnieParam->stTaskBuf.u64VirAddr = 0;
    }

    if(0!=pstNnieParam->stStepBuf.u64PhyAddr && 0!=pstNnieParam->stStepBuf.u64VirAddr)
    {
        HI_MPI_SYS_MmzFree(pstNnieParam->stTaskBuf.u64PhyAddr,(void*)pstNnieParam->stTaskBuf.u64VirAddr);
        pstNnieParam->stStepBuf.u64PhyAddr = 0;
        pstNnieParam->stStepBuf.u64VirAddr = 0;
    }
    return HI_SUCCESS;
}

HI_S32 nnie::Net::NNIE_ParamInit(NNIE_CFG_S *pstNnieCfg, NNIE_PARAM_S *pstNnieParam) {
    HI_U32 i = 0, j = 0;
    HI_U32 u32TotalSize = 0;
    HI_U32 u32TotalTaskBufSize = 0;
    HI_U32 u32TmpBufSize = 0;
    HI_S32 s32Ret = HI_SUCCESS;
    HI_U32 u32Offset = 0;
    HI_U64 u64PhyAddr = 0;
    HI_U8 *pu8VirAddr = NULL;
    NNIE_BLOB_SIZE_S astBlobSize[SVP_NNIE_MAX_NET_SEG_NUM] = {0};

    /*fill forward info*/
    s32Ret = NNIE_FillForwardInfo(pstNnieCfg,pstNnieParam);
    //SAMPLE_SVP_CHECK_EXPR_RET(HI_SUCCESS != s32Ret,s32Ret,SAMPLE_SVP_ERR_LEVEL_ERROR,
    //                          "Error,SAMPLE_SVP_NNIE_FillForwardCtrl failed!\n");

    /*Get taskInfo and Blob mem size*/
    s32Ret = NNIE_GetTaskAndBlobBufSize(pstNnieCfg,pstNnieParam,&u32TotalTaskBufSize,
                                        &u32TmpBufSize,astBlobSize,&u32TotalSize);
    //SAMPLE_SVP_CHECK_EXPR_RET(HI_SUCCESS != s32Ret,s32Ret,SAMPLE_SVP_ERR_LEVEL_ERROR,
    //                          "Error,SAMPLE_SVP_NNIE_GetTaskAndBlobBufSize failed!\n");

    /*Malloc mem*/
    s32Ret = HI_MPI_SYS_MmzAlloc_Cached((HI_U64*)&u64PhyAddr,(void**)&pu8VirAddr, "SAMPLE_NNIE_TASK",NULL, u32TotalSize);
    if(HI_SUCCESS != s32Ret){
        printf("Error,Malloc memory failed!\n");
    }

    memset(pu8VirAddr, 0, u32TotalSize);
    HI_MPI_SYS_MmzFlushCache(u64PhyAddr,(void*)pu8VirAddr,u32TotalSize);

    /*fill taskinfo mem addr*/
    pstNnieParam->stTaskBuf.u32Size = u32TotalTaskBufSize;
    pstNnieParam->stTaskBuf.u64PhyAddr = u64PhyAddr;
    pstNnieParam->stTaskBuf.u64VirAddr = (HI_U64)(HI_UL)pu8VirAddr;

    /*fill Tmp mem addr*/
    pstNnieParam->stTmpBuf.u32Size = u32TmpBufSize;
    pstNnieParam->stTmpBuf.u64PhyAddr = u64PhyAddr+u32TotalTaskBufSize;
    pstNnieParam->stTmpBuf.u64VirAddr = (HI_U64)(HI_UL)pu8VirAddr+u32TotalTaskBufSize;

    /*fill forward ctrl addr*/
    for(i = 0; i < pstNnieParam->pstModel->u32NetSegNum; i++)
    {
        if(SVP_NNIE_NET_TYPE_ROI == pstNnieParam->pstModel->astSeg[i].enNetType)
        {
            pstNnieParam->astForwardWithBboxCtrl[i].stTmpBuf = pstNnieParam->stTmpBuf;
            pstNnieParam->astForwardWithBboxCtrl[i].stTskBuf.u64PhyAddr= pstNnieParam->stTaskBuf.u64PhyAddr+u32Offset;
            pstNnieParam->astForwardWithBboxCtrl[i].stTskBuf.u64VirAddr= pstNnieParam->stTaskBuf.u64VirAddr+u32Offset;
            pstNnieParam->astForwardWithBboxCtrl[i].stTskBuf.u32Size= pstNnieParam->au32TaskBufSize[i];
        }
        else if(SVP_NNIE_NET_TYPE_CNN == pstNnieParam->pstModel->astSeg[i].enNetType ||
                SVP_NNIE_NET_TYPE_RECURRENT == pstNnieParam->pstModel->astSeg[i].enNetType)
        {


            pstNnieParam->astForwardCtrl[i].stTmpBuf = pstNnieParam->stTmpBuf;
            pstNnieParam->astForwardCtrl[i].stTskBuf.u64PhyAddr= pstNnieParam->stTaskBuf.u64PhyAddr+u32Offset;
            pstNnieParam->astForwardCtrl[i].stTskBuf.u64VirAddr= pstNnieParam->stTaskBuf.u64VirAddr+u32Offset;
            pstNnieParam->astForwardCtrl[i].stTskBuf.u32Size= pstNnieParam->au32TaskBufSize[i];
        }
        u32Offset += pstNnieParam->au32TaskBufSize[i];
    }

    /*fill each blob's mem addr*/
    u64PhyAddr =  u64PhyAddr+u32TotalTaskBufSize+u32TmpBufSize;
    pu8VirAddr = pu8VirAddr+u32TotalTaskBufSize+u32TmpBufSize;
    for(i = 0; i < pstNnieParam->pstModel->u32NetSegNum; i++)
    {
        /*first seg has src blobs, other seg's src blobs from the output blobs of
        those segs before it or from software output results*/
        if(0 == i)
        {
            for(j = 0; j < pstNnieParam->pstModel->astSeg[i].u16SrcNum; j++)
            {
                if(j!=0)
                {
                    u64PhyAddr += astBlobSize[i].au32SrcSize[j-1];
                    pu8VirAddr += astBlobSize[i].au32SrcSize[j-1];
                }
                pstNnieParam->astSegData[i].astSrc[j].u64PhyAddr = u64PhyAddr;
                pstNnieParam->astSegData[i].astSrc[j].u64VirAddr = (HI_U64)(HI_UL)pu8VirAddr;
            }
            u64PhyAddr += astBlobSize[i].au32SrcSize[j-1];
            pu8VirAddr += astBlobSize[i].au32SrcSize[j-1];
        }

        /*fill the mem addrs of each seg's output blobs*/
        for(j = 0; j < pstNnieParam->pstModel->astSeg[i].u16DstNum; j++)
        {
            if(j!=0)
            {
                u64PhyAddr += astBlobSize[i].au32DstSize[j-1];
                pu8VirAddr += astBlobSize[i].au32DstSize[j-1];
            }
            pstNnieParam->astSegData[i].astDst[j].u64PhyAddr = u64PhyAddr;
            pstNnieParam->astSegData[i].astDst[j].u64VirAddr = (HI_U64)(HI_UL)pu8VirAddr;
        }
        u64PhyAddr += astBlobSize[i].au32DstSize[j-1];
        pu8VirAddr += astBlobSize[i].au32DstSize[j-1];
    }
    return s32Ret;
}

HI_S32 nnie::Net::NNIE_UnloadModel(NNIE_MODEL_S *pstNnieModel) {
    if(0!=pstNnieModel->stModelBuf.u64PhyAddr && 0!=pstNnieModel->stModelBuf.u64VirAddr)
    {
        HI_MPI_SYS_MmzFree(pstNnieModel->stModelBuf.u64PhyAddr,
                           reinterpret_cast<void *>(pstNnieModel->stModelBuf.u64VirAddr));
        pstNnieModel->stModelBuf.u64PhyAddr = 0;
        pstNnieModel->stModelBuf.u64VirAddr = 0;
    }
    return HI_SUCCESS;
}

HI_S32 nnie::Net::NNIE_LoadModel(const char *pszModelFile, NNIE_MODEL_S *pstNnieModel) {
    HI_S32 s32Ret = HI_INVALID_VALUE;
    HI_U64 u64PhyAddr = 0;
    HI_U8 *pu8VirAddr = NULL;
    HI_SL slFileSize = 0;
    /*Get model file size*/
    FILE *fp=fopen(pszModelFile,"rb");
    if(nullptr == fp){
        printf("Error, open model file failed!\n");
        return s32Ret;
    }
    s32Ret = fseek(fp,0L,SEEK_END);
    if(-1 == s32Ret){
        printf("Error, fseek failed!\n");
        if (NULL != fp)
        {
            fclose(fp);
        }
        return s32Ret;
    }
    slFileSize = ftell(fp);
    s32Ret = fseek(fp,0L,SEEK_SET);

    /*malloc model file mem*/
    s32Ret = HI_MPI_SYS_MmzAlloc((HI_U64*)&u64PhyAddr,(void**)&pu8VirAddr, "SAMPLE_NNIE_MODEL", NULL, slFileSize);

    if(s32Ret != HI_SUCCESS){
        printf("Error(%#x),Malloc memory failed!\n",s32Ret);
        if (NULL != fp)
        {
            fclose(fp);
        }
        return HI_FAILURE;
    }

    pstNnieModel->stModelBuf.u32Size = (HI_U32)slFileSize;
    pstNnieModel->stModelBuf.u64PhyAddr = u64PhyAddr;
    pstNnieModel->stModelBuf.u64VirAddr = (HI_U64)(HI_UL)pu8VirAddr;

    s32Ret = fread(pu8VirAddr,slFileSize,1,fp);
    if(s32Ret != 1){
        printf("Error,read model file failed!\n");
        HI_MPI_SYS_MmzFree(pstNnieModel->stModelBuf.u64PhyAddr,
                           reinterpret_cast<void *>(pstNnieModel->stModelBuf.u64VirAddr));
        pstNnieModel->stModelBuf.u32Size  = 0;
        if (NULL != fp)
        {
            fclose(fp);
        }
        return HI_FAILURE;
    }

    /*load model*/
    s32Ret = HI_MPI_SVP_NNIE_LoadModel(&pstNnieModel->stModelBuf,&pstNnieModel->stModel);
    if(s32Ret != HI_SUCCESS){
        printf("Error,read model file failed!\n");
        HI_MPI_SYS_MmzFree(pstNnieModel->stModelBuf.u64PhyAddr,
                           reinterpret_cast<void *>(pstNnieModel->stModelBuf.u64VirAddr));
        pstNnieModel->stModelBuf.u32Size  = 0;
        if (NULL != fp)
        {
            fclose(fp);
        }
        return HI_FAILURE;
    }

    fclose(fp);
    return HI_SUCCESS;


}

HI_S32 nnie::Net::NNIE_Forward(NNIE_PARAM_S *pstNnieParam, NNIE_INPUT_DATA_INDEX_S *pstInputDataIdx,
                               NNIE_PROCESS_SEG_INDEX_S *pstProcSegIdx, HI_BOOL bInstant) {
    HI_S32 s32Ret = HI_SUCCESS;
    HI_U32 i = 0, j = 0;
    HI_BOOL bFinish = HI_FALSE;
    SVP_NNIE_HANDLE hSvpNnieHandle = 0;
    HI_U32 u32TotalStepNum = 0;

    HI_MPI_SYS_MmzFlushCache(pstNnieParam->astForwardCtrl[pstProcSegIdx->u32SegIdx].stTskBuf.u64PhyAddr,
                             (HI_VOID *) pstNnieParam->astForwardCtrl[pstProcSegIdx->u32SegIdx].stTskBuf.u64VirAddr,
                             pstNnieParam->astForwardCtrl[pstProcSegIdx->u32SegIdx].stTskBuf.u32Size);

    /*set input blob according to node name*/
    if(pstInputDataIdx->u32SegIdx != pstProcSegIdx->u32SegIdx)
    {
        for(i = 0; i < pstNnieParam->pstModel->astSeg[pstProcSegIdx->u32SegIdx].u16SrcNum; i++)
        {
            for(j = 0; j < pstNnieParam->pstModel->astSeg[pstInputDataIdx->u32SegIdx].u16DstNum; j++)
            {
                if(0 == strncmp(pstNnieParam->pstModel->astSeg[pstInputDataIdx->u32SegIdx].astDstNode[j].szName,
                                pstNnieParam->pstModel->astSeg[pstProcSegIdx->u32SegIdx].astSrcNode[i].szName,
                                SVP_NNIE_NODE_NAME_LEN))
                {
                    pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astSrc[i] =
                            pstNnieParam->astSegData[pstInputDataIdx->u32SegIdx].astDst[j];
                    break;
                }
            }
            if(j == pstNnieParam->pstModel->astSeg[pstInputDataIdx->u32SegIdx].u16DstNum){
                printf("Error,can't find %d-th seg's %d-th src blob!\n",pstProcSegIdx->u32SegIdx,i);
            }
            /*SAMPLE_SVP_CHECK_EXPR_RET((j == pstNnieParam->pstModel->astSeg[pstInputDataIdx->u32SegIdx].u16DstNum),
                                      HI_FAILURE,SAMPLE_SVP_ERR_LEVEL_ERROR,"Error,can't find %d-th seg's %d-th src blob!\n",
                                      pstProcSegIdx->u32SegIdx,i);*/
        }
    }

    /*NNIE_Forward*/
    s32Ret = HI_MPI_SVP_NNIE_Forward(&hSvpNnieHandle,
                                     pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astSrc,
                                     pstNnieParam->pstModel, pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst,
                                     &pstNnieParam->astForwardCtrl[pstProcSegIdx->u32SegIdx], bInstant);
    if(HI_SUCCESS != s32Ret){
        printf("Error,HI_MPI_SVP_NNIE_Forward failed!\n");
    }
    //SAMPLE_SVP_CHECK_EXPR_RET(HI_SUCCESS != s32Ret,s32Ret,SAMPLE_SVP_ERR_LEVEL_ERROR,
    //                          "Error,HI_MPI_SVP_NNIE_Forward failed!\n");

    if(bInstant)
    {
        /*Wait NNIE finish*/
        while(HI_ERR_SVP_NNIE_QUERY_TIMEOUT == (s32Ret = HI_MPI_SVP_NNIE_Query(pstNnieParam->astForwardCtrl[pstProcSegIdx->u32SegIdx].enNnieId,
                                                                               hSvpNnieHandle, &bFinish, HI_TRUE)))
        {
            usleep(100);
            printf("HI_MPI_SVP_NNIE_Query Query timeout!\n");
        }
    }

    bFinish = HI_FALSE;
    for(i = 0; i < pstNnieParam->astForwardCtrl[pstProcSegIdx->u32SegIdx].u32DstNum; i++)
    {
        if(SVP_BLOB_TYPE_SEQ_S32 == pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].enType)
        {
            for(j = 0; j < pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u32Num; j++)
            {
                u32TotalStepNum += *((HI_U32*)(pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].unShape.stSeq.u64VirAddrStep)+j);
            }
            HI_MPI_SYS_MmzFlushCache(pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u64PhyAddr,
                                     (HI_VOID *) pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u64VirAddr,
                                     u32TotalStepNum*pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u32Stride);

        }
        else
        {

            HI_MPI_SYS_MmzFlushCache(pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u64PhyAddr,
                                     (HI_VOID *) pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u64VirAddr,
                                     pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u32Num*
                                     pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].unShape.stWhc.u32Chn*
                                     pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].unShape.stWhc.u32Height*
                                     pstNnieParam->astSegData[pstProcSegIdx->u32SegIdx].astDst[i].u32Stride);
        }
    }

    return s32Ret;
}

HI_S32 nnie::Net::NNIE_FillForwardInfo(NNIE_CFG_S *pstNnieCfg, NNIE_PARAM_S *pstNnieParam) {
    HI_U32 i = 0, j = 0;
    HI_U32 u32Offset = 0;
    HI_U32 u32Num = 0;

    for(i = 0; i < pstNnieParam->pstModel->u32NetSegNum; i++)
    {
        /*fill forwardCtrl info*/
        if(SVP_NNIE_NET_TYPE_ROI == pstNnieParam->pstModel->astSeg[i].enNetType)
        {
            pstNnieParam->astForwardWithBboxCtrl[i].enNnieId = pstNnieCfg->aenNnieCoreId[i];
            pstNnieParam->astForwardWithBboxCtrl[i].u32SrcNum = pstNnieParam->pstModel->astSeg[i].u16SrcNum;
            pstNnieParam->astForwardWithBboxCtrl[i].u32DstNum = pstNnieParam->pstModel->astSeg[i].u16DstNum;
            pstNnieParam->astForwardWithBboxCtrl[i].u32ProposalNum = 1;
            pstNnieParam->astForwardWithBboxCtrl[i].u32NetSegId = i;
            pstNnieParam->astForwardWithBboxCtrl[i].stTmpBuf = pstNnieParam->stTmpBuf;
            pstNnieParam->astForwardWithBboxCtrl[i].stTskBuf.u64PhyAddr= pstNnieParam->stTaskBuf.u64PhyAddr+u32Offset;
            pstNnieParam->astForwardWithBboxCtrl[i].stTskBuf.u64VirAddr= pstNnieParam->stTaskBuf.u64VirAddr+u32Offset;
            pstNnieParam->astForwardWithBboxCtrl[i].stTskBuf.u32Size= pstNnieParam->au32TaskBufSize[i];
        }
        else if(SVP_NNIE_NET_TYPE_CNN == pstNnieParam->pstModel->astSeg[i].enNetType ||
                SVP_NNIE_NET_TYPE_RECURRENT== pstNnieParam->pstModel->astSeg[i].enNetType)
        {


            pstNnieParam->astForwardCtrl[i].enNnieId = pstNnieCfg->aenNnieCoreId[i];
            pstNnieParam->astForwardCtrl[i].u32SrcNum = pstNnieParam->pstModel->astSeg[i].u16SrcNum;
            pstNnieParam->astForwardCtrl[i].u32DstNum = pstNnieParam->pstModel->astSeg[i].u16DstNum;
            pstNnieParam->astForwardCtrl[i].u32NetSegId = i;
            pstNnieParam->astForwardCtrl[i].stTmpBuf = pstNnieParam->stTmpBuf;
            pstNnieParam->astForwardCtrl[i].stTskBuf.u64PhyAddr= pstNnieParam->stTaskBuf.u64PhyAddr+u32Offset;
            pstNnieParam->astForwardCtrl[i].stTskBuf.u64VirAddr= pstNnieParam->stTaskBuf.u64VirAddr+u32Offset;
            pstNnieParam->astForwardCtrl[i].stTskBuf.u32Size= pstNnieParam->au32TaskBufSize[i];
        }
        u32Offset += pstNnieParam->au32TaskBufSize[i];

        /*fill src blob info*/
        for(j = 0; j < pstNnieParam->pstModel->astSeg[i].u16SrcNum; j++)
        {
            /*Recurrent blob*/
            if(SVP_BLOB_TYPE_SEQ_S32 == pstNnieParam->pstModel->astSeg[i].astSrcNode[j].enType)
            {
                pstNnieParam->astSegData[i].astSrc[j].enType = pstNnieParam->pstModel->astSeg[i].astSrcNode[j].enType;
                pstNnieParam->astSegData[i].astSrc[j].unShape.stSeq.u32Dim = pstNnieParam->pstModel->astSeg[i].astSrcNode[j].unShape.u32Dim;
                pstNnieParam->astSegData[i].astSrc[j].u32Num = pstNnieCfg->u32MaxInputNum;
                pstNnieParam->astSegData[i].astSrc[j].unShape.stSeq.u64VirAddrStep = pstNnieCfg->au64StepVirAddr[i*NNIE_EACH_SEG_STEP_ADDR_NUM];
            }
            else
            {
                pstNnieParam->astSegData[i].astSrc[j].enType = pstNnieParam->pstModel->astSeg[i].astSrcNode[j].enType;
                pstNnieParam->astSegData[i].astSrc[j].unShape.stWhc.u32Chn = pstNnieParam->pstModel->astSeg[i].astSrcNode[j].unShape.stWhc.u32Chn;
                pstNnieParam->astSegData[i].astSrc[j].unShape.stWhc.u32Height = pstNnieParam->pstModel->astSeg[i].astSrcNode[j].unShape.stWhc.u32Height;
                pstNnieParam->astSegData[i].astSrc[j].unShape.stWhc.u32Width = pstNnieParam->pstModel->astSeg[i].astSrcNode[j].unShape.stWhc.u32Width;
                pstNnieParam->astSegData[i].astSrc[j].u32Num = pstNnieCfg->u32MaxInputNum;
            }
        }

        /*fill dst blob info*/
        if(SVP_NNIE_NET_TYPE_ROI == pstNnieParam->pstModel->astSeg[i].enNetType)
        {
            u32Num = pstNnieCfg->u32MaxRoiNum*pstNnieCfg->u32MaxInputNum;
        }
        else
        {
            u32Num = pstNnieCfg->u32MaxInputNum;
        }

        for(j = 0; j < pstNnieParam->pstModel->astSeg[i].u16DstNum; j++)
        {
            if(SVP_BLOB_TYPE_SEQ_S32 == pstNnieParam->pstModel->astSeg[i].astDstNode[j].enType)
            {
                pstNnieParam->astSegData[i].astDst[j].enType = pstNnieParam->pstModel->astSeg[i].astDstNode[j].enType;
                pstNnieParam->astSegData[i].astDst[j].unShape.stSeq.u32Dim =
                        pstNnieParam->pstModel->astSeg[i].astDstNode[j].unShape.u32Dim;
                pstNnieParam->astSegData[i].astDst[j].u32Num = u32Num;
                pstNnieParam->astSegData[i].astDst[j].unShape.stSeq.u64VirAddrStep =
                        pstNnieCfg->au64StepVirAddr[i*NNIE_EACH_SEG_STEP_ADDR_NUM+1];
            }
            else
            {
                pstNnieParam->astSegData[i].astDst[j].enType = pstNnieParam->pstModel->astSeg[i].astDstNode[j].enType;
                pstNnieParam->astSegData[i].astDst[j].unShape.stWhc.u32Chn = pstNnieParam->pstModel->astSeg[i].astDstNode[j].unShape.stWhc.u32Chn;
                pstNnieParam->astSegData[i].astDst[j].unShape.stWhc.u32Height = pstNnieParam->pstModel->astSeg[i].astDstNode[j].unShape.stWhc.u32Height;
                pstNnieParam->astSegData[i].astDst[j].unShape.stWhc.u32Width = pstNnieParam->pstModel->astSeg[i].astDstNode[j].unShape.stWhc.u32Width;
                pstNnieParam->astSegData[i].astDst[j].u32Num = u32Num;
            }
        }
    }
    return HI_SUCCESS;
}

HI_S32 nnie::Net::NNIE_GetTaskAndBlobBufSize(NNIE_CFG_S *pstNnieCfg, NNIE_PARAM_S *pstNnieParam,
                                             HI_U32 *pu32TotalTaskBufSize, HI_U32 *pu32TmpBufSize,
                                             NNIE_BLOB_SIZE_S *astBlobSize, HI_U32 *pu32TotalSize) {
    HI_S32 s32Ret = HI_SUCCESS;
    HI_U32 i = 0, j = 0;
    HI_U32 u32TotalStep = 0;

    /*Get each seg's task buf size*/
    s32Ret = HI_MPI_SVP_NNIE_GetTskBufSize(pstNnieCfg->u32MaxInputNum, pstNnieCfg->u32MaxRoiNum,
                                           pstNnieParam->pstModel, pstNnieParam->au32TaskBufSize,pstNnieParam->pstModel->u32NetSegNum);
    //SAMPLE_SVP_CHECK_EXPR_RET(HI_SUCCESS != s32Ret,s32Ret,SAMPLE_SVP_ERR_LEVEL_ERROR,
    //                          "Error,HI_MPI_SVP_NNIE_GetTaskSize failed!\n");

    /*Get total task buf size*/
    *pu32TotalTaskBufSize = 0;
    for(i = 0; i < pstNnieParam->pstModel->u32NetSegNum; i++)
    {
        *pu32TotalTaskBufSize += pstNnieParam->au32TaskBufSize[i];
    }

    /*Get tmp buf size*/
    *pu32TmpBufSize = pstNnieParam->pstModel->u32TmpBufSize;
    *pu32TotalSize += *pu32TotalTaskBufSize + *pu32TmpBufSize;

    /*calculate Blob mem size*/
    for(i = 0; i < pstNnieParam->pstModel->u32NetSegNum; i++)
    {
        if(SVP_NNIE_NET_TYPE_RECURRENT == pstNnieParam->pstModel->astSeg[i].enNetType)
        {
            for(j = 0; j < pstNnieParam->astSegData[i].astSrc[0].u32Num; j++)
            {
                u32TotalStep += *((HI_S32*)(HI_UL)pstNnieParam->astSegData[i].astSrc[0].unShape.stSeq.u64VirAddrStep+j);
            }
        }
        /*the first seg's Src Blob mem size, other seg's src blobs from the output blobs of
        those segs before it or from software output results*/
        if(i == 0)
        {
            NNIE_GetBlobMemSize(&(pstNnieParam->pstModel->astSeg[i].astSrcNode[0]),
                                pstNnieParam->pstModel->astSeg[i].u16SrcNum,u32TotalStep,&(pstNnieParam->astSegData[i].astSrc[0]),
                                NNIE_ALIGN_16, pu32TotalSize, &(astBlobSize[i].au32SrcSize[0]));
        }

        /*Get each seg's Dst Blob mem size*/
        NNIE_GetBlobMemSize(&(pstNnieParam->pstModel->astSeg[i].astDstNode[0]),
                            pstNnieParam->pstModel->astSeg[i].u16DstNum,u32TotalStep,&(pstNnieParam->astSegData[i].astDst[0]),
                            NNIE_ALIGN_16, pu32TotalSize, &(astBlobSize[i].au32DstSize[0]));
    }
    return s32Ret;

}

void nnie::Net::NNIE_GetBlobMemSize(SVP_NNIE_NODE_S *astNnieNode, HI_U32 u32NodeNum, HI_U32 u32TotalStep,
                                    SVP_BLOB_S *astBlob, HI_U32 u32Align, HI_U32 *pu32TotalSize, HI_U32 *au32BlobSize) {
    HI_U32 i = 0;
    HI_U32 u32Size = 0;
    HI_U32 u32Stride = 0;

    for(i = 0; i < u32NodeNum; i++)
    {
        if(SVP_BLOB_TYPE_S32== astNnieNode[i].enType||SVP_BLOB_TYPE_VEC_S32== astNnieNode[i].enType||
           SVP_BLOB_TYPE_SEQ_S32== astNnieNode[i].enType)
        {
            u32Size = sizeof(HI_U32);
        }
        else
        {
            u32Size = sizeof(HI_U8);
        }
        if(SVP_BLOB_TYPE_SEQ_S32 == astNnieNode[i].enType)
        {
            if(NNIE_ALIGN_16 == u32Align)
            {
                u32Stride = NNIE_ALIGN16(astNnieNode[i].unShape.u32Dim*u32Size);
            }
            else
            {
                u32Stride = NNIE_ALIGN32(astNnieNode[i].unShape.u32Dim*u32Size);
            }
            au32BlobSize[i] = u32TotalStep*u32Stride;
        }
        else
        {
            if(NNIE_ALIGN_16 == u32Align)
            {
                u32Stride = NNIE_ALIGN16(astNnieNode[i].unShape.stWhc.u32Width*u32Size);
            }
            else
            {
                u32Stride = NNIE_ALIGN32(astNnieNode[i].unShape.stWhc.u32Width*u32Size);
            }
            au32BlobSize[i] = astBlob[i].u32Num*u32Stride*astNnieNode[i].unShape.stWhc.u32Height*
                              astNnieNode[i].unShape.stWhc.u32Chn;
        }
        *pu32TotalSize += au32BlobSize[i];
        astBlob[i].u32Stride = u32Stride;
    }
}

void nnie::Net::nnie_model_info_init() {
    for(int seg_id =0;seg_id < this->nnie_model.stModel.u32NetSegNum;seg_id++){
        int dst_num = this->nnie_model.stModel.astSeg[seg_id].u16DstNum;
        int src_num = this->nnie_model.stModel.astSeg[seg_id].u16SrcNum;
        for(int dst_id = 0; dst_id<dst_num;dst_id++){
            printf("[NNIE][OUTPUT][ID:%d][TOTAL_ID:%d][Type:%d][Name:%s][Shape(WxHxC):%d x %d x %d][Stride:%d]\n",
                   dst_id,dst_num,
                   this->nnie_model.stModel.astSeg[seg_id].astDstNode[dst_id].enType,
                   this->nnie_model.stModel.astSeg[seg_id].astDstNode[dst_id].szName,
                   this->nnie_model.stModel.astSeg[seg_id].astDstNode[dst_id].unShape.stWhc.u32Width,
                   this->nnie_model.stModel.astSeg[seg_id].astDstNode[dst_id].unShape.stWhc.u32Height,
                   this->nnie_model.stModel.astSeg[seg_id].astDstNode[dst_id].unShape.stWhc.u32Chn,
                   this->nnie_param.astSegData[seg_id].astDst[dst_id].u32Stride);
            NNIE_LAYER_S layer;
            layer.u64PhyAddr = &(this->nnie_param.astSegData[seg_id].astDst[dst_id].u64PhyAddr);
            layer.u64VirAddr = &(this->nnie_param.astSegData[seg_id].astDst[dst_id].u64VirAddr);
            layer.u32Stride = &(this->nnie_param.astSegData[seg_id].astDst[dst_id].u32Stride);
            layer.channel = this->nnie_model.stModel.astSeg[seg_id].astDstNode[dst_id].unShape.stWhc.u32Chn;
            layer.width = this->nnie_model.stModel.astSeg[seg_id].astDstNode[dst_id].unShape.stWhc.u32Width;
            layer.height = this->nnie_model.stModel.astSeg[seg_id].astDstNode[dst_id].unShape.stWhc.u32Height;
            layer.data_type = this->nnie_model.stModel.astSeg[seg_id].astDstNode[dst_id].enType;
            layer.layer_id = dst_id;
            layer.segment_id = seg_id;
            this->output_layer.insert(std::make_pair(this->nnie_model.stModel.astSeg[seg_id].astDstNode[dst_id].szName,layer));
            this->output_layer_vec.push_back(layer);
        }
        for(int src_id = 0;src_id<src_num;src_id++){
            printf("[NNIE][INPUT][ID:%d][TOTAL_ID:%d][Type:%d][Name:%s][Shape(WxHxC):%d x %d x %d][Stride:%d]\n",
                   src_id,src_num,
                   this->nnie_model.stModel.astSeg[seg_id].astSrcNode[src_id].enType,
                   this->nnie_model.stModel.astSeg[seg_id].astSrcNode[src_id].szName,
                   this->nnie_model.stModel.astSeg[seg_id].astSrcNode[src_id].unShape.stWhc.u32Width,
                   this->nnie_model.stModel.astSeg[seg_id].astSrcNode[src_id].unShape.stWhc.u32Height,
                   this->nnie_model.stModel.astSeg[seg_id].astSrcNode[src_id].unShape.stWhc.u32Chn,
                   this->nnie_param.astSegData[seg_id].astSrc[src_id].u32Stride);
            //this->nnie_param.astSegData->astSrc
            NNIE_LAYER_S layer;
            layer.u64PhyAddr = &(this->nnie_param.astSegData[seg_id].astSrc[src_id].u64PhyAddr);
            layer.u64VirAddr = &(this->nnie_param.astSegData[seg_id].astSrc[src_id].u64VirAddr);
            layer.u32Stride = &(this->nnie_param.astSegData[seg_id].astSrc[src_id].u32Stride);
            layer.channel = this->nnie_model.stModel.astSeg[seg_id].astSrcNode[src_id].unShape.stWhc.u32Chn;
            layer.width = this->nnie_model.stModel.astSeg[seg_id].astSrcNode[src_id].unShape.stWhc.u32Width;
            layer.height = this->nnie_model.stModel.astSeg[seg_id].astSrcNode[src_id].unShape.stWhc.u32Height;
            layer.data_type = this->nnie_model.stModel.astSeg[seg_id].astSrcNode[src_id].enType;
            layer.layer_id = src_id;
            layer.segment_id = seg_id;

            this->input_layer.insert(std::make_pair(this->nnie_model.stModel.astSeg[seg_id].astSrcNode[src_id].szName,layer));
            this->input_layer_vec.push_back(layer);
        }
    }
}

void nnie::Net::set_input(const char *layer_name, nnie::Mat<uint8_t> &data) {
    NNIE_LAYER_S layer = input_layer[layer_name];
    //printf("[NNIE][INFO] Load input from mat data, expecting %dx%d got %dx%d\n",
    //       layer.width,layer.height,data.cols,data.rows);
    if(data.rows == layer.height && data.cols == layer.width && data.channels() == layer.channel){
        if(*layer.u32Stride==layer.width){
            //printf("set_input, single \n");
            memcpy((uint8_t*)*layer.u64VirAddr,data.get_data_pointer(),layer.width*layer.height*layer.channel*sizeof(u_char));
        }else{
            for(int i=0;i<layer.height*layer.channel;i++){
                memcpy((u_char *)(*layer.u64VirAddr)+(*layer.u32Stride)*i,data.get_data_pointer()+i*data.cols,
                       layer.width*sizeof(u_char));
            }
        }
    }else{
        printf("[NNIE][ERROR] Failed to load input from mat data, wrong size, expecting %dx%d got %dx%d\n",
               layer.width,layer.height,data.cols,data.rows);
    }
    //printf("[NNIE][INFO] Load input from mat data, success!\n",
    //       layer.width,layer.height,data.cols,data.rows);
    HI_MPI_SYS_MmzFlushCache(*layer.u64PhyAddr, reinterpret_cast<void *>(*layer.u64VirAddr),
                             layer.width * layer.height * layer.channel * sizeof(u_char));
}

void nnie::Net::set_input(const char *layer_name, VIDEO_FRAME_INFO_S frame) {
    NNIE_LAYER_S layer = input_layer[layer_name];
    if(layer.width == frame.stVFrame.u32Width && layer.height == frame.stVFrame.u32Height){
        (*layer.u64VirAddr) = frame.stVFrame.u64VirAddr[0];
        (*layer.u64PhyAddr) = frame.stVFrame.u64PhyAddr[0];
        (*layer.u32Stride) = frame.stVFrame.u32Stride[0];
    }else{
        printf("[NNIE][ERROR] Failed to set input as frame data, wrong size, expecting %dx%d got %dx%d\n",
               layer.width,layer.height,frame.stVFrame.u32Width,frame.stVFrame.u32Height);
    }
}

void nnie::Net::forward(const char *input_layer_name) {
    NNIE_LAYER_S layer = input_layer[input_layer_name];
    NNIE_INPUT_DATA_INDEX_S stInputDataIdx;
    NNIE_PROCESS_SEG_INDEX_S stProcSegIdx;
    stProcSegIdx.u32NodeIdx = layer.layer_id;
    stProcSegIdx.u32SegIdx = layer.segment_id;
    stInputDataIdx.u32SegIdx = layer.segment_id;
    stInputDataIdx.u32NodeIdx = layer.layer_id;
    //printf("Forward: %d %d\n",layer.layer_id,layer.segment_id);
    if(HI_SUCCESS != NNIE_Forward(&this->nnie_param,&stInputDataIdx,&stProcSegIdx,HI_TRUE)){
        printf("Forward failed\n");
    }
}

nnie::Mat<float> nnie::Net::extract(const char *layer_name) {
    NNIE_LAYER_S layer = output_layer[layer_name];
    nnie::Mat<float> ret(layer.width, layer.height, layer.channel, MAT_TYPE_FLOAT);
    float* ptr = ret.get_data_pointer();
    HI_S32* data_ptr = (HI_S32*)*layer.u64VirAddr;
    uint stride = (*layer.u32Stride)/sizeof(HI_S32);
    float quant_base = 1.0f/QUANT_BASE;
    for(int c=0;c<layer.channel;c++){
        for(int h=0;h<layer.height;h++){
            for(int w=0;w<layer.width;w++){
                ptr[w]=data_ptr[w]*quant_base;
                //ptr[w]=float(data_ptr[w]);
                //printf("%d - %d - %d - %f\n",c*layer.height*layer.width+layer.width*h+w,c*layer.height*stride+stride*h+w,data_ptr[c*layer.height*stride+stride*h+w],(float(data_ptr[c*layer.height*stride+stride*h+w])/QUANT_BASE));
            }
            ptr+=layer.width;
            data_ptr+=stride;
        }
    }
    return ret;
}

nnie::Mat<float> nnie::Net::extract(int id) {
    NNIE_LAYER_S layer = output_layer_vec[id];
    nnie::Mat<float> ret(layer.width, layer.height, layer.channel, MAT_TYPE_FLOAT);
    float* ptr = ret.get_data_pointer();
    HI_S32* data_ptr = (HI_S32*)*layer.u64VirAddr;
    uint stride = (*layer.u32Stride)/sizeof(HI_S32);
    float quant_base = 1.0f/QUANT_BASE;
    for(int c=0;c<layer.channel;c++){
        for(int h=0;h<layer.height;h++){
            for(int w=0;w<layer.width;w++){
                ptr[w]=data_ptr[w]*quant_base;
                //ptr[w]=float(data_ptr[w]);
                //printf("%d - %d - %d - %f\n",c*layer.height*layer.width+layer.width*h+w,c*layer.height*stride+stride*h+w,data_ptr[c*layer.height*stride+stride*h+w],(float(data_ptr[c*layer.height*stride+stride*h+w])/QUANT_BASE));
            }
            //memcpy(ptr,data_ptr,layer.width);
            ptr+=layer.width;
            data_ptr+=stride;
        }
    }
    return ret;
}

nnie::Net::Net(void *wk_data, long length) {
    memset(&this->nnie_model,0,sizeof(NNIE_MODEL_S));
    memset(&this->nnie_param,0,sizeof(NNIE_PARAM_S));
    NNIE_CFG_S nnie_cfg;
    nnie_cfg.u32MaxInputNum = 1; //max input image num in each batch
    nnie_cfg.u32MaxRoiNum = 0;
    nnie_cfg.aenNnieCoreId[0] = SVP_NNIE_ID_0; //set NNIE core for 0-th Seg
    this->nnie_config = nnie_cfg;
    NNIE_LoadModel(wk_data,length,&this->nnie_model);
    this->nnie_param.pstModel = &this->nnie_model.stModel;
    NNIE_ParamInit(&this->nnie_config, &nnie_param);
    this->nnie_model_info_init();
}

HI_S32 nnie::Net::NNIE_LoadModel(void *data, long len, NNIE_MODEL_S *pstNnieModel) {
    HI_S32 s32Ret = HI_INVALID_VALUE;
    HI_U64 u64PhyAddr = 0;
    HI_U8 *pu8VirAddr = NULL;
    HI_SL slFileSize = len;
    /*Get model file size*/

    /*malloc model file mem*/
    s32Ret = HI_MPI_SYS_MmzAlloc((HI_U64*)&u64PhyAddr,(void**)&pu8VirAddr, "NNIE_MODEL", NULL, slFileSize);

    if(s32Ret != HI_SUCCESS){
        printf("Error(%#x),Malloc memory failed!\n",s32Ret);
        return HI_FAILURE;
    }

    pstNnieModel->stModelBuf.u32Size = (HI_U32)slFileSize;
    pstNnieModel->stModelBuf.u64PhyAddr = u64PhyAddr;
    pstNnieModel->stModelBuf.u64VirAddr = (HI_U64)(HI_UL)pu8VirAddr;

    memcpy(pu8VirAddr,data,len);
    /*load model*/
    s32Ret = HI_MPI_SVP_NNIE_LoadModel(&pstNnieModel->stModelBuf,&pstNnieModel->stModel);
    if(s32Ret != HI_SUCCESS){
        printf("Error,read model file failed!\n");
        HI_MPI_SYS_MmzFree(pstNnieModel->stModelBuf.u64PhyAddr,
                           reinterpret_cast<void *>(pstNnieModel->stModelBuf.u64VirAddr));
        pstNnieModel->stModelBuf.u32Size  = 0;
        return HI_FAILURE;
    }

    return HI_SUCCESS;
}

void load_yuv_blob(NNIE_LAYER_S* blob, const nnie::Mat<u_char>& image, IVE_IMAGE_TYPE_E type){
    IVE_HANDLE iveHandle;
    IVE_SRC_IMAGE_S srcImage;
    // IVE_DST_IMAGE_S dstImage;
    IVE_CSC_CTRL_S stCscCtrl;
    HI_U32 u32Height = blob->height, u32Width = blob->width, u32Chn = blob->channel;
    IVE_DST_IMAGE_S dstImage;
    HI_MPI_SYS_MmzAlloc_Cached(&dstImage.au64PhyAddr[0], (void **)&dstImage.au64VirAddr[0], "BSDDstImg",
                               HI_NULL, u32Height * u32Width * u32Chn);
    memset(&srcImage, 0, sizeof(IVE_SRC_IMAGE_S));
    srcImage.enType = type; //IVE_IMAGE_TYPE_YUV420SP
    srcImage.au32Stride[0] = *(blob->u32Stride);
    srcImage.au32Stride[1] = *(blob->u32Stride);
    srcImage.u32Height = u32Height;
    srcImage.u32Width = u32Width;
    srcImage.au64PhyAddr[0] = *(blob->u64PhyAddr);
    srcImage.au64PhyAddr[1] = *(blob->u64PhyAddr) + u32Width * u32Height;
    srcImage.au64VirAddr[0] = *(blob->u64VirAddr);
    srcImage.au64VirAddr[1] = *(blob->u64VirAddr) + u32Width * u32Height;
    dstImage.enType = IVE_IMAGE_TYPE_U8C3_PLANAR;
    dstImage.u32Width     = u32Width;
    dstImage.u32Height    = u32Height;
    dstImage.au32Stride[0] = (((u32Width + 15) >> 4) << 4);
    dstImage.au32Stride[1] = (((u32Width + 15) >> 4) << 4);
    dstImage.au32Stride[2] = (((u32Width + 15) >> 4) << 4);
    dstImage.au64PhyAddr[1] = dstImage.au64PhyAddr[0]+u32Width*u32Height;
    dstImage.au64PhyAddr[2] = dstImage.au64PhyAddr[1]+u32Width*u32Height;
    dstImage.au64VirAddr[1] = dstImage.au64VirAddr[0]+u32Width*u32Height;
    dstImage.au64VirAddr[2] = dstImage.au64VirAddr[1]+u32Width*u32Height;
    memcpy(reinterpret_cast<void *>(dstImage.au64VirAddr[0]), image.get_data_pointer(), u32Height * u32Width * u32Chn);
    HI_S32 s32Ret;

    HI_U8* aps32ImageBlob = NULL;
    aps32ImageBlob = (HI_U8*)(HI_UL)(dstImage.au64VirAddr[0]);
    stCscCtrl.enMode = IVE_CSC_MODE_VIDEO_BT601_RGB2YUV;
    s32Ret = HI_MPI_IVE_CSC(&iveHandle,&dstImage,&srcImage,&stCscCtrl,HI_TRUE);
    HI_BOOL is_finish;
    HI_MPI_IVE_Query(iveHandle, &is_finish, HI_TRUE);
    if(s32Ret != HI_SUCCESS){
        printf("WARNING:FAILED!,%#x\n",s32Ret);
    }
    HI_MPI_SYS_MmzFlushCache(srcImage.au64PhyAddr[0], reinterpret_cast<void *>(srcImage.au64VirAddr[0]), u32Width * u32Height * 1.5);
    HI_MPI_SYS_MmzFree(dstImage.au64PhyAddr[0], reinterpret_cast<void *>(dstImage.au64VirAddr[0]));
}

void nnie::Net::set_input(int id, nnie::Mat<uint8_t> &data) {
    NNIE_LAYER_S layer = input_layer_vec[id];
    //printf("[NNIE][INFO] Load input from mat data, expecting %dx%d got %dx%d\n",
    //       layer.width,layer.height,data.cols,data.rows);
    if(data.rows == layer.height && data.cols == layer.width && data.channels() == layer.channel){
        if(layer.data_type == SVP_BLOB_TYPE_YVU420SP || layer.data_type == SVP_BLOB_TYPE_YVU422SP){
            load_yuv_blob(&layer,data,layer.data_type == SVP_BLOB_TYPE_YVU420SP?IVE_IMAGE_TYPE_YUV420SP:IVE_IMAGE_TYPE_YUV422SP);
        }else{
            if(*layer.u32Stride==layer.width){
                //printf("set_input, single \n");
                memcpy((uint8_t*)*layer.u64VirAddr,data.get_data_pointer(),layer.width*layer.height*layer.channel*sizeof(u_char));
            }else{
                for(int i=0;i<layer.height*layer.channel;i++){
                    memcpy((u_char *)(*layer.u64VirAddr)+(*layer.u32Stride)*i,data.get_data_pointer()+i*data.cols,
                           layer.width*sizeof(u_char));
                }
            }
        }
    }else{
        printf("[NNIE][ERROR] Failed to load input from mat data, wrong size, expecting %dx%d got %dx%d\n",
               layer.width,layer.height,data.cols,data.rows);
    }
    //printf("[NNIE][INFO] Load input from mat data, success!\n",
    //       layer.width,layer.height,data.cols,data.rows);
    HI_MPI_SYS_MmzFlushCache(*layer.u64PhyAddr, reinterpret_cast<void *>(*layer.u64VirAddr),
                             layer.width * layer.height * layer.channel * sizeof(u_char));
}

void nnie::Net::forward(int id) {
    NNIE_LAYER_S layer = input_layer_vec[id];
    NNIE_INPUT_DATA_INDEX_S stInputDataIdx;
    NNIE_PROCESS_SEG_INDEX_S stProcSegIdx;
    stProcSegIdx.u32NodeIdx = layer.layer_id;
    stProcSegIdx.u32SegIdx = layer.segment_id;
    stInputDataIdx.u32SegIdx = layer.segment_id;
    stInputDataIdx.u32NodeIdx = layer.layer_id;
    //printf("Forward: %d %d\n",layer.layer_id,layer.segment_id);
    if(HI_SUCCESS != NNIE_Forward(&this->nnie_param,&stInputDataIdx,&stProcSegIdx,HI_TRUE)){
        printf("Forward failed\n");
    }
}




