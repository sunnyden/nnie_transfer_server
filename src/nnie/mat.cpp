//
// Created by sunny on 19/1/2020.
//

#include "nnie/mat.h"
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cmath>

namespace nnie{
    template<typename T>
    Mat<T>::Mat(int width,int height,int channel,DATA_TYPE data_type) {
        this->type = data_type;
        this->data = new T[width*height*channel];
        this->cols = width;
        this->rows = height;
        this->chns = channel;
        this->w = width;
        this->h = height;
        if(this->type == MAT_TYPE_U8_FROM_PACKAGE){
            this->type = MAT_TYPE_U8;
        }
        memory_size = sizeof(T)*chns*w*h;
    }
    template<typename T>
    int Mat<T>::channels() {
        return this->chns;
    }
    template<typename T>Mat<T>::~Mat() {
        //printf("Deinit mat %d %d\n",this->rows,this->cols);
        if(this->data != nullptr && !ext_data){
            //printf("Deinit mat~~ not null\n");
            delete[] this->data;
            //printf("Deinit mat~~ done\n");
            this->data = nullptr;
        }
    }

    template<typename T>
    Mat<T>::Mat(int width, int height, int channel, T *orig_data, DATA_TYPE data_type) {
        this->type = data_type;
        this->data = orig_data;
        this->cols = width;
        this->rows = height;
        this->chns = channel;
        this->w = width;
        this->h = height;
        this->ext_data = true;
        memory_size = sizeof(T)*chns*w*h;
    }

    template<typename T>
    T* Mat<T>::get_data_pointer() const{
        return this->data;
    }

    template<typename T>
    Mat<T>
    Mat<T>::create_mat_from_data(int width, int height, int channel, T *orig_data, DATA_TYPE data_type, int stride) {
        Mat<T> ret(width, height, channel, data_type);
        T* ptr = ret.get_data_pointer();
        if(data_type != MAT_TYPE_U8_FROM_PACKAGE){
            if(stride == -1 || stride == width){
                memcpy(ptr,orig_data,width*height*channel*sizeof(T));
            }else if(stride > width){
                for(int i=0;i<height*channel;i++){
                    memcpy(ptr+width*i,orig_data+stride*width,width*sizeof(T));
                }
            }
        }else{
            // if it is a packaged data pointer, we need to perform an o(w*h*c) operation, currently.
            if(stride == -1 || stride >= width){
                if(stride ==-1){
                    stride = width * channel;
                }
                for(int c=0;c<channel;c++){
                    for(int h=0;h<height;h++){
                        for(int w=0;w<width; w++){
                            ptr[width*height*c+h*width+w]=orig_data[stride*h+w*channel+c];
                        }
                    }
                }
            }
        }
        return ret;
    }

    template<typename T>
    T *Mat<T>::channel(int chn) {
        /*T* ptr = new T[this->rows*this->cols];
        //memset(ptr,0,512*sizeof(T));
        memcpy(ptr,this->data+(chn*rows*cols),this->rows*this->cols*sizeof(T));*/

        return this->data+(chn*rows*cols);
    }

    template<typename T>
    Mat<T>
    Mat<T>::create_mat_from_data_resize(int width, int height, int channel, int new_width, int new_height, T *orig_data,
                                        DATA_TYPE data_type, int stride, bool transpose,bool scale_lock) {
        Mat<T> ret(new_width, new_height, channel, data_type);
        float stride_w = (float)width / (float)new_width;
        float stride_h = (float)height / (float)new_height;
        if(transpose){
            stride_w = (float)width / (float)new_height;
            stride_h = (float)height / (float)new_width;
        }
        if(scale_lock){
            if((float)width/(float)height>(float)new_width/(float)new_height){
                stride_h = stride_w;
            }else{
                stride_w = stride_h;
            }
        }
        if(!transpose){
            if(stride == -1 || stride >= width){
                if(stride ==-1){
                    stride = width * channel;
                }
                int cur_h=0,cur_w=0;
                for(int c=0;c<channel;c++){
                    T* ptr=ret.channel(c);
                    for(int h=0;h<new_height;h++){
                        for(int w=0;w<new_width; w++){
                            cur_h = int((float)h*stride_h);
                            cur_w = int((float)w*stride_w);
                            if(cur_w>=width||cur_h>=height){
                                break;
                            }
                            ptr[w]=orig_data[stride*cur_h+channel*cur_w+c];
                        }
                        if(cur_h>=height){
                            break;
                        }
                        ptr+=new_width;
                    }
                }
            }
        }else{
            if(stride == -1 || stride >= width){
                if(stride ==-1){
                    stride = width * channel;
                }
                int cur_h=0,cur_w=0;
                for(int c=0;c<channel;c++){
                    T* ptr=ret.channel(c);
                    for(int h=0;h<new_height;h++){
                        for(int w=0;w<new_width; w++){
                            cur_h = int((float)w*stride_h);
                            cur_w = int((float)h*stride_w);
                            if(cur_h>=height || cur_w>=width){
                                break;
                            }
                            ptr[w]=orig_data[stride*cur_h+channel*cur_w+c];
                        }
                        if(cur_w>=width){
                            break;
                        }
                        ptr+=new_width;

                    }
                }
            }
        }

        return ret;
    }

    template<typename T>
    size_t Mat<T>::size() const{
        return memory_size;
    }

    // currently supported data types
    template class Mat<int>;
    template class Mat<float>;
    template class Mat<u_char>;
    //template class Mat<double>; not supported
}


