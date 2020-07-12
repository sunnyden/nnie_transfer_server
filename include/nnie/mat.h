//
// Created by sunny on 19/1/2020.
//

#ifndef HISILICON_MAT_H
#define HISILICON_MAT_H

#include <sys/param.h>

namespace nnie{
    typedef enum{
        MAT_TYPE_U8,
        MAT_TYPE_S32,
        MAT_TYPE_FLOAT,
        MAT_TYPE_U8_FROM_PACKAGE, // means BGR,BGR,BGR
    }DATA_TYPE;
    template <typename T>
    class Mat {
    public:
        int cols{},rows{},w{},h{};
        /**
         * Creates an empty
         * @param width
         * @param height
         * @param channel
         * @param data_type
         */
        Mat(int width,int height,int channel,DATA_TYPE data_type);

        /**
         *
         * Create Mat using existing data.
         * Input: width height channel, data pointer, data type
         * This constructor will NOT copy the data in the pointer, you need to handle the memory of the pointer by
         * yourself.
         *
         * @param width
         * @param height
         * @param channel
         * @param orig_data
         * @param data_type
         */
        Mat(int width,int height,int channel,T* orig_data,DATA_TYPE data_type);
        /**
         * Destructor, will release the data pointer memory space if it is not assigned from outer space.
         */
        ~Mat();

        T* get_data_pointer() const;
        int channels();
        /**
         * Create Mat from external data. Note that this function will copy the data from the specified pointer.
         * @param width
         * @param height
         * @param channel
         * @param orig_data
         * @param data_type
         * @param stride
         * @return
         */
        static Mat<T> create_mat_from_data(int width,int height,int channel,T* orig_data,DATA_TYPE data_type, int stride=-1);

        /**
         * Create Mat from external data. Note that this function will copy the data from the specified pointer.
         * @param width
         * @param height
         * @param channel
         * @param orig_data
         * @param data_type
         * @param stride
         * @return
         */
        static Mat<T> create_mat_from_data_resize(int width,int height,int channel,int new_width,int new_height,T* orig_data,DATA_TYPE data_type, int stride=-1,bool transpose = false,bool scale_lock=false);

        /**
         * Copy the channel data to the pointer.
         * You need to handle the returned pointer pointed memory space by yourself.
         * @param chn
         * @return pointer to the data.
         */
        T* channel(int chn);

        size_t size() const;
    private:
        T *data;
        int chns = 0;
        bool ext_data = false;
        T *squeeze_data = nullptr;
        size_t memory_size = 0;
        DATA_TYPE type{};
    };


}



#endif //HISILICON_MAT_H
