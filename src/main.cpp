#include <nnie/ModelManager.h>
#include "network/NetworkBase.h"
void protocol_process(int operation_code, void *data, long len, int fd) {

}

int main(int argc,char** argv) {
    HI_MPI_SYS_Init();
    std::cout << "Srv Init"<< std::endl;
    int port = argc > 1?atoi(argv[1]):7777;
    NetworkBase server(port);
    ModelManager manager;
    server.register_callback(0,[&](void *data, long len, int fd){
        auto model = new nnie::Net(data,len);
        int package_size = 12+(model->input_layer.size() + model->output_layer.size())* sizeof(NNIE_LAYER_EXPORT);
        int struct_size = sizeof(NNIE_LAYER_EXPORT);
        u_char header[12];
        memcpy(header,server.header,4);
        memcpy(header+4,&package_size, sizeof(int));
        memcpy(header+8,&struct_size, sizeof(int));
        send(fd,header, 12,0);
        for(auto& layer:model->input_layer){
            NNIE_LAYER_EXPORT layerInfo;
            layerInfo.is_input = 1;
            layerInfo.channel = layer.second.channel;
            layerInfo.height = layer.second.height;
            layerInfo.width = layer.second.width;
            layerInfo.layer_id = layer.second.layer_id;
            layerInfo.segment_id = layer.second.segment_id;
            memset(layerInfo.name,0, sizeof(layerInfo.name));
            memcpy(layerInfo.name,layer.first.data(),layer.first.size());
            for(int i = 0;i<model->input_layer_vec.size();i++){
                if(model->input_layer_vec[i].layer_id == layerInfo.layer_id && model->input_layer_vec[i].segment_id == layerInfo.segment_id){
                    layerInfo.id = i;
                }
            }
            send(fd,&layerInfo, sizeof(layerInfo),0);
        }
        for(auto& layer:model->output_layer){
            NNIE_LAYER_EXPORT layerInfo;
            layerInfo.is_input = 0;
            layerInfo.channel = layer.second.channel;
            layerInfo.height = layer.second.height;
            layerInfo.width = layer.second.width;
            layerInfo.layer_id = layer.second.layer_id;
            layerInfo.segment_id = layer.second.segment_id;
            memset(layerInfo.name,0, sizeof(layerInfo.name));
            memcpy(layerInfo.name,layer.first.data(),layer.first.size());
            for(int i = 0;i<model->output_layer_vec.size();i++){
                if(model->output_layer_vec[i].layer_id == layerInfo.layer_id && model->output_layer_vec[i].segment_id == layerInfo.segment_id){
                    layerInfo.id = i;
                }
            }
            send(fd,&layerInfo, sizeof(layerInfo),0);
        }
        manager.create(model,fd);
    });

    server.register_callback(1,[&](void *data, long len, int fd){
        // four stage pipeline
        // parse
        auto model = manager.access(fd);
        int blob_count = *(int*)(data);
        int offset = sizeof(int);
        for(int i=0;i<blob_count;i++){
            int blob_id = *(int*)((u_char*)data+offset);
            int length = *(int*)((u_char*)data+offset+4);
            offset+=8;
            nnie::Mat<u_char> blob(model->input_layer_vec[blob_id].width,model->input_layer_vec[blob_id].height,model->input_layer_vec[blob_id].channel,nnie::MAT_TYPE_U8);
            if(length == blob.size()){
                memcpy(blob.get_data_pointer(),(u_char*)data+offset,length);
            }
            model->set_input(blob_id,blob);
            offset+=length;
        }
        // forward
        int forward_id = *(int*)((u_char*)data+offset);
        model->forward(forward_id);

        // extract and build packet
        offset=12;
        int output_blob_count = model->output_layer_vec.size();
        auto *result = static_cast<u_char *>(malloc(offset));
        memcpy(result,server.header,4);
        memcpy(result+8,&output_blob_count,4);

        for(int i=0;i<output_blob_count;i++){
            auto blob = model->extract(i);
            result = static_cast<u_char *>(realloc(result, offset + 8 + blob.size()));
            *(int*)((u_char*)result+offset)=i;
            *(int*)((u_char*)result+offset+4)=blob.size();
            offset+=8;
            memcpy(result+offset,blob.get_data_pointer(),blob.size());
            offset+=blob.size();
        }
        memcpy(result+4,&offset,4);

        // send
        int length = offset;
        const int buf_size = 524288;
        for(int i=0;i<length;i+=buf_size){
            size_t packet_len = std::min(length-i,buf_size);
            send(fd,result+i,packet_len,0);
        }
        free(result);
    });

    server.setOnDisconnectListener([&](int fd){
        printf("disconnected\n");
        manager.remove(fd);
    });
    server.start();
    getchar();
    HI_MPI_SYS_Exit();
    return 0;
}
