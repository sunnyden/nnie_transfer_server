//
// Created by 邓昊晴 on 8/7/2020.
//
#include <vector>
#include <ctime>
#include "network/NetworkBase.h"
#define BUF_SIZE 2048
void *NetworkBase::service(NetworkBase *transport_server) {
    std::vector<pthread_t> thread_pool;
    while(transport_server->is_running){
        auto connectFd = accept(transport_server->listenFd, nullptr, nullptr);
        if(connectFd == -1){
            continue;
        }
        auto *connection = new Connection;
        connection->server = transport_server;
        connection->fd = connectFd;
        pthread_create(&connection->thread, nullptr, reinterpret_cast<void *(*)(void *)>(request_processor), (void*)connection);
        thread_pool.push_back(connection->thread);
    }
    for(auto& thread:thread_pool){
        pthread_join(thread, nullptr);
    }
    return nullptr;
}

void *NetworkBase::request_processor(NetworkBase::Connection *connection) {
    long offset = 0;
    auto *buffer = static_cast<u_char *>(malloc(BUF_SIZE));
    long begin_timestamp = std::time(nullptr);
    while(connection->server->is_running){
        auto count = recv(connection->fd, buffer+offset, BUF_SIZE - (offset % BUF_SIZE), MSG_DONTWAIT);
        if(count > 0 || offset > 0) {
            offset += std::max(count,0);
            if (offset % BUF_SIZE == 0) {
                buffer = static_cast<u_char *>(realloc(buffer, offset + BUF_SIZE));
            }
            bool valid_data = true;
            for(int i = 0;i<4;i++){
                if(buffer[i] != connection->server->header[i]){
                    valid_data = false;
                }
            }

            if(!valid_data){
                offset = 0;
                free(buffer);
                buffer = static_cast<u_char *>(malloc(BUF_SIZE));
                continue;
            }
            //printf("Offset=%ld \n",offset);
            if(offset>8){
                unsigned int package_length;
                memcpy(&package_length,buffer+4, sizeof(unsigned int));
                //printf("Package Length=%u \n",package_length);
                if(offset >= package_length){
                    // process data
                    int operation_code = -1;
                    memcpy(&operation_code,buffer+8, sizeof(unsigned int));
                    //printf("OP Code:%d Callbacks:%lu\n",operation_code,connection->server->callbacks.size());
                    //connection->server->protocol_process(operation_code,buffer+8,offset-8,connection->fd);
                    for(auto &callback : connection->server->callbacks){
                        if(callback.first == operation_code){
                            callback.second(buffer+12,package_length-12,connection->fd);
                        }
                    }

                    auto new_buffer = static_cast<u_char *>(malloc(offset - package_length + BUF_SIZE));
                    if(offset>package_length){
                        memcpy(new_buffer,buffer+package_length,offset-package_length);
                    }
                    offset -= package_length;
                    free(buffer);
                    buffer = new_buffer;
                }
            }

            begin_timestamp = std::time(nullptr);
        }else{
            long end_timestamp = std::time(nullptr);
            if(end_timestamp - begin_timestamp > 10){
                break;
            }
        }
    }
    connection->server->onDisconnectCallback(connection->fd);
    close(connection->fd);
    free(buffer);
    return nullptr;
}

void NetworkBase::start() {
    is_running = true;
    pthread_create(&working_thread, nullptr, reinterpret_cast<void *(*)(void *)>(service), (void*)this);
}

void NetworkBase::stop() {
    is_running = false;
    close(listenFd);
    pthread_join(working_thread, nullptr);
}

void NetworkBase::register_callback(int operation_code, const std::function<void(void*,long,int)>& callback) {
    this->callbacks.emplace(operation_code,callback);
}

void NetworkBase::setOnDisconnectListener(const std::function<void(int fd)> &listener) {
    this->onDisconnectCallback = listener;
}



