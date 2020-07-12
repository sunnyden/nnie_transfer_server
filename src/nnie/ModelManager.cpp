//
// Created by 邓昊晴 on 9/7/2020.
//

#include <ctime>
#include <pthread.h>
#include "nnie/ModelManager.h"

nnie::Net *ModelManager::access(int fd) {
    try{
        this->model_list.at(fd).second = std::time(nullptr);
        return this->model_list.at(fd).first;
    }catch(std::exception& ex){
        return nullptr;
    }
}

void ModelManager::remove(int fd) {
    try{
        delete this->model_list.at(fd).first;
        this->model_list.erase(fd);
    }catch (std::exception& exception){

    }
}

void ModelManager::create(nnie::Net *net,int fd) {
    remove(fd);
    model_list.emplace(fd,std::make_pair(net,std::time(nullptr)));
}

ModelManager::ModelManager() {
    pthread_mutex_init(&index_lock, nullptr);
}

ModelManager::~ModelManager() {
    pthread_mutex_destroy(&index_lock);
}

void ModelManager::optimize() {
    long current_time = std::time(nullptr);
    for(auto& model:model_list){
        if(current_time - model.second.second > timeout && timeout != -1){
            delete model.second.first;
            model_list.erase(model.first);
        }
    }
}

int ModelManager::getTimeout() const {
    return timeout;
}

void ModelManager::setTimeout(int timeout) {
    ModelManager::timeout = timeout;
}
