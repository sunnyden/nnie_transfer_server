//
// Created by 邓昊晴 on 9/7/2020.
//

#ifndef NNIE_TRANSFER_SERVER_MODELMANAGER_H
#define NNIE_TRANSFER_SERVER_MODELMANAGER_H

#include "nnie/net.h"
class ModelManager {
public:
    ModelManager();
    nnie::Net* access(int fd);
    void remove(int fd);
    void create(nnie::Net* net,int fd);
    int getTimeout() const;
    void setTimeout(int timeout);
    ~ModelManager();
    void optimize();
private:
    int model_index{};
    std::map<int,std::pair<nnie::Net*,long>> model_list;
    int timeout = 300;
    pthread_mutex_t index_lock{};
};


#endif //NNIE_TRANSFER_SERVER_MODELMANAGER_H
