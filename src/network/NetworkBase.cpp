//
// Created by 邓昊晴 on 8/7/2020.
//


#include "network/NetworkBase.h"

NetworkBase::NetworkBase(int port) {
    bool is_error = false;
    listenFd = socket(AF_INET, SOCK_STREAM, 0);
    if(listenFd == -1){
        printf("create socket error: %s(errno: %d)\n",strerror(errno),errno);
        is_error = true;
    }
    memset(&srvAddr, 0, sizeof(srvAddr));
    srvAddr.sin_family = AF_INET;
    srvAddr.sin_addr.s_addr = htonl(INADDR_ANY);
    srvAddr.sin_port = htons(port);
    if(bind(listenFd, (struct sockaddr*)&srvAddr, sizeof(srvAddr)) == -1){
        printf("bind socket error: %s(errno: %d)\n",strerror(errno),errno);
        is_error = true;
    }
    if(listen(listenFd, 10) == -1){
        printf("listen socket error: %s(errno: %d)\n",strerror(errno),errno);
        is_error = true;
    }
    assert(!is_error);
}

NetworkBase::~NetworkBase() {
    if(is_running){
        is_running = false;
        close(listenFd);
        pthread_join(working_thread, nullptr);
    }

}




