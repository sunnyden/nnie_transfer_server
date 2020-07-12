//
// Created by 邓昊晴 on 8/7/2020.
//

#ifndef NNIE_TRANSPORT_SERVER_NETWORKBASE_H
#define NNIE_TRANSPORT_SERVER_NETWORKBASE_H

#include <cerrno>
#include <cstdio>
#include <cassert>
#include <cstring>
#include <iostream>
#include <exception>
#include <pthread.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <vector>
#include <functional>
#include <map>

class NetworkBase {
public:
    explicit NetworkBase(int port);
    void start();
    //virtual void protocol_process(int operation_code, void* data, long len, int fd);
    void register_callback(int operation_code, const std::function<void(void *data, long len, int fd)>& callback);
    void setOnDisconnectListener(const std::function<void(int fd)>& listener);
    void stop();
    unsigned char header[4] = {0x30,0x00,0xFA,0xCA};
    ~NetworkBase();
private:
    typedef struct{
        NetworkBase* server;
        int fd;
        pthread_t thread;
    }Connection;
    sockaddr_in srvAddr{};
    int listenFd;
    bool is_running = false;
    pthread_t working_thread{};
    std::map<int,std::function<void(void*,long,int)>>callbacks;
    static void* service(NetworkBase* transport_server);
    static void* request_processor(Connection* connection);
    std::function<void(int fd)> onDisconnectCallback = [](int){};
};



#endif //NNIE_TRANSPORT_SERVER_NETWORKBASE_H
