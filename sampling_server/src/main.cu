#include "server.h"
#include <stdlib.h>
#include <iostream>

int main(int argc, char** argv){

    std::cout<<"Start Sampling Server\n";
    Server* server = NewGPUServer();
    std::vector<int> fanout;
    fanout.push_back(25);
    fanout.push_back(10);
    server->Initialize(atoi(argv[1]), fanout);//gpu number, default 1; in memory, default true
    server->PreSc();   
    if (atoi(argv[2])){//if 0, profiling only; else, running main loop
        server->Run();
        server->Finalize();
    }
}