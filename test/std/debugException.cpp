
#include "../customized_terminate.hpp"
#include <gtest/gtest.h>

namespace {
    // invoke set_terminate as part of global constant initialization
    static const bool SET_TERMINATE = std::set_terminate(customized_terminate);
}

int throw_exception() {
    // throw an unhandled runtime error
    throw std::runtime_error("RUNTIME ERROR!");
    return 0;
}

int foo2() {
    throw_exception();
    return 0;
}

int foo1() {
    foo2();
    return 0;
}

int main(int argc, char ** argv) {
    struct sigaction sigact;

    sigact.sa_sigaction = crit_err_hdlr;
    sigact.sa_flags = SA_RESTART | SA_SIGINFO;

    if (sigaction(SIGABRT, &sigact, (struct sigaction *)NULL) != 0) {
        std::cerr << "error setting handler for signal " << SIGABRT
                  << " (" << strsignal(SIGABRT) << ")\n";
        exit(EXIT_FAILURE);
    }

    foo1();

    exit(EXIT_SUCCESS);
}
