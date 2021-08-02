#ifndef CUSTOMIZED_TERMINATE_HPP
#define CUSTOMIZED_TERMINATE_HPP

// https://stackoverflow.com/questions/2443135/how-do-i-find-where-an-exception-was-thrown-in-c
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#ifndef __USE_GNU
#define __USE_GNU
#endif

#include <execinfo.h>
#include <signal.h>

void customized_terminate(void);

// This structure mirrors the one found in /usr/include/asm/ucontext.h
typedef struct _sig_ucontext {
   unsigned long     uc_flags;
   struct ucontext   *uc_link;
   stack_t           uc_stack;
   struct sigcontext uc_mcontext;
   sigset_t          uc_sigmask;
} sig_ucontext_t;

void crit_err_hdlr(int sig_num, siginfo_t * info, void * ucontext);

void customized_terminate();

#endif // CUSTOMIZED_TERMINATE_HPP
