/*
 Copyright (c) 2015 Siyu Lei, Silviu Maniu, Luyi Mo (University of Hong Kong)

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE.
 */

#ifndef oim_common_h
#define oim_common_h

#include <random>
#include <sys/time.h>
#include <memory>
#include <unistd.h>
#include <ios>
#include <iostream>
#include <fstream>
#include <string>

#include "InfluenceDistribution.h"
#include "SingleInfluence.h"


#define THETA_OFFSET 5
#define MAX_R 10000000
#define NUM_THREADS_MAX 4

extern double sampling_time;
extern double choosing_time;
extern double reused_ratio;

typedef long long int64;

typedef struct {
  unsigned long source;
  unsigned long target;
  unsigned int trial;
} TrialType;

/**
  Builds a seed using nanoseconds to avoid same results.
*/
int seed_ns() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (int)ts.tv_nsec;
}

typedef unsigned long long timestamp_t;

/**
  Returns number of microseconds since the Epoch.
*/
static timestamp_t get_timestamp() {
  struct timeval now;
  gettimeofday(&now, NULL);
  return  now.tv_usec + (timestamp_t)now.tv_sec * 1000000;
}

double sqr(double t) {
  return t * t;
}

void process_mem_usage(double &vm_usage, double &resident_set) {
   using std::ios_base;
   using std::ifstream;
   using std::string;

   vm_usage     = 0.0;
   resident_set = 0.0;

   // 'file' stat seems to give the most reliable results
   ifstream stat_stream("/proc/self/stat", ios_base::in);

   // dummy vars for leading entries in stat that we don't care about
   string pid, comm, state, ppid, pgrp, session, tty_nr;
   string tpgid, flags, minflt, cminflt, majflt, cmajflt;
   string utime, stime, cutime, cstime, priority, nice;
   string O, itrealvalue, starttime;

   // the two fields we want
   unsigned long vsize;
   long rss;

   stat_stream >> pid >> comm >> state >> ppid >> pgrp >> session >> tty_nr
               >> tpgid >> flags >> minflt >> cminflt >> majflt >> cmajflt
               >> utime >> stime >> cutime >> cstime >> priority >> nice
               >> O >> itrealvalue >> starttime >> vsize >> rss; // don't care about the rest

   stat_stream.close();

   long page_size_kb = sysconf(_SC_PAGE_SIZE) / 1024; // in case x86-64 is configured to use 2MB pages
   vm_usage     = vsize / 1024.0;
   resident_set = rss * page_size_kb;
}

double disp_mem_usage() {
  double vm, rss;
  process_mem_usage(vm, rss);
  vm /= 1024;
  rss /= 1024;
  return rss;
}

#endif
