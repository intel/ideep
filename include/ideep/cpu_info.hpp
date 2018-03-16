/*
 *Copyright (c) 2018 Intel Corporation.
 *
 *Permission is hereby granted, free of charge, to any person obtaining a copy
 *of this software and associated documentation files (the "Software"), to deal
 *in the Software without restriction, including without limitation the rights
 *to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *copies of the Software, and to permit persons to whom the Software is
 *furnished to do so, subject to the following conditions:
 *
 *The above copyright notice and this permission notice shall be included in
 *all copies or substantial portions of the Software.
 *
 *THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 *THE SOFTWARE.
 *
 */


#ifndef _CPU_INFO_HPP_
#define _CPU_INFO_HPP_

#include <set>
#include <vector>
#include <thread>
#include <fstream>
#include <cstring>
#include <omp.h>
#include <sched.h>

namespace ideep {

struct Processor {
  Processor() : processor(0), physicalId(0),
    siblings(0), coreId(0), cpuCores(0),
    speedMHz(0) {}

  unsigned processor;
  unsigned physicalId;
  unsigned siblings;
  unsigned coreId;
  unsigned cpuCores;
  unsigned speedMHz;
};

class CpuInfoInterface {
public:
  virtual ~CpuInfoInterface() {}
  virtual const char *getFirstLine() = 0;
  virtual const char *getNextLine() = 0;
};

class CpuInfo : public CpuInfoInterface {
public:
  CpuInfo() {
    loadContentFromFile("/proc/cpuinfo");
  }

  explicit CpuInfo(const char *content) {
    loadContent(content);
  }

  ~CpuInfo() {
    delete [] fileContentBegin;
  }

  virtual const char *getFirstLine() {
    currentLine = fileContentBegin < fileContentEnd ? fileContentBegin : NULL;
    return getNextLine();
  }

  virtual const char *getNextLine() {
    if (!currentLine) {
      return NULL;
    }

    const char *savedCurrentLine = currentLine;
    while (*(currentLine++)) {
    }

    if (currentLine >= fileContentEnd) {
      currentLine = NULL;
    }

    return savedCurrentLine;
  }

private:
  const char *fileContentBegin;
  const char *fileContentEnd;
  const char *currentLine;

  void loadContentFromFile(const char *fileName) {
    std::ifstream file(fileName);
    std::string content(
            (std::istreambuf_iterator<char>(file)),
            (std::istreambuf_iterator<char>()));

    loadContent(content.c_str());
  }

  void loadContent(const char *content) {
    size_t contentLength = strlen(content);
    char *contentCopy = new char[contentLength + 1];
    snprintf(contentCopy, contentLength + 1, "%s", content);

    parseLines(contentCopy);

    fileContentBegin = contentCopy;
    fileContentEnd = &contentCopy[contentLength];
    currentLine = NULL;
  }

  void parseLines(char *content) {
    for (; *content; content++) {
      if (*content == '\n') {
        *content = '\0';
      }
    }
  }
};

class CollectionInterface {
public:
  virtual ~CollectionInterface() {}
  virtual unsigned getProcessorSpeedMHz() = 0;
  virtual unsigned getTotalNumberOfSockets() = 0;
  virtual unsigned getTotalNumberOfCpuCores() = 0;
  virtual unsigned getNumberOfProcessors() = 0;
  virtual const Processor &getProcessor(unsigned processorId) = 0;
};

class Collection : public CollectionInterface {
public:

  explicit Collection(CpuInfoInterface *cpuInfo) : cpuInfo(*cpuInfo) {
    totalNumberOfSockets = 0;
    totalNumberOfCpuCores = 0;
    currentProcessor = NULL;

    processors.reserve(96);

    parseCpuInfo();
    collectBasicCpuInformation();
  }

  virtual unsigned getProcessorSpeedMHz() {
    return processors.size() ? processors[0].speedMHz : 0;
  }

  virtual unsigned getTotalNumberOfSockets() {
    return totalNumberOfSockets;
  }

  virtual unsigned getTotalNumberOfCpuCores() {
    return totalNumberOfCpuCores;
  }

  virtual unsigned getNumberOfProcessors() {
    return processors.size();
  }

  virtual const Processor &getProcessor(unsigned processorId) {
    return processors[processorId];
  }

private:
  CpuInfoInterface &cpuInfo;
  unsigned totalNumberOfSockets;
  unsigned totalNumberOfCpuCores;
  std::vector<Processor> processors;
  Processor *currentProcessor;

  Collection(const Collection &collection);
  Collection &operator =(const Collection &collection);

  void parseCpuInfo() {
    const char *cpuInfoLine = cpuInfo.getFirstLine();
    for (; cpuInfoLine; cpuInfoLine = cpuInfo.getNextLine()) {
      parseCpuInfoLine(cpuInfoLine);
    }
  }

  void parseCpuInfoLine(const char *cpuInfoLine) {
    int delimiterPosition = strcspn(cpuInfoLine, ":");

    if (cpuInfoLine[delimiterPosition] == '\0') {
      currentProcessor = NULL;
    } else {
      parseValue(cpuInfoLine, &cpuInfoLine[delimiterPosition + 2]);
    }
  }

  void parseValue(const char *fieldName, const char *valueString) {
    if (!currentProcessor) {
      appendNewProcessor();
    }

    if (beginsWith(fieldName, "processor")) {
      currentProcessor->processor = parseInteger(valueString);
    }

    if (beginsWith(fieldName, "physical id")) {
      currentProcessor->physicalId = parseInteger(valueString);
    }

    if (beginsWith(fieldName, "siblings")) {
      currentProcessor->siblings = parseInteger(valueString);
    }

    if (beginsWith(fieldName, "core id")) {
      currentProcessor->coreId = parseInteger(valueString);
    }

    if (beginsWith(fieldName, "cpu cores")) {
      currentProcessor->cpuCores = parseInteger(valueString);
    }

    if (beginsWith(fieldName, "model name")) {
      currentProcessor->speedMHz = extractSpeedFromModelName(valueString);
    }
  }

  void appendNewProcessor() {
    processors.push_back(Processor());
    currentProcessor = &processors.back();
  }

  bool beginsWith(const char *lineBuffer, const char *text) const {
    while (*text) {
      if (*(lineBuffer++) != *(text++)) {
        return false;
      }
    }

    return true;
  }

  unsigned parseInteger(const char *text) const {
    return atol(text);
  }

  /* Function extracts CPU speed from model name. If unit is not set it is
     assumed that values below 100 are specified in GHz, otherwise MHz */
  unsigned extractSpeedFromModelName(const char *text) const {
    text = strstr(text, "@");
    if (!text) {
      return 0;
    }

    char *unit;
    double speed = strtod(&text[1], &unit);

    while (isspace(*unit)) {
      unit++;
    }

    bool isMHz = !strncmp(unit, "MHz", 3);
    bool isGHz = !strncmp(unit, "GHz", 3);
    bool isGHzPossible = (speed < 100);

    if (isGHz || (isGHzPossible && !isMHz)) {
      return 1000 * speed + 0.5;
    } else {
      return speed + 0.5;
    }
  }

  void collectBasicCpuInformation() {
    std::set<unsigned> uniquePhysicalId;
    std::vector<Processor>::iterator processor = processors.begin();
    for (; processor != processors.end(); processor++) {
      uniquePhysicalId.insert(processor->physicalId);
      updateCpuInformation(*processor, uniquePhysicalId.size());
    }
  }

  void updateCpuInformation(const Processor &processor,
        unsigned numberOfUniquePhysicalId) {
    if (totalNumberOfSockets == numberOfUniquePhysicalId) {
      return;
    }

    totalNumberOfSockets = numberOfUniquePhysicalId;
    totalNumberOfCpuCores += processor.cpuCores;
  }
};

/* The OpenMpManager class is responsible for determining a set of all of
   available CPU cores and delegating each core to perform other tasks. The
   first of available cores is delegated for background threads, while other
   remaining cores are dedicated for OpenMP threads. Each OpenMP thread owns
   one core for exclusive use. The number of OpenMP threads is then limited
   to the number of available cores minus one. The amount of CPU cores may
   be limited by system eg. when numactl was used. */

static const char *openMpEnvVars[] = {
  "OMP_CANCELLATION", "OMP_DISPLAY_ENV", "OMP_DEFAULT_DEVICE", "OMP_DYNAMIC",
  "OMP_MAX_ACTIVE_LEVELS", "OMP_MAX_TASK_PRIORITY", "OMP_NESTED",
  "OMP_NUM_THREADS", "OMP_PROC_BIND", "OMP_PLACES", "OMP_STACKSIZE",
  "OMP_SCHEDULE", "OMP_THREAD_LIMIT", "OMP_WAIT_POLICY", "GOMP_CPU_AFFINITY",
  "GOMP_DEBUG", "GOMP_STACKSIZE", "GOMP_SPINCOUNT", "GOMP_RTEMS_THREAD_POOLS",
  "KMP_AFFINITY", "KMP_NUM_THREADS", "MIC_KMP_AFFINITY",
  "MIC_OMP_NUM_THREADS", "MIC_OMP_PROC_BIND", "PHI_KMP_AFFINITY",
  "PHI_OMP_NUM_THREADS", "PHI_KMP_PLACE_THREADS", "MKL_NUM_THREADS",
  "MKL_DYNAMIC", "MKL_DOMAIN_NUM_THREADS"
};

class OpenMpManager {
public:
  static void setGpuEnabled() {
    OpenMpManager &openMpManager = get_instance();
    openMpManager.isGpuEnabled = true;
  }

  static void setGpuDisabled() {
    OpenMpManager &openMpManager = get_instance();
    openMpManager.isGpuEnabled = false;
  }

  //static void printVerboseInformation();

  static unsigned getProcessorSpeedMHz() {
    OpenMpManager &openMpManager = get_instance();
    return openMpManager.collection.getProcessorSpeedMHz();
  }

  static bool isMajorThread(void) {
    OpenMpManager &openMpManager = get_instance();
    return (std::this_thread::get_id() == openMpManager.mainThreadId);
  }

  void bindOpenMpThreads() {
    OpenMpManager &openMpManager = get_instance();

    if (!openMpManager.isThreadsBindAllowed())
      return;

    openMpManager.setOpenMpThreadNumberLimit();
    #pragma omp parallel
    {
      unsigned logicalCoreId = omp_get_thread_num();
      openMpManager.bindCurrentThreadToLogicalCoreCpu(logicalCoreId);
    }
  }

  // Ideally bind given thread to secondary logical core, if
  // only one thread exists then bind to primary one
  static void bindCurrentThreadToNonPrimaryCoreIfPossible() {
    OpenMpManager &openMpManager = get_instance();
    if (openMpManager.isThreadsBindAllowed()) {
      int totalNumberOfAvailableCores = CPU_COUNT(&openMpManager.currentCoreSet);
      int logicalCoreToBindTo = totalNumberOfAvailableCores > 1 ? 1 : 0;
      openMpManager.bindCurrentThreadToLogicalCoreCpus(logicalCoreToBindTo);
    }
  }

private:
  std::thread::id mainThreadId;
  Collection &collection;

  bool isGpuEnabled;
  bool isAnyOpenMpEnvVarSpecified;
  cpu_set_t currentCpuSet;
  cpu_set_t currentCoreSet;

  explicit OpenMpManager(Collection *collection) :
    mainThreadId(std::this_thread::get_id()),
    collection(*collection) {
    getOpenMpEnvVars();
    getCurrentCpuSet();
    getCurrentCoreSet();
  }

  OpenMpManager(const OpenMpManager &openMpManager);
  OpenMpManager &operator =(const OpenMpManager &openMpManager);

  static OpenMpManager &get_instance() {
    static CpuInfo cpuInfo;
    static Collection collection(&cpuInfo);
    static OpenMpManager openMpManager(&collection);
    return openMpManager;
  }

  void getOpenMpEnvVars() {
    static const unsigned numberOfOpenMpEnvVars =
      sizeof(openMpEnvVars) / sizeof(openMpEnvVars[0]);

    isAnyOpenMpEnvVarSpecified = false;
    for (unsigned i = 0; i < numberOfOpenMpEnvVars; i++) {
      if (getenv(openMpEnvVars[i])) {
        isAnyOpenMpEnvVarSpecified = true;
      }
    }
  }

  void getCurrentCpuSet() {
    if (sched_getaffinity(0, sizeof(currentCpuSet), &currentCpuSet)) {
      getDefaultCpuSet(&currentCpuSet);
    }
  }

  void getDefaultCpuSet(cpu_set_t *defaultCpuSet) {
    CPU_ZERO(defaultCpuSet);
    unsigned numberOfProcessors = collection.getNumberOfProcessors();
    for (unsigned processorId = 0; processorId < numberOfProcessors; processorId++) {
      CPU_SET(processorId, defaultCpuSet);
    }
  }

  /* Function getCurrentCoreSet() fills currentCoreSet variable with a set of
     available CPUs, where only one CPU per core is chosen. When multiple CPUs
     of single core are used, function is selecting only first one of all
     available. */
  void getCurrentCoreSet() {
    unsigned numberOfProcessors = collection.getNumberOfProcessors();
    unsigned totalNumberOfCpuCores = collection.getTotalNumberOfCpuCores();

    cpu_set_t usedCoreSet;
    CPU_ZERO(&usedCoreSet);
    CPU_ZERO(&currentCoreSet);

    for (unsigned processorId = 0; processorId < numberOfProcessors; processorId++) {
      if (CPU_ISSET(processorId, &currentCpuSet)) {
        unsigned coreId = processorId % totalNumberOfCpuCores;
        if (!CPU_ISSET(coreId, &usedCoreSet)) {
          CPU_SET(coreId, &usedCoreSet);
          CPU_SET(processorId, &currentCoreSet);
        }
      }
    }
  }

  void selectAllCoreCpus(cpu_set_t *set, unsigned physicalCoreId) {
    unsigned numberOfProcessors = collection.getNumberOfProcessors();
    unsigned totalNumberOfCpuCores = collection.getTotalNumberOfCpuCores();

    unsigned processorId = physicalCoreId % totalNumberOfCpuCores;
    while (processorId < numberOfProcessors) {
      if (CPU_ISSET(processorId, &currentCpuSet)) {
        CPU_SET(processorId, set);
      }

      processorId += totalNumberOfCpuCores;
    }
  }

  unsigned getPhysicalCoreId(unsigned logicalCoreId) {
    unsigned numberOfProcessors = collection.getNumberOfProcessors();

    for (unsigned processorId = 0; processorId < numberOfProcessors; processorId++) {
      if (CPU_ISSET(processorId, &currentCoreSet)) {
        if (!logicalCoreId--) {
          return processorId;
        }
      }
    }

    return 0;
  }

  bool isThreadsBindAllowed() {
    return !isAnyOpenMpEnvVarSpecified && !isGpuEnabled;
  }

  // Limit of threads to number of logical cores available
  void setOpenMpThreadNumberLimit() {
    omp_set_num_threads(CPU_COUNT(&currentCoreSet));
  }

  void bindCurrentThreadToLogicalCoreCpu(unsigned logicalCoreId) {
    unsigned physicalCoreId = getPhysicalCoreId(logicalCoreId);
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(physicalCoreId, &set);
    sched_setaffinity(0, sizeof(set), &set);
  }

  void bindCurrentThreadToLogicalCoreCpus(unsigned logicalCoreId) {
    unsigned physicalCoreId = getPhysicalCoreId(logicalCoreId);

    cpu_set_t set;
    CPU_ZERO(&set);
    selectAllCoreCpus(&set, physicalCoreId);
    sched_setaffinity(0, sizeof(set), &set);
  }
};

}
#endif
