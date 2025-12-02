#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

cd ramulator2

cp -f ../src/main.cpp                       src/
cp -f ../src/base.h                         src/base/
cp -f ../src/request.h                      src/base/
cp -f ../src/AgenX_NDP_linear_mappers.cpp   src/addr_mapper/impl/
cp -f ../src/AgentXNDP.cpp                  src/dram/impl/
cp -f ../src/controller.h                   src/dram_controller/
cp -f ../src/AgentX_NDP_controller.cpp      src/dram_controller/impl/
cp -f ../src/AgentX_Trace_Recorder.cpp      src/dram_controller/impl/plugin/
cp -f ../src/AgentX_All_Bank_refresh.cpp    src/dram_controller/impl/refresh/
cp -f ../src/no_refresh.cpp                 src/dram_controller/impl/refresh/
cp -f ../src/NDP_scheduler.cpp              src/dram_controller/impl/scheduler/
cp -f ../src/NDP_loadstore_trace.cpp        src/frontend/impl/memory_trace/
cp -f ../src/memory_system.h                src/memory_system/
cp -f ../src/AgentX_NDP_system.cpp          src/memory_system/impl/

echo "All *cpp *h templates applied."

cd ..

SRC_DIR="src/CMakeLists"

for f in "$SRC_DIR"/*__CMakeLists.txt; do
    filename=$(basename "$f")

    core=${filename%__CMakeLists.txt}

    rel_dir=${core//__//}

    cp -f "$f" "$SRC_DIR/CMakeLists.txt"

    cp -f "$SRC_DIR/CMakeLists.txt" "$rel_dir/CMakeLists.txt"

    rm -f "$SRC_DIR/CMakeLists.txt"

    echo "Applied $filename → ramulator2/$rel_dir/CMakeLists.txt"
done

echo "All CMakeLists templates applied."