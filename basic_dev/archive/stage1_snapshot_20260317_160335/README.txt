Stage1 snapshot archived on 2026-03-17 16:03:35 CST.

Purpose:
- Preserve the current basic_dev implementation before manual PX4 avoidance integration.

Captured files:
- basic_dev/src/basic_dev/src/basic_dev.cpp
- basic_dev/src/basic_dev/include/basic_dev.hpp
- basic_dev/src/basic_dev/CMakeLists.txt
- basic_dev/src/basic_dev/package.xml
- basic_dev/Dockerfile
- basic_dev/setup.bash

Restore approach:
- Copy files back from this snapshot into the working tree.
- Or diff against this snapshot while integrating PX4 logic.
