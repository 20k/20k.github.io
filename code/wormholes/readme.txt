To build, add ../common to the include search paths. You'll also need to link against ../common/toolkit/clock.cpp, ../common/toolkit/fs_helpers.cpp, and ../common/toolkit/opencl.cpp, ./accretion_disk.cpp, ./blackbody.cpp

Link libraries are:

-lsfml-window
-lsfml-system
-lsfml-graphics
-lglew32
-lopengl32
-lopencl