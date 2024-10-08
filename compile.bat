
rem Windows build
cmake -S ./CSource -B ./_cmake/ -DCMAKE_BUILD_TYPE=Release
cd ./_cmake
cmake --build . --config Release
cd ../

rem Linux build
wsl cmake -S ./CSource -B ./_cmake_wsl/ -DCMAKE_BUILD_TYPE=Release
cd ./_cmake_wsl
wsl cmake --build . --config Release
cd ../
