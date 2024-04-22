#!\bin\bash

version=$(git describe --tags --abbrev=0)
# get rid of the leading v
version=${version:1}
# store the build hash
build=$(git rev-parse --short HEAD)
# store the date
date=$(git show -s --format=%ci)

rm -r _deploy/
mkdir _deploy/

Robocopy.exe SMArtIInt ./_deploy/SMArtIInt //e //xd ./SMArtIInt/Resources/ExampleNeuralNets/venv/ ./SMArtIInt/Resources/ExampleNeuralNets/NNHeatTransfer/ht_log ./SMArtIInt/Resources/ExampleNeuralNets/NNHeatTransfer/large* ./SMArtIInt/Resources/ExampleNeuralNets/NNHeatTransfer/small* ./SMArtIInt/Resources/ExampleNeuralNets/.idea

sed -i -e "s/%build%/$build/g" ./_deploy/SMArtIInt/package.mo
sed -i -e "s/%version%/$version/g" ./_deploy/SMArtIInt/package.mo
sed -i -e "s/%date%/$date/g" ./_deploy/SMArtIInt/package.mo

#clean up
rm ./_deploy/SMArtIInt/Resources/Library/win64/*
cp ./SMArtIInt/Resources/Library/win64/tensorflowlite_c.dll ./_deploy/SMArtIInt/Resources/Library/win64/
rm ./_deploy/SMArtIInt/Resources/Library/win64/*
cp ./SMArtIInt/Resources/Library/linux64/tensorflowlite_c.so ./_deploy/SMArtIInt/Resources/Library/linux64/

# Windows build
#cmake -S ./CSource -B ./_deploy/_cmake/ -G "MinGW Makefiles"
cmake -S ./CSource -B ./_deploy/_cmake/ -DCMAKE_BUILD_TYPE=Release
cd ./_deploy/_cmake
cmake --build . --config Release
cd ../../

# Linux build
wsl cmake -S ./CSource -B ./_deploy/_cmake_wsl/ -DCMAKE_BUILD_TYPE=Release
cd ./_deploy/_cmake_wsl
wsl cmake --build . --config Release
cd ../../


# pack everything
cp ./README.md ./_deploy/
cp LICENSE ./_deploy/
cd _deploy/
zip -r ../SMArtIInt_$version"_"$build.zip ./SMArtIInt/

