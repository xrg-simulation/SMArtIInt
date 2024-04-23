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

# in case we run on a windows machine we do not have rsync - therefore use find in combination with mkdir and cp to copy the required files
find ./SMArtIInt/* -type d ! \( -path "*venv*" -or -path "*idea*" -or -path *_log* -or -name data_*.pckl \)  -exec mkdir -p ./_deploy/{} \;
find ./SMArtIInt/* -type f ! \( -path "*venv*" -or -path "*idea*" -or -path *_log* -or -name data_*.pckl \)  -exec cp {} ./_deploy/{} \;

sed -i -e "s/%build%/$build/g" ./_deploy/SMArtIInt/package.mo
sed -i -e "s/%version%/$version/g" ./_deploy/SMArtIInt/package.mo
sed -i -e "s/%date%/$date/g" ./_deploy/SMArtIInt/package.mo

#clean up
rm ./_deploy/SMArtIInt/Resources/Library/win64/*
rm ./_deploy/SMArtIInt/Resources/Library/win64/*

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

# copy additional required libs
cp ./SMArtIInt/Resources/Library/win64/tensorflowlite_c.dll ./_deploy/SMArtIInt/Resources/Library/win64/
cp ./SMArtIInt/Resources/Library/linux64/libtensorflowlite_c.so ./_deploy/SMArtIInt/Resources/Library/linux64/

# pack everything
cp ./README.md ./_deploy/
cp LICENSE ./_deploy/
cd _deploy/
# clean the zip file first
zip -d ../SMArtIInt_$version"_"$build.zip
zip -r ../SMArtIInt_$version"_"$build.zip ./SMArtIInt README.md LICENSE

echo "Press any key to continue..."
# -s: Do not echo input coming from a terminal
# -n 1: Read one character
read -s -n 1