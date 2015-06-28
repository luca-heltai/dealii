#/bin/sh -f
 
# things to do for travis-ci in the before_install section
 
if ( test "`uname -s`" = "Darwin" )
then
  #cmake v2.8.12 is installed on the Mac workers now
  #brew update
  #brew install cmake
  echo
else
  #install a newer cmake since at this time Travis only has version 2.8.7
  sudo apt-get install build-essential
  echo "yes" | sudo add-apt-repository ppa:kalakris/cmake
  sudo add-apt-repository -y ppa:saiarcot895/chromium-beta
  sudo apt-get update -qq
  sudo apt-get install cmake
  sudo apt-get install ninja-build

  # Trilinos
  sudo add-apt-repository -y ppa:nschloe/trilinos-nightly
  sudo add-apt-repository -y ppa:nschloe/netcdf-nightly
  sudo apt-get update -qq
  sudo apt-get install libopenmpi-dev openmpi-bin
  # The -dev versions are definitely not needed since Trilinos doesn't expose
  # any of the interface of libptscotch. This can be fixed by the PRIVATE
  # keyword in the appropriate target_link_libraries() in Trilinos.
  sudo apt-get install libptscotch-dev libmumps-dev libboost-program-options-dev
  sudo apt-get install libboost-system-dev liblapack-dev binutils-dev
  sudo apt-get install libhdf5-openmpi-7 libhdf5-openmpi-dev
  sudo apt-get install libnetcdf-dev
  sudo apt-get install libboost-dev libtbb-dev # should be added by Trilinos
  sudo apt-get install cmake-data cmake
  sudo apt-get install trilinos-dev
fi
