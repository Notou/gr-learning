INCLUDE(FindPkgConfig)
PKG_CHECK_MODULES(PC_LEARNING learning)

FIND_PATH(
    LEARNING_INCLUDE_DIRS
    NAMES learning/api.h
    HINTS $ENV{LEARNING_DIR}/include
        ${PC_LEARNING_INCLUDEDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/include
          /usr/local/include
          /usr/include
)

FIND_LIBRARY(
    LEARNING_LIBRARIES
    NAMES gnuradio-learning
    HINTS $ENV{LEARNING_DIR}/lib
        ${PC_LEARNING_LIBDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/lib
          ${CMAKE_INSTALL_PREFIX}/lib64
          /usr/local/lib
          /usr/local/lib64
          /usr/lib
          /usr/lib64
)

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(LEARNING DEFAULT_MSG LEARNING_LIBRARIES LEARNING_INCLUDE_DIRS)
MARK_AS_ADVANCED(LEARNING_LIBRARIES LEARNING_INCLUDE_DIRS)

