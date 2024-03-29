find_package(PkgConfig)

PKG_CHECK_MODULES(PC_GR_learning gnuradio-learning)

FIND_PATH(
    GR_learning_INCLUDE_DIRS
    NAMES gnuradio/learning/api.h
    HINTS $ENV{learning_DIR}/include
        ${PC_learning_INCLUDEDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/include
          /usr/local/include
          /usr/include
)

FIND_LIBRARY(
    GR_learning_LIBRARIES
    NAMES gnuradio-learning
    HINTS $ENV{learning_DIR}/lib
        ${PC_learning_LIBDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/lib
          ${CMAKE_INSTALL_PREFIX}/lib64
          /usr/local/lib
          /usr/local/lib64
          /usr/lib
          /usr/lib64
          )

include("${CMAKE_CURRENT_LIST_DIR}/gnuradio-learningTarget.cmake")

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(GR_learning DEFAULT_MSG GR_learning_LIBRARIES GR_learning_INCLUDE_DIRS)
MARK_AS_ADVANCED(GR_learning_LIBRARIES GR_learning_INCLUDE_DIRS)
