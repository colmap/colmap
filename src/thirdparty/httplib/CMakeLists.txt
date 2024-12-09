COLMAP_ADD_LIBRARY(
    NAME colmap_httplib
    SRCS
        httplib.h httplib.cc
    PRIVATE_LINK_LIBS
        OpenSSL::SSL
        OpenSSL::Crypto
)
target_compile_definitions(colmap_httplib PUBLIC CPPHTTPLIB_OPENSSL_SUPPORT)
